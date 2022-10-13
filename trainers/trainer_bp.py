import datetime
import logging

import learn2learn as l2l
import numpy as np
import torch
from learn2learn import detach_module
from torch import nn, optim
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmeta.utils.data import BatchMetaDataLoader

from data.data_utils import BPDataset
from utils.utils import unscale, evaluate


def save_model(model, params):
    torch.save(model, params['log_dir'] + '_' + params['model'] + '_' + str(params['seed'])+ '.pth')

def load_model(params):
    return torch.load(params['log_dir'] + '_' + params['model'] + '_' + str(params['seed'])+ '.pth')


def get_train_val_sampler(num_tasks):
    indices = list(range(num_tasks))
    np.random.seed(43)
    np.random.shuffle(indices)
    split = int(0.75 * num_tasks)
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    logging.info('Using {} instances for training'.format(len(train_indices)))
    logging.info('Using {} instances for validation'.format(len(val_indices)))

    return train_sampler, val_sampler


def train_maml(
        model,
        params,
):
    # load the dataset
    num_tasks = params['num_tasks_training']
    tasks_per_batch = params['meta_batch_size']
    adapt_lr = params['adapt_lr']
    meta_lr = params['meta_lr']
    adapt_steps = params['adapt_steps']
    num_epochs = params['epochs_maml']
    shots = params['shots']
    params['maml_bias'] = str(model.head[2].bias).split('\n')[1]

    tasksets = BPDataset(dataset = params['dataset'], num_tasks=num_tasks, meta_train=True, steps_ahead=params['steps_ahead'],
                         static_config= params['static_config'])
    params['num_tasks_training'] = tasksets.num_tasks
    scaler = tasksets.get_scaler()
    # dataloader = BatchMetaDataLoader(tasksets, batch_size=tasks_per_batch)

    train_sampler, val_sampler = get_train_val_sampler(tasksets.num_tasks)

    train_dataloader = BatchMetaDataLoader(tasksets, batch_size=tasks_per_batch, sampler=train_sampler, shuffle=False)
    val_dataloader = BatchMetaDataLoader(tasksets, batch_size=len(val_sampler), sampler=val_sampler, shuffle=False)

    # create the model
    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True).to(params['device'])
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = nn.MSELoss(reduction='mean')
    params['tb_dir'] = 'runs/' + params['log_dir'] + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(params['tb_dir'])
    last_loss = None
    with torch.backends.cudnn.flags(enabled=False):
        for e in range(num_epochs):
            logging.info('Epoch:  {}'.format(e + 1))

            if (e > 0 and e % params['es_epoch'] == 0):
                # First time
                if last_loss == None:
                    last_loss = val_loss
                    logging.info('saving model for the first time!')
                    save_model(maml, params)

                # No significant differnce between current and last time in terms of loss, early stop ..!!
                elif (last_loss - val_loss) < params['early_stop_diff']:
                    logging.info('Early stop at epoch: {}'.format(e))
                    logging.info('Early stop loss: {}, val loss: {}'.format(last_loss, val_loss))
                    maml_from_file = load_model(params)
                    return scaler, maml_from_file
                # Significant improve in loss
                elif (last_loss - val_loss) > params['early_stop_diff']:
                    logging.info('Loss improved, saving model and setting loss')
                    save_model(maml, params)
                    last_loss = val_loss
                else:
                    logging.info('Loss did not improve significantly')


            # for each iteration
            for iter, batch in enumerate(train_dataloader):  # num_tasks/batch_size
                meta_train_loss = 0.0

                # for each task in the batch
                effective_batch_size = batch[0].shape[0]

                for i in range(effective_batch_size):

                    learner = maml.clone() #.to(params['device'])
                    learner.train()

                    # divide the data into support and query sets
                    train_inputs, train_targets = batch[0][i].to(params['dtype']).to(params['device']), batch[1][i].to(params['dtype']).to(params['device'])
                    x_support, y_support = train_inputs[:shots], train_targets[:shots]
                    x_query, y_query = train_inputs[shots:], train_targets[shots:]

                    for _ in range(adapt_steps):  # adaptation_steps
                        learner.reset_hidden_state()
                        support_preds = learner(x_support)
                        support_loss = lossfn(support_preds.flatten(), y_support.flatten())
                        learner.adapt(support_loss)
                    # reset hidden state before continuing with the query ( for fair comparison with the models that do not see a query)
                    learner.reset_hidden_state()
                    query_preds = learner(x_query)
                    query_loss = lossfn(query_preds.flatten(), y_query.flatten())
                    meta_train_loss += query_loss

                meta_train_loss = meta_train_loss / effective_batch_size

                opt.zero_grad()
                meta_train_loss.backward()
                opt.step()

                if iter % 100 == 0:
                    # ...log the running loss
                    writer.add_scalar('meta train loss',
                                      meta_train_loss,
                                      e * len(train_dataloader) + iter)
                    print('Iteration:', iter, 'Meta Train Loss', meta_train_loss.item())
                    logging.info('Iteration: {}, Meta Train Loss {}'.format(iter, meta_train_loss.item()))

                    if (iter > 0 or e > 1):
                        for name, param in maml.named_parameters():
                            if param.requires_grad:
                                writer.add_histogram('maml_weight/' + name, param, e * len(train_dataloader) + iter)
                                writer.add_histogram('maml_grad/' + name, param.grad, e * len(train_dataloader) + iter)

                    # Validation loss
                    val_loss = 0.0
                    with torch.no_grad():
                        for _, val_batch in enumerate(val_dataloader):
                            maml.reset_hidden_state(val_batch[0].shape[0])
                            val_inputs, val_targets = val_batch[0].to(params['dtype'])[:,shots:,:].to(params['device']), val_batch[1].to(params['dtype'])[:,shots:,:].to(params['device'])

                            val_preds = maml(val_inputs)
                            val_loss += lossfn(val_preds, val_targets)
                    logging.info('Iteration: {}, val loss {}'.format(iter, val_loss.item()))
                    writer.add_scalar('maml val loss',
                                      val_loss.item(),
                                      e * len(train_dataloader) + iter)

    return scaler, maml


def train_baseline(scaler,
                   model,
                   params
                   ):
    # load the dataset
    # The same dataset used for the MAML experiment
    logging.info('-------start-baseline-------')
    num_epochs = params['epochs_baseline']
    num_tasks = params['num_tasks_training']
    shots = params['shots']
    batch_size = params['batch_size']
    params['baseline_bias'] = str(model.head[2].bias).split('\n')[1] # str(model.head[6].bias).split('\n')[1]

    lr = params['lr']
    tasksets = BPDataset(dataset = params['dataset'], num_tasks=num_tasks, meta_train=True,
                         steps_ahead=params['steps_ahead'],
                         static_config= params['static_config'])
    params['num_tasks_training'] = tasksets.num_tasks
    # Use scaler from MAML experiment to avoid calculating again
    tasksets.set_scaler(scaler)

    train_sampler, val_sampler = get_train_val_sampler(tasksets.num_tasks)

    train_dataloader = BatchMetaDataLoader(tasksets, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    val_dataloader = BatchMetaDataLoader(tasksets, batch_size=len(val_sampler), sampler=val_sampler, shuffle=False)

    baseline = model.to(params['device'])
    opt = optim.Adam(baseline.parameters(), lr)
    lossfn = nn.MSELoss(reduction='mean')
    writer = SummaryWriter(params.get('tb_dir', 'runs/' + params['log_dir'] + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    last_loss = None
    for e in range(num_epochs):
        logging.info('Epoch:  {}'.format(e))

        if (e > 0 and e % params['es_epoch'] == 0):
            # First time
            if last_loss == None:
                last_loss = val_loss
                logging.info('saving model for the first time!')
                save_model(baseline, params)

            # No significant differnce between current and last time in terms of loss, early stop ..!!
            elif (last_loss - val_loss) < params['early_stop_diff']:
                logging.info('Early stop at epoch: {}'.format(e))
                logging.info('Early stop loss: {}, val loss: {}'.format(last_loss, val_loss))
                baseline = load_model(params)

                return baseline
            # Significant improve in loss
            elif (last_loss - val_loss) > params['early_stop_diff']:
                logging.info('Loss improved, saving model and setting loss')
                save_model(baseline, params)
                last_loss = val_loss
            else:
                logging.info('Loss did not improve significantly')


        # for each iteration
        for iter, batch in enumerate(train_dataloader):  # num_tasks/batch_size
            # divide the data into support and query sets
            baseline.train()
            baseline.reset_hidden_state(batch[0].shape[0])

            train_inputs, train_targets = batch[0].to(params['dtype']).to(params['device']), batch[1].to(params['dtype']).to(params['device'])

            preds = baseline(train_inputs)
            loss = lossfn(preds, train_targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if iter % 100 == 0:
                print('iteration:', iter, 'baseline train loss', loss.item())
                logging.info('iteration: {}, baseline train loss {}'.format(iter, loss.item()))

                writer.add_scalar('baseline train loss',
                                  loss,
                                  e * len(train_dataloader) + iter)


                for name, param in baseline.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram('baseline_weight/' + name, param, e * len(train_dataloader) + iter)
                        writer.add_histogram('baseline_grad/' + name, param.grad, e * len(train_dataloader) + iter)
                val_loss = 0.0
                with torch.no_grad():
                    for _, val_batch in enumerate(val_dataloader):
                        baseline.reset_hidden_state(val_batch[0].shape[0])

                        val_inputs, val_targets = val_batch[0].to(params['dtype'])[:,shots:,:].to(params['device']), val_batch[1].to(params['dtype'])[:,shots:,:].to(params['device'])

                        val_preds = baseline(val_inputs)
                        val_loss += lossfn(val_preds, val_targets)
                logging.info('Iteration: {}, baseline val loss {}'.format(iter, val_loss.item()))
                writer.add_scalar('baseline val loss',
                                  val_loss.item(),
                                  e * len(train_dataloader) + iter)
    logging.info('--------end-baseline-------')
    return baseline


def eval_test_set(model, scaler, params, results_dict, meta_train=False):
    tasksets = BPDataset(dataset = params['dataset'], num_tasks=params['num_tasks_test'], meta_train=meta_train,
                         steps_ahead=params['steps_ahead'],
                         static_config= params['static_config'], include_static=params['with_static'])
    # Inputs are scaled now
    tasksets.set_scaler(scaler)

    params['num_tasks_test'] = tasksets.num_tasks
    logging.info('test_num_tasks = {}'.format(tasksets.num_tasks))
    # At test time we always use a batch size of 1
    dataloader = BatchMetaDataLoader(tasksets, batch_size=1, shuffle=False)

    for iter, batch in enumerate(dataloader):  # num_tasks/batch_size
        # for each task in the batch
        effective_batch_size = batch[0].shape[0]
        for i in range(effective_batch_size):
            params['test_instance'] = iter
            inputs = batch[0][i].to(params['dtype']).to(params['device'])
            outputs = batch[1][i].to(params['dtype']).to(params['device'])

            shots = params['shots']
            xsupport, ysupport = inputs[:shots], outputs[:shots]
            xquery, yquery = inputs[shots:], outputs[shots:]

            support = (xsupport, ysupport)
            query = (xquery, yquery)

            eval_on_task(model, scaler, support, query, params, results_dict)


def clone_model(model, params):
    weights_path = 'model_tmp_' + params['log_dir'] + '_' + str(params['seed']) + '.pth'
    torch.save(model, weights_path)
    return torch.load(weights_path)


def eval_on_task(model, scaler, support, query, params, results_dict):
    shots = params['shots']

    # train_inputs, train_targets = test_instance[0], test_instance[1]
    fit_res = fit_and_eval(model, support, query, params)
    xquery, yquery = query
    assert (len(xquery) == len(yquery) == shots)
    # Evaluation is done
    # We can unscale
    # 1 refers to the window size here
    xtest, ytest = unscale(scaler, xquery.detach().cpu()[:,:params['steps_ahead']+1], yquery.detach().cpu() )
    # xtest, ytest = train_inputs[shots:], train_targets[shots:]

    for n, res, loss in fit_res:
        # Unscale results
        scaled_preds = np.squeeze(res.detach().cpu().numpy(), axis=0)  # np.squeeze(res.detach().numpy(), axis=-1)
        # Note: xtest has already been unscaled, but we are not using it anymore so we don't care
        _, unscaled_preds = unscale(scaler, xtest, scaled_preds)

        # results = evaluate(ytest, unscaled_preds)

        evaluate_and_log(ytest, unscaled_preds, params, results_dict, n)
        # for m in ['mae', 'rmse']:
        #     compound_key = m + ' ' + params['model'] + ' ' + str(n)
        #     results_dict[compound_key] = results_dict.get(compound_key, list()) + [float(results[m])]



def fit_and_eval(m, support, query, params):
    fits = params['fits']

    xtrain, ytrain = support
    xtest, ytest = query

    if params['save_models']:
        from pathlib import Path
        # k= number of instances used for training - size of training set (# tasks) - experiment (e.g. maml, fsl)
        models_path = './saved_models/{}/k{}-training-{}-test-{}-maml-{}-tl-{}/mbs-{}-adapt-{}/{}/{}/'.format(
            params['log_dir'],
            params['shots'],
            params['num_tasks_training'],
            params['num_tasks_test'],
            params['epochs_maml'],
            params['epochs_baseline'],
            params['meta_batch_size'],
            params['adapt_steps'],
            params['model'],
            params['seed'])

        Path(models_path).mkdir(parents=True, exist_ok=True)
        torch.save(m, models_path + 'maml_model.pth')

    if 'baseline' in params['model']:
        model = clone_model(m, params) # m.clone()
    elif 'maml' in params['model']:
        model = m.clone(first_order=True)
    else:
        raise ValueError

    def get_loss(res):
        return nn.MSELoss(reduction='mean')(res.flatten(), ytest.flatten())
        # return F.mse_loss(res, V(ytest[:, None]).unsqueeze(1)).cpu().data.numpy()  # [0]

    fit_res = []
    # Evaluating the model without doing steps
    if 0 in fits:
        model.reset_hidden_state()
        model.eval()
        with torch.no_grad():
            results = model(xtest)
        results = torch.nan_to_num(results)
        fit_res.append((0, results, get_loss(results)))

    for i in range(np.max(fits)):
        fit_patient(model, (xtrain, ytrain), params)
        if i + 1 in fits:
            if params['save_models']:
                torch.save(model, models_path + str(params['test_instance']) + '_' + str(i + 1) + '.pth')


            model.eval()
            # Rest hidden state before evaluating
            model.reset_hidden_state()
            with torch.no_grad():
                results = model(xtest)
            results = torch.nan_to_num(results)
            fit_res.append(
                (
                    i + 1,
                    results,
                    get_loss(results)
                )
            )

    return fit_res


def fit_patient(model, train_data, params):
    model.reset_hidden_state()
    model.train()
    x, y = train_data
    if 'baseline' in params['model']:
        opt = optim.Adam(model.parameters(), params['baseline_finetune_lr'])
        loss = nn.MSELoss(reduction='mean')(model(x).flatten(), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    elif 'maml' in params['model']:
        loss = nn.MSELoss(reduction='mean')(model(x).flatten(), y.flatten())
        model.adapt(loss)
    else:
        raise ValueError

    return loss.data.cpu().numpy()


def log_params(params):
    logging.info('-----params-----')
    for k, v in params.items():
        if k == 'static_config':
            continue
        logging.info(k + ' : ' + str(v))
    logging.info('-----params-----')


def log_end_results(results, test_size):
    for k, v in results.items():
        logging.info('{} : {} : {}'.format(k, np.mean(v), np.std(v)))
        print('{} : {}'.format(k, np.mean(v)))
        assert (len(v) == test_size)


def evaluate_and_log(ytrue, preds, params, results_dict, steps):
    results = evaluate(ytrue, preds)
    for m in ['mae', 'rmse']:
        compound_key = m + ' ' + params['model'] + ' ' + str(steps)
        results_dict[compound_key] = results_dict.get(compound_key, list()) + [float(results[m])]
