# Reconstruction script for models without static data in input


import datetime
import logging
import os
import re
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmeta.utils.data import BatchMetaDataLoader

from data.data_utils import BPDataset
from data.static_data_utils import get_static_config
from trainers.trainer_bp import eval_on_task, fit_patient
from trainers.trainer_bp import get_train_val_sampler
from utils.utils import set_seed
from trainers.models import APGenerator, amp_fn, phase_fn

from utils.model_reconstruction_utils import evaluate, infos_from_dirs, StaticReprDataset, SineEncoder, \
    nearest_neighbour, SXEncoder, SineEncoder3, overwrite_model_weights_generic, get_pers_repr_generic, get_best_seed

import warnings

# warnings.simplefilter("error")
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params = {}
    params['debugging'] = False
    # Encoder or not # TODO
    params['encoder'] = True

    # Set seed for reproducibility
    seed = 0 #if params['debugging'] else int(sys.argv[1])
    set_seed(seed)
    params['seed'] = seed

    params['underlying_seed'] = get_best_seed() # extracts best seed from ./results.md

    # Encoder
    embedding_dim = 128
    hidden_dim = 256
    learning_rate = 1e-3
    num_epochs = 150
    params['batch_size'] = 32
    # Early stopping
    # if diff in loss < params['early_stop_diff'], stop early
    params['early_stop_diff'] = .0
    # Check val_loss every params['es_epoch'] epochs
    params['es_epoch'] = 10

    # Dataset options
    num_tasks = 24694
    num_test_tasks = 24694
    n = num_tasks
    wanted_sa = 3
    steps_ahead = wanted_sa
    params['dataset'] = 'bp'
    params['with_static'] = False # TODO
    params['static_config'] = get_static_config(128, params['dataset'])



    # Finetuning
    params['ft_points'] = 0
    params['random_order'] = False
    # Corresponds to no finetuning
    min_ft_points = 0
    # Corresponds to finetuning w/ support set
    max_ft_points = 10 # TODO
    params['ft_steps'] = 0 if params['ft_points'] == 0 else 10
    params['repr_params'] = 'head'
    params['syn_pt_ft'] = False # TODO
    # Set default type to float64
    torch.set_default_dtype(torch.float64)
    params['dtype'] = torch.float64

    enc = 'enc' if params['encoder'] else 'no_enc'

    logging.basicConfig(
        filename=os.path.basename(__file__)[:-3] + '_' + enc + '_' + params['dataset'] + '_' + str(wanted_sa) + '.log',
        level=logging.INFO,
        format='%(filename)s:%(asctime)s:%(message)s', force=True)

    if params['debugging']:
        num_tasks = 30
        num_epochs = 1
        logging.disable('INFO')

    ws = 1

    logging.info('seed: {}'.format(seed))
    logging.info('*** started preparing data ***')

    # Where the personalized models are
    dir = './saved_models/eval_train_bp_3/k10-training-24694-test-24694-maml-50-tl-50/mbs-10-adapt-10/maml/{}/'.format(params['underlying_seed'])

    # Data info
    params['dir'] = dir
    params['steps_ahead'] = steps_ahead
    params['training_instances'] = num_tasks
    params['device'] = device
    params['shots'] = 10
    params['save_models'] = False
    params['log_dir'] = './logs_encoder/'
    params['model'] = 'encoder_' + str(params['dataset']) + '_sa' + str(steps_ahead)

    tasksets = BPDataset(dataset=params['dataset'], num_tasks=num_tasks, meta_train=True,
                         steps_ahead=steps_ahead,
                         static_config=params['static_config'], include_static=params['with_static'], random_order=params['random_order'])
    scaler = tasksets.get_scaler()
    n = len(tasksets)
    if params['encoder']:
        # Second pass through to actually encode data
        # Inputs (those might have different lens), because of comorbidities [7:]
        X = []
        # Outputs (personalized representation)
        Y = []


        for i in range(n):
            X.append(tasksets[i].s_oh)
            Y.append(get_pers_repr_generic(i, dir, params['device'], only=params['repr_params']))

        logging.info('*** finished preparing data ***')

        assert (len(X) == len(Y))

        params['enc_input_dim'] = len(X[0])
        params['enc_output_dim'] = len(Y[0])
        params['enc_num_training_instances'] = len(X)


        logging.info('*** started training encoder ***')
        # Train and val sampler for encoder
        train_sampler, val_sampler = get_train_val_sampler(tasksets.num_tasks)

        static_repr_dataset = StaticReprDataset(X, Y)
        train_dataloader = DataLoader(static_repr_dataset, sampler=train_sampler, batch_size=params['batch_size'])
        val_dataloader = DataLoader(static_repr_dataset, sampler=val_sampler, batch_size=params['batch_size'])

        encoder = SineEncoder(input_dim=len(X[0]),
                          hidden_dim=hidden_dim, output_dim=len(Y[0])).to(device)

        # tasksets now contain
        logging.info('embedding dim: {}'.format(embedding_dim))
        logging.info('hidden dim:    {}'.format(hidden_dim))
        logging.info('learning rate:    {}'.format(learning_rate))
        logging.info('training epochs:    {}'.format(num_epochs))
        logging.info('** encoder-begin **')
        logging.info(encoder)
        logging.info('** encoder-end **')

        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
        writer = SummaryWriter('encoder_runs' + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        logging.info('*** starting training encoder ***')
        last_loss = None
        for e in range(num_epochs):
            logging.info('epoch: {}'.format(e))
            if (e > 0 and e % params['es_epoch'] == 0):
                # First time
                if last_loss == None:
                    last_loss = mae
                    logging.info('saving model for the first time!')
                    torch.save(encoder, params['model'] + '.pth')

                # No significant differnce between current and last time in terms of loss, early stop ..!!
                elif (last_loss - mae) < params['early_stop_diff']:
                    logging.info('Early stop at epoch: {} !!!'.format(e))
                    logging.info('Early stop loss: {}, val loss: {}'.format(last_loss, mae))
                    encoder = torch.load(params['model'] + '.pth')
                    break
                # Significant improve in loss
                elif (last_loss - mae) > params['early_stop_diff']:
                    logging.info('Loss improved, saving model and setting loss')
                    torch.save(encoder, params['model'] + '.pth')
                    last_loss = mae
                else:
                    logging.info('Loss did not improve significantly')

            # for each iteration
            for iter, batch in enumerate(train_dataloader):  # num_tasks/batch_size
                # divide the data into support and query sets
                encoder.train()
                train_inputs, train_targets = batch[0].to(device), batch[1].to(device)

                preds = encoder(train_inputs)
                loss = loss_fn(preds, train_targets.double())
                # print('step: {}, loss: {}'.format(e * len(train_dataloader) + iter, loss))

                if iter % 500 == 0:
                    # Train errors
                    rmse_train, mae_train = evaluate(encoder, train_dataloader, params['device'])

                    writer.add_scalar('encoder training rmse',
                                      rmse_train,
                                      e * len(train_dataloader) + iter)

                    writer.add_scalar('encoder training mae',
                                      mae_train,
                                      e * len(train_dataloader) + iter)
                    logging.info('iteration: {},  train mae:    {}'.format(e * len(train_dataloader) + iter, mae_train))
                    logging.info(
                        'iteration: {},  train rmse:    {}'.format(e * len(train_dataloader) + iter, rmse_train))

                    # val errors
                    rmse, mae = evaluate(encoder, val_dataloader, params['device'])

                    writer.add_scalar('encoder val rmse',
                                      rmse,
                                      e * len(train_dataloader) + iter)

                    writer.add_scalar('encoder val mae',
                                      mae,
                                      e * len(train_dataloader) + iter)

                    logging.info('iteration: {},  val mae:    {}'.format(e * len(train_dataloader) + iter, mae))
                    logging.info('iteration: {},  val rmse:    {}'.format(e * len(train_dataloader) + iter, rmse))

                    writer.add_scalar(' train_mae val_mae',
                                      mae_train / mae,
                                      e * len(train_dataloader) + iter)

                if iter > 0 and iter % 100 == 0:
                    for name, param in encoder.named_parameters():
                        if param.requires_grad:
                            writer.add_histogram('encoder_grad/' + name, param.grad, e * len(train_dataloader) + iter)

                            writer.add_histogram('encoder_weight/' + name, param, e * len(train_dataloader) + iter)

                opt.zero_grad()
                loss.backward()
                opt.step()

        logging.info('*** finished training encoder ***')
    logging.info('*** started evaluating on test set ***')
    #torch.save(encoder, './enc_{}.pth'.format(params['underlying_seed']))


    # Now we have the encoder, we can test / evaluate
    test_tasksets = BPDataset(dataset=params['dataset'], num_tasks=num_test_tasks, meta_train=False,
                              steps_ahead=steps_ahead,
                              static_config=params['static_config'], include_static=params['with_static'], random_order=params['random_order'])
    test_tasksets.set_scaler(scaler)
    print('#Test sets:  ', str(len(test_tasksets)))
    logging.info('#instances in test set: {}'.format(len(test_tasksets)))

    #m0 = torch.load(dir + 'maml_model.pth', map_location=device)
    # m0 = torch.load('theta.head_{}.pth'.format(params['underlying_seed'])).to(device)

    # Go through data instances
    results_dict = {}
    dataloader = BatchMetaDataLoader(test_tasksets, batch_size=1, shuffle=False)


    for h in range(min_ft_points, max_ft_points+1):
        m0 = torch.load('./maml_{}.pth'.format(params['underlying_seed'])).to(device)

        # Finetuning
        params['ft_points'] = h
        params['fits'] = [0] if h == 0 else [10]
        params['model'] = 'maml_' + enc + '_' + str(params['dataset']) + '_' + str(params['ft_points'])

        for iter, batch in enumerate(dataloader):  # num_tasks/batch_size
            # for each task in the batch
            effective_batch_size = batch[0].shape[0]
            for i in range(effective_batch_size):
                inputs, outputs = batch[0][i].to(params['dtype']).to(params['device']), batch[1][i].to(
                    params['dtype']).to(params['device'])
            # Get static data
            params['test_instance'] = iter
            t = tasksets[iter]
            all_features = t.s_oh
            all_features = np.expand_dims(all_features, 0)

            # Cloning here already because we might change the model with the encoder
            m_i = m0.clone(first_order=True)

            if params['encoder']:
                with torch.no_grad():
                    # Generate representations from static data
                    repr = encoder(torch.tensor(all_features).to(device))
                    # Overwrite the bias of the last layer
                    m_i = overwrite_model_weights_generic(m_i, repr, params['dtype'], only=params['repr_params'])




            shots = params['shots']
            # first ft_points are for finetuning
            xsupport, ysupport = inputs[:params['ft_points']], outputs[:params['ft_points']]
            # last 'shots' points for evaluating
            xquery, yquery = inputs[shots:], outputs[shots:]

            support = (xsupport, ysupport)
            query = (xquery, yquery)

            if params['syn_pt_ft']:
                # Finetune
                # Construct a semi-artificial data point
                loss_fn = nn.MSELoss()

                m_i.train()
                # Use first instance of test/query set
                x_1 = xquery.detach().cpu().numpy()[0].copy()
                # Set all interventions 0.
                x_1[1:1 + wanted_sa] = (wanted_sa) * [0.]
                # Set target values to input values
                y_1 = steps_ahead * [x_1[0]]


                # perturb target values
                y_1 = y_1 * np.random.uniform(low=0.97, high=1.03, size=(len(y_1),))

                x = torch.tensor(x_1).to(params['dtype']).to(params['device'])
                x = x.reshape(1, -1)
                y = torch.tensor(y_1).to(params['dtype']).to(params['device'])

                for j in range(10):
                    m_i.reset_hidden_state()
                    pred = m_i(x)
                    finetuning_loss = loss_fn(y.flatten(), pred.flatten())
                    m_i.adapt(finetuning_loss)

            eval_on_task(m_i, scaler, support, query, params, results_dict)

    logging.info('** params ***')
    for p, value in params.items():
        if p == 'static_config':
            continue
        logging.info('{} : {}'.format(p, value))
        print('{} : {}'.format(p, value))
    logging.info('** params ***')

    logging.info('** results ***')
    for e in ['mae', 'rmse']:
        logging.info('*** results in {} ***'.format(e.upper()))
        print('*** results in {} ***'.format(e.upper()))

        for k, v in results_dict.items():
            if e in k:
                # logging.info('{}, mean: {} +/- {}'.format(k, np.mean(v), np.std(v)))
                logging.info('| {} | {} | {} |'.format(k, np.round(np.mean(v), 2), np.round(np.std(v), 2)))
                print('| {} | {} | {} |'.format(k, np.round(np.mean(v), 2), np.round(np.std(v), 2)))

        logging.info('*** results in {} ***'.format(e.upper()))
        print('*** results in {} ***'.format(e.upper()))

    logging.info('*** finished evaluating on test set ***')