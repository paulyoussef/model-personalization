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

from data.data_utils import BPDataset
from trainers.trainer_sine_anil import eval_on_task  # TODO check this
from trainers.trainer_sine_anil import get_train_val_sampler
from utils.utils import set_seed
from trainers.models import APGenerator, amp_fn, phase_fn

from utils.model_reconstruction_utils import evaluate, infos_from_dirs, StaticReprDataset, SineEncoder, \
    nearest_neighbour, SXEncoder, SineEncoder3, overwrite_model_weights_generic, get_pers_repr_generic, get_best_seed

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params = {}
    params['debugging'] = False
    # Encoder or not
    params['encoder'] = True

    # Set seed for reproducibility
    seed = 0  # if params['debugging'] else int(sys.argv[1])
    set_seed(seed)
    params['seed'] = seed

    params['underlying_seed'] = get_best_seed()  # extracts best seed from ./results.md

    # Encoder
    embedding_dim = 128
    hidden_dim = 256
    learning_rate = 2e-5 # 1e-3
    num_epochs = 150
    params['batch_size'] = 32
    # Early stopping
    # if diff in loss < params['early_stop_diff'], stop early
    params['early_stop_diff'] = .0
    # Check val_loss every params['es_epoch'] epochs
    params['es_epoch'] = 10

    # Dataset options
    num_tasks = 8000
    num_test_tasks = 2000
    n = num_tasks
    wanted_sa = 1
    steps_ahead = wanted_sa
    params['dataset'] = 'sx'
    params['with_static'] = True
    params['random_order'] = True
    params['swap_support_query'] = False
    params['noise_std'] = .5
    params['static_config'] = None

    # Finetuning
    params['ft_points'] = 0
    params['ft_steps'] = 0 if params['ft_points'] == 0 else 10

    # Set default type to float64
    torch.set_default_dtype(torch.float64)
    params['dtype'] = torch.float64

    enc = 'enc' if params['encoder'] else 'no_enc'
    finetuning = 'ft' if params['ft_points'] > 0 else 'no_ft'

    logging.basicConfig(
        filename=os.path.basename(__file__)[:-3] + '_' + enc + '_' + finetuning + '_' + params['dataset'] + '_' + str(
            wanted_sa) + '.log',
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
    dir = './saved_models/eval_train_sx_1/k10-training-8000-test-8000-maml-50-tl-50/mbs-10-adapt-10/maml-{}-False/'.format(
        params['underlying_seed'])

    # Data info
    params['dir'] = dir
    params['steps_ahead'] = steps_ahead
    params['training_instances'] = num_tasks
    params['device'] = device
    params['shots'] = 10
    params['save_models'] = False
    params['fits'] = [0]
    params['log_dir'] = './logs_encoder/'
    params['model'] = 'encoder_' + str(params['dataset']) + '_sa' + str(steps_ahead)

    if params['encoder']:

        tasksets = BPDataset(dataset=params['dataset'], num_tasks=num_tasks, meta_train=True,
                             steps_ahead=steps_ahead,
                             static_config=params['static_config'], include_static=params['with_static'],
                             noise_std=params['noise_std'], random_order=params['random_order'])

        # Second pass through to actually encode data
        # Inputs (those might have different lens), because of comorbidities [7:]
        X = []
        # Outputs (personalized representation)
        Y = []

        for i in range(n):
            X.append(tasksets[i].input_nn)
            Y.append(get_pers_repr_generic(i, dir, params['device']))

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
    torch.save(encoder, './enc_{}.pth'.format(params['underlying_seed']))

    # Now we have the encoder, we can test / evaluate
    test_tasksets = BPDataset(dataset=params['dataset'], num_tasks=num_test_tasks, meta_train=False,
                              steps_ahead=steps_ahead,
                              static_config=params['static_config'], include_static=params['with_static'],
                              noise_std=params['noise_std'], random_order=params['random_order'])

    print('#Test sets:  ', str(len(test_tasksets)))
    logging.info('#instances in test set: {}'.format(len(test_tasksets)))

    # m0 = torch.load(dir + 'maml_model.pth', map_location=device)
    m0 = torch.load('theta_{}.pth'.format(params['underlying_seed'])).to(device)
    # Go through data instances
    results_dict = {}
    for i, t in enumerate(test_tasksets):
        # Get static data
        params['test_instance'] = i
        all_features = t.input_nn
        all_features = np.expand_dims(all_features, 0)

        m_i = m0.clone()

        if params['encoder']:
            with torch.no_grad():
                # Generate representations from static data
                repr = encoder(torch.tensor(all_features).to(device))
                # Overwrite the bias of the last layer
                m_i = overwrite_model_weights_generic(m_i, repr, params['dtype'])
        # Inputs have shape (20, 128)
        inputs, targets = t.get_scaled_inputs_targets()
        inputs = torch.tensor(np.array(inputs)).to(params['dtype']).to(params['device'])
        targets = torch.tensor(np.array(targets)).to(params['dtype']).to(params['device'])

        if params['ft_points'] > 0:
            # loss function
            loss_fn = nn.MSELoss(reduction='mean')
            # inputs, targets
            ft_inputs, ft_targets = inputs[:params['ft_points']], targets[:params['ft_points']]

            for _ in range(params['ft_steps']):
                ft_preds = m_i(ft_inputs)
                loss = loss_fn(ft_preds, ft_targets)
                m_i.adapt(loss)

        eval_on_task(m_i, None, (inputs, targets), params, results_dict)

    logging.info('** params ***')
    for p, value in params.items():
        if p == 'static_config':
            continue
        logging.info('{} : {}'.format(p, value))
        print('{} : {}'.format(p, value))
    logging.info('** params ***')

    logging.info('** results ***')
    for k, v in results_dict.items():
        print(k, len(v))
        print('| {} | {} | {} |'.format(k, np.round(np.mean(v), 2), np.round(np.std(v), 2)))
        logging.info('{}, mean: {} +/- {}'.format(k, np.mean(v), np.std(v)))
    logging.info('** results ***')

    logging.info('*** finished evaluating on test set ***')

    exit(0)
