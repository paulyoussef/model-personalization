# Reconstruction script for models without static data in input


import datetime
import logging
import os
import re
import sys
import numpy as np
import torch
from learn2learn import detach_module
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmeta.utils.data import BatchMetaDataLoader

from data.data_utils import BPDataset
from data.static_data_utils import get_static_config
from trainers.trainer_bp import eval_on_task, fit_patient  # TODO check this
from trainers.trainer_bp import get_train_val_sampler
from utils.utils import set_seed
from trainers.models import APGenerator, amp_fn, phase_fn

from utils.model_reconstruction_utils import evaluate, infos_from_dirs, StaticReprDataset, SineEncoder, \
    nearest_neighbour, SXEncoder, SineEncoder3, overwrite_model_weights_generic, get_pers_repr_generic, get_best_seed

import warnings

# warnings.simplefilter("error")
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

import copy

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params = {}

    # Set seed for reproducibility
    seed = 0 #if params['debugging'] else int(sys.argv[1])
    set_seed(seed)
    params['seed'] = seed


    # Dataset options
    num_tasks = 24694
    num_test_tasks = 100
    n = num_tasks
    wanted_sa = 3
    steps_ahead = 3





    # Set default type to float64
    torch.set_default_dtype(torch.float64)
    params['dtype'] = torch.float64

    enc = 'no_enc'
    params['dataset'] = 'bp'

    logging.basicConfig(
        filename=os.path.basename(__file__)[:-3] + '_' + enc + '_' + params['dataset'] + '_' + str(wanted_sa) + '.log',
        level=logging.INFO,
        format='%(filename)s:%(asctime)s:%(message)s', force=True)

    # Where the personalized models are




    # before_loop_params = copy.deepcopy(params)
    results_dict = {}

    for h in range(0,5):
        # Finetuning
        params['random_order'] = False
        params['underlying_seed'] = get_best_seed()  # extracts best seed from ./results.md

        params['dataset'] = 'bp'
        params['with_static'] = False
        params['static_config'] = get_static_config(128, params['dataset'])
        params['underlying_seed'] = get_best_seed()  # extracts best seed from ./results.md

        # Data info
        params['dir'] = dir
        params['steps_ahead'] = wanted_sa
        params['training_instances'] = num_tasks
        params['device'] = device
        params['shots'] = 10
        params['save_models'] = False
        params['log_dir'] = './logs_encoder/'
        params['model'] = 'encoder_' + str(params['dataset']) + '_sa' + str(steps_ahead)
        params['fits'] = [0]
        tasksets = BPDataset(dataset=params['dataset'], num_tasks=num_tasks, meta_train=True,
                             steps_ahead=steps_ahead,
                             static_config=params['static_config'], include_static=params['with_static'],
                             random_order=params['random_order'])
        scaler = tasksets.get_scaler()




        # if h > 0:
        #     for k, v in before_loop_params.items():
        #         print('old value: {}'.format(v))
        #             print('new value: {}'.format(params[k]))


        test_tasksets = BPDataset(dataset=params['dataset'], num_tasks=num_test_tasks, meta_train=False,
                                  steps_ahead=steps_ahead,
                                  static_config=params['static_config'], include_static=params['with_static'], random_order=params['random_order'])
        test_tasksets.set_scaler(scaler)
        dataloader = BatchMetaDataLoader(test_tasksets, batch_size=1, shuffle=False)


        # Finetuning
        params['ft_points'] = h
        params['ft_steps'] = 10
        params['model'] = 'maml_' + enc + '_' + str(params['dataset']) + '_' + str(params['ft_points'])

        for iter, batch in enumerate(dataloader):  # num_tasks/batch_size
            # for each task in the batch
            effective_batch_size = batch[0].shape[0]
            for i in range(effective_batch_size):
                inputs, targets = batch[0][i].to(params['dtype']).to(params['device']), batch[1][i].to(
                    params['dtype']).to(params['device'])
            # Get static data
            params['test_instance'] = iter
            m0 = torch.load('./maml_{}.pth'.format(params['underlying_seed'])).to(device)
            # print(m0.head[3].bias)

            m_i = m0.clone(first_order=True)
            # print(m_i.head[3].bias)

            # if iter == 23:
            #     print(inputs)
            if params['ft_points'] > 0:
                # inputs, targets
                ft_inputs, ft_targets = inputs[:params['ft_points']], targets[:params['ft_points']]
                for _ in range(params['ft_steps']):
                    assert (len(ft_inputs) == h)
                    assert (len(ft_targets) == h)
                    fit_patient(m_i, (ft_inputs, ft_targets), params)

            eval_on_task(m_i, scaler, (inputs, targets), params, results_dict)

    logging.info('** params ***')
    for p, value in params.items():
        if p == 'static_config':
            continue
        logging.info('{} : {}'.format(p, value))
        print('{} : {}'.format(p, value))
    logging.info('** params ***')

    logging.info('** results ***')
    logging.info('*** results in MAE ***')
    print('*** results in MAE ***')

    for k, v in results_dict.items():
        if 'mae' in k:
            assert(len(v) == len(test_tasksets))
            logging.info('{}, mean: {} +/- {}'.format(k, np.mean(v), np.std(v)))
            print('| {} | {} | {} |'.format(k, np.round(np.mean(v), 2), np.round(np.std(v), 2)))

    logging.info('*** results in RMSE ***')
    print('*** results in RMSE ***')

    for k, v in results_dict.items():
        if 'rmse' in k:
            assert(len(v) == len(test_tasksets))
            logging.info('{}, mean: {} +/- {}'.format(k, np.mean(v), np.std(v)))
            print('| {} | {} | {} |'.format(k, np.round(np.mean(v), 2), np.round(np.std(v), 2)))
    logging.info('*** results ***')

    logging.info('*** finished evaluating on test set ***')