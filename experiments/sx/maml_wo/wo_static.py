import logging
import os
import sys
import torch
import copy
from trainers.models import MLP, APGenerator, amp_fn, phase_fn, MLPAnil
from trainers.trainer_sine import train_maml, train_baseline, eval_test_set, log_end_results, log_params
from utils.utils import set_seed, set_common_params, compare_params
from data.data_utils import BPDataset

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set default type to float64 to increase precision
    params = {}
    torch.set_default_dtype(torch.float64)
    exp = False

    # Data params
    params['dataset'] = 'sx'
    params['with_static'] = False
    params['random_order'] = True
    params['noise_std'] = .5
    params['static_config'] = None
    params['static_dim'] = 0
    params['num_tasks_training'] = 8000
    params['num_tasks_test'] = 2000

    params['device'] = device
    params['model_type'] = 'mlp'
    # Seed
    #######
    params['seed'] = 0 if exp else int(sys.argv[1])
    params['steps_ahead'] = 1

    set_common_params(params,  os.path.basename(__file__), exp)

    logging.basicConfig(filename=params['log_dir'] + '.log', level=logging.INFO, force=True,
                        format='%(filename)s:%(asctime)s:%(message)s')
    logging.info('----------start-of-run----------')
    # set seed
    set_seed(params['seed'])
    params['model'] = 'maml'
    training_set = BPDataset(dataset=params['dataset'], num_tasks=params['num_tasks_training'], meta_train=True,
                             steps_ahead=params['steps_ahead'],
                             static_config=params['static_config'], noise_std=params['noise_std'],
                             random_order=params['random_order'], include_static=params['with_static'])
    params['num_tasks_training'] = training_set.num_tasks

    _, maml = train_maml(training_set,
                         MLPAnil(input_dim=1,
                             hidden_dim=params['hidden_dim'],
                             output_dim=1).to(device), params)

    if not params['debugging']:
        torch.save(maml, './theta_{}.pth'.format(params['seed']))

    # set seed again to make sure the model is initialized similarly
    set_seed(params['seed'])

    params['model'] = 'baseline'

    baseline = train_baseline(training_set,
                              MLPAnil(input_dim=1,
                                  hidden_dim=params['hidden_dim'],
                                  output_dim=1).to(device), params)

    if not params['debugging']:
        torch.save(baseline, './tl_{}.pth'.format(params['seed']))

    test_set = BPDataset(dataset=params['dataset'], num_tasks=params['num_tasks_test'], meta_train=False,
                         steps_ahead=params['steps_ahead'],
                         static_config=params['static_config'], noise_std=params['noise_std'],
                         random_order=params['random_order'], include_static=params['with_static'])

    results_dict = {}
    params['model'] = 'maml'
    copied_params = copy.deepcopy(maml)
    eval_test_set(test_set, maml, None, params, results_dict)
    compare_params(copied_params, maml)


    params['model'] = 'baseline'
    eval_test_set(test_set, baseline, None, params, results_dict)
    log_params(params)
    log_end_results(results_dict, params['num_tasks_test'])
    logging.info('----------end-of-run----------')
