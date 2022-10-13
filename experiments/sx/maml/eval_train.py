import logging
import os
import sys
import torch
import copy
from trainers.models import APGenerator, amp_fn, phase_fn, MLPAnil
from trainers.trainer_sine import train_maml, train_baseline, eval_test_set, log_end_results, log_params
from utils.model_reconstruction_utils import get_best_seed
from utils.utils import set_seed, set_common_params, compare_params
from data.data_utils import BPDataset

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set default type to float64 to increase precision
    params = {}
    torch.set_default_dtype(torch.float64)
    exp = False

    # Data params
    params['dataset'] = 'sx'
    params['with_static'] = True
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
    params['seed'] = 0 if exp else get_best_seed() # Shouldn't make a difference, but just to get the best underlying seed
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
                             random_order=params['random_order'])
    params['num_tasks_training'] = training_set.num_tasks

    maml = torch.load('./theta_{}.pth'.format(params['seed']))

    results_dict = {}
    params['model'] = 'maml'
    params['save_models'] = True

    copied_params = copy.deepcopy(maml)
    eval_test_set(training_set, maml, None, params, results_dict, meta_train=True)
    compare_params(copied_params, maml)

    log_params(params)
    log_end_results(results_dict, params['num_tasks_test'])
    logging.info('----------end-of-run----------')
