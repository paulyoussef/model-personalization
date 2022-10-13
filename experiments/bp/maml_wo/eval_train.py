import copy
import logging
import os
import sys
import torch

from data.data_utils import BPDataset
from trainers.models import LSTMGeneric
from trainers.trainer_bp import train_maml, train_baseline, log_end_results, log_params, eval_test_set
from utils.model_reconstruction_utils import get_best_seed
from utils.utils import set_seed, num_params, set_common_params, compare_params
from data.static_data_utils import get_static_config


if __name__ == '__main__':

    import warnings

    # warnings.simplefilter("error")
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", UserWarning)

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set default type to float64 to increase precision
    torch.set_default_dtype(torch.float64)
    exp = False

    params = {}

    params['dtype'] = torch.float64
    params['dataset'] = "bp"
    params['steps_ahead'] = 3


    params['static_config'] = None
    params['static_dim'] = 0
    params['static_output_dim'] = 0

    params['num_tasks_training'] = 30000
    params['num_tasks_test'] = 30000
    params['device'] = device
    params['model_type'] = 'lstm'
    set_common_params(params,  os.path.basename(__file__), exp)


    params['seed'] =  get_best_seed()


    logging.basicConfig(filename=params['log_dir'] + '.log', level=logging.INFO, force=True,
                        format='%(filename)s:%(asctime)s:%(message)s')
    logging.info('----------start-of-run----------')
    # set seed
    set_seed(params['seed'])
    params['model'] = 'maml'
    tasksets = BPDataset(dataset = params['dataset'], num_tasks=params['num_tasks_training'], meta_train=True, steps_ahead=params['steps_ahead'],
                         static_config= params['static_config'])
    params['num_tasks_training'] = len(tasksets)
    scaler = tasksets.get_scaler()
    set_seed(params['seed'])

    maml = torch.load('./maml_{}.pth'.format(params['seed']))

    results_dict = {}
    params['model'] = 'maml'
    params['save_models'] = True

    copied_maml = copy.deepcopy(maml)
    eval_test_set(maml, scaler, params, results_dict, meta_train=True)
    compare_params(copied_maml, maml)


    log_params(params)
    log_end_results(results_dict, params['num_tasks_test'])
    logging.info('----------end-of-run----------')
