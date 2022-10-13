import copy
import logging
import os
import sys
import torch

from trainers.models import LSTMStaticAnil, LSTMOP
from trainers.trainer_bp import train_maml, train_baseline, log_end_results, log_params, eval_test_set
from utils.utils import set_seed, num_params, set_common_params, compare_params
from data.static_data_utils import get_static_config


if __name__ == '__main__':

    import warnings

    # warnings.simplefilter("error")
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", UserWarning)

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set default type to float64 to increase precision
    torch.set_default_dtype(torch.float64)
    exp = True

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


    params['seed'] = 0 if exp else int(sys.argv[1])


    logging.basicConfig(filename=params['log_dir'] + '.log', level=logging.INFO, force=True,
                        format='%(filename)s:%(asctime)s:%(message)s')
    logging.info('----------start-of-run----------')
    # set seed
    set_seed(params['seed'])
    params['model'] = 'maml'

    scaler, maml = train_maml(
        LSTMOP(device, input_dim=params['static_dim']+ params['window_size'] + params['steps_ahead'],
                   hidden_dim=params['hidden_dim'],
                   output_dim=params['steps_ahead'], static_input_dim= abs(params['static_dim']), static_output_dim= params['static_output_dim'] ), params)
    # set seed again to make sure the model is initialized similarly
    set_seed(params['seed'])

    # baseline = train_baseline(
    #     scaler,
    #     LSTMStaticAnil(device, input_dim=params['static_dim'] + params['window_size'] + params['steps_ahead'],
    #                    hidden_dim=params['hidden_dim'],
    #                    output_dim=params['steps_ahead'], static_input_dim=abs(params['static_dim']), static_output_dim= params['static_output_dim']), params)
    #
    # params['num_params'] = num_params(baseline)
    # params['model'] = num_params(baseline)


    if not params['debugging']:
        torch.save(maml, './maml_{}.pth'.format(params['seed']))
        # torch.save(baseline, './baseline_{}.pth'.format(params['seed']))

    results_dict = {}
    params['model'] = 'maml'
    copied_maml = copy.deepcopy(maml)
    eval_test_set(maml, scaler, params, results_dict)
    compare_params(copied_maml, maml)

    # params['model'] = 'baseline'
    # eval_test_set(baseline, scaler, params, results_dict)

    log_params(params)
    log_end_results(results_dict, params['num_tasks_test'])
    logging.info('----------end-of-run----------')
