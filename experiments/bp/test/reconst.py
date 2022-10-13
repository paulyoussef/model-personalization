
import os
import torch
import numpy as np
from torchmeta.utils.data import BatchMetaDataLoader

from data.data_utils import BPDataset
from trainers.trainer_bp import eval_on_task, eval_test_set
import warnings

# warnings.simplefilter("error")
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = {}
    # general
    params['dtype'] = torch.float64
    torch.set_default_dtype(params['dtype'])

    params['device'] = device
    params['save_models'] = False
    # dataset
    params['dataset'] = 'bp'
    params['num_tasks_training'] = 100
    params['num_tasks_test'] = 10
    params['steps_ahead'] = 3
    params['static_config'] = None

    training_set = BPDataset(dataset=params['dataset'], num_tasks=params['num_tasks_training'], meta_train=True,
                             steps_ahead=params['steps_ahead'],
                             static_config=params['static_config'])
    params['num_tasks_training'] = len(training_set)
    scaler = training_set.get_scaler()


    test_set = BPDataset(dataset=params['dataset'], num_tasks=params['num_tasks_test'], meta_train=False,
                             steps_ahead=params['steps_ahead'],
                             static_config=params['static_config'])
    test_set.set_scaler(scaler)

    results_dict = {}

    dataloader = BatchMetaDataLoader(test_set, batch_size=1, shuffle=False)

    model = torch.load('maml_0.pth')
    print(model.head[2].bias)

    # finetuning
    params['shots'] = 10
    params['fits'] = [10]
    params['model'] = 'maml_0'
    eval_test_set(model, scaler, params, results_dict, meta_train=False)
    # for iter, batch in enumerate(dataloader):  # num_tasks/batch_size
    #     # for each task in the batch
    #     effective_batch_size = batch[0].shape[0]
    #     assert (effective_batch_size == 1)
    #     for i in range(effective_batch_size):
    #         params['test_instance'] = iter
    #         inputs = batch[0][i].to(params['dtype']).to(params['device'])
    #         outputs = batch[1][i].to(params['dtype']).to(params['device'])
    #
    #
    #         xsupport, ysupport = inputs[:params['shots']], outputs[:params['shots']]
    #         xquery, yquery = inputs[params['shots']:], outputs[params['shots']:]
    #
    #         support = (xsupport, ysupport)
    #         query = (xquery, yquery)
    #
    #         eval_on_task(model, scaler, support, query, params, results_dict)

    print('*** results in MAE ***')
    for k, v in results_dict.items():
        if 'mae' in k:
            assert (len(v) == params['num_tasks_test'])
            print('| {} | {} | {} |'.format(k, np.mean(v), np.std(v)))

    print('*** results in RMSE ***')
    for k, v in results_dict.items():
        if 'rmse' in k:
            assert (len(v) == params['num_tasks_test'])
            print('| {} | {} | {} |'.format(k, np.mean(v), np.std(v)))

    print ('*** params ***')
    for k, v in params.items():
        print('{}: {}'.format(k, v))
