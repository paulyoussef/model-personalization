import os
from torchmeta.utils.data import BatchMetaDataLoader

from data.data_utils import BPDataset
from data.static_data_utils import get_static_config
import torch
import numpy as np

torch.set_default_dtype(torch.float64)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
params = {}
params['dtype'] = torch.float64
params['device'] = device
params['dataset'] = 'bp'
params['with_static'] = False
params['static_config'] = get_static_config(128, params['dataset'])
params['num_tasks_training'] = 30000
params['steps_ahead'] = 3
random_order=params['random_order'] = False

tasksets = BPDataset(dataset = params['dataset'], num_tasks=params['num_tasks_training'], meta_train=True, steps_ahead=params['steps_ahead'],
                         static_config= params['static_config'], random_order=params['random_order'])

test_tasksets = BPDataset(dataset=params['dataset'], num_tasks=24694, meta_train=False,
                          steps_ahead=params['steps_ahead'],
                          static_config=params['static_config'], include_static=params['with_static'],
                          random_order=params['random_order'])

params['num_tasks_training'] = len(tasksets)
scaler = tasksets.get_scaler()

dataloader = BatchMetaDataLoader(test_tasksets, batch_size=1, shuffle=False)

for iter, batch in enumerate(dataloader):  # num_tasks/batch_size
    # for each task in the batch
    effective_batch_size = batch[0].shape[0]
    inputs_np, targets_np = test_tasksets[iter].get_scaled_inputs_targets()
    for i in range(effective_batch_size):
        inputs, outputs = batch[0][i].to(params['dtype']).to(params['device']), batch[1][i].to(params['dtype']).to(params['device'])

    print('inputs: ---{}---'.format(iter))
    print(inputs.numpy())
    print(np.array(inputs_np))
    print('outputs:---{}---'.format(iter))
    print(np.array(outputs))
    print(targets_np)

    if iter == 10:
        break

