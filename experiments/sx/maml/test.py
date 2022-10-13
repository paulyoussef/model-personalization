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

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set default type to float64 to increase precision
    params = {}
    torch.set_default_dtype(torch.float64)
    exp = False


    logging.info('----------end-of-run----------')
