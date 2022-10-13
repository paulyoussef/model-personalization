import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from data.data_utils import BPDataset

def encode_age(age):
    return float(age) / 100.


def encode_gender(g):
    if g.lower() == 'f':
        return [1., 0.]
    elif g.lower() == 'm':
        return [0., 1.]
    else:
        raise ValueError

class StaticDataset(Dataset):
    def __init__(self, X):
        self.x = X

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.x[idx])


def oh_encode_words(most_common_cmrbdts):
    '''
    creates a mapping between most common cmrbdts (and ethnicities) and oh vectors
    :param most_common_cmrbdts:
    :return: a dictionary
    '''
    # Add other
    most_common_cmrbdts['other'] = -1
    oh_dict = {}
    vec_len = len(list(most_common_cmrbdts.keys()))
    for i, k in enumerate(list(most_common_cmrbdts.keys())):
        oh_vec = np.zeros(vec_len)
        oh_vec[i] = 1.
        oh_dict[k] = oh_vec
    return oh_dict


def encode_cmrbdts_eths(cmrbdts, e, encoded_keys, cmrbdts_eths_oh):
    '''

    :param cmrbdts: cmrbdts (raw)
    :param e: ethnicity
    :param encoded_keys: words we have in our dictionary (x most common cmrbdts and ethnicities)
    :param cmrbdts_eths_oh: mapping from keys to oh-vectors
    :return: oh-vector for a specific patient contains ethnicity and comorbidities
    '''
    cmrbdts_split = preprocess_sentence(cmrbdts).split(';')
    cmrbdts_split = [x.strip() for x in cmrbdts_split]
    # Cmrbdts and eths
    cmrbdts_eth = cmrbdts_split + [e]

    oh_vecs = []
    included_others = False
    for x in cmrbdts_eth:
        # If it is not in our dictionary, encode as other (but only once)
        if x not in encoded_keys and not included_others:
            x = 'other'
            included_others = True
        elif x not in encoded_keys and included_others:
            continue

        x_oh = cmrbdts_eths_oh[x]
        assert (np.sum(x_oh) == 1.)
        oh_vecs.append(x_oh)

    return np.sum(np.array(oh_vecs), axis=0)


def preprocess_sentence(s):
    # Original preprocessing contained only the first line
    s = s.replace('?', '')
    # Added rules: cause a drop in performance :/
    # s = s.replace(' /', '/')
    # s = s.replace('GASTROINTESTINAL BLEED', 'GI BLEED')

    return s


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_static_config(m, ds):
    # Default dataset options
    params = {}
    params['steps_ahead'] = 3
    params['training_instances'] = 30000

    tasksets = BPDataset(ds, num_tasks=params['training_instances'], meta_train=True,
                         steps_ahead=params['steps_ahead'])
    eths = []
    all_cmrbdts = []
    for i in range(len(tasksets)):
        gender, e, age, cmrbdts = tasksets[i].static
        cmrbdts = preprocess_sentence(cmrbdts)
        cmrbdts_split = cmrbdts.split(';')

        for x in cmrbdts_split:
            all_cmrbdts.append(x.strip())
        eths.append(e.strip())

    cmrbdts_counter = Counter(all_cmrbdts)

    most_common_cmrbdts_lst = cmrbdts_counter.most_common(m)
    most_common_cmrbdts = dict(most_common_cmrbdts_lst)
    # add ethnicities
    for e in eths:
        most_common_cmrbdts[e] = -1

    # in-vocabulary words : keys before adding 'other'
    iv_words = set(most_common_cmrbdts.keys())
    # Dict that contains a mapping from cmrbdts(and eths) to oh-vectors
    cmrbdts_eths_oh = oh_encode_words(most_common_cmrbdts)

    config = {}
    config['iv_words'] = iv_words
    config['cmrbdts_eths_oh'] = cmrbdts_eths_oh

    return config

# if __name__ == '__main__':
#
#     iv_words, cmrbdts_eths_oh = get_static_config(128)
#     exit(0)
#     # Just for testing
