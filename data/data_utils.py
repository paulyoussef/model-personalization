import logging

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data.hp_dataset import HPSet
from data.synthetic_datasets import Stat2persSinusoid, SXSet
from torchmeta.utils.data import Task, MetaDataset


class BPTask(Task):
    def __init__(self, patients, index, static_config=None, scaler=None, include_static = True):
        self._inputs = None
        self._targets = None
        self.id = index
        self.scaler = scaler
        self.static_config = static_config
        # Include static is only used to NOT include the static data in the input although this is available (i.e. because static_config != None)
        self.include_static = include_static
        self.encoded_features = None
        self.repr = None

        self.ints = np.array(patients[str(self.id) + ' - ints'])
        # self.time = np.expand_dims(np.array([i for i in range(len(self.ints))], dtype=np.float32), 1)
        self.outs = np.array(patients[str(self.id) + ' - outs'])
        self.static = patients[str(self.id) + ' - static']
        self.tmp = []

        self.covs = np.array(patients[str(self.id) + ' - covs'])
        self.covs = self.covs.reshape(-1, 1)
        for i in range(len(self.ints)):
            tmp_ints = np.expand_dims(self.ints[i], 1)
            tmp_covs = np.expand_dims(self.covs[i], 1)
            self.tmp.append(np.vstack((tmp_covs, tmp_ints)))


        self.tmp = np.array(self.tmp)
        tmp_shape = self.tmp.shape
        if len(tmp_shape) > 2:
            self.tmp = self.tmp.reshape(tmp_shape[0], tmp_shape[1] * tmp_shape[2])
        self._inputs = self.tmp

        self._targets = self.outs
        self.inputs = self._inputs
        self.targets = self._targets

        self.s_encoded = None
        if static_config is not None:
            from data.static_data_utils import encode_cmrbdts_eths, encode_gender, encode_age

            self.iv_words = self.static_config['iv_words']
            self.cmrbdts_eths_oh = self.static_config['cmrbdts_eths_oh']

            for k, v in self.static_config['cmrbdts_eths_oh'].items():
                self.static_config['cmrbdts_eths_oh'][k] = np.array(v)

            # # add encoded static data to input
            # gender, e, age, cmrbdts = self.static
            # # One-hot encoding for gender and age
            # gender_oh = encode_gender(gender)
            # age_oh = encode_age(age)
            # # Int-embedding for ethnicities and comorbidities
            # e_ints = sent2int(e, self.word2int, 1)
            # cmrbs_ints = sent2int(cmrbdts, self.word2int, self.max_cmrbdts)
            # all_features = list(map(lambda x: float(x), gender_oh)) + [age_oh] + list(
            #     map(lambda x: float(x), e_ints)) + list(map(lambda x: float(x), cmrbs_ints))

            gender, e, age, cmrbdts = self.static
            # One-hot encoding for gender and age
            gender_oh = encode_gender(gender)
            age_oh = encode_age(age)
            # Int-embedding for ethnicities and comorbidities
            cmrbs_ints = encode_cmrbdts_eths(cmrbdts, e, self.iv_words, self.cmrbdts_eths_oh)
            # all_features encoded
            all_features = list(map(lambda x: float(x), gender_oh)) + [age_oh] + list(
                map(lambda x: float(x), cmrbs_ints))
            self.s_oh = all_features

            self.s_encoded = self.s_oh

            # if autoencoder is not None:
            #     self.s_encoded = self.autoencoder.encode(
            #         torch.Tensor(all_features).to(list(self.autoencoder.parameters())[0].device)).to(
            #         torch.float64).cpu().detach().numpy().reshape(-1, )

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        # Those represent one data point for one task/patient
        input, target = self._inputs[index], self._targets[index]

        if self.scaler != None:
            unscaled = list(input) + list(target)
            unscaled = np.array(unscaled)
            unscaled = unscaled.reshape(1, len(unscaled))
            scaled = self.scaler.transform(unscaled)
            scaled = scaled.flatten().tolist()

            input, target = scaled[:len(input)], scaled[len(input):]

            if self.s_encoded is not None and self.include_static:
                d = np.array(input)
                input = np.hstack((d, self.s_encoded))
            return np.array(input), np.array(target)

        return (input, target)

    def get_scaled_inputs_targets(self):
        inputs = []
        targets = []
        for i in range(self.__len__()):
            inputs.append(self.__getitem__(i)[0])
            targets.append(self.__getitem__(i)[1])

        return (inputs, targets)


class BPDataset(MetaDataset):
    def __init__(self, dataset, num_tasks=1000000, meta_train=True, steps_ahead=3, num_data_points=20, static_config=None, include_static = True, noise_std=0., random_order = False, noise_targets = False):
        self.dataset = dataset
        self.meta_train = meta_train
        self.num_tasks = num_tasks

        if dataset == 'sx':
            if self.meta_train:
                self.tasksets = SXSet(split = 'train',num_samples_per_task=num_data_points, num_tasks=num_tasks, noise_std = noise_std, seed=0, random_order = random_order, noise_targets = noise_targets, with_static = include_static)#, transform= sine_scale, target_transform=sine_scale)
            else:
                self.tasksets = SXSet(split = 'test',num_samples_per_task=num_data_points, num_tasks=num_tasks, noise_std = noise_std, seed=1, random_order = random_order, noise_targets = noise_targets, with_static = include_static)#, transform= sine_scale, target_transform=sine_scale)

            self.num_tasks = len(self.tasksets)
            self.scaler = None
            self.static_config = static_config
            self.include_static = include_static

        elif dataset == 'bp':

            self.scaler = None
            self.static_config = static_config
            self.include_static = include_static
            if self.meta_train:
                self.patients = few_shot_learning_dataset(dataset, 'training', num_data_points, steps_ahead)
            elif not self.meta_train:
                self.patients = few_shot_learning_dataset(dataset, 'test', num_data_points, steps_ahead)

            MAX_SIZE = self.patients['nr_tasks']
            self.num_tasks = int(min(num_tasks, MAX_SIZE))


        else:
            raise ValueError
    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        if self.dataset == 'si':
            return self.tasksets[index]
        elif self.dataset == 'sx':
            return self.tasksets[index]
        elif self.dataset == 'hp':
            return self.tasksets[index]

        task = BPTask(self.patients, index, self.static_config, self.scaler, include_static= self.include_static)
        return task

    def get_scaler(self):
        if self.dataset == 'si' or self.dataset == 'sx':
            return None
        scaler = MinMaxScaler()
        all_data = []
        for i in range(int(self.num_tasks)):
            task = BPTask(self.patients, i)
            inputs = task.inputs
            targets = task.targets
            if len(targets.shape) == 1:
                targets = np.expand_dims(task.targets, axis=1)
            unscaled = np.hstack((inputs, targets))

            self.inputs_dim = inputs.shape[1]
            self.targets_dim = targets.shape[1]
            all_data.append(unscaled)
        tmp = np.array(all_data)
        tmp = tmp.reshape(tmp.shape[0] * tmp.shape[1], tmp.shape[2])
        assert (tmp.shape[1] == self.inputs_dim + self.targets_dim)

        scaler.fit(tmp)
        self.scaler = scaler
        return scaler

    def set_scaler(self, scaler):
        self.scaler = scaler


def check_index(param):
    if param == 0:
        return None
    else:
        return param


def few_shot_learning_dataset(dataset, part, num_data_points=20, steps_ahead=3):



    # Length of output sequence = Length of input sequence - ws + 1
    # Length of output sequence = Length of input sequence -ws -steps_ahead + 2
    combined_data = joblib.load('DATA_DIR/data/{}/combined_{}.pkl'.format(dataset, part))
    ws = 1
    trajectory_length = num_data_points + ws + steps_ahead - 2
    sequences_length = combined_data['sequences_length']
    # Pick patients who have a trajectory length of at least 'trajectory_length'
    targeted_indices = np.squeeze(np.argwhere(sequences_length >= trajectory_length))

    covs = combined_data['covariates'][targeted_indices]
    ints = combined_data['interventions'][targeted_indices]
    outs = combined_data['outcomes'][targeted_indices]
    static = combined_data['static'][targeted_indices]

    data = {}
    data['nr_tasks'] = len(targeted_indices)
    # time_steps = [i for i in range(trajectory_length)]
    contains_ones = 0
    # count patients where there are 0 and 1 in both halves
    ones_zeros_first_second = 0
    ones_zeros_first = 0
    for i in range(len(covs)):
        interventions = ints[i][:trajectory_length]

        # remove last steps_ahead -1 covs because those are outputs in the last data instance

        data[str(i) + ' - covs'] = list(
            to_windows(list(covs[i][:trajectory_length])[:check_index(-(steps_ahead - 1))], ws))
        data[str(i) + ' - ints'] = list(
            to_windows(list(ints[i][:trajectory_length][check_index(ws - 1):]), steps_ahead))
        # data[str(i) + ' - time'] = to_windows([x for x in range(trajectory_length)], ws)
        # data[str(i) + ' - outs'] = list(outs[i][:trajectory_length])[ws-1:]
        # remove first ws-1 outs because those are inputs  in the first data instance
        data[str(i) + ' - outs'] = list(to_windows(outs[i][:trajectory_length][check_index(ws - 1):], steps_ahead))
        data[str(i) + ' - static'] = static[i]

        assert (len(data[str(i) + ' - ints']) == len(data[str(i) + ' - outs']))

    logging.info('# all trajectories: {}'.format(len(targeted_indices)))
    return data


def few_shot_learning_dataset_old(part, num_data_points=20, ws=3, steps_ahead=3, no_covs=False):
    # Length of output sequence = Length of input sequence - ws + 1
    # Length of output sequence = Length of input sequence -ws -steps_ahead + 2
    combined_data = joblib.load('DATA_DIR/data/bp/combined_{}.pkl'.format(part))

    trajectory_length = num_data_points + ws + steps_ahead - 2
    sequences_length = combined_data['sequences_length']
    # Pick patients who have a trajectory length of at least 'trajectory_length'
    targeted_indices = np.squeeze(np.argwhere(sequences_length >= trajectory_length))

    covs = combined_data['covariates'][targeted_indices]
    ints = combined_data['interventions'][targeted_indices]
    outs = combined_data['outcomes'][targeted_indices]
    static = combined_data['static'][targeted_indices]

    data = {}
    data['nr_tasks'] = len(targeted_indices)
    # time_steps = [i for i in range(trajectory_length)]
    contains_ones = 0
    # count patients where there are 0 and 1 in both halves
    ones_zeros_first_second = 0
    ones_zeros_first = 0
    for i in range(len(covs)):
        interventions = ints[i][:trajectory_length]
        # if 1. in interventions:
        #     contains_ones = contains_ones + 1
        #     #print(i)

        if 0. in interventions[:int(trajectory_length / 2)] and 1. in interventions[:int(trajectory_length / 2)]:
            ones_zeros_first += 1
            if 0. in interventions[int(trajectory_length / 2):] and 1. in interventions[int(trajectory_length / 2):]:
                ones_zeros_first_second += 1

        # remove last steps_ahead -1 covs because those are outputs in the last data instance
        if not no_covs:
            data[str(i) + ' - covs'] = list(
                to_windows(list(covs[i][:trajectory_length])[:check_index(-(steps_ahead - 1))], ws))
        data[str(i) + ' - ints'] = list(
            to_windows(list(ints[i][:trajectory_length][:check_index(-(steps_ahead - 1))]), ws))
        data[str(i) + ' - time'] = to_windows([x for x in range(trajectory_length)], ws)
        # data[str(i) + ' - outs'] = list(outs[i][:trajectory_length])[ws-1:]
        # remove first ws-1 outs because those are inputs  in the first data instance
        data[str(i) + ' - outs'] = list(to_windows(outs[i][:trajectory_length][check_index(ws - 1):], steps_ahead))
        data[str(i) + ' - static'] = static[i]

        assert (len(data[str(i) + ' - ints']) == len(data[str(i) + ' - outs']))

    logging.info('#trajectories with 0s and 1s in first have: {}, {}'.format(ones_zeros_first,
                                                                             np.round(ones_zeros_first / len(
                                                                                 targeted_indices), 2)))
    logging.info('#trajectories with 0s and 1s in both halves: {}, {}'.format(ones_zeros_first_second,
                                                                              np.round(ones_zeros_first_second / len(
                                                                                  targeted_indices), 2)))
    logging.info('# all trajectories: {}'.format(len(targeted_indices)))
    return data


def to_windows(seq, ws):
    # Edge cases
    if ws == 1:
        return np.expand_dims(np.array(seq), 1).tolist()
    if len(seq) < ws:
        raise ValueError('Window size is too big for this sequence')

    ws_seq = []
    seq = seq
    c = 0
    while True:
        if c + ws > len(seq):
            return ws_seq
        # convert seq[] to list to avoid having any ndarrays that would prevent dumping the file in json format
        ws_seq.append(list(seq[c:c + ws]))
        c += 1


