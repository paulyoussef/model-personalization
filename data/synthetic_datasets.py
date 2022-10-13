import numpy as np

from torchmeta.utils.data import Task, MetaDataset
from numpy.random import default_rng
import torch



class SXSet(MetaDataset):

    def __init__(self, split, num_samples_per_task, num_tasks=1000000,
                 noise_std=None, transform=None, target_transform=None,
                 dataset_transform=None, seed=None, random_order=False, noise_targets = False, with_static = True):
        super(SXSet, self).__init__(meta_split='train',
                                                target_transform=target_transform, dataset_transform=dataset_transform)
        self.num_samples_per_task = num_samples_per_task
        self.num_tasks = num_tasks
        self.transform = transform
        self.random_order = random_order
        self._input_range = np.array([-5.0, 5.0])
        self.noise_std = noise_std
        self.noise_targets = noise_targets
        self.split = split
        self.with_static = with_static
        range_start = 0
        range_end = 1
        input_dim = 127


        self.np_random = default_rng(seed=seed)
        self.inputs_nn = np.round(self.np_random.uniform(range_start, range_end, size=(self.num_tasks, input_dim)))
        # To ensure all are between 0 and 1
        self.inputs_nn_normalized = self.inputs_nn #(self.inputs_nn - range_start)/(range_end - range_start)

        assert (np.min(self.inputs_nn_normalized) >= 0)
        assert (np.max(self.inputs_nn_normalized) <= 1)

        self.amp_nn = torch.load('(./data/sx/amp_nn.pth')
        self.phase_nn = torch.load('./data/sx/phase_nn.pth')

        with torch.no_grad():
            self.amplitudes = self.amp_nn(torch.from_numpy(self.inputs_nn)).numpy()
            self.phases = self.phase_nn(torch.from_numpy(self.inputs_nn)).numpy()

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        amplitude, phase = self.amplitudes[index], self.phases[index]
        amp_noise, phase_noise = 0., 0.

        assert (amplitude >= 0.1 and amplitude <= 5.)
        assert (phase >= 0 and phase <= np.pi)
        task = SXTask(self.split, index, self.inputs_nn_normalized[index], amplitude, phase, amp_noise, phase_noise, self._input_range, self.num_samples_per_task,
                            seed = self.num_tasks + index, random_order=self.random_order, rand_state = self.np_random, noise_std = self.noise_std, noise_targets=self.noise_targets, with_static=self.with_static)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task


class SXTask(Task):
    def __init__(self, split, index, input_nn, amplitude, phase, amp_noise, phase_noise, input_range,
                 num_samples, seed=None, random_order=False, rand_state = None, noise_std = 0., noise_targets = False, with_static = True):
        super(SXTask, self).__init__(index, None)  # Regression task
        # True static values
        self.amplitude = amplitude
        self.phase = phase
        # Noise to be added to the true values
        self.amp_noise = amp_noise
        self.phase_noise = phase_noise

        self.input_range = input_range
        self.num_samples = num_samples
        self.seed = seed
        self.input_nn = input_nn
        self.noise_std = noise_std
        self.noise_targets = noise_targets
        self.split = split
        self.with_static = with_static
        # num tasks is used to always generate the same inputs for the same task
        #self.np_random = np.random.RandomState(self.seed)
        self.np_random = default_rng(seed)
        self._inputs = self.np_random.uniform(input_range[0], input_range[1],
                                         size=(num_samples, 1))

        if not random_order:
            self._inputs = np.sort(self._inputs, axis=0)
        else:
            self.np_random.shuffle(self._inputs)

        self._targets = amplitude * np.sin(self._inputs - phase)

        if self.noise_targets:
            if self.split == 'train':
                self._targets += np.expand_dims(self.np_random.standard_normal(len(self._targets)), 1) * noise_std
            elif self.split == 'test':
                # Adds noise only to the first half (support)
                self._targets += np.expand_dims(np.hstack(
                    (self.np_random.standard_normal(len(self._targets) // 2), np.zeros(len(self._targets) // 2))),
                                                1) * noise_std
            else:
                raise ValueError

        # self.noisy_amplitude = self.amplitude + self.amp_noise
        # self.noisy_phase = self.phase + self.phase_noise

        # Alternatively:
        # tmp1 = rand_state.randn(1)[0]*noise_std
        # tmp2 = rand_state.randn(1)[0]*noise_std
        # # print('i: {}: directly from State: {}, from upper class: {}'.format(index, tmp1, self.amp_noise))
        # self.noisy_amplitude = self.amplitude + tmp1
        # self.noisy_phase = self.phase + tmp2


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        input, target = self._inputs[index], self._targets[index]

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.with_static:
            input_sine = np.expand_dims(input, 1)
            inputs = np.squeeze(np.concatenate((input_sine, np.expand_dims(self.input_nn, 1))))
            assert(len(inputs) ==128)
        else:
            inputs = input
            assert(len(inputs) == 1)


        return (inputs, target)

    def get_scaled_inputs_targets(self):
        inputs = []
        targets = []

        for i in range(self.__len__()):
            inputs.append(self.__getitem__(i)[0])

            targets.append(self.__getitem__(i)[1])

        return (inputs, targets)
