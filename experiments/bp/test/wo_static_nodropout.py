import copy
import logging
import os
import sys
import torch

from trainers.models import LSTMGeneric, LSTMGeneric2
from trainers.trainer_bp import train_maml, train_baseline, log_end_results, log_params, eval_test_set
from utils.utils import set_seed, num_params, set_common_params, compare_params
from data.static_data_utils import get_static_config
import copy


class Features2(nn.Module):
    def __init__(self, device, input_dim=3, hidden_dim=64, output_dim=3, static_input_dim=16, static_output_dim=4):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_size = hidden_dim
        self.device = device
        self.static_input_dim = static_input_dim
        self.static_output_dim = static_output_dim
        self.lstm = nn.LSTM(input_dim - static_input_dim, hidden_dim)


        # contains the hidden state, and the hidden cell
        # both have the shape: (num_layers, batch_size, ...)
        # hidden state: H_out projection size if it is > 0 otherwise hidden_size
        # hidden cell: H_cell hidden size
        self.hidden = (torch.zeros(1, 1, self.hidden_layer_size, device=device),
                       torch.zeros(1, 1, self.hidden_layer_size, device=device))


        # MLP for encoding static part
        if self.static_input_dim > 0:
            self.static_mlp = nn.Sequential(
                nn.Linear(static_input_dim, 32),
                nn.GELU(),
                nn.Linear(32, self.static_output_dim),
                nn.GELU())

    def forward(self, input_seq):
        # inp = torch.randn(batch_size, seq_len, input_dim)
        # hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
        # cell_state = torch.randn(n_layers, batch_size, hidden_dim)
        # hidden = (hidden_state, cell_state)

        if len(input_seq.shape) <= 2:
            batch_size = 1
        else:
            batch_size = input_seq.shape[0]

        input_seq = input_seq.view(-1, batch_size, self.input_dim)

        if self.static_input_dim > 0:
            static, dynamic = input_seq[:, :, -self.static_input_dim:], input_seq[:, :,
                                                                        :self.input_dim - self.static_input_dim],
            assert (static.shape[2] == self.static_input_dim)

            # Input should have the shape: (seq_len, batch_size, input_size)
            lstm_out, self.hidden = self.lstm(dynamic, self.hidden)

            # Output has the shape (seq_len, batch_size, output_size)

            # self.lstm return a tuple x
            # x[0] is a tensor of shape (1, 1, hidden_layer_size)
            # x[1] is a tuple that contains two tensors of the same shape (1, 1, hidden_layer_size)
            lstm_out = lstm_out.view(batch_size, -1, self.hidden_layer_size)
            fnn_out = self.static_mlp(static.view(batch_size, -1, self.static_input_dim))

            # combining both
            s_d = torch.cat((lstm_out, fnn_out), 2)
            return s_d
        else:
            lstm_out, self.hidden = self.lstm(input_seq, self.hidden)
            lstm_out = lstm_out.view(batch_size, -1, self.hidden_layer_size)
            return lstm_out

    def reset_hidden_state(self, batch_size=1):
        self.hidden = (torch.zeros(1, batch_size, self.hidden_layer_size, device=self.device),
                       torch.zeros(1, batch_size, self.hidden_layer_size, device=self.device))

class LSTMGeneric2(nn.Module):
    def __init__(self, device, input_dim=3, hidden_dim=64, output_dim=3, static_input_dim=16, static_output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_size = hidden_dim
        self.device = device
        self.static_input_dim = static_input_dim
        self.static_output_dim = static_output_dim

        self.features = Features2(device = self.device, input_dim = self.input_dim, output_dim = self.output_dim, hidden_dim = self.hidden_layer_size, static_input_dim = self.static_input_dim, static_output_dim= self.static_output_dim)
        self.features.to(self.device)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + self.static_output_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
            )

    def forward(self, input_seq):
        s_d = self.features(input_seq)

        return self.head(s_d)

    def reset_hidden_state(self, batch_size=1):
        self.features.reset_hidden_state(batch_size)
if __name__ == '__main__':

    import warnings

    # warnings.simplefilter("error")
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", UserWarning)

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
    params['num_tasks_test'] = 100
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
        LSTMGeneric2(device, input_dim=params['static_dim']+ params['window_size'] + params['steps_ahead'],
                   hidden_dim=params['hidden_dim'],
                   output_dim=params['steps_ahead'], static_input_dim= abs(params['static_dim']), static_output_dim= params['static_output_dim'] ), params)
    # set seed again to make sure the model is initialized similarly
    set_seed(params['seed'])


    torch.save(maml, './maml_{}.pth'.format(params['seed']))
    params['fits'] = [0, 10]

    log_params(params)

    # results_dict = {}
    # params['model'] = 'maml'
    # print(maml.head[3].bias)
    # eval_test_set(maml, scaler, params, results_dict)
    #
    # log_end_results(results_dict, params['num_tasks_test'])
    copied_params = copy.deepcopy(params)
    logging.info('----------end-of-run----------')
    set_seed(params['seed'])
    scaler, maml2 = train_maml(
        LSTMGeneric2(device, input_dim=params['static_dim'] + params['window_size'] + params['steps_ahead'],
                    hidden_dim=params['hidden_dim'],
                    output_dim=params['steps_ahead'], static_input_dim=abs(params['static_dim']),
                    static_output_dim=params['static_output_dim']), params)
    print(compare_params(maml, maml2))

    results_dict = {}
    params['model'] = 'maml2'
    print(maml2.head[2].bias)
    print(maml2.lr)
    maml2.reset_hidden_state()
    eval_test_set(maml2, scaler, params, results_dict)

    log_end_results(results_dict, params['num_tasks_test'])
    logging.info('----------end-of-run----------')
    set_seed(params['seed'])

    results_dict = {}
    params['model'] = 'maml2'
    print(maml2.head[2].bias)
    maml2.reset_hidden_state()
    eval_test_set(maml2, scaler, params, results_dict)
    print(maml2.lr)

    log_end_results(results_dict, params['num_tasks_test'])
    logging.info('----------end-of-run----------')
