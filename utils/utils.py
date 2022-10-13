import logging
import random
import joblib
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from torch.utils.data import Dataset

from analysis.val_loss import get_best_val_loss_baseline, get_best_val_loss
from utils.model_reconstruction_utils import get_best_seed, get_best_seed_bl


def load_dataset(part, window_size, data_dir):
    return np.array(joblib.load('{}/x_{}_{}.pkl'.format(data_dir, part, str(window_size)))), np.array(joblib.load(
        '{}/y_{}_{}.pkl'.format(data_dir, part, str(window_size))))


def load_training_set(window_size, data_dir='../data'):
    return load_dataset('training', window_size, data_dir)


def load_test_set(window_size, data_dir='../data'):
    return load_dataset('test', window_size, data_dir)

def shuffle_same_order(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def normalize_rmse(rmse):
    from numpy.random import seed
    seed(0)

    MAX = 330
    MIN = 14
    return np.round((rmse/(MAX-MIN))*100,2)

def evaluate(unscaled_targets, unscaled_preds):
    y_true = np.array(unscaled_targets)
    y_preds = np.nan_to_num(np.array(unscaled_preds))

    rmse = mean_squared_error(y_true, y_preds , squared=False)

    mae = mean_absolute_error(y_true, y_preds)

    nrmse = normalize_rmse(rmse)
    # print('Micro-averaged results:')
    # print('MAE {}'.format(mae))
    # print('RMSE {}'.format(rmse))
    # print('Normalized RMSE: {}'.format(nrmse))
    results = {}
    results['mae'] = mae
    results['rmse'] = rmse
    results['normlaized rmse'] = nrmse
    return results


def evaluate_no_rounding(unscaled_targets, unscaled_preds):
    '''

    :param unscaled_targets:
    :param unscaled_preds:
    :return: return the results without rounding them and as float
    '''
    y_true = np.array(unscaled_targets)
    y_preds = np.array(unscaled_preds)
    rmse = mean_squared_error(y_true, y_preds , squared=False)
    mae = mean_absolute_error(y_true, y_preds)
    nrmse = normalize_rmse(rmse)

    results = {}
    results['mae'] = mae
    results['rmse'] = rmse
    results['normlaized rmse'] = nrmse
    return results


def get_scaled_dataset(ws, train_eval_split = 0.75 , data_dir='DATA_DIR/data/'):
    x_unscaled, y_unscaled = load_training_set(ws, data_dir=data_dir)
    y_unscaled = y_unscaled.reshape(y_unscaled.shape[0], 1)
    # Join data to normalize
    dataset = np.hstack((x_unscaled, y_unscaled))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # First ws*2 columns are the features
    X = dataset[:, :ws * 2]
    # Last column is the output
    Y = dataset[:, ws * 2]
    X, Y = shuffle_same_order(X, Y)

    data_len = len(Y)
    train_index = int(data_len * train_eval_split)

    train_x = X[:train_index]
    train_y = Y[:train_index]
    ####
    val_x = X[train_index:]
    val_y = Y[train_index:]

    print('train_x.shape: ', train_x.shape)
    print('train_y.shape: ', train_y.shape)
    print('val_x.shape: ', val_x.shape)
    print('val_y.shape: ', val_y.shape)

    return train_x, train_y, val_x, val_y, scaler

def evaluate_on_test_set(model, scaler,  ws, reshape_function = None, data_dir='DATA_DIR/data/',):
    x_test_unscaled, y_test_unscaled = load_test_set(ws, data_dir=data_dir)
    # Join inputs and outputs
    y_test_unscaled = y_test_unscaled.reshape(y_test_unscaled.shape[0], 1)
    # Combine X and Y
    test_set_unscaled = np.hstack((x_test_unscaled, y_test_unscaled))
    # Targets
    unscaled_targets = test_set_unscaled[:, ws * 2]

    # Normalize
    test_set_scaled = scaler.transform(test_set_unscaled)
    scaled_inputs = test_set_scaled[:, :ws * 2]

    if reshape_function != None:
        old_shape = scaled_inputs.shape
        scaled_inputs = reshape_function(scaled_inputs)

    scaled_preds = model.predict(scaled_inputs)
    if  len(scaled_preds.shape) == 1:
        scaled_preds = np.expand_dims(scaled_preds, axis=1)

    if reshape_function != None:
        scaled_inputs = scaled_inputs.reshape(old_shape)

    scaled_test_set_preds = np.hstack((scaled_inputs, scaled_preds))
    unscaled_preds = scaler.inverse_transform(scaled_test_set_preds)[:, ws * 2]

    return evaluate(unscaled_targets, unscaled_preds)


def get_scaled_dataset_no_treatments(ws, train_eval_split = 0.75 , data_dir='DATA_DIR/data/'):
    x_unscaled, y_unscaled = load_training_set(ws, data_dir='DATA_DIR/data/')
    x_unscaled = x_unscaled[:, :ws]
    y_unscaled = y_unscaled.reshape(y_unscaled.shape[0], 1)
    # Join data to normalize
    dataset = np.hstack((x_unscaled, y_unscaled))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # First ws columns are the features
    X = dataset[:, :ws]
    # Last column is the output
    Y = dataset[:, -1]
    X, Y = shuffle_same_order(X, Y)

    data_len = len(Y)
    train_index = int(data_len * train_eval_split)

    train_x = X[:train_index]
    train_y = Y[:train_index]
    ####
    val_x = X[train_index:]
    val_y = Y[train_index:]

    print('train_x.shape: ', train_x.shape)
    print('train_y.shape: ', train_y.shape)
    print('val_x.shape: ', val_x.shape)
    print('val_y.shape: ', val_y.shape)

    return train_x, train_y, val_x, val_y, scaler

def evaluate_on_test_set_no_treatment(model, scaler,  ws,  data_dir='DATA_DIR/data/' ):
    x_test_unscaled, y_test_unscaled = load_test_set(ws, data_dir=data_dir)
    x_test_unscaled = x_test_unscaled[:, :ws]
    # Join inputs and outputs
    y_test_unscaled = y_test_unscaled.reshape(y_test_unscaled.shape[0], 1)

    test_set_unscaled = np.hstack((x_test_unscaled, y_test_unscaled))
    unscaled_targets = test_set_unscaled[:, ws]

    # Normalize
    test_set_scaled = scaler.transform(test_set_unscaled)
    test_set_inputs = test_set_scaled[:, :ws]
    scaled_preds = model.predict(test_set_inputs)
    # Combine scaled inputs and predictions
    scaled_test_set_preds = np.hstack((test_set_inputs, scaled_preds))
    # Unscaled inputs and predictions and get predictions
    unscaled_preds = scaler.inverse_transform(scaled_test_set_preds)[:, ws]

    return evaluate(unscaled_targets, unscaled_preds)

def write_results(results, model_name, ws, batch_size, training_epochs, steps_ahead=1, results_file_path = '../results/results.csv'):

    mae = results['mae']
    rmse = results['rmse']
    nrmse = results['normlaized rmse']

    results_and_model_details = ', '.join([model_name, str(ws), str(steps_ahead), rmse, nrmse, mae, str(batch_size), str(training_epochs)]) + '\n'

    print('results and model details: ', results_and_model_details)
    with open(results_file_path, "a+") as f:
        f.write(results_and_model_details)

def load_steps_ahead_pred_part(part, window_size, steps_ahead, data_dir = 'DATA_DIR/data/multi-step-ahead/'):
    return np.load('{}/x_{}_{}_{}.npy'.format(data_dir, part,window_size,steps_ahead)), np.load('{}/y_{}_{}_{}.npy'.format(data_dir, part,window_size,steps_ahead))

def load_steps_ahead_pred_dataset(window_size, steps_ahead, data_dir = 'DATA_DIR/data/multi-step-ahead/'):
    return load_steps_ahead_pred_part('training', window_size, steps_ahead, data_dir), load_steps_ahead_pred_part('test', window_size, steps_ahead, data_dir)

def get_scaled_msa_dataset(ws, steps_ahead, train_eval_split = 0.75 , data_dir='DATA_DIR/data/multi-step-ahead/'):
    # msa: multi-step ahead
    x_unscaled, y_unscaled = load_steps_ahead_pred_part('training', ws,steps_ahead, data_dir=data_dir)
    # Join data to normalize
    dataset = np.hstack((x_unscaled, y_unscaled))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # First ws*2 columns are the features
    X = dataset[:, :ws * 2]
    # Last steps ahead columns is the output
    Y = dataset[:, ws * 2:]
    X, Y = shuffle_same_order(X, Y)

    data_len = len(Y)
    train_index = int(data_len * train_eval_split)

    train_x = X[:train_index]
    train_y = Y[:train_index]
    ####
    val_x = X[train_index:]
    val_y = Y[train_index:]

    print('train_x.shape: ', train_x.shape)
    print('train_y.shape: ', train_y.shape)
    print('val_x.shape: ', val_x.shape)
    print('val_y.shape: ', val_y.shape)

    return train_x, train_y, val_x, val_y, scaler

def evaluate_on_msa_test_set(model, scaler,  ws, steps_ahead, reshape_function = None, data_dir='DATA_DIR/data/multi-step-ahead/'):
    x_test_unscaled, y_test_unscaled =load_steps_ahead_pred_part('test', ws, steps_ahead, data_dir=data_dir)
    print('# of test instances, evaluate on msa test set :', len(y_test_unscaled))
    # Join inputs and outputs
    # Combine X and Y
    test_set_unscaled = np.hstack((x_test_unscaled, y_test_unscaled))
    # Targets
    unscaled_targets = test_set_unscaled[:, ws * 2:]

    # Normalize
    test_set_scaled = scaler.transform(test_set_unscaled)
    scaled_inputs = test_set_scaled[:, :ws * 2]

    if reshape_function != None:
        old_shape = scaled_inputs.shape
        scaled_inputs = reshape_function(scaled_inputs)

    scaled_preds = model.predict(scaled_inputs)
    if  len(scaled_preds.shape) == 1:
        scaled_preds = np.expand_dims(scaled_preds, axis=1)

    if reshape_function != None:
        scaled_inputs = scaled_inputs.reshape(old_shape)

    scaled_test_set_preds = np.hstack((scaled_inputs, scaled_preds))
    unscaled_preds = scaler.inverse_transform(scaled_test_set_preds)[:, ws * 2:]

    return evaluate(unscaled_targets, unscaled_preds)


def add_current_results(results, current_results):

    for k, v in current_results.items():
        results[k] = np.append(results[k] , v)
        #results[k].append(float(v))

def analyse_patient_wise_results(results):
    def apply_and_round(f, values):
        return np.round(f(values), 2)
    print('Macro-averaged results:')
    for k, v in results.items():
        print('---')
        print(k.upper())
        print('mean:    {}'.format(apply_and_round(np.mean, v)))
        print('std dev:    {}'.format(apply_and_round(np.std, v)))
        print('min:    {}'.format(apply_and_round(np.min, v)))
        print('max:    {}'.format(apply_and_round(np.max, v)))

    sns.histplot(data=results['mae'], binwidth=1, element='poly')
    plt.xlabel('Mean Absolute Error');
    plt.show()

    print('Done histogram')
def plot_trajectories(id, mae, targets, preds, x_test):
    plt.figure()
    x = list(range(len(targets)))
    plt.plot(x, targets, label='Targets')
    plt.plot(x, preds, label='Predictions')
    plt.plot(x, [x[-1]*20 for x in x_test], 'x', label='Interventions')
    plt.title('mean absolute error: {}'.format(mae))
    plt.ylim(bottom=13)
    # plt.xlim(-1, 100)
    plt.legend()
    plots_path = './trajectories/'
    Path(plots_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_path + str(mae) + '_' + str(id) + '.png')
    plt.show()
    plt.close('all')

def eval_patient_wise(model, scaler, ws, steps_ahead, reshape_function = None, toPlot = (lambda x: False)):
    combined_data = joblib.load('DATA_DIR/data/combined_raw_data/combined_test.pkl')

    covs = combined_data['covariates']
    ints = combined_data['interventions']
    outs = combined_data['outcomes']
    sequences_length = combined_data['sequences_length']
    inputs = []
    outputs = []

    results = {}
    results['rmse'] = np.array([])
    results['normlaized rmse'] = np.array([])
    results['mae'] = np.array([])
    # In case you want to check the number instances
    #counter = 0
    for i in range(len(covs)):
        seq_len = int(sequences_length[i])
        current_covs = covs[i][:seq_len]
        current_ints = ints[i][:seq_len]
        current_outs = outs[i][:seq_len]

        for j in range(seq_len):
            if (j+ws)  > seq_len or (j+ws+steps_ahead) > seq_len:
                break
            # Input for one instance
            current_x = np.concatenate((current_covs[j:j+ws], current_ints[j:j+ws]), axis = 0)
            # Output for one instance
            current_y = current_outs[j+ws -1: j+ws + steps_ahead -1 ]

            inputs.append(current_x)
            outputs.append(current_y)
            #counter += 1
        # Starts evaluation for this patient
        xtest = np.array(inputs)
        ytest = np.array(outputs)

        # Combine data
        test = np.hstack((xtest, ytest))
        # Transform data
        test_scaled = scaler.transform(test)
        # Use inputs for prediction
        scaled_inputs = test_scaled[:, :ws * 2]

        if reshape_function != None:
            scaled_preds = model.predict(reshape_function(scaled_inputs))
        else:
            scaled_preds = model.predict(scaled_inputs)
        # Join inputs and prediction to unscale
        if len(scaled_preds.shape) == 1:
            scaled_preds = np.expand_dims(scaled_preds, axis=1)
        scaled_inputs_preds = np.hstack((scaled_inputs, scaled_preds))
        # Unscale
        unscaled_preds = scaler.inverse_transform(scaled_inputs_preds)[:, ws * 2:]

        # In case, you want to aggregate all predictions and targets
        # all_preds += list(unscaled_preds.flatten())
        # all_truths += list(ytest.flatten())

        current_results = evaluate_no_rounding(ytest.flatten(), unscaled_preds.flatten())
        if toPlot(np.round(current_results['mae'], 2)):
            plot_trajectories(i, np.round(current_results['mae'], 2), ytest.flatten(), unscaled_preds.flatten(), xtest)
        add_current_results(results, current_results)
        # Reset inputs/outputs arrays
        inputs = []
        outputs = []

    #print('# of test instances: ', counter)
    analyse_patient_wise_results(results)
    return results


def patient_wise_data_generator(scaler,  ws, steps_ahead):
    '''
    Generates data for each patient separately
    :param scaler: scaler
    :param ws: window size
    :param steps_ahead: how many steps ahead
    :return:
    '''
    data_path = 'DATA_DIR/data/combined_raw_data/combined_training.pkl'
    combined_data = joblib.load(data_path)
    print('Loading data from {}'.format(data_path.split('/')[-1]))

    covs = combined_data['covariates']
    ints = combined_data['interventions']
    outs = combined_data['outcomes']
    sequences_length = combined_data['sequences_length']


    # In case you want to check the number instances
    #counter = 0
    for i in range(len(covs)):
        inputs = []
        outputs = []

        seq_len = int(sequences_length[i])
        current_covs = covs[i][:seq_len]
        current_ints = ints[i][:seq_len]
        current_outs = outs[i][:seq_len]

        for j in range(seq_len):
            if (j+ws) > seq_len or (j+ws+steps_ahead) > seq_len:
                break
            # Input for one instance
            current_x = np.concatenate((current_covs[j:j+ws], current_ints[j:j+ws]), axis = 0)
            # Output for one instance
            current_y = current_outs[j+ws -1: j+ws + steps_ahead -1 ]

            inputs.append(current_x)
            outputs.append(current_y)
            #counter += 1
        # Starts evaluation for this patient
        x = np.array(inputs)
        y = np.array(outputs)

        # Combine data
        train = np.hstack((x, y))
        # Transform data
        train_scaled = scaler.transform(train)
        # Use inputs for prediction
        scaled_inputs = train_scaled[:, :ws * 2]
        scaled_outputs = train_scaled[:, ws*2:]

        yield scaled_inputs, scaled_outputs

def evaluate_stateful_lstm(model, scaler, ws, steps_ahead, reshape_function = None):
    data_path = 'DATA_DIR/data/combined_raw_data/combined_test.pkl'
    combined_data = joblib.load(data_path)
    print('Loading data from {}'.format(data_path.split('/')[-1]))


    covs = combined_data['covariates']
    ints = combined_data['interventions']
    outs = combined_data['outcomes']
    sequences_length = combined_data['sequences_length']
    inputs = []
    outputs = []

    results = {}
    results['rmse'] = np.array([])
    results['normlaized rmse'] = np.array([])
    results['mae'] = np.array([])
    # In case you want to check the number instances
    all_targets = []
    all_preds = []
    #counter = 0
    for i in range(len(covs)):
        seq_len = int(sequences_length[i])
        current_covs = covs[i][:seq_len]
        current_ints = ints[i][:seq_len]
        current_outs = outs[i][:seq_len]

        for j in range(seq_len):
            if (j+ws)  > seq_len or (j+ws+steps_ahead) > seq_len:
                break
            # Input for one instance
            current_x = np.concatenate((current_covs[j:j+ws], current_ints[j:j+ws]), axis = 0)
            # Output for one instance
            current_y = current_outs[j+ws -1: j+ws + steps_ahead -1 ]

            inputs.append(current_x)
            outputs.append(current_y)
            #counter += 1
        # Starts evaluation for this patient
        xtest = np.array(inputs)
        ytest = np.array(outputs)

        # Combine data
        test = np.hstack((xtest, ytest))
        # Transform data
        test_scaled = scaler.transform(test)
        # Use inputs for prediction
        scaled_inputs = test_scaled[:, :ws * 2]

        if reshape_function != None:
            scaled_preds = model.predict(reshape_function(scaled_inputs), batch_size=1)
            model.reset_states()
        else:
            scaled_preds = model.predict(reshape_function(scaled_inputs), batch_size=1)
            model.reset_states()

        # Join inputs and prediction to unscale
        if len(scaled_preds.shape) == 1:
            scaled_preds = np.expand_dims(scaled_preds, axis=1)
        scaled_inputs_preds = np.hstack((scaled_inputs, scaled_preds))
        # Unscale
        unscaled_preds = scaler.inverse_transform(scaled_inputs_preds)[:, ws * 2:]

        # In case, you want to aggregate all predictions and targets
        all_preds += list(unscaled_preds.flatten())
        all_targets += list(ytest.flatten())

        current_results = evaluate_no_rounding(ytest.flatten(), unscaled_preds.flatten())
        add_current_results(results, current_results)
        # Reset inputs/outputs arrays
        inputs = []
        outputs = []

    #print('# of test instances: ', counter)
    analyse_patient_wise_results(results)
    results = evaluate(all_targets, all_preds)
    return results


def unscale(scaler, x, y):
    '''
    Unscale inputs and targets using the given target
    :param scaler:
    :param x: scaled inputs
    :param y: scaled targets
    :return:
    '''
    y = np.array(y)
    if len(y.shape) == 1:
        # Expand dimension to match x
        y = np.expand_dims(y, axis=-1)

    # if y.shape[1] > 1:
    #     raise ValueError('Target has more than one value...')

    scaled = np.hstack((x,y))
    unscaled = scaler.inverse_transform(scaled)

    inputs = unscaled[:, :x.shape[1]]
    targets = unscaled[:, x.shape[1]:]

    assert(inputs.shape == x.shape)
    assert(targets.shape == y.shape)

    return inputs, targets



def step_wise_data_loader(part):

    '''

    :param part: 'training' or test
    :return:
    '''
    train_data_path =  'DATA_DIR/data/bp/combined_training.pkl'
    test_data_path =  'DATA_DIR/data/bp/combined_test.pkl'

    # Load training data to create a scaler - disregarding 'part' argument
    combined_data = joblib.load(train_data_path)
    print('Loading data from {} for scaling'.format(train_data_path.split('/')[-1]))

    covs = combined_data['covariates']
    ints = combined_data['interventions']
    outs = combined_data['outcomes']
    sequences_length = combined_data['sequences_length']

    all_covs = []
    for i in range(len(covs)):
        seq_len = int(sequences_length[i])
        all_covs += covs[i][:seq_len].tolist()
    #all_covs += outs[-1][:int(sequences_length[-1])].tolist()


    covs_scaler = MinMaxScaler()
    all_covs = np.array(all_covs)
    all_covs = np.expand_dims(all_covs, -1)
    covs_scaler.fit(all_covs)

    # If part was 'test', load test data
    if part == 'test':
        print('Loading data from {}'.format(test_data_path.split('/')[-1]))
        combined_data = joblib.load(test_data_path)
        covs = combined_data['covariates']
        ints = combined_data['interventions']
        outs = combined_data['outcomes']
        sequences_length = combined_data['sequences_length']


    # In case you want to check the number instances
    #counter = 0
    import random
    indices = [x for x in range(len(covs))]
    for _ in range(len(covs)):
        i = random.choice(indices)
        seq_len = int(sequences_length[i])
        current_covs = covs[i][:seq_len]
        current_ints = ints[i][:seq_len]
        current_outs = outs[i][:seq_len]

        for j in range(seq_len):
            state = covs_scaler.transform(current_covs[j].reshape(1,1)).flatten()
            desired_state = covs_scaler.transform(current_outs[j].reshape(1,1)).flatten()
            target_action = current_ints[j]
            last_state = j == (seq_len-1)
            
            yield state[-1], desired_state[-1], target_action, last_state



class BPOffRLDataset(Dataset):

    def __init__(self, train):
        self.items = []
        train_data_path = 'DATA_DIR/data/bp/combined_training.pkl'
        test_data_path = 'DATA_DIR/data/bp/combined_test.pkl'

        # Load training data to create a scaler - disregarding 'part' argument
        combined_data = joblib.load(train_data_path)
        print('Loading data from {} for scaling'.format(train_data_path.split('/')[-1]))

        covs = combined_data['covariates']
        ints = combined_data['interventions']
        outs = combined_data['outcomes']
        sequences_length = combined_data['sequences_length']

        all_covs = []
        for i in range(len(covs)):
            seq_len = int(sequences_length[i])
            all_covs += covs[i][:seq_len].tolist()
        # all_covs += outs[-1][:int(sequences_length[-1])].tolist()

        covs_scaler = MinMaxScaler()
        all_covs = np.array(all_covs)
        all_covs = np.expand_dims(all_covs, -1)
        covs_scaler.fit(all_covs)

        if not train:
            print('Loading data from {}'.format(test_data_path.split('/')[-1]))
            combined_data = joblib.load(test_data_path)
            covs = combined_data['covariates']
            ints = combined_data['interventions']
            outs = combined_data['outcomes']
            sequences_length = combined_data['sequences_length']

            # In case you want to check the number instances
            # counter = 0
        import random
        indices = [x for x in range(len(covs))]
        for i in range(len(covs)):
            #i = random.choice(indices)
            seq_len = int(sequences_length[i])
            current_covs = covs[i][:seq_len]
            current_ints = ints[i][:seq_len]
            current_outs = outs[i][:seq_len]


            states = []
            desired_states = []
            target_actions = []
            last_action = -1

            for j in range(seq_len):
                state = covs_scaler.transform(current_covs[j].reshape(1, 1)).flatten()[-1]
                states.append(state)

                desired_state = covs_scaler.transform(current_outs[j].reshape(1, 1)).flatten()[-1]
                desired_states.append(desired_state)

                target_action = current_ints[j]
                target_actions.append(last_action)

                last_state = j == (seq_len - 1)
                # What is the output here ?
                input = list(zip(states, desired_states, target_actions))
                output = target_action
                self.items.append((torch.tensor(input), torch.tensor(output)))
                last_action = target_action
                if last_state:
                    states = []
                    desired_states = []
                    target_actions = []
                    last_action = -1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_fsl_log_file(path, print_values, best_seeds = False, num_seeds = 5):
    with open(path) as f:
        lines = f.readlines()

    if best_seeds:
        maml_seed = get_best_val_loss(path) # get_best_seed(path[:path.rfind('/')+1] + 'results.md')
        bl_seed = get_best_val_loss_baseline(path) #get_best_seed_bl(path[:path.rfind('/')+1] + 'results.md')
        print('maml seed: {}'.format(maml_seed))
        print('baseline seed: {}'.format(bl_seed))


    dict = {}
    # standard deviation across patients
    std_dict = {}
    seeds = []
    for l in lines:
        if 'seed' in l:
            l_split = l.split(':')
            seeds.append(l_split[-1].strip())
        if ('maml' in l or 'baseline' in l) and ('mae' in l or 'rmse' in l):
            l_split = l.split(':')
            key = l_split[-3].strip()
            try:
                value = float(l_split[-2].strip())
                std = float(l_split[-1].strip())
            except:
                print(l)
                return
            dict[key] = dict.get(key, list()) + [value]
            std_dict[key] = std_dict.get(key, list()) + [std]

    print('found seeds: ' + str(seeds))
    for m in ['mae', 'rmse']:
        print('| model |' + m + ' |std dev|' )
        print('| :---: | :---: | :---: |')
        for k, v in dict.items():
            if m in k:
                assert (len(v) == num_seeds)
                # assert (len(v) == len(seeds))
                # assert (len(std_dict[k]) == len(seeds))

                print('| {} | {} +/- {} | {} |'.format(k.replace(m, '').strip(), np.round(np.mean(v),2), np.round(np.std(v),4), np.round(np.mean(std_dict[k]),2)))
                if print_values:
                    if best_seeds:
                        if 'maml' in k:
                            index = maml_seed
                        elif 'baseline' in k:
                            index = bl_seed
                        else:
                            raise ValueError
                        print('|{} {}| {} | {}|'.format(k.replace(m, '').strip(), 'best', np.round(v[index], 2), np.round(std_dict[k][index], 2)))
                    else:
                        print(v)

        print(' ')

def num_params(model):
    return sum(p.numel() for p in model.parameters())

def set_common_params(params, file_name, exp):
    params['dtype'] = torch.float64
    # if diff in loss < params['early_stop_diff'], stop early
    params['early_stop_diff'] = 0.0
    # Check val_loss every params['es_epoch'] epochs
    params['es_epoch'] = 3
    # Epochs
    params['epochs_maml'] = 50
    params['epochs_baseline'] = 50
    params['meta_batch_size'] = 10
    params['shots'] = 10

    params['hidden_dim'] = 64
    params['adapt_steps'] = 10
    params['window_size'] = 1
    params['fits'] = (0, 10)
    params['adapt_lr'] = 0.01
    params['meta_lr'] = 0.001
    # For baseline
    params['lr'] = 0.001
    params['baseline_finetune_lr'] = 0.0001
    params['batch_size'] = 32
    params['save_models'] = False
    params['log_dir'] = file_name[:-3] + '_' + params['dataset'] + '_' + str(params['steps_ahead'])
    params['debugging'] = exp
    params['swap_support_query'] = False

    if exp:
        # Overwriting some params for testing
        params['epochs_maml'] = 1
        params['epochs_baseline'] = 1
        params['meta_batch_size'] = 2
        params['num_tasks_training'] = 100
        params['num_tasks_test'] = 10
        params['hidden_dim'] = 32
        params['steps'] = (0, 1)
        params['save_models'] = False
        logging.disable('INFO')



def compare_params(m1, m2):
    assert(m1 != m2)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def print_params(m):
    for n, p in m.named_parameters():
        print('{} : {}'.format(n, p))