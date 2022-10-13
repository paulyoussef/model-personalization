import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']


def simple_imputer(df, train_subj):
    idx = pd.IndexSlice
    df = df.copy()
    # Uses the mean and count of all columns
    df_out = df.loc[:, idx[:, ['mean', 'count']]]
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()
    global_means = df_out.loc[idx[train_subj, :], idx[:, 'mean']].mean(axis=0)
    # Fill NA/NaN values using the specified method.
    df_out.loc[:, idx[:, 'mean']] = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).fillna(
        method='ffill'
    ).groupby(ID_COLS).fillna(icustay_means).fillna(global_means)

    df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(float)
    df_out.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)

    is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
    hours_of_absence = is_absent.cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[is_absent == 0].fillna(method='ffill')
    time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)

    df_out.sort_index(axis=1, inplace=True)
    return df_out

def combine_data(static, ints, values, sequences_length):
    '''
    combines static data, interventions and values in one dictionary
    :param static: static data
    :param ints: interventions
    :param values: e.g. mbp
    :param sequences_length:
    :return:
    '''
    dict = {}

    covs = []
    outcomes = []
    interventions = []
    assert (len(static) == len(ints) == len(values))

    for i in range(len(static)):
        seq_len = sequences_length[i]
        # Covariates are all the observed values over time, except the last one
        c = np.concatenate((values[i][:seq_len-1] , np.array([0]) , values[i][seq_len:]), axis=0)
        covs.append(c)
        # padded tail (should only contain zeros)
        tail = values[i][seq_len:]
        assert(len(np.argwhere(tail != 0)) == 0)
        #Outcomes are all the observed values over time, except the first one
        o = np.concatenate((values[i][1:] , np.array([0])), axis=0)
        outcomes.append(o)

        # We omit the last intervention as we don't observe its effect
        intervention = np.concatenate( (ints[i][:seq_len-1],  np.array([-1]), ints[i][seq_len:]),  axis= 0)
        interventions.append(intervention)

    dict['static'] = preprocess_static_data(static)
    dict['covariates'] = np.array(covs)
    dict['outcomes'] = np.array(outcomes)
    dict['interventions'] = np.array(interventions)
    dict['sequences_length'] = np.array(sequences_length) - 1

    return dict

def get_max_len(X):
    subject_id_vals = list(X.index.get_level_values(0))
    cntr = Counter(subject_id_vals)
    max_len = cntr.most_common(1)[0][1]
    return max_len
def get_nr_patients(lst):
    cntr = Counter(lst)
    max_len = cntr.most_common(1)[0][1]
    return len(set(lst)), max_len

def get_covariates(ids, X, covariate):
    '''
    extracts the mean blood pressure for all instances
    :param X: blood pressure values (imputed)
    :return:
    '''
    subject_id_vals = sorted(ids[('subject_id')])
    nr_patients = len(subject_id_vals)

    max_len = get_max_len(X)
    covariates = np.zeros(shape=(nr_patients, max_len))
    # 1's where we still have values otherwise 0's
    active_entries = np.zeros(shape=(nr_patients, max_len))
    # How long each sequence is
    sequences_length = []
    counter = 0
    for i in subject_id_vals:
        # mean blood pressure values
        mbp_values = np.array(X.loc[i][(covariate, 'mean')])
        sequences_length.append(mbp_values.shape[0])
        # pad to reach max length
        mbp_values_padded = np.zeros(max_len)
        mbp_values_padded[:mbp_values.shape[0]] = mbp_values
        covariates[counter] = mbp_values_padded

        #  active entries
        active_entries[counter][:mbp_values.shape[0]] = np.ones(shape=(mbp_values.shape[0],))
        # sanity check
        assert(np.sum(active_entries[counter]) == mbp_values.shape[0])
        # add values to blood pressure
        counter = counter + 1

    return covariates, active_entries, sequences_length



def get_interventions(ids, Y, intervention):
    '''

    :param ids: list of wanted ids
    :param Y: interventions
    :return:
    '''
    ids = sorted(ids[('subject_id')])
    nr_patietns, max_len = get_nr_patients(Y.index.get_level_values(0))
    all_treatments = np.zeros((len(ids), max_len))

    counter = 0
    for id in ids:
        treatments = Y[(intervention)][id]
        treatments_padded = np.ones(max_len)*-1
        treatments_padded[:treatments.shape[0]] = treatments
        all_treatments[counter] = treatments_padded
        counter += 1

    return all_treatments


def get_static_data(ids, static):
    '''

    :param ids:
    :param static:
    :return:
    '''
    ids = sorted(ids[('subject_id')])
    all_static_features = set(['gender', 'ethnicity', 'age', 'insurance', 'admittime', 'diagnosis_at_admission', 'dischtime', 'discharge_location', 'fullcode_first', 'dnr_first', 'fullcode', 'dnr', 'dnr_first_charttime', 'cmo_first', 'cmo_last', 'cmo', 'deathtime', 'intime', 'outtime', 'los_icu', 'admission_type', 'first_careunit', 'mort_icu', 'mort_hosp','hospital_expire_flag', 'hospstay_seq', 'readmission_30', 'max_hours'])
    # maybe: 'admission_type'
    kept_static_features = ['gender', 'ethnicity', 'age', 'diagnosis_at_admission']

    tmp = static.copy()
    columns_to_drop = all_static_features.difference(kept_static_features)
    tmp = tmp.drop(labels=list(columns_to_drop), axis=1)

    filtered_static_data = []

    for i in ids:
        features = tmp.loc[i].to_numpy()
        features = np.reshape(features, -1)
        filtered_static_data.append(features)

    return np.array(filtered_static_data)
def categorize_age(age):
    return age

def categorize_ethnicity(ethnicity):
    if 'AMERICAN INDIAN' in ethnicity:
        ethnicity = 'AMERICAN INDIAN'
    elif 'ASIAN' in ethnicity:
        ethnicity = 'ASIAN'
    elif 'WHITE' in ethnicity:
        ethnicity = 'WHITE'
    elif 'HISPANIC' in ethnicity:
        ethnicity = 'HISPANIC/LATINO'
    elif 'BLACK' in ethnicity:
        ethnicity = 'BLACK'

    return ethnicity

def preprocess_static_data(static):
    static_preprocessed = []

    for s in static:
        s_pp = [s[0], categorize_ethnicity(s[1]), categorize_age(s[2]), s[3]]
        static_preprocessed.append(s_pp)

    return np.array(static_preprocessed)

if __name__ == "__main__":
    # Code is based on code from https://github.com/MLforHealth/MIMIC_Extract/blob/master/notebooks/Baselines%20for%20Intervention%20Prediction%20-%20Vasopressor.ipynb

    INTERVENTION = 'vaso'
    COVARIATE = 'mean blood pressure'
    RANDOM = 0


    DATAFILE = './all_hourly_data.h5'
    X = pd.read_hdf(DATAFILE, 'vitals_labs')
    Y = pd.read_hdf(DATAFILE, 'interventions')
    static = pd.read_hdf(DATAFILE, 'patients')

    Y = Y[[INTERVENTION]]

    train_ids, test_ids = train_test_split(static.reset_index(), test_size=0.2,
                                           random_state=RANDOM, stratify=static['mort_hosp'])

    X_clean = simple_imputer(X, train_ids['subject_id'])
    joblib.dump(X_clean, 'X_imputed.pkl')
    # TRAINING DATA

    # static data
    static_train = get_static_data(train_ids, static)

    # interventions
    interventions_train = get_interventions(train_ids, Y, intervention= INTERVENTION)
    print('interventions_train shape: ', interventions_train.shape)

    # Covariates / Outcomes imputed...
    x_clean = joblib.load('X_imputed.pkl')
    covs_train, active_entries, sequences_length = get_covariates(train_ids, x_clean, covariate=COVARIATE)
    print('covariates_train shape: ', covs_train.shape)


    # Combine data in one dictionary
    combined_data_training = combine_data(static_train, interventions_train, covs_train, sequences_length)
    joblib.dump(combined_data_training, 'data/combined_training_static_{}_{}.pkl'.format(INTERVENTION, COVARIATE))

    print(len(train_ids))

    # TEST DATA
    static_test = get_static_data(test_ids, static)
    interventions_test = get_interventions(test_ids, Y, intervention= INTERVENTION)
    print('interventions_test shape: ', interventions_test.shape)
    covs_test, active_entries_test, sequences_length_test = get_covariates(test_ids, x_clean, covariate=COVARIATE)
    print('covariates_test shape: ', covs_test.shape)

    combined_data_training = combine_data(static_test, interventions_test, covs_test, sequences_length_test)
    joblib.dump(combined_data_training, 'data/combined_test_static_{}_{}.pkl'.format(INTERVENTION, COVARIATE))

    print(len(test_ids))


