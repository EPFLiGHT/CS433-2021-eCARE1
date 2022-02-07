# File with unmodified functions from the epoct_simplified code
#
# author of the original code Cecile Trottet
#
# date 23.12.2021
#

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from torch.utils.data import Dataset, random_split
import copy
#import wandb

# functions
def normalize(qst, X_train, X_valid, X_test, features, log=False):
    """
    normalize train, validation and test datasets using mean and variance
    of train dataset
    :param qst: questionnaire dataset object
    :param X_train: training data
    :param X_valid: validation data
    :param X_test: test data
    :param features: boolean, true if variables to normalize are features
    :param log: whether to apply a log transformation
    :return: normalized train, validation and test sets
    """

    if features:
        # remove columns containing only nans in X_cont_train from X_cont_train, X_cont_valid and X_cont_test
        features_to_remove = [index for index, feature in enumerate(np.all(np.isnan(X_train), axis=0)) if feature]
        # shift indices in decision tree structure
        for feature in features_to_remove:
            for key in qst.tree_struct.keys():
                qst.tree_struct[key] = [elem if elem < feature else elem - 1 for elem in qst.tree_struct[key]]
        X_valid = X_valid[:, ~np.all(np.isnan(X_train), axis=0)]
        X_test = X_test[:, ~np.all(np.isnan(X_train), axis=0)]
        X_train = X_train[:, ~np.all(np.isnan(X_train), axis=0)]
        # change qst.num_continuous_features if some columns have been dropped
        qst.num_continuous_features = X_train.shape[1]

    min = np.nanmin(X_train, axis=0)
    max = np.nanmax(X_train, axis=0)
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0)
    X_train[:, std > 0] = (X_train[:, std > 0] - mean[std > 0]) / std[std > 0]
    X_valid[:, std > 0] = (X_valid[:, std > 0] - mean[std > 0]) / std[std > 0]
    X_test[:, std > 0] = (X_test[:, std > 0] - mean[std > 0]) / std[std > 0]
    X_train[:, std == 0] = (X_train[:, std == 0] - mean[std == 0])
    X_valid[:, std == 0] = (X_valid[:, std == 0] - mean[std == 0])
    X_test[:, std == 0] = (X_test[:, std == 0] - mean[std == 0])

    return X_train, X_valid, X_test, mean, std, min, max

def preprocess_data_epoct(qst, valid_size=0.2, test_size=0.2, fraction = 0):
    """
    Split questionnaire dataset into training, validation and test sets. Normalize the
    data and apply imputation strategy.
    :param qst: QuestionnaireDataset object
    :param valid_size: proportion of data to be used as validation set
    :param test_size: proportion of data to be used as test set
    :param tensors: boolean, whether to convert datasets to torch tensors
    :param imput_strategy: imputation strategy to be used to replace np.nans in features
    :param log: whether to apply log transformation
    :param method: wether to use normalization or min max scaling
    :param all_indices : indices of the train/valid/test sets in epoct and epoct+ data, not used anymore
    :param fraction: fraction of training set to completely drop
    :return: preprocessed X_cont_train, X_cont_valid, X_cont_test, y_train, y_valid, y_test
    """

    training_data, validation_data, testing_data = qst.ml_dataset(valid_size, test_size, fraction)


    X_train, X_valid, X_test = qst.questionnaire_data[training_data.indices], qst.questionnaire_data[validation_data.indices], \
                               qst.questionnaire_data[testing_data.indices]
    qst.remap_test_valid_set()

    y_train, y_valid, y_test = qst.labels[training_data.indices], qst.labels[validation_data.indices], \
                               qst.labels[testing_data.indices]
    

    X_train, X_valid, X_test, mean, std, min_, max_ = normalize(qst, X_train, X_valid, X_test, features=False)
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_valid = torch.from_numpy(X_valid.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))

    y_train, y_valid, y_test = torch.LongTensor(y_train), torch.LongTensor(y_valid), \
                               torch.LongTensor(y_test)

    return (X_train, X_valid, X_test,
           y_train, y_valid, y_test)


def get_optimizer_all_levels_simplified(initial_state, encoders, binary_decoders,
                   lr_encoders,
                  lr_binary_decoders, lr, diseases = True):
    """
    instantiate the optimizer with the parameters for the modules it has to optimize
    :param encoders_to_train: which encoders to train
    :param initial_state: initial state
    :param encoders: encoder modules
    :param binary_decoders: binary decoder modules
    :param lr_encoders: learning rate for encoders
    :param lr_binary_decoders: learning rate for disease decoders
    :param lr: learning rate for other parameters (unused)
    :param diseases: boolean, whether to train disease decoders (usually true)
    :return:
    """
    parameters_optimizer = [{'params' : [initial_state.state_value], 'lr' :lr}]

    parameters_optimizer.extend({'params' : list(encoders[feature_name].parameters()), 'lr' : lr_encoders[feature_name]}
                                 for feature_name in encoders.keys())
    if diseases:
        parameters_optimizer.extend(
            [{'params': list(binary_decoders[disease].parameters()), 'lr': lr_binary_decoders} for disease in binary_decoders.keys()])

    optimizer = torch.optim.Adam(parameters_optimizer, lr=lr)

    return optimizer

def shuffle(array):
    array = list(array)
    return random.sample(array, len(array))

def train_modules_epoct_data_all_levels_simplified(qst, X_train, y_train, X_valid, y_valid, initial_state,
                                                    encoders,
                                                    disease_decoders):
    """
    Simplified training process with no feature decoding
    :param qst: questionnaire object
    :param X_cont_train: continuous training features
    :param y_train: targets train
    :param X_cont_valid: continuous validation features
    :param y_valid: targets valid
    :param X_cat_train: categorical training features
    :param X_cat_valid: categorical validation features
    :param initial_state: initial state
    :param encoders: encoder modules
    :param disease_decoders: disease decoder modules
    :param encoders_to_train: encoders we want to train (usually all of them)
    :param save: boolean to save the model
    :param seed: random seed
    :return:
    """

    # parameters for classical encoders and decoders
    training_parameters = qst.training_parameters
    lr_encoders = training_parameters['lr_encoders']
    lr_diseases_decoder = training_parameters['lr_diseases_decoder']
    lr = training_parameters['lr']
    n_epochs = training_parameters['n_epochs']

    batch_size = training_parameters['batch_size']

    # loss for classification
    criterion = nn.NLLLoss()

    # get the optimizer for the parameters we train in that level
    optimizer = get_optimizer_all_levels_simplified(initial_state, encoders, disease_decoders,
                   lr_encoders,
                  lr_diseases_decoder, lr, diseases=True)

    for j in range(n_epochs):
        # to store the sum of all the disease decoding losses after each encoding step
        disease_loss = 0
        # select random batch of indices (without replacement)
        batch_indices = np.random.choice(range(len(X_train)), batch_size, replace=False)
        # init state
        state = initial_state(len(X_train))
        # iterate over levels in tree
        for level in qst.train_group_order_features.keys():
            # apply encoders
            for feature_group in shuffle(qst.train_group_order_features[level].keys()):
                # select patients
                patients = np.intersect1d(qst.train_group_order_features[level][feature_group], batch_indices)
                if len(patients) > 0:
                    # apply shuffled encoders (we shuffle encoders in same level in tree)
                    for feature_name in shuffle(feature_group):
                        state[patients, :] = encoders[feature_name](state[patients, :],
                                                                                                                        X_train[
                                                                                                                            patients, qst.feature_names.index(
                                                                                                                                feature_name)].view(-1,
                                                                                                                                                    1))


                    # after having encoded each feature we try to predict the diseases
                    for name in qst.disease_names:
                        disease_loss += criterion(disease_decoders[name](state[batch_indices, :]), y_train[batch_indices, qst.disease_names.index(name)])

        #wandb.log({'epoch': j,  'disease_loss': disease_loss})


        # Differentiate loss and make an optimizer step
        optimizer.zero_grad()
        disease_loss.backward()
        for param_dict in optimizer.param_groups:
            for p in param_dict['params']:
                torch.nn.utils.clip_grad_norm_(p, max_norm=training_parameters['gradient_clipping'])

        optimizer.step()


        if j%50 == 0:
            print(f'epoch : {j} : disease_loss {disease_loss}')

        # MONITORING ON VALIDATION DATA
        if j %10 == 0:
            with torch.no_grad():
                disease_loss_valid = 0
                valid_state = initial_state(len(X_valid))

                for level in qst.valid_group_order_features.keys():
                    for feature_group in qst.valid_group_order_features[level].keys():
                        patients = qst.valid_group_order_features[level][feature_group]
                        if len(patients) > 0:
                            for feature_name in feature_group:
                                valid_state[patients, :] = encoders[feature_name](
                                    valid_state[patients, :],
                                    X_valid[patients, qst.feature_names.index(
                                        feature_name)].view(-1, 1))

                            for index, name in enumerate(qst.disease_names):
                                disease_loss_valid += criterion(disease_decoders[name](valid_state), y_valid[:, index])
                if j % 50 == 0:
                    print(f'epoch : {j} : disease_loss_valid {disease_loss_valid}')

                #wandb.log({'epoch': j, 'disease_loss_valid': disease_loss_valid})
    return
    


# questionnaire class
class QuestionnaireDataset(Dataset):
    """
    Parent base class to handle the data
    """

    def __init__(self, questionnaire_data, labels, num_classes, possible_target_values):
        super().__init__()
        # questionnaire data set
        self.questionnaire_data = questionnaire_data
        # targets
        self.labels = labels
        self.target_variable = {"name": "disease", "type": "ordinal", "num_classes": num_classes,
                                "possible_values": possible_target_values,
                                "value_count": self.number_labels_per_leafnode()}
        self.training_data = None
        self.testing_data = None
        self.validation_data = None

    def __len__(self):
        return len(self.questionnaire_data)

    def __getitem__(self, patient_index):
        """
        Return a traditional / flat representation of the datapoint
        np.nan for missing data
        """

        return self.questionnaire_data[patient_index, :], self.labels[patient_index]

    def number_labels_per_leafnode(self):
        """

        :return: dict with possible labels as keys and number of data points with that label
        as value
        """
        return {label: len(self.labels[label][self.labels[label] == 1]) for label in self.labels.columns}

    def ml_dataset(self, valid_size=0.2, test_size=0.2, fraction = 0):
        """
        Split, standardise and handle missing values of train, validation and test sets.
        :param valid_size: proportion of data to be used as validation set
        :param test_size: proportion of data to be used as test set
        :param fraction: fraction of training data to totally throw away

        """
        # split into train, validation and test
        training_data, validation_data, testing_data = random_split(self, [len(self) -
                                                                           math.ceil(test_size * len(self)) -
                                                                           math.ceil(valid_size * len(self)),
                                                                           math.ceil(valid_size * len(self)),
                                                                           math.ceil(test_size * len(self))],
                                                                    generator=torch.Generator().manual_seed(0))

        indices_to_keep = np.random.choice(training_data.indices, size = int(len(training_data.indices)-len(training_data.indices) * fraction), replace = False)
        training_data.indices = indices_to_keep
        self.training_data = training_data
        self.testing_data = testing_data
        self.validation_data = validation_data
        return training_data, validation_data, testing_data

class EPOCTQuestionnaireDataset(QuestionnaireDataset):
    """
    child of QuestionnaireDataset class, specific to e-POCT data set
    """

    def __init__(self, questionnaire_data, labels, num_classes, possible_target_values):
        super().__init__(questionnaire_data, labels, num_classes, possible_target_values)
        self.raw_data = questionnaire_data.sort_index(axis=1)
        self.raw_labels = labels.sort_index(axis=1)
        self.feature_names = sorted(list(questionnaire_data.columns))
        self.questionnaire_data = questionnaire_data.sort_index(axis=1).to_numpy()
        self.labels = labels.sort_index(axis=1).to_numpy()
        self.disease_names = sorted(list(labels.columns))
        self.complete_columns, self.dict_equivalences, self.dict_implications = self.get_order_df()
        self.order_features = self.get_order_features()
        self.group_order_features = self.group_patients_by_order_features()
        # to store, train/valid/test indices from epoct and epoct plus data sets
        self.all_indices = None
        self.num_available_features = self.raw_data.count(axis=1)

    def get_order_df(self):
        """
        get the equivalences and implications (i.e. which features come at the same time in
        the decision tree and which are the ones implying others) of the features of the data set
        :return: the features that are available for everyone (i.e. corresponding to triage questions,
        a dict with the features as keys and the equivalent features as values,
        a dict with the features as keys and the predecessors in the tree as values
        """
        n_patients = len(self)
        # get features available for all patients (triage questions)
        complete_columns = [column for column in self.raw_data.columns if len(self.raw_data[column].dropna()) == n_patients]
        # for each feature, get set of patients with value for that feature
        df_features = pd.DataFrame({'feature': self.raw_data.columns})
        df_features['patient_set'] = [set(self.raw_data[feature].dropna().index) for feature in df_features['feature']]
        # dict of features such that if featureA is in dict[featureB], then featureA comes after featureB in tree , i.e. AnB=A!=B and thus A -->B but B -/-> A
        dict_implications = {feature: [] for feature in df_features['feature']}
        # dict of features that are always asked for the same group of patients and thus can be shuffled : AnB=A=B i.e. A --> B and B-->A
        dict_equivalences = {feature: [] for feature in df_features['feature']}
        for feature_A in df_features['feature']:
            A = df_features[df_features['feature'] == feature_A]['patient_set'].item()
            for feature_B in df_features['feature']:
                B = df_features[df_features['feature'] == feature_B]['patient_set'].item()
                if feature_A != feature_B:
                    if ((A & B) == A) and ((A & B) == B):
                        # A and B are equivalent
                        dict_equivalences[feature_B].append(feature_A)
                    elif (A & B) == A:
                        dict_implications[feature_B].append(feature_A)

        return complete_columns, dict_equivalences, dict_implications

    def get_order_features(self):
        """
        get the order of the features for each patient
        :return: dict with patients as keys and ordered list of features as values
        """
        order_features = {patient: {} for patient in range(len(self))}
        i = 0
        for patient in range(len(self)):
            if i % 1000 == 0:
                print('patient:', i)
            i += 1
            # get non nan features
            non_nan_features = list(self.raw_data.loc[patient][self.raw_data.notna().loc[patient]].index)
            # keep only keys that are present for that patient
            index = 0
            while len(non_nan_features) > 0:
                next_feature = self.get_next_feature(non_nan_features)
                order_features[patient][index] = sorted(next_feature)
                for feature in next_feature:
                    non_nan_features.remove(feature)
                index += 1

        return order_features

    def get_next_feature(self, non_nan_features):
        """
        Greedily retrieve the order of the features in the non missing features
        :param non_nan_features:
        :return: features that are not implied by any other features (i.e. root of subtree)
        """
        # get the features implied by each feature in non nan feature
        dict_implications_patient = {key: self.dict_implications[key].copy() for key in non_nan_features if key in self.dict_implications.keys()}
        # for each feature in dict_implications_patient.keys() drop the features in its implication list that are missing
        for key in dict_implications_patient.keys():
            for value in dict_implications_patient[key]:
                if value not in non_nan_features:
                    dict_implications_patient[key].remove(value)
        # get the features that are implied by some other feature in the non missing features
        features_implied_by_other = []
        for key in dict_implications_patient.keys():
            features_implied_by_other.extend(dict_implications_patient[key])
        features_implied_by_other = set(features_implied_by_other).intersection(non_nan_features)
        # get the features that are not in features_implied_by_other, i.e. the features that are the highest in the subtree
        next_feature = list(set(non_nan_features) - features_implied_by_other)

        return next_feature

    def group_patients_by_order_features(self):
        """
        create a nested dictionary with the levels in the tree as keys, and as values a dictionnary with the possible features at that
        level in the tree and a list of patients with values for these features at that level in the tree
        {level_0 : {(feature_1, feature_2) : [patient_1, patient_3} }
        :return: nested dictionary
        """
        # longest path in tree
        max_depth = max([len(self.order_features[patient]) for patient in self.order_features.keys()])
        # find different orders of questions
        df_question_levels = pd.DataFrame()
        # get the possible features in each level in the tree
        for question_level in range(max_depth):
            # find different groups of features
            df_question_levels['level_' + str(question_level)] = [
                self.order_features[patient][question_level] if len(self.order_features[patient]) > question_level else [] for patient in
                self.order_features.keys()]
        # create the nested dict
        question_group_dict = {level: {} for level in range(max_depth)}
        for index, column in enumerate(df_question_levels.columns):
            question_groups = df_question_levels[column].apply(tuple).unique()
            for group in question_groups:
                if group:
                    # for each level in the tree for each feature group in that level, get the list of patients with values
                    question_group_dict[index][group] = [patient for patient in range(len(self)) if df_question_levels.loc[patient, column] == list(group)]

        return question_group_dict

    def remap_test_valid_set(self):
        # mappings with keys corresponding to indices in raw data and values to indices in train/valid/test data respectively
        self.traininig_data_indices_mapping = {self.training_data.indices[index]: index for index in range(len(self.training_data.indices))}
        self.validation_data_indices_mapping = {self.validation_data.indices[index]: index for index in range(len(self.validation_data.indices))}
        self.testing_data_indices_mapping = {self.testing_data.indices[index]: index for index in range(len(self.testing_data.indices))}
        # reverse mappings with keys corresponding to indices in train/valid/test data respectively and values to indices in raw data
        self.traininig_data_indices_reverse_mapping = {value: key for key, value in self.traininig_data_indices_mapping.items()}
        self.validation_data_indices_reverse_mapping = {value: key for key, value in self.validation_data_indices_mapping.items()}
        self.testing_data_indices_reverse_mapping = {value: key for key, value in self.testing_data_indices_mapping.items()}
        group_order_train_features = {level: {feature_group: [] for feature_group in self.group_order_features[level].keys()} for level in
                                      self.group_order_features.keys()}
        group_order_valid_features = {level: {feature_group: [] for feature_group in self.group_order_features[level].keys()} for level in
                                      self.group_order_features.keys()}
        group_order_test_features = {level: {feature_group: [] for feature_group in self.group_order_features[level].keys()} for level in
                                     self.group_order_features.keys()}
        self.order_train_features = {self.traininig_data_indices_mapping[patient]: copy.deepcopy(self.order_features[patient]) for patient in
                                     self.training_data.indices}
        self.order_test_features = {self.testing_data_indices_mapping[patient]: copy.deepcopy(self.order_features[patient]) for patient in
                                    self.testing_data.indices}
        self.order_valid_features = {self.validation_data_indices_mapping[patient]: copy.deepcopy(self.order_features[patient]) for patient in
                                     self.validation_data.indices}
        for level in self.group_order_features.keys():
            for feature_group in self.group_order_features[level].keys():
                for patient in self.group_order_features[level][feature_group]:
                    if patient in self.training_data.indices:
                        group_order_train_features[level][feature_group].append(self.traininig_data_indices_mapping[patient])
                    elif patient in self.validation_data.indices:
                        group_order_valid_features[level][feature_group].append(self.validation_data_indices_mapping[patient])
                    elif patient in self.testing_data.indices:
                        group_order_test_features[level][feature_group].append(self.testing_data_indices_mapping[patient])
        self.train_group_order_features = group_order_train_features
        self.valid_group_order_features = group_order_valid_features
        self.test_group_order_features = group_order_test_features
        return
    def keep_training_indices(self, indices_to_keep):
        for level in self.train_group_order_features.keys():
            for feature_group in self.train_group_order_features[level].keys():
                self.train_group_order_features[level][feature_group] = list(set(self.train_group_order_features[level][feature_group]).intersection(indices_to_keep))

        return
#modules
class EpoctEncoder(nn.Module):
    """
    Basic feature encoder
    """

    def __init__(self, STATE_SIZE, hidden_size=32):
        super(EpoctEncoder, self).__init__()
        self.fc1 = nn.Linear(1 + STATE_SIZE, hidden_size)
        self.fc2 = nn.Linear(hidden_size, STATE_SIZE)

    def forward(self, state, x):
        x = F.relu(self.fc1(torch.cat([x, state], axis=-1)))
        return state + self.fc2(x)

    def init_weights(self, exponent=3):
        """
        Initialize first layers to random numbers close to 0, the biases to 0 and the last layer to 0
        :param exponent: parameters are initialized in a range of magnitude [-10^(-exp), 10^(-exp)]
        :return:
        """
        stdv = 1. / math.sqrt(self.fc1.weight.size(1))
        self.fc1.weight.data.uniform_(-stdv * 10 ** (-exponent), stdv * 10 ** (-exponent))
        if self.fc1.bias is not None:
            self.fc1.bias.data.fill_(0.00)
        self.fc2.weight.data.fill_(0.00)
        if self.fc2.bias is not None:
            self.fc2.bias.data.fill_(0.00)


class EpoctBinaryDecoder(nn.Module):
    """
    Categorical individual disease decoder
    """

    def __init__(self, STATE_SIZE, hidden_size=10):
        super(EpoctBinaryDecoder, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 2, bias=True)

    def forward(self, x):
        return F.log_softmax(self.fc1(x), dim=1)


class EpoctScalarDecoder(nn.Module):
    """
    Feature decoder
    """

    def __init__(self, STATE_SIZE, hidden_size=10):
        super(EpoctScalarDecoder, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class EpoctDistributionDecoder(nn.Module):
    """ Feature distribution decoder approximating mean and variance
    of distribution"""
    def __init__(self, STATE_SIZE, hidden_size = 10):
        super(EpoctDistributionDecoder, self).__init__()
        self.fc1_mu = nn.Linear(STATE_SIZE, hidden_size, bias=True)
        self.fc2_mu = nn.Linear(hidden_size, 1, bias=True)
        self.fc1_sigma = nn.Linear(STATE_SIZE, hidden_size, bias=True)
        self.fc2_sigma = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        mu = self.fc2_mu(F.relu(self.fc1_mu(x)))
        log_sigma = self.fc2_sigma(F.relu(self.fc1_sigma(x)))
        return mu, log_sigma

class InitState(nn.Module):
    """Initial state"""
    def __init__(self, STATE_SIZE):
        super(InitState, self).__init__()
        self.STATE_SIZE = STATE_SIZE
        self.state_value = torch.nn.Parameter(torch.randn([1, STATE_SIZE], requires_grad=True))
    def forward(self, n_data_points):
        init_tensor = torch.tile(self.state_value, [n_data_points,1])
        return init_tensor
