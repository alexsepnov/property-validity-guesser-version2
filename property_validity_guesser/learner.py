import os
import numpy as np
import property_validity_guesser.logistic_regression as logistic_regression
import property_validity_guesser.preprocessor as preprocessor

data_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../data")

RESIDUAL_EPS = 1.0e-5
L1_PENALTY = 1.0e-3
LEARNING_RATE = 1.0e-1
MAX_ITER = 20000
PROBABILITY_THRESHOLD = 0.5

##==============================================================================
def build_location_blacklist(blacklist_filename, blacklist_object_filename):
    '''
    :param blacklist_filename:
    :param blacklist_object_filename:
    :return:
    '''
    location_blacklist = preprocessor.LocationBlacklist()
    location_blacklist.process_file(data_path + "/" + blacklist_filename)
    location_blacklist.write_to_file(data_path + "/" + blacklist_object_filename)

##==============================================================================
def load_location_blacklist(blacklist_object_filename):
    '''
    :param blacklist_object_filename:
    :return:
    '''
    location_blacklist = preprocessor.LocationBlacklist.\
        load_from_file(data_path + "/" + blacklist_object_filename)
    return location_blacklist

##==============================================================================
def build_learning_data(features_data_filename, location_blacklist_object, seed=0):
    '''
    :param features_data_filename:
    :param location_blacklist_object:
    :param seed:
    :return:
    '''
    data_preprocessor = preprocessor.PreprocessorLearning()
    data_preprocessor.load_location_blacklist(location_blacklist_object)
    data_preprocessor.process_file(data_path + "/" + features_data_filename)
    x, y = data_preprocessor.get_data()

    if seed == 0:
        return x, y
    else:
        ndata = y.shape[0]
        np.random.seed(seed)
        p = np.random.permutation(ndata)
        return x[p], y[p]

##==============================================================================
def build_model(x, y, train_ratio, learning_model_object_filename,
                print_loss=False, print_loss_interval=500):
    '''
    :param x:
    :param y:
    :param train_ratio:
    :param learning_model_object_filename:
    :param print_loss:
    :param print_loss_interval:
    :return:
    '''

    ndata = y.shape[0]
    train = int(train_ratio * ndata)

    x_train = x[0:train]
    y_train = y[0:train]

    learning_model = logistic_regression.LogRegL1(residual_eps=RESIDUAL_EPS,
                     l1_penalty=L1_PENALTY, learning_rate=LEARNING_RATE,
                     max_iter=MAX_ITER, probability_threshold=PROBABILITY_THRESHOLD)
    learning_model.train(x_train, y_train, print_loss=print_loss,
                         print_loss_interval=print_loss_interval)
    learning_model.write_to_file(data_path + "/" + learning_model_object_filename)

##==============================================================================
def load_model(learning_model_object_filename):
    '''
    :param learning_model_object_filename:
    :return:
    '''
    learning_model = logistic_regression.LogRegL1.\
        load_from_file(data_path + "/" + learning_model_object_filename)
    return learning_model

##==============================================================================
def predict(learning_model, input_row):
    '''
    :param learning_model:
    :param input_row:
    :return:
    '''
    label_i, _ = learning_model.predict(input_row)
    return label_i


