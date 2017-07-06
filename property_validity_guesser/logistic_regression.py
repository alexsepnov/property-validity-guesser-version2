import numpy as np
import pickle

##==============================================================================
class LogRegParam:
    max_iteration = 200         # Max iterations
    epsilon = 1.0e-6            # Convergence threshold, suggested <= 1.0e-4
    learning_rate = 5.0e-3      # Learning rate
    l1_penalty = 1.0e-4         # L1 penalty weight
    probability_threshold = 0.5

##==============================================================================
class LogRegL1(object):
    def __init__(self, residual_eps=LogRegParam.epsilon,
                       l1_penalty=LogRegParam.l1_penalty,
                       learning_rate=LogRegParam.learning_rate,
                       max_iter=LogRegParam.max_iteration,
                       probability_threshold=LogRegParam.probability_threshold):

        '''
        :param residual_eps: residual limit for convergence
        :param max_iter: maximum iteration
        :param learning_rate: learning rate value
        :param l1_penalty: L1 regularization value
        :param probability_threshold: minimum probability value to be defined as true/1
        '''

        self.residual_eps = residual_eps
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.l1_penalty = l1_penalty
        self.probability_threshold = probability_threshold

        self.weight = None
        self.n_feature = 0

    @staticmethod
    def l2_norm_vector(vector1, vector2):
        '''
        :param vector1: n-dim vector input 1
        :param vector2: n-dim vector input 2
        :return: L2 Norm value
        '''
        return np.linalg.norm(vector1 - vector2)

    @staticmethod
    def sigmoid_func(weight, feature):
        '''
        :param weight: n-dim weight
        :param feature: n-dim feature
        :return: sigmoid value
        '''
        return 1.0e0 / (1.0e0 + np.exp(-weight.dot(feature)))

    @staticmethod
    def predict_probability(weight, x_i):
        '''
        :param weight: n-dim weights
        :param x_i: the i-th of input feature data input with size n-dim
        :return: 1. probability (sigmoid value)
                 2. feature --> x_i + bias feature(=1)
        '''
        feat = x_i.copy()
        feat = np.append(feat, 1.0)
        return LogRegL1.sigmoid_func(weight, feat), feat

    @staticmethod
    def loss_function(weight, x, y):
        '''
        :param weight:
        :param x: input feature data
        :param y: input label data
        :return: loss value
        '''
        loss = 0.0
        n_data = x.shape[0]
        for i in range(n_data):
            sigmoid, _ = LogRegL1.predict_probability(weight, x[i])
            loss -= y[i] * np.log(sigmoid) + (1 - y[i]) * np.log(1 - sigmoid)

        return loss / n_data

    def train(self, x, y, print_loss=True, print_loss_interval=100):
        '''
        :param x: input feature data
        :param y: input label data
        :param print_loss: boolean value to print/or not the loss during learning
        :return: print loss
        '''

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        n_data = x.shape[0]
        self.n_feature = x.shape[1]
        self.weight = np.zeros(self.n_feature+1)
        l1_total = np.zeros(self.n_feature+1)

        normal_residual = 1.0
        yu = 0.0

        iteration = 0
        while normal_residual > self.residual_eps and iteration <= self.max_iter:
            iteration += 1

            old_weight = self.weight.copy()
            yu += self.learning_rate * self.l1_penalty

            for i in range(0, n_data):
                predict, xt = LogRegL1.predict_probability(self.weight, x[i])
                self.weight -= LogRegParam.learning_rate * (predict - y[i]) * xt

                for j in range(0, self.n_feature+1):
                    wt = self.weight[j]  #this is deep copy

                    if self.weight[j] > 0.e0:
                        self.weight[j] = max(0.0e0, self.weight[j] - (yu + l1_total[j]))
                    elif self.weight[j] < 0.0e0:
                        self.weight[j] = min(0.0e0, self.weight[j] + (yu - l1_total[j]))

                    l1_total[j] += (self.weight[j] - wt)

            normal_residual = LogRegL1.l2_norm_vector(old_weight, self.weight)

            if iteration % print_loss_interval == 0 and print_loss:
                print 'iter :', iteration, 'convergence: ', normal_residual, \
                      'loss: ', LogRegL1.loss_function(self.weight, x, y)

        if print_loss:
            print 'final iter :', iteration, 'convergence: ', normal_residual, \
                  'loss: ', LogRegL1.loss_function(self.weight, x, y)

        self.print_weight()

    def get_weight(self):
        '''
        :return: weight
        '''
        return self.weight

    def print_weight(self):
        '''
        :return: print weight
        '''
        print "weight :", self.weight

    def predict(self, x_i):
        '''
        :param x: input feature data
        :return: label, probability
        '''

        if not isinstance(x_i, np.ndarray):
            x_i = np.array(x_i)

        probability, _ = LogRegL1.predict_probability(self.weight, x_i)
        label = 1 if probability >= self.probability_threshold else 0

        return label, probability

    def predict_testing(self, x, y, print_accuracy=True):
        '''
        :param x: input feature data
        :param y: input label data
        :param print_accuracy: boolean value to print/or not the accuracy etc
        :return: number of true positive data(tp)
                 number of true negative data(tn)
                 number of false positive data(fp)
                 number of false negative data(fn)
        '''
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        ndata_real_label_1 = 0

        for i, x_i in enumerate(x):
            prob_t, _ = LogRegL1.predict_probability(self.weight, x_i)

            if y[i] == 1:
                ndata_real_label_1 += 1

                if prob_t >= self.probability_threshold:
                    tp += 1
                else:
                    fn += 1
            else:
                if prob_t >= self.probability_threshold:
                    fp += 1
                else:
                    tn += 1

        if print_accuracy:
            print 'label 1 data = ', ndata_real_label_1, ' of total data: ', x.shape[0], \
                  '(', 100.0 * ndata_real_label_1 / x.shape[0], '%)'

            print 'accuracy  = ', float(tp + tn) / float(tp + tn + fp + fn), \
                  '(', tp + tn, '/', tp + tn + fp + fn, ')'

            if(tp + fp) > 0:
                print 'precision = ', float(tp) / float(tp + fp)
            else:
                print 'precision = ERROR, (tp + fp) == 0'

            if(tp + fn) > 0:
                print 'recall    = ', float(tp) / float(tp + fn)
            else:
                print 'recall    = ERROR, (tp + fn) == 0'

            print 'tp = ', tp, 'tn = ', tn, 'fp = ', fp, 'fn = ', fn

        return tp, tn, fp, fn

    def write_to_file(self, filename):
        pickle.dump(self, open(filename, "wb"))

    @staticmethod
    def load_from_file(filename):
        logistic_regression_object = pickle.load(open(filename, "rb"))
        return logistic_regression_object
