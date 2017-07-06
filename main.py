import csv
import sys
import property_validity_guesser.learner as learner
import property_validity_guesser.preprocessor as preprocessor

csv.field_size_limit(sys.maxsize)

csv.register_dialect(
    'tsv_dialect',
    delimiter='\t',
    quotechar='"',
    doublequote=True,
    skipinitialspace=True,
    lineterminator='\r\n',
    quoting=csv.QUOTE_MINIMAL)

RESIDUAL_EPS = 1.0e-5
L1_PENALTY = 1.0e-3
LEARNING_RATE = 1.0e-1
MAX_ITER = 20000

N_RANDOM_TEST = 3
TRAIN_RATIO = 0.9

LABEL_OUT = ["Correct", "Incorrect"]  # label 0 = correct, label 1 = incorrect


##==============================================================================
def predict_testing_property_validity_guesser(learning_model, x, y):
    '''
    :param learning_model: object for logistic regression
    :param x: input feature data
    :param y: input label data
    :param print_accuracy: boolean value to print/or not the accuracy etc
    :return: print result metrics
             result_label_string
    '''
    tp = 0 #true positive
    fp = 0 #false positive
    fn = 0 #false negative
    tn = 0 #true negative

    ndata_real_label_1 = 0 #label to count how many data with label "1" over total data

    for i, x_i in enumerate(x):
        label_i = learner.predict(learning_model, x_i)

        if y[i] == 1:
            ndata_real_label_1 += 1

            if label_i == 1:
                tp += 1
            else:
                fn += 1
        else:
            if label_i == 1:
                fp += 1
            else:
                tn += 1

    print 'n-data with label 1 = ', ndata_real_label_1, \
          ' of total data: ', x.shape[0], \
          '(', 100.0 * ndata_real_label_1 / x.shape[0], '%)'

    print 'accuracy  = ', float(tp + tn) / float(tp + tn + fp + fn), \
          '(', tp + tn, '/', tp + tn + fp + fn, ')'

    if (tp + fp) > 0:
        print 'precision = ', float(tp) / float(tp + fp)
    else:
        print 'precision = ERROR, (tp + fp) == 0'

    if (tp + fn) > 0:
        print 'recall    = ', float(tp) / float(tp + fn)
    else:
        print 'recall    = ERROR, (tp + fn) == 0'

    print 'tp = ', tp, 'tn = ', tn, 'fp = ', fp, 'fn = ', fn

##==============================================================================
if __name__ == "__main__":
    learner.build_location_blacklist("property_blacklist.csv", "location_blacklist.p")
    location_blacklist = learner.load_location_blacklist("location_blacklist.p")

    print "================================================="
    print "=============== accuracy testing ================"
    print "================================================="
    for k in range(N_RANDOM_TEST):
        seed = int(str(k+1) + str(k+2) + str(k+3) + str(k+4) + str(k+5) + str(k+6))
        print k, "seed :", seed

        ## build learning data
        x, y = learner.build_learning_data("data_reorganized.tsv", location_blacklist, seed)

        ## build learning model
        learner.build_model(x, y, TRAIN_RATIO, "learning_model.p", print_loss=True)
        learning_model = learner.load_model("learning_model.p")

        ## predict accuracy
        ndata = y.shape[0]
        train = int(TRAIN_RATIO * ndata)

        x_test = x[train:ndata]
        y_test = y[train:ndata]
        predict_testing_property_validity_guesser(learning_model, x_test, y_test)

        print "________________________________________________"


    print "================================================="
    print "================ short testing =================="
    print "================================================="

    x, y = learner.build_learning_data("data_reorganized.tsv",location_blacklist)

    learner.build_model(x, y, 1.0, "learning_model.p", print_loss=True)
    learning_model = learner.load_model("learning_model.p")

    preprocessor_row = preprocessor.PreprocessorRow()
    preprocessor_row.load_location_blacklist(location_blacklist)

    with open("data/short_data_reorganized_for_testing.tsv") as csvfile:
        reader = csv.reader(csvfile, dialect='tsv_dialect')
        next(reader, None) # skip the headers

        i = 0
        for row in reader:
            property_id, xt = preprocessor_row.process_row(row)
            label_i = learner.predict(learning_model, xt)

            print "property id :", property_id, \
                  " :: real label:", LABEL_OUT[y[i]], \
                  "predicted label:", LABEL_OUT[label_i]
            i += 1


