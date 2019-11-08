import numpy as np

from sklearn import naive_bayes, svm
from sklearn.metrics import confusion_matrix, classification_report

if __name__ == "__main__":

    # Get training and testing files
    train_full_mixed_features_path = 'datasets/parsed/train_full_mixed_features.csv'
    train_full_number_occurrences_path = 'datasets/parsed/train_full_number_occurrences.csv'
    train_full_ordered_instructions_path = 'datasets/parsed/train_full_ordered_instructions.csv'
    train_partial_mixed_features_path = 'datasets/parsed/train_partial_mixed_features.csv'
    train_partial_number_occurrences_path = 'datasets/parsed/train_partial_number_occurrences.csv'
    train_partial_ordered_instructions_path = 'datasets/parsed/train_partial_ordered_instructions.csv'
    test_full_mixed_features_path = 'datasets/parsed/test_full_mixed_features.csv'
    test_full_number_occurrences_path = 'datasets/parsed/test_full_number_occurrences.csv'
    test_full_ordered_instructions_path = 'datasets/parsed/test_full_ordered_instructions.csv'
    test_partial_mixed_features_path = 'datasets/parsed/test_partial_mixed_features.csv'
    test_partial_number_occurrences_path = 'datasets/parsed/test_partial_number_occurrences.csv'
    test_partial_ordered_instructions_path = 'datasets/parsed/test_partial_ordered_instructions.csv'

    # splitting into features(x) and labels(y)
    train_full_mixed_features_x = list()
    train_full_mixed_features_y = list()
    train_full_number_occurrences_x = list()
    train_full_number_occurrences_y = list()
    train_full_ordered_instructions_x = list()
    train_full_ordered_instructions_y = list()
    train_partial_mixed_features_x = list()
    train_partial_mixed_features_y = list()
    train_partial_number_occurrences_x = list()
    train_partial_number_occurrences_y = list()
    train_partial_ordered_instructions_x = list()
    train_partial_ordered_instructions_y = list()
    test_full_mixed_features_x = list()
    test_full_mixed_features_y = list()
    test_full_number_occurrences_x = list()
    test_full_number_occurrences_y = list()
    test_full_ordered_instructions_x = list()
    test_full_ordered_instructions_y = list()
    test_partial_mixed_features_x = list()
    test_partial_mixed_features_y = list()
    test_partial_number_occurrences_x = list()
    test_partial_number_occurrences_y = list()
    test_partial_ordered_instructions_x = list()
    test_partial_ordered_instructions_y = list()

    with open(train_full_mixed_features_path, mode='r') as fd_train_full_mf, \
        open(train_full_number_occurrences_path, mode='r') as fd_train_full_no, \
        open(train_full_ordered_instructions_path, mode='r') as fd_train_full_oi, \
        open(train_partial_mixed_features_path, mode='r') as fd_train_partial_mf, \
        open(train_partial_number_occurrences_path, mode='r') as fd_train_partial_no, \
        open(train_partial_ordered_instructions_path, mode='r') as fd_train_partial_oi, \
        open(test_full_mixed_features_path, mode='r') as fd_test_full_mf, \
        open(test_full_number_occurrences_path, mode='r') as fd_test_full_no, \
        open(test_full_ordered_instructions_path, mode='r') as fd_test_full_oi, \
        open(test_partial_mixed_features_path, mode='r') as fd_test_partial_mf, \
        open(test_partial_number_occurrences_path, mode='r') as fd_test_partial_no, \
        open(test_partial_ordered_instructions_path, mode='r') as fd_test_partial_oi:

        for line in fd_train_full_mf:

            splitted = line.split(',')

            train_full_mixed_features_x.append(list(int(x) for x in splitted[0].split(" ")))
            train_full_mixed_features_y.append(int(splitted[1]))

        for line in fd_train_full_no:

            splitted = line.split(',')

            train_full_number_occurrences_x.append(list(int(x) for x in splitted[0].split(" ")))
            train_full_number_occurrences_y.append(int(splitted[1]))

        for line in fd_train_full_oi:

            splitted = line.split(',')
            
            train_full_ordered_instructions_x.append(list(int(x) for x in splitted[0].split(" ")))
            train_full_ordered_instructions_y.append(int(splitted[1]))

        for line in fd_train_partial_mf:

            splitted = line.split(',')

            train_partial_mixed_features_x.append(list(int(x) for x in splitted[0].split(" ")))
            train_partial_mixed_features_y.append(int(splitted[1]))

        for line in fd_train_partial_no:

            splitted = line.split(',')

            train_partial_number_occurrences_x.append(list(int(x) for x in splitted[0].split(" ")))
            train_partial_number_occurrences_y.append(int(splitted[1]))

        for line in fd_train_partial_oi:

            splitted = line.split(',')

            train_partial_ordered_instructions_x.append(list(int(x) for x in splitted[0].split(" ")))
            train_partial_ordered_instructions_y.append(int(splitted[1]))

        for line in fd_test_full_mf:

            splitted = line.split(',')

            test_full_mixed_features_x.append(list(int(x) for x in splitted[0].split(" ")))
            test_full_mixed_features_y.append(int(splitted[1]))

        for line in fd_test_full_no:

            splitted = line.split(',')

            test_full_number_occurrences_x.append(list(int(x) for x in splitted[0].split(" ")))
            test_full_number_occurrences_y.append(int(splitted[1]))

        for line in fd_test_full_oi:

            splitted = line.split(',')
            
            test_full_ordered_instructions_x.append(list(int(x) for x in splitted[0].split(" ")))
            test_full_ordered_instructions_y.append(int(splitted[1]))

        for line in fd_test_partial_mf:

            splitted = line.split(',')

            test_partial_mixed_features_x.append(list(int(x) for x in splitted[0].split(" ")))
            test_partial_mixed_features_y.append(int(splitted[1]))

        for line in fd_test_partial_no:

            splitted = line.split(',')

            test_partial_number_occurrences_x.append(list(int(x) for x in splitted[0].split(" ")))
            test_partial_number_occurrences_y.append(int(splitted[1]))

        for line in fd_test_partial_oi:

            splitted = line.split(',')

            test_partial_ordered_instructions_x.append(list(int(x) for x in splitted[0].split(" ")))
            test_partial_ordered_instructions_y.append(int(splitted[1]))
            
    inp = input("which model do you want to try?:(nb for naive-bayes, svm for support vector machine)\n")

    if inp == 'nb':

        # full mf
        gaussian_model_full_mf = naive_bayes.GaussianNB()
        gaussian_model_full_mf.fit(train_full_mixed_features_x, train_full_mixed_features_y)

        # printing results
        print("Training test accuracy with naive-bayes, using mixed features, full mapping: {}".
            format(gaussian_model_full_mf.score(train_full_mixed_features_x, train_full_mixed_features_y)))
        print("Testing test accuracy with naive-bayes, using mixed features, full mapping: {}".
            format(gaussian_model_full_mf.score(test_full_mixed_features_x, test_full_mixed_features_y)))

        y_pred = gaussian_model_full_mf.predict(test_full_mixed_features_x)
        print(confusion_matrix(test_full_mixed_features_y, y_pred))
        print(classification_report(test_full_mixed_features_y, y_pred))

        # partial mf
        gaussian_model_partial_mf = naive_bayes.GaussianNB()
        gaussian_model_partial_mf.fit(train_partial_mixed_features_x, train_partial_mixed_features_y)

        # printing results
        print("Training test accuracy with naive-bayes, using mixed features, partial mapping: {}".
            format(gaussian_model_partial_mf.score(train_partial_mixed_features_x, train_partial_mixed_features_y)))
        print("Testing test accuracy with naive-bayes, using mixed features, partial mapping: {}".
            format(gaussian_model_partial_mf.score(test_partial_mixed_features_x, test_partial_mixed_features_y)))

        y_pred = gaussian_model_partial_mf.predict(test_partial_mixed_features_x)
        print(confusion_matrix(test_partial_mixed_features_y, y_pred))
        print(classification_report(test_partial_mixed_features_y, y_pred))

        # full no
        gaussian_model_full_no = naive_bayes.GaussianNB()
        gaussian_model_full_no.fit(train_full_number_occurrences_x, train_full_number_occurrences_y)

        # printing results
        print("Training test accuracy with naive-bayes, using number occurrences, full mapping: {}".
            format(gaussian_model_full_no.score(train_full_number_occurrences_x, train_full_number_occurrences_y)))
        print("Testing test accuracy with naive-bayes, using number occurrences, full mapping: {}".
            format(gaussian_model_full_no.score(test_full_number_occurrences_x, test_full_number_occurrences_y)))

        y_pred = gaussian_model_full_no.predict(test_full_number_occurrences_x)
        print(confusion_matrix(test_full_number_occurrences_y, y_pred))
        print(classification_report(test_full_number_occurrences_y, y_pred))

        # partial no
        gaussian_model_partial_no = naive_bayes.GaussianNB()
        gaussian_model_partial_no.fit(train_partial_number_occurrences_x, train_partial_number_occurrences_y)

        # printing results
        print("Training test accuracy with naive-bayes, using number occurrences, partial mapping: {}".
            format(gaussian_model_partial_no.score(train_partial_number_occurrences_x, train_partial_number_occurrences_y)))
        print("Testing test accuracy with naive-bayes, using number occurrences, partial mapping: {}".
            format(gaussian_model_partial_no.score(test_partial_number_occurrences_x, test_partial_number_occurrences_y)))

        y_pred = gaussian_model_partial_no.predict(test_partial_number_occurrences_x)
        print(confusion_matrix(test_partial_number_occurrences_y, y_pred))
        print(classification_report(test_partial_number_occurrences_y, y_pred))

        # full oi
        gaussian_model_full_oi = naive_bayes.GaussianNB()
        gaussian_model_full_oi.fit(train_full_ordered_instructions_x, train_full_ordered_instructions_y)

        # printing results
        print("Training test accuracy with naive-bayes, using ordered instructions, full mapping: {}".
            format(gaussian_model_full_oi.score(train_full_ordered_instructions_x, train_full_ordered_instructions_y)))
        print("Testing test accuracy with naive-bayes, using ordered instructions, full mapping: {}".
            format(gaussian_model_full_oi.score(test_full_ordered_instructions_x, test_full_ordered_instructions_y)))

        y_pred = gaussian_model_full_oi.predict(test_full_ordered_instructions_x)
        print(confusion_matrix(test_full_ordered_instructions_y, y_pred))
        print(classification_report(test_full_ordered_instructions_y, y_pred))

        # partial oi
        gaussian_model_partial_oi = naive_bayes.GaussianNB()
        gaussian_model_partial_oi.fit(train_partial_ordered_instructions_x, train_partial_ordered_instructions_y)

        # printing results
        print("Training test accuracy with naive-bayes, using ordered instructions, partial mapping: {}".
            format(gaussian_model_partial_oi.score(train_partial_ordered_instructions_x, train_partial_ordered_instructions_y)))
        print("Testing test accuracy with naive-bayes, using ordered instructions, partial mapping: {}".
            format(gaussian_model_partial_oi.score(test_partial_ordered_instructions_x, test_partial_ordered_instructions_y)))

        y_pred = gaussian_model_partial_oi.predict(test_partial_ordered_instructions_x)
        print(confusion_matrix(test_partial_ordered_instructions_y, y_pred))
        print(classification_report(test_partial_ordered_instructions_y, y_pred))

    elif inp == 'svm':

        # full mf
        svm_model_full_mf = svm.SVC(gamma='scale')
        svm_model_full_mf.fit(train_full_mixed_features_x, train_full_mixed_features_y)

        # printing results
        print("Training test accuracy with svm, using mixed features, full mapping: {}".
            format(svm_model_full_mf.score(train_full_mixed_features_x, train_full_mixed_features_y)))
        print("Testing test accuracy with svm, using mixed features, full mapping: {}".
            format(svm_model_full_mf.score(test_full_mixed_features_x, test_full_mixed_features_y)))

        y_pred = svm_model_full_mf.predict(test_full_mixed_features_x)
        print(confusion_matrix(test_full_mixed_features_y, y_pred))
        print(classification_report(test_full_mixed_features_y, y_pred))

        # partial mf
        svm_model_partial_mf = svm.SVC(gamma='scale')
        svm_model_partial_mf.fit(train_partial_mixed_features_x, train_partial_mixed_features_y)

        # printing results
        print("Training test accuracy with svm, using mixed features, partial mapping: {}".
            format(svm_model_partial_mf.score(train_partial_mixed_features_x, train_partial_mixed_features_y)))
        print("Testing test accuracy with svm, using mixed features, partial mapping: {}".
            format(svm_model_partial_mf.score(test_partial_mixed_features_x, test_partial_mixed_features_y)))

        y_pred = svm_model_partial_mf.predict(test_partial_mixed_features_x)
        print(confusion_matrix(test_partial_mixed_features_y, y_pred))
        print(classification_report(test_partial_mixed_features_y, y_pred))

        # full no
        svm_model_full_no = svm.SVC(gamma='scale')
        svm_model_full_no.fit(train_full_number_occurrences_x, train_full_number_occurrences_y)

        # printing results
        print("Training test accuracy with svm, using number occurrences, full mapping: {}".
            format(svm_model_full_no.score(train_full_number_occurrences_x, train_full_number_occurrences_y)))
        print("Testing test accuracy with svm, using number occurrences, full mapping: {}".
            format(svm_model_full_no.score(test_full_number_occurrences_x, test_full_number_occurrences_y)))

        y_pred = svm_model_full_no.predict(test_full_number_occurrences_x)
        print(confusion_matrix(test_full_number_occurrences_y, y_pred))
        print(classification_report(test_full_number_occurrences_y, y_pred))

        # partial no
        svm_model_partial_no = svm.SVC(gamma='scale')
        svm_model_partial_no.fit(train_partial_number_occurrences_x, train_partial_number_occurrences_y)

        # printing results
        print("Training test accuracy with svm, using number occurrences, partial mapping: {}".
            format(svm_model_partial_no.score(train_partial_number_occurrences_x, train_partial_number_occurrences_y)))
        print("Testing test accuracy with svm, using number occurrences, partial mapping: {}".
            format(svm_model_partial_no.score(test_partial_number_occurrences_x, test_partial_number_occurrences_y)))

        y_pred = svm_model_partial_no.predict(test_partial_number_occurrences_x)
        print(confusion_matrix(test_partial_number_occurrences_y, y_pred))
        print(classification_report(test_partial_number_occurrences_y, y_pred))

        # full oi
        svm_model_full_oi = svm.SVC(gamma='scale')
        svm_model_full_oi.fit(train_full_ordered_instructions_x, train_full_ordered_instructions_y)

        # printing results
        print("Training test accuracy with svm, using ordered instructions, full mapping: {}".
            format(svm_model_full_oi.score(train_full_ordered_instructions_x, train_full_ordered_instructions_y)))
        print("Testing test accuracy with svm, using ordered instructions, full mapping: {}".
            format(svm_model_full_oi.score(test_full_ordered_instructions_x, test_full_ordered_instructions_y)))

        y_pred = svm_model_full_oi.predict(test_full_ordered_instructions_x)
        print(confusion_matrix(test_full_ordered_instructions_y, y_pred))
        print(classification_report(test_full_ordered_instructions_y, y_pred))

        # partial oi
        svm_model_partial_oi = svm.SVC(gamma='scale')
        svm_model_partial_oi.fit(train_partial_ordered_instructions_x, train_partial_ordered_instructions_y)

        # printing results
        print("Training test accuracy with svm, using ordered instructions, partial mapping: {}".
            format(svm_model_partial_oi.score(train_partial_ordered_instructions_x, train_partial_ordered_instructions_y)))
        print("Testing test accuracy with svm, using ordered instructions, partial mapping: {}".
            format(svm_model_partial_oi.score(test_partial_ordered_instructions_x, test_partial_ordered_instructions_y)))

        y_pred = svm_model_partial_oi.predict(test_partial_ordered_instructions_x)
        print(confusion_matrix(test_partial_ordered_instructions_y, y_pred))
        print(classification_report(test_partial_ordered_instructions_y, y_pred))
