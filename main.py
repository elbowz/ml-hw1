#!/usr/bin/env python3

'''
# How to install dependencies on Arch Linux
sudo pip install -U scikit-learn
sudo pip install -U matplotlib
sudo pip install -U pandas
sudo pacman -S tk
'''

import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pprint import pprint

__default_ts_name = 'wine'
__default_test_size = 0.20

__ts_opts = {
    'wine': {
        'url': "./wine.data",
        'columns': (
            'Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
            'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
            'Proline'),
        'x_slice': (slice(None, None), slice(1, None)),
        'y_slice': (slice(None, None), 0),
    },
    'breast-cancer': {
        'url': "./breast-cancer-wisconsin.data",
        'columns': ('Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                    'Normal Nucleoli',
                    'Mitoses', 'Class'),
        'x_slice': (slice(None, None), slice(1, -1)),
        'y_slice': (slice(None, None), -1),
    },
    'letters': {
        'url': "./letter-recognition.data",
        'columns': (
            'lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr',
            'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx'),
        'x_slice': (slice(None, None), slice(1, None)),
        'y_slice': (slice(None, None), 0),

    },
    'poker': {
        'url': "./poker-hand-testing-500k.data",
        'columns': ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'),
        'x_slice': (slice(None, None), slice(0, -1)),
        'y_slice': (slice(None, None), -1),
    },
}


def knn(X_train, y_train, X_test, y_test, n_neighbors, metric='euclidean'):
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    return accuracy_score(y_test, y_pred)


# TODO: could be written better (no more time...times is end)
def proximity(X_train, y_train, X_test, y_test, metric='euclidean'):
    dist = DistanceMetric.get_metric(metric)

    X_grouped_by_y = {}

    # Grouping X by y in a dictionary
    for i in range(len(y_train)):

        if y_train[i] in X_grouped_by_y:
            X_grouped_by_y[y_train[i]].append(X_train[i].tolist())
        else:
            X_grouped_by_y[y_train[i]] = [X_train[i].tolist()]

    y_pred = []
    y_list = list(X_grouped_by_y.keys())

    # Foreach X compute the metric distance
    for X_test_item in X_test:

        distance = []

        # Append the X_test_item to each X_grouped_by_y list and compute the distance for each one
        for y, X_grouped in X_grouped_by_y.items():
            X_grouped.append(X_test_item.tolist())
            euclidian_result = dist.pairwise(X_grouped)
            X_grouped.pop()

            # Distance is on latest row (we have append the X_test_item in last position)
            distance.append(sum(euclidian_result[-1:][0]) / len(euclidian_result[-1:][0]))

        winner_index = distance.index(min(distance))
        y_pred.append(y_list[winner_index])

    return accuracy_score(y_test, y_pred)


def main(argv):
    print('Usage: {} [wine|breast-cancer][test_size]\n'.format(argv[0]))

    test_size = __default_test_size
    trainingset_selected_name = __default_ts_name

    if len(argv) > 1 and argv[1] in __ts_opts:
        trainingset_selected_name = argv[1]

        if len(argv) > 2:
            test_size = float(argv[2])

    ts_selected_opts = __ts_opts[trainingset_selected_name]

    print('\nTrainingSet selected: ' + ts_selected_opts['url'])

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(ts_selected_opts['url'], names=ts_selected_opts['columns'])

    print('\nFirst five rows of TrainingSet:\n')
    print(dataset.head())

    # Remove row with question marks
    dataset = dataset[~(dataset.astype(str) == '?').any(1)]

    print('\nDataSet Length: {}'.format(len(dataset)))

    # wine.data
    X = dataset.iloc[ts_selected_opts['x_slice'][0], ts_selected_opts['x_slice'][1]]
    y = dataset.iloc[ts_selected_opts['y_slice'][0], ts_selected_opts['y_slice'][1]]

    print('\nInput:\n')
    print(X.head())
    print('\nOutput:\n')
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size)

    k_max = 31

    metrics = ('euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming', 'canberra', 'braycurtis')

    # Knn
    for metric in metrics:

        print('\nRunning KNN with ' + metric)
        hit_rate = []

        # Calculating error for K values between 1 and 40
        for k in range(1, k_max):
            current_hit_rate = knn(X_train, y_train, X_test, y_test, k, metric)

            print('Accuracy for k={0:>2}: {1:.4f}'.format(k, current_hit_rate))
            hit_rate.append(current_hit_rate)

        plt.figure(figsize=(12, 6))
        plt.plot(range(k_max - 1), hit_rate, color='red', linestyle='dashed', marker='o', markerfacecolor='blue',
                 markersize=10)
        plt.title('HitRate Knn ({}, K [1,{}]'.format(metric, k_max - 1))
        plt.xlabel('K Value')
        plt.ylabel('HitRate')

        plt.show(block=False)
        plt.pause(0.001)
        input("Press [enter] to continue.")

    hit_rate = []

    # Proximity
    for metric in metrics:
        print('\nRunning Proximity with ' + metric)

        current_hit_rate = proximity(X_train, y_train, X_test, y_test, metric)

        print('Accuracy for metric={0}: {1:.4f}'.format(metric, current_hit_rate))
        hit_rate.append(current_hit_rate)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(metrics)), hit_rate, color='red', linestyle='dashed', marker='o', markerfacecolor='blue',
             markersize=10)
    plt.title('HitRate Proximity')
    plt.xlabel('Metric')
    plt.ylabel('HitRate')

    plt.show(block=False)
    plt.pause(0.001)
    input("Press [enter] to continue.")

    exit(0)


if __name__ == "__main__":
    main(sys.argv)
