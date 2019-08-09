#!/usr/bin/env python3

from lib import *
import matplotlib.pyplot as plt


def main():
    training_set_rate = 80
    category_position = 0
    file_csv = './wine.data'

    # Split data set
    data_set, training_set, test_set = load_dataset(file_csv, training_set_rate, category_position)
    print('Whole data set: ' + str(len(data_set)))
    print('Train set: ' + str(len(training_set)))
    print('Test set: ' + str(len(test_set)))

    k_max = 30
    error = []

    # Calculating error for K values between 1 and k_max
    for k in range(1, k_max):

        predictions = []

        # Printed table columns
        print('\n N  Predicted  TestSet  Result')

        for x in range(len(test_set)):
            neighbors = get_neighbors(training_set, test_set[x], k, category_position)
            result = get_response(neighbors, category_position)
            predictions.append(result)

            print('{:>2}  {:^9}  {:^7}  {}'
                  .format(x, result, test_set[x][category_position], ('👍' if result == test_set[x][category_position] else '👎')))

        hit_rate = get_hit_rate(test_set, predictions, category_position)
        print('\nAccuracy for k={0}: {1:.2f}%'.format(k, hit_rate))

        error.append(hit_rate)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, k_max), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=6)
    plt.title('Hit Rate for K Value')
    plt.xlabel('K Value')
    plt.ylabel('Hit Rate')

    plt.show()


main()
