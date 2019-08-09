import csv
import random
import math
import operator
#import numpy as np
from pprint import pprint


def load_dataset(filename, training_set_rate, ignore_pos=-1):
    with open(filename, 'rt') as csvfile:

        lines = csv.reader(csvfile)
        data_set = list(lines)

        # Strip lines with ?
        data_set = [row for row in data_set if '?' not in row]

        # Numeber of dataset row and number of features (csv line length)
        dataset_len = len(data_set)
        features_len = len(data_set[0])

        # Convert csv string to float
        for x in range(dataset_len):

            for y in range(features_len):

                if y is not ignore_pos:
                    data_set[x][y] = float(data_set[x][y])

        training_set = []
        training_set_len = int(round(dataset_len / 100 * training_set_rate))

        # Create the TrainingSet respecting the rate
        while training_set_len:

            # Splitting the training set randomly
            random_input_row = random.randint(0, dataset_len - 1)

            # add to training set respecting the training_set_rate and if not just present
            if training_set_len and (data_set[random_input_row] not in training_set):
                training_set.append(data_set[random_input_row])
                training_set_len -= 1

        # Fill the TestSet with the remain input (not in TrainingSet)
        test_set = [x for x in data_set if x not in training_set]

    return data_set, training_set, test_set


def euclidean_distance(instance1, instance2, ignore_pos=-1):
    distance = 0

    for x in range(len(instance1)):
        if x is not ignore_pos:
            distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(training_set, test_instance, k, category_position=0):
    distances = []

    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], category_position)
        distances.append((training_set[x], dist))

    distances.sort(key=lambda a: a[1])

    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])

    return neighbors


def get_response(neighbors, category_position=0):
    cateory_votes = {}

    for x in range(len(neighbors)):

        response = neighbors[x][category_position]

        if response in cateory_votes:
            cateory_votes[response] += 1
        else:
            cateory_votes[response] = 1

    sorted_votes = sorted(cateory_votes.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_votes[0][0]


def get_hit_rate(test_set, predictions, category_position=0):
    correct = 0

    for x in range(len(test_set)):
        if test_set[x][category_position] == predictions[x]:
            correct += 1

    return (correct / float(len(test_set))) * 100.0
