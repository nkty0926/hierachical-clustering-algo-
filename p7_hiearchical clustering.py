'''
// Main File:        pokemon_stats.py
// Semester:         CS 540 Fall 2020
// Authors:          Tae Yong Namkoong
// CS Login:         namkoong
// NetID:            kiatvithayak
// References:       TA's & Peer Mentor's Office Hours
                     https://docs.python.org/3/library/csv.html
                     https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
                     https://stackoverflow.com/questions/9838861/scipy-linkage-format
                     https://stackoverflow.com/questions/15951711/how-to-compute-cluster-assignments-from-linkage-distance-matrices-in-scipy-in-py
                     https://docs.python.org/3/library/csv.html#csv.DictReader
                     https://towardsdatascience.com/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019
                     https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318

'''
import csv
import math
import numpy as np
from scipy.cluster.hierarchy import linkage

def load_data(filepath):
    """
    This method takes in a string with a path to a CSV file formatted as in the link above, and returns the first 20 data points
    (without the Generation and Legendary columns but retaining all other columns) in a single structure.
    :param filepath: path of the file
    :return: list of data in dictionary
    """
    # produce dictionaries in a list
    data = []
    # open filepath
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # remove generation and generation columns
            row.pop('Generation')
            row.pop('Legendary')
            # convert columns with numerical data to int
            for key in row:
                try:
                    row[key] = int(row[key])
                except:
                    pass
            data.append(row)
            #only should have the first 20 Pokemon in this structure
            data = data[0:20]
    return data

def calculate_x_y(stats):
    """
    This method takes in one row from the data loaded from the previous function, calculates the corresponding x, y
    values for that Pokemon as specified above, and returns them in a single structure.
    :param stats: row of data from load_data function
    :return: x, y value of the row as a tuple
    """
    x,y = 0,0
    for key, value in stats.items():
        # calculate x value
        if (key == 'Attack') or (key == 'Sp. Atk') or (key == 'Speed'):
            x += value
        # calculate y value
        elif (key == 'Defense') or (key == 'Sp. Def') or (key == 'HP'):
            y += value
        # return tuple (x,y)
    return (x, y)

def recursive_hac(cluster_list, matrix, idx):
    """
    This method is a helper recursive method for the hac function that counts # of clusterings
    :param cluster_list: list of clusters
    :param matrix: matrix
    :param idx: new index to update
    :return: recursively call this function until end
    """
    length = len(cluster_list)
    if length <= 1:
        return
    # computer shortest distance of cluster lists for distance array
    distance_arr = np.empty((length, length))
    for i in range(0, length - 1):
        for j in range(i + 1, length):
            distance_arr[i][j] = shortest_distance(cluster_list[i]['cluster'], cluster_list[j]['cluster'])

    # continuously find minimum distance
    minimum = distance_arr[0][1]
    location = (0, 1)
    for i in range(0, length - 1):
        for j in range(i + 1, length):
            # Tie Breaking: In the event that there are multiple pairs of points with equal distance for
            # the next cluster
            if distance_arr[i][j] < minimum:
                minimum = distance_arr[i][j]
                location = (i, j)
    a, b = location

    # we prefer the pair with the smallest second cluster index.
    # update new_row in cluster a and b
    new_row = [cluster_list[a]['idx'], cluster_list[b]['idx'], minimum, len(cluster_list[a]['cluster'])
               + len(cluster_list[b]['cluster'])]
    matrix.append(new_row)

    # update clusters by popping b first
    cluster_2 = cluster_list.pop(b)
    cluster_1 = cluster_list.pop(a)
    cluster = {'cluster': cluster_1['cluster'] + cluster_2['cluster'], 'idx': idx}

    cluster_list.append(cluster)
    # increment idx
    idx += 1
    # recursively call this function until end
    recursive_hac(cluster_list, matrix, idx)

def hac(dataset):
    """
    This method performs single linkage hierarchical agglomerative clustering on the Pokemon with the (x,y) feature
    representation, and returns a data structure representing the clustering.
    param: dataset: a m x n data structure
    return: a m-1 x 4 matrix representing the clustering
    """
    matrix = []
    cluster = []
    # update single linkage hac recursively
    for i in range(len(dataset)):
        dict = {'cluster': [tuple(dataset[i])], 'idx': i}
        cluster.append(dict)
    length = len(dataset)

    # recursively call helper function
    recursive_hac(cluster, matrix, length)
    output = np.array(matrix)
    return output

def distance(point_1, point_2):
    """
    This function calculates the euclidean distance
    :param point_1: first point
    :param point_2: second point
    :return: euclidean distance
    """
    # compute the euclidean distance between point_1 and point_2 using given equation
    distance = math.sqrt(((point_1[0] - point_2[0]) ** 2) + ((point_1[1] - point_2[1]) ** 2))
    return distance

def shortest_distance(cluster_1, cluster_2):
    """
    This is a helper method that computers the shortest distance between clusters
    :param cluster_1: first cluster
    :param cluster_2: second cluster
    :return: shortest distance between cluster_1 and cluster_2
    """
    # begin with first pt of each cluster
    single_linkage = distance(cluster_1[0], cluster_2[0])
    # iterate through each points in each cluster
    for point_1 in cluster_1:
        for point_2 in cluster_2:
            # compute distance
            temp = distance(point_1, point_2)
            # update min continuously
            if temp < single_linkage:
                single_linkage = temp

    return single_linkage



