from sklearn.metrics import rand_score
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import statistics as s

from keras.datasets import mnist as MNIST
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm
import itertools


class KMeans:
    """
    KMeans class for clustering images.

    Attributes:
        num_clusters (int): number of clusters required
        num_iterations (int): number of iteration of the Lloyd's algorithm
        cluster_centers (list): to store the centers of each of the num_clusters clusters
        point_clusters (list): cluster ids assigned to each point
        clusters (list): contains the indices of the points in each cluster
    """

    def __init__(self, num_clusters=10, num_iterations=20):
        self.num_clusters = num_clusters  # number of clusters required
        # number of iteration of the Lloyd's algorithm
        self.num_iterations = num_iterations
        self.cluster_centers = []  # to store the centers of each of the num_clusters clusters
        self.point_clusters = []  # cluster ids assigned to each point
        # contains the indices of the points in each cluster
        self.clusters = [[] for _ in range(num_clusters)]

    def euclidean_distance(self, image1, image2):
        """
        Computes the Euclidean distance between two images.
        """
        return np.linalg.norm(image1 - image2)

    def assign_probabilities(self, points):
        """
        Assigns probabilities to the points in order to pick them in the k-means++ initialization.
        """
        distances = []
        for point in points:
            min_distance = np.inf
            for center in self.cluster_centers:
                distance = self.euclidean_distance(point, center)
                if distance < min_distance:
                    min_distance = distance
            distances.append(np.square(min_distance))
        sum_of_distances = sum(distances)
        probabilities = [distance / sum_of_distances for distance in distances]
        return probabilities

    def initialize_means(self, points):
        """
        Initializes the cluster centers using the k-means++ algorithm.
        """
        center = points[0]
        self.cluster_centers.append(center)
        for i in range(1, self.num_clusters):
            probabilities = np.array(self.assign_probabilities(points))
            next_center_index = np.random.choice(
                len(points), p=probabilities, size=1)[0]
            next_center = points[next_center_index]
            self.cluster_centers.append(next_center)

    def assign_clusters(self, points):
        """
        Assigns each point to the nearest cluster based on the distance to the cluster centers.
        """
        self.point_clusters = [None] * len(points)
        for i, point in enumerate(points):
            min_distance = np.inf
            cluster_index = 0
            for j, center in enumerate(self.cluster_centers):
                distance = self.euclidean_distance(center, point)
                if distance < min_distance:
                    cluster_index = j
                    min_distance = distance
            self.point_clusters[i] = cluster_index
            self.clusters[cluster_index].append(i)

    def update_centers(self, points):
        """
        Updates the cluster centers by computing the mean of the points in each cluster.
        """
        for i, cluster in enumerate(self.clusters):
            sum_of_points = 0
            for j in cluster:
                sum_of_points += points[j]
            mean_point = sum_of_points / len(cluster)
            self.cluster_centers[i] = mean_point

    def fit(self, points):
        """
        Fits the KMeans model to the given points.
        """
        self.initialize_means(points)
        for i in range(self.num_iterations):
            self.assign_clusters(points)
            self.update_centers(points)

    def rand_score_value(self, labels):
        """
        Computes the Rand score between the true labels and the assigned clusters.
        """
        return rand_score(labels, self.point_clusters)


class KCenters:
    """ 
    K-Centers class for clustering images.

    Attributes:
        num_clusters (int): number of clusters required
        num_iterations (int): number of iteration of the Lloyd's algorithm
        cluster_centers (list): to store the centers of each of the num_clusters clusters
        point_clusters (list): cluster ids assigned to each point
        clusters (list): contains the indices of the points in each cluster
    """

    def __init__(self, num_clusters=10, num_iterations=10):
        self.point_to_cluster_number = None
        self.clusters = None
        self.numbers = None
        self.dataset = None

        self.K = num_clusters
        self.num_samples = None
        self.mean_indices = []
        self.centers = []
        self.num_iterations = num_iterations

    def euclidean_distance(self, image1, image2):
        """
        Computes the Euclidean distance between two images.
        """

        return np.linalg.norm(image1 - image2)

    def initialize_centers(self, X):
        """
        Initializes the cluster centers using the k-means++ algorithm.
        """

        first_center = np.random.randint(0, self.num_samples - 1)
        center_indices = [first_center]

        for _ in range(1, self.K):

            max_distance = -np.inf
            max_distance_point_id = -1

            for i in range(self.num_samples):
                if i not in center_indices:
                    closest_center_dist = np.inf
                    for c_id in center_indices:
                        currennt_d = self.euclidean_distance(X[i], X[c_id])
                        if currennt_d < closest_center_dist:
                            closest_center_dist = currennt_d

                    if closest_center_dist > max_distance:
                        max_distance = closest_center_dist
                        max_distance_point_id = i
            center_indices.append(max_distance_point_id)

        return np.array(center_indices)

    def find_closest_center(self, i):
        """ 
        Finds the closest center to the i-th point in the dataset.
        """
        minimum_distance = np.inf
        center = -1

        for j in range(len(self.centers)):
            current_distance = self.euclidean_distance(
                self.dataset[i], self.centers[j])
            if(current_distance < minimum_distance):
                minimum_distance = current_distance
                center = j

        return center

    def assign_points_to_closest_centers(self, X):
        """ 
        Assigns each point to the closest center.
        """
        point_to_cluster_map = []

        for i in range(self.num_samples):
            closest_center = self.find_closest_center(i)
            point_to_cluster_map.append(closest_center)

        clusters = [[] for i in range(self.K)]

        for i in range(self.num_samples):
            clusters[point_to_cluster_map[i]].append(i)
        self.point_to_cluster_number = point_to_cluster_map
        self.clusters = clusters

    def find_farthest_cluster_point(self, i):
        """ 
        Finds the farthest point from the i-th cluster center.
        """
        current_center = self.mean_indices[i]
        max_distance_from_currennt_center = -np.inf
        new_center_id = -1

        for j in (self.clusters[i]):
            currennt_dist = self.euclidean_distance(
                self.dataset[j], self.dataset[current_center])

            if(currennt_dist > max_distance_from_currennt_center):
                max_distance_from_currennt_center = currennt_dist
                new_center_id = j

        return new_center_id

    def assign_centers_to_farthest_cluster_points(self, X):
        """ 
        Assigns each cluster center to the farthest point in the cluster.
        """
        new_centroids_ids = []

        for p in range(len(self.clusters)):
            new_center = self.find_farthest_cluster_point(p)
            new_centroids_ids.append(new_center)

        new_centroids = X[np.array(new_centroids_ids)]
        self.mean_indices = new_centroids_ids

        return new_centroids

    def fit(self, X):
        """ 
        Fits the K-Centers model to the given points.
        """
        self.dataset = X
        self.num_samples = X.shape[0]

        self.mean_indices = self.initialize_centers(X)
        self.centers = X[self.mean_indices]
        self.assign_points_to_closest_centers(X)

        i = 0
        while(i < self.num_iterations):
            new_centroids = self.assign_centers_to_farthest_cluster_points(X)
            self.centroids = new_centroids
            self.assign_points_to_closest_centers(X)
            i += 1

    def convert_cluster_number_to_class(self, y):
        """ 
        Converts the cluster number to the class value.
        """
        self.numbers = np.array([-1 for i in range(y.size)])

        for clust in self.clusters:
            sz = len(clust)

            class_values_in_cluster = [y[clust[i]] for i in range(sz)]
            class_value = s.mode(class_values_in_cluster)

            mask = np.array(clust)
            self.numbers[mask] = class_value

    def rand_score_value(self, y):
        """ 
        Computes the Rand score value.
        """
        score = rand_score(y, self.point_to_cluster_number)
        return score


class SingleLinkage:
    """
    SingleLinkage class for clustering images.

    Attributes:
        num_clusters (int): number of clusters required
        assign_clusters (list): cluster ids assigned to each point
        clusters (dict): contains the indices of the points in each cluster
        cluster_distances (dict): contains the distances between each pair of clusters
        cluster_ids (list): contains the ids of the clusters
    """

    def __init__(self, num_clusters=10):
        self.num_clusters = num_clusters
        self.assign_clusters = []
        self.clusters = {}
        self.cluster_distances = {}
        self.cluster_ids = []

    def distance(self, img1, img2):
        """ 
        Computes the distance between two images.
        """
        return np.linalg.norm(img1 - img2, ord=2)

    def init_cluster_distances(self, X):
        """ 
        Initializes the cluster distances between each pair of clusters.
        """
        dict_ = {}
        index_pairs = list(itertools.combinations(range(len(X)), 2))
        for x in index_pairs:
            dict_[x] = self.distance(X[x[0]], X[x[1]])
        self.cluster_distances = dict_

    def find_cluster_difference(self, min_cluster):
        """ 
        Finds the difference between the clusters.
        """
        for x in self.cluster_ids:
            if x == min_cluster[1] or x == min_cluster[0]:
                continue
            tup1 = tuple(sorted((x, min_cluster[0])))
            tup2 = tuple(sorted((x, min_cluster[1])))
            self.cluster_distances[tup1] = min(
                self.cluster_distances[tup1], self.cluster_distances[tup2])
            self.cluster_distances.pop(tup2)
        del self.cluster_distances[min_cluster]

    def fit(self, X):
        """ 
        Fits the SingleLinkage model to the given points.
        """
        n = len(X)
        self.init_cluster_distances(X)
        self.clusters = {}
        for x in range(n):
            self.clusters[x] = [x]
            self.cluster_ids.append(x)

        while len(self.clusters) > self.num_clusters:
            min_value = min(self.cluster_distances.values())
            min_cluster = tuple(
                [k for k, v in self.cluster_distances.items() if v == min_value][0])
            self.find_cluster_difference(min_cluster)
            self.clusters[min_cluster[0]].extend(self.clusters[min_cluster[1]])
            del self.clusters[min_cluster[1]]
            self.cluster_ids.remove(min_cluster[1])

        self.assign_clusters = np.zeros(n, dtype=np.int32)
        count = 0
        for x in self.clusters.values():
            for j in x:
                self.assign_clusters[j] = count
            count += 1

    def rand_score_value(self, y):
        """ 
        Computes the Rand score between the true labels and the assigned clusters.
        """
        return(rand_score(y, self.assign_clusters))
