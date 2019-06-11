#!./.env/bin/python
"""This script allows user to check clustering stability"""
import multiprocessing as mp
from functools import partial

import numpy as np


class Stability:
    """Clustering Stability"""
    def __init__(self, data, clusterer):
        """stability object init

        Arguments:
            X {numpy.ndarray} -- original data that will be used for clustering
             clusterer {object} -- Object similar to a sklearn.cluster CLASS
             object, should have the function fit_predict, and that function
             should return cluster labels, Eg. sklearn.cluster.MiniBatchKMeans,
             not an object of MiniBatchKMeans
        """
        self.data = data
        self.clusterer = clusterer

    @staticmethod
    def resample(data, number_of_resamples, dilution_factor, random_state):
        """create resample indices of the data

        Arguments:
            data {numpy.ndarray} -- data that needs to be resampled
            number_of_resamples {int} -- number of resamples to be created
            dilution_factor {float} -- a number between 0 and 1, which decides
             the number of samples in each resample

        Returns:
            resample_indices {list} -- list of list of indices which correspond
             to rows in self.data
            resamples {list} -- list of resamples
        """
        np.random.seed(random_state)
        sample_size = int(dilution_factor * data.shape[0])
        resample_indices = [np.random.choice(np.arange(data.shape[0]),
                                             size=sample_size,
                                             replace=False) for i in
                            range(number_of_resamples)]
        resamples = [data[indices] for indices in resample_indices]

        return resample_indices, resamples

    @staticmethod
    def match_labels(label, labels):
        """Matches labels, used in create_connectivity_matrix
        
        Arguments:
            label {int or string} -- label assigned to a particular sample
            labels {numpy.ndarray} -- array of labels

        
        Returns:
            row {numpy.ndarray} -- array of 0s and 1s indicating matches
        """
        row = label == labels
        return row
    
    @staticmethod
    def create_connectivity_matrix(labels, n_jobs=None):
        """Creates connectivity matrix for a set of cluster labels

        Arguments:
            labels {list} -- list of labels of size n
            
        Keyword Arguments:
        n_jobs {int} -- number of jobs to run in parallel for matrix 
         creation, None means 1, -1 means use all processors 

        Returns:
            matrix {numpy.ndarray} -- connectivity matrix of size n x n
        """

        labels = np.array(labels)
        if n_jobs:
            if n_jobs == -1:
                n_jobs = mp.cpu_count()
            elif n_jobs > mp.cpu_count():
                raise ValueError("n_jobs exceeds number of cores available")
            pool = mp.Pool(processes=n_jobs)
            rows = pool.map(partial(Stability.match_labels, labels), labels)
        else:
            rows = [label == labels for label in labels]

        matrix = np.vstack(rows)
        matrix = matrix.astype(np.int16)
        return matrix

    def compute_stability_score(self, number_of_resamples, dilution_factor,
                                params, random_state=None, original_labels=None, 
                                n_jobs=None, sample_size=None):
        """Computes stability score for a given set of params

        Arguments:
            number_of_resamples {int} -- number of resamples to be created
            dilution_factor {float} -- a number between 0 and 1, which decides
             the number of samples in each resample
            params {dict} -- dictionary containing parameters required for 
             self.clusterer object

        Keyword Arguments:
            random_state {int} -- if int, random_state is the seed used by the
             random number generator (default: {None})
            original_labels {numpy.ndarray} -- original labels for the data 
             given at the time of initialization for the given clusterer. If 
             not passed, then its recomputed (default: {None})
            n_jobs {int} -- number of jobs to run in parallel for matrix 
             creation, refer create_connectivity_matrix docstring 
             (default: {None})
            sample_size {int} -- number of samples to consider when computing 
             score, not to be confused with the number of resamples. The
             resamples will be drawn from this sample. (default: {None})

        Returns:
            {float} -- a stability score for the clustering algorithm with the
             given set of parameters for self.data
        """
        if original_labels is None:
            original_labels = self.clusterer(**params).fit_predict(self.data)
        if sample_size:
            label_indices = np.random.choice(np.arange(self.data.shape[0]), 
                                             size=sample_size,
                                             replace=False)
            data = self.data[label_indices]
            labels = original_labels[label_indices]
        else:
            data = self.data
            labels = original_labels
        original_mat = Stability.create_connectivity_matrix(labels, n_jobs)
        sample_indices, samples = self.resample(data, number_of_resamples,
                                                dilution_factor, random_state)
        scores = []
        for i, sample in enumerate(samples):
            indices = sample_indices[i]
            sample_labels = self.clusterer(**params).fit_predict(sample)
            resample_mat = Stability.create_connectivity_matrix(sample_labels,
                                                                n_jobs)
            original_resample_mat = original_mat[indices][:, indices]
            score = ((original_resample_mat + resample_mat) / 2).mean()
            scores.append(score)
        stability_score = np.mean(scores)
        return stability_score
