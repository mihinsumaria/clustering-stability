#!./.env/bin/python
"""This script allows user to check clustering stability"""
import numpy as np


class Stability:
    """Clustering Stability"""
    def __init__(self, data, clusterer):
        """stability object init

        Arguments:
            X {numpy.ndarray} -- original data that will be used for clustering
             clusterer {object} -- Object similar to a sklearn.cluster class
             object, should have the function fit_predict, and that function
             should return cluster labels
        """
        self.data = data
        self.clusterer = clusterer

    def resample(self, number_of_resamples, dilution_factor, random_state):
        """create resample indices of the data

        Arguments:
            number_of_resamples {int} -- number of resamples to be created
            dilution_factor {float} -- a number between 0 and 1, which decides
             the number of samples in each resample

        Returns:
            resample_indices {list} -- list of list of indices which correspond
             to rows in self.data
            resamples {list} -- list of resamples
        """
        np.random.seed(random_state)
        sample_size = int(dilution_factor * self.data.shape[0])
        resample_indices = [np.random.choice(np.arange(self.data.shape[0]),
                                             size=sample_size,
                                             replace=False) for i in
                            range(number_of_resamples)]
        resamples = [self.data[indices] for indices in resample_indices]

        return resample_indices, resamples

    @staticmethod
    def create_connectivity_matrix(labels):
        """Creates connectivity matrix for a set of cluster labels

        Arguments:
            labels {list} -- list of labels of size n

        Returns:
            matrix {numpy.ndarray} -- connectivity matrix of size n x n
        """

        labels = np.array(labels)
        matrix = np.vstack([label == labels for label in labels])
        matrix = matrix.astype(int)
        return matrix

    def compute_stability_score(self, params, number_of_resamples,
                                dilution_factor, random_state=None):
        """Computes stability score for a given set of params

        Arguments:
            params {dict} -- dictionary containing function parameters
             corresponding to self.clusterer.fit_predict
            number_of_resamples {int} -- number of resamples to be created
            dilution_factor {float} -- a number between 0 and 1, which decides
             the number of samples in each resample

        Keyword Arguments:
            random_state {int} -- if int, random_state is the seed used by the
             random number generator (default: {None})

        Returns:
            {float} -- a stability score for the clustering algoritm with the
             given set of parameters for self.data
        """
        original_labels = self.clusterer(**params).fit_predict(self.data)
        original_mat = Stability.create_connectivity_matrix(original_labels)
        sample_indices, samples = self.resample(number_of_resamples,
                                                dilution_factor, random_state)
        scores = []
        for i, sample in enumerate(samples):
            indices = sample_indices[i]
            sample_labels = self.clusterer(**params).fit_predict(sample)
            resample_mat = Stability.create_connectivity_matrix(sample_labels)
            original_resample_mat = original_mat[indices][:, indices]
            score = ((original_resample_mat + resample_mat) / 2).mean()
            scores.append(score)
        stability_score = np.mean(scores)
        return stability_score
