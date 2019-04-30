#!/Users/mihin/miniconda3/bin/python
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

    def resample(self, number_of_resamples, dilution_factor):
        """create resample indices of the data

        Arguments:
            number_of_resamples {int} -- number of resamples to be created
            dilution_factor {float} -- a number between 0 and 1, which decides
            the number of samples in each resample

        Returns:
            resample_indices {list} -- list of list of indices which correspond
            to rows in self.data
        """
        sample_size = int(dilution_factor * self.data.shape[0])
        resample_indices = [np.random.choice(np.arange(self.data.shape[0]),
                                             size=sample_size,
                                             replace=False) for i in
                            range(number_of_resamples)]

        return resample_indices

    @staticmethod
    def create_connectivity_matrix(labels1, labels2):
        """Creates connectivity matrix for two sets of cluster labels

        Arguments:
            labels1 {list} -- List of Labels 1
            labels2 {list} -- List of Labels 2

        Raises:
            ValueError: raised if lengths of labels1 and labels2 do not match

        Returns:
            numpy.ndarray -- connectivity matrix
        """
        if len(labels1) != len(labels2):
            raise ValueError('Lengths of labels1 and labels2 do not match:'
                             ' {} and {} respectively'.format(len(labels1),
                                                              len(labels2)))
        labels1 = np.array(labels1)
        labels2 = np.array(labels2)
        for i, label in enumerate(labels1):
            if not i:
                matrix = label == labels2
            else:
                matrix = np.vstack((matrix, label == labels2))
        matrix = matrix.astype(int)
        return matrix
