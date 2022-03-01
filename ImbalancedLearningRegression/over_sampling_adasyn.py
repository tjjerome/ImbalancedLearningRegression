# He, H., Bai, Y., Garcia, E. A., & Li, S. (2008, June).
# ADASYN: Adaptive synthetic sampling approach for imbalanced
# learning. In 2008 IEEE international joint conference on neural
# networks (IEEE world congress on computational intelligence)
# (pp. 1322-1328). IEEE.

## load dependencies - third party
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm

## load dependencies - internal
import matplotlib.pyplot as plt
from sklearn import neighbors

## generate synthetic observations
def adasyn(X, label, beta, K, threshold=1):

    """
    Adaptively generating minority data samples according to their distributions.
    More synthetic data is generated for minority class samples that are harder to learn.
    
    Inputs
         -----
         X:  Input features, X, sorted by the minority examples on top.  Minority example should also be labeled as 1
         label:  Labels, with minority example labeled as 1
         beta:  Degree of imbalance desired.  Neg:Pos. A 1 means the positive and negative examples are perfectly balanced.
         K:  Amount of neighbours to look at
         threshold:  Amount of imbalance rebalance required for algorithm

    """

    ms = int(sum(label))
    ml = len(label) - ms

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X, label)

    # calculate the degree of class imbalance. If degree of class imbalance is violated, continue.
    d = np.divide(ms, ml)

    if d > threshold:
        return print("The data set is not imbalanced enough.")

    # if the minority data set is below the maximum tolerated threshold, generate data.
    # Beta is the desired balance level parameter.  Beta > 1 means u want more of the imbalanced type, vice versa.
    G = (ml - ms) * beta

    # Step 2b, find the K nearest neighbours of each minority class example in euclidean distance.
    # Find the ratio ri = majority_class in neighbourhood / K
    Ri = []
    Minority_per_xi = []
    for i in range(ms):
        xi = X[i, :].reshape(1, -1)
        # Returns indices of the closest neighbours, and return it as a list
        neighbours = clf.kneighbors(xi, n_neighbors=K, return_distance=False)[0]
        # Skip classifying itself as one of its own neighbours
        # neighbours = neighbours[1:]

        # Count how many belongs to the majority class
        count = 0
        for value in neighbours:
            if value > ms:
                count += 1

        Ri.append(count / K)

        # Find all the minority examples
        minority = []
        for value in neighbours:
            # Shifted back 1 because indices start at 0
            if value <= ms - 1:
                minority.append(value)

        Minority_per_xi.append(minority)

    # normalize ri's so their sum equals to 1
    Rhat_i = []
    for ri in Ri:
        rhat_i = ri / sum(Ri)
        Rhat_i.append(rhat_i)

    assert(sum(Rhat_i) > 0.99)

    # calculate the number of synthetic data examples that will be generated for each minority example
    Gi = []
    for rhat_i in Rhat_i:
        gi = round(rhat_i * G)
        Gi.append(int(gi))

    ## generate synthetic examples
    syn_data = []
    for i in range(ms):
        xi = X[i, :].reshape(1, -1)
        for j in range(Gi[i]):
            # If the minority list is not empty
            if Minority_per_xi[i]:
                index = np.random.choice(Minority_per_xi[i])
                xzi = X[index, :].reshape(1, -1)
                si = xi + (xzi - xi) * np.random.uniform(0, 1)
                syn_data.append(si)

    # Test the new generated data
    test = []
    for values in syn_data:
        a = clf.predict(values)
        test.append(a)

    # Build the data matrix
    data = []
    for values in syn_data:
        data.append(values[0])

    # Concatenate the positive labels with the newly made data
    labels = np.ones([len(data), 1])
    data = np.concatenate([labels, data], axis=1)

    # Concatenate with old data
    org_data = np.concatenate([label.reshape(-1, 1), X], axis=1)
    data = np.concatenate([data, org_data])

    return data