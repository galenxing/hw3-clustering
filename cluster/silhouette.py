import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        s = []
        for point, label in zip(X,y):
            bs = []

            for i in range(np.max(y)+1):
                if i == label:
                    c = X[y==i]
                    a = np.mean(cdist(c,[point]))
                else:
                    c = X[y==i]
                    bs.append(np.mean(cdist(c,[point])))
                    
            b = np.min(np.array(bs))
            s.append((b-a)/np.max((a,b)))

        return np.array(s)
    