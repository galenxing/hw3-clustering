import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100, seed: int = 5):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if k>0:
            self.k = k
        else:
            raise ValueError('k needs to be greater than 0.')
            
        self.tol = tol
        self.max_iter = max_iter
        np.random.seed(seed)        
        

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """


        maxes = np.max(mat,axis=0)
        normed = mat/maxes

        ass = np.random.randint(0, self.k, normed.shape[0])

        q=0
        sse_prev = np.inf
        err = np.inf
        sse_diff = np.inf
        
        while (q < self.max_iter) and (sse_diff > self.tol):
            centers = []
            for i in range(self.k):
                cluster = normed[ass == i]
                centers.append(np.mean(cluster, axis=0))

            tmp_centers = []
            for center in centers:
                if np.isnan(center).any():
                    tmp_centers.append(np.random.random(mat.shape[1]))
                else:
                    tmp_centers.append(center)

            centers = np.array(tmp_centers)

            ass = np.argmin(cdist(normed,np.array(centers)),axis=1)
            
            sse = 0
            for i in range(self.k):
                cluster = mat[ass == i]
                sse += np.sum((cluster - centers[i]*maxes)**2)
            sse = sse/mat.shape[0]
            
#             sse_diff = sse_prev-sse
            sse_prev = sse

            q+=1
            
        self.error = sse
        self.centers = centers * maxes

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        return np.argmin(cdist(mat,self.centers),axis=1)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centers
        
