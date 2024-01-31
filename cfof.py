"""
    CFOF anomaly score for outlier detection
    Fabrizio Angiulli, 02/06/2020

"""
import numpy as np
import random

class CFOF:


    def __compute_distances(self, X):
        """    Returns the symmetric matrix of distances between each pair of points in X
               Each row of X represent a distinct point 
        """
        dp = -2 * X@X.transpose()
        n2 = (X**2).sum(axis=1)
        n2t = np.array([n2]).transpose()
        return dp + n2 + n2t




    def fit_predict(self, ds, k = [0.05]):
        """    Computes the CFOF anomaly scores of the points in the dataset ds
            k is a tuple of neighborhood parameters
        """
        if not isinstance(ds,np.ndarray) or len(ds.shape) != 2:
            raise ValueError('Bad dataset')
        n = len(ds)
        #n = ds.shape[0]
        
        #if len(k) == 0:
        #    k = (0.05,)
        nk = len(k)
        for j in range(nk):
            if k[j] < 0 or k[j] > n:
                raise ValueError('Invalid neighborhood')
        dst = self.__compute_distances(ds)
        nns = dst.argsort(axis=1)
        
        ##score = np.array([[n]*n]*nk)
        #score = np.reshape(np.arange(nk*n, dtype=np.float), (nk,n))
        #score[:] = n
        score = n*np.ones((nk,n))
        
        #count = np.array([0]*n)
        count = np.zeros(n)
        
        ind = [0]*nk
        for j in range(nk):
            ind[j] = {*range(n)}
        kpos = 0
        while sum([len(w) for w in ind]) > 0 and kpos < n:
            count = count + np.bincount(nns[:,kpos], minlength=n)
            for j in range(nk):
                kk = k[j]*n if k[j] < 1 else k[j] 
                kind = [i for i in ind[j] if count[i] >= kk and score[j][i] == n]
                for i in kind:
                    score[j][i] = kpos + 1
                    ind[j].remove(i)
            kpos = kpos + 1
        if nk == 1:
            score = score.reshape(n)
        return score / n

    def fit_predict_memory(self, ds, k=[0.05]):
        """    Computes the CFOF anomaly scores of the points in the dataset ds
            k is a tuple of neighborhood parameters
            Compute distances between data points once at each time to reduce memory occupation
        """
        if not isinstance(ds, np.ndarray) or len(ds.shape) != 2:
            raise ValueError('Bad dataset')
        n = len(ds)


        nk = len(k)
        for j in range(nk):
            if k[j] < 0 or k[j] > n:
                raise ValueError('Invalid neighborhood')


        XT = ds.T
        XT2 = np.sum(XT**2,axis=0)

        dst = np.zeros(n)
        for i in range(n):
            dst[:] = np.linalg.norm(ds[i, :]) ** 2 - 2 * XT + XT2
            nns = dst.argsort(axis=1)



        #dst = self.__compute_distances(ds)
        nns = dst.argsort(axis=1)



        score = n * np.ones((nk, n))

        # count = np.array([0]*n)
        count = np.zeros(n)

        ind = [0] * nk
        for j in range(nk):
            ind[j] = {*range(n)}
        kpos = 0
        while sum([len(w) for w in ind]) > 0 and kpos < n:
            count = count + np.bincount(nns[:, kpos], minlength=n)
            for j in range(nk):
                kk = k[j] * n if k[j] < 1 else k[j]
                kind = [i for i in ind[j] if count[i] >= kk and score[j][i] == n]
                for i in kind:
                    score[j][i] = kpos + 1
                    ind[j].remove(i)
            kpos = kpos + 1
        if nk == 1:
            score = score.reshape(n)
        return score / n





    def test(n = 1000, d = 2):
        ##X = np.array([[0.]*d]*n, dtype=np.float)
        #X = np.reshape(np.arange(n*d, dtype=np.float), (n,d))
        X = np.zeros((n,d))
        for i in range(n):
            for j in range(d):
                X[i,j] = random.random()
        return fit_predict(X, k=[2,3,4,5]), X





X = np.ones((10,2))
clf = CFOF()
clf.fit_predict(X)





