# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
class XMeans:
    """
    x-means法を行うクラス
    """

    def __init__(self, k_init = 1, **k_means_args):
        """
        k_init : The initial number of clusters applied to KMeans()
        """
        self.k_init = k_init
        self.k_means_args = k_means_args

    def fit(self, X):
        """
        x-means法を使ってデータXをクラスタリングする
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        self.__clusters = [] 

        clusters = self.Cluster.build(X, KMeans(self.k_init, **self.k_means_args).fit(X))
        self.__recursively_split(clusters)

        self.labels_ = np.empty(X.shape[0], dtype = np.intp)
        for i, c in enumerate(self.__clusters):
            self.labels_[c.index] = i

        self.cluster_centers_ = np.array([c.center for c in self.__clusters])
        self.cluster_log_likelihoods_ = np.array([c.log_likelihood() for c in self.__clusters])
        self.cluster_sizes_ = np.array([c.size for c in self.__clusters])

        return self

    def __recursively_split(self, clusters):
        """
        引数のclustersを再帰的に分割する
        clusters : list-like object, which contains instances of 'XMeans.Cluster'
        """
        for cluster in clusters:
            if cluster.size <= 3:
                self.__clusters.append(cluster)
                continue
            
            
            k_means = KMeans(2, **self.k_means_args).fit(cluster.data)
            c1, c2 = self.Cluster.build(cluster.data, k_means, cluster.index)
            
            if (c1.size == 1 or c2.size==1):
                self.__clusters.append(cluster)
                #self.__recursively_split([c1, c2])
                return
           
            beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(np.linalg.det(c1.cov) + np.linalg.det(c2.cov))
            alpha = 0.5 / stats.norm.cdf(beta)
            bic = -2 * (cluster.size * np.log(alpha) + c1.log_likelihood() + c2.log_likelihood()) + 2 * cluster.df * np.log(cluster.size)
            
            if bic < cluster.bic():
                self.__recursively_split([c1, c2])
            else:
                self.__clusters.append(cluster)

    class Cluster:
        """
        k-means法によって生成されたクラスタに関する情報を持ち、尤度やBICの計算を行うクラス
        """

        @classmethod
        def build(cls, X, k_means, index = None):
            if index == None:
                index = np.array(range(0, X.shape[0]))
            labels = range(0, k_means.get_params()["n_clusters"])

            return tuple(cls(X, index, k_means, label) for label in labels)

        # index: Xの各行におけるサンプルが元データの何行目のものかを示すベクトル
        def __init__(self, X, index, k_means, label):
            self.data = np.copy(X[k_means.labels_ == label])
            self.index = np.copy( index[k_means.labels_ == label])
            self.size = self.data.shape[0]
            self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
            self.center = np.copy (k_means.cluster_centers_[label])
            self.cov = np.cov(self.data.T)


        def log_likelihood(self):
            ll = 0

            for x in self.data:
                ll += stats.multivariate_normal.logpdf(x, self.center, self.cov, allow_singular=True)
            

            return ll

        def bic(self):
            bic_param = -2 * self.log_likelihood() + self.df * np.log(self.size)
            return bic_param
###############################################################################
def clear_outliers (X, outliers_fraction = 0.005, contamination=0.1):
     # !!!!!!!!!!!!!!!!!!!!!!!!
#    clf = EllipticEnvelope(contamination=contamination, assume_centered=True)
#    clf.fit(X)
#    y_pred = clf.decision_function(X).ravel()
#    threshold = stats.scoreatpercentile(y_pred, 100 * outliers_fraction)
#    y_pred = y_pred > threshold
#    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (X.size > 20):
        y_pred = np.ones_like(X, dtype = bool)
    else:
        y_pred = np.zeros_like(X, dtype = bool)
    
    
    return y_pred

    

## librarary for filtering    
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data) # lfilter(b, a, data)
    return y 

###############################################################################
def moving_average(x, n, mode='simple'):
    x = np.asarray(x)
    if mode=='simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a =  np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a
###############################################################################
def get_argextremums(signal):
    dif = np.diff(signal)
    dif[dif < 0] = -2
    dif[dif > 0] = 2
    dif[dif == 0] = 1
    	
    lm = np.diff(dif)
    ext_ind = np.argwhere(lm != 0)
    ext_ind += 1
    ext = np.zeros_like(dif)
    ext[ext_ind] = dif[ext_ind]
    lmax_ind = np.argwhere(ext < 0)
    lmin_ind = np.argwhere(ext > 0)
    return (lmax_ind, lmin_ind)
    
    