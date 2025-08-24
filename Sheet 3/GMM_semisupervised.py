import numpy as np
from scipy.stats import multivariate_normal

class GMM_SS():
    def __init__(self) -> None:
        pass

    
    def fit(self, unsup_feature: np.ndarray, sup_feature: np.ndarray, sup_labels: np.ndarray, num_gaussians:int, alpha: float = 0.2, iters: int = 10):
        self.m_unsup = len(unsup_feature) #number of unsupervised examples
        self.m_sup = len(sup_feature)     #number of supervised examples
        self.k = num_gaussians            #number of gaussians modelling to
        self.n = len(sup_feature.T)       #dimensionality of features
        self.alpha = alpha                #hyperparameter alpha
        
        self.means = [np.ones(self.n)] * self.k
        self.covs = [np.diag(np.ones(self.n))] * self.k
        self.phis = np.ones(self.k) / self.k

        combined_features = np.vstack((unsup_feature, sup_feature))

        for _ in range(iters):
            W = self.build_weight_matrix(unsup_feature, sup_feature, sup_labels)
            col_sum = np.sum(W, axis=0)
            self.phis = col_sum/(self.alpha * self.m_sup + self.m_unsup)
            for j in range(self.k):
                self.means[j] = np.dot(W[:,j], combined_features) / col_sum[j]
                curr = np.zeros((self.n, self.n))
                for i in range(self.m_unsup + self.m_sup):
                    z = combined_features[i] - self.means[j]
                    curr += W[i][j] * np.outer(z,z)
                self.covs[j] = curr / col_sum[j] + 1e-6 * np.eye(self.n)

        return self.means, self.covs, self.phis

    def build_weight_matrix(self, unsup_feature: np.ndarray, sup_feature: np.ndarray, sup_labels: np.ndarray) -> np.ndarray:
        mat = np.zeros((self.m_unsup + self.m_sup, self.k))
        for row in range(self.m_unsup):
            curr = 0
            for col in range(self.k):
                mat[row][col] = self.gaussian(unsup_feature[row], self.means[col], self.covs[col]) * self.phis[col]
                curr += mat[row][col]
            mat[row] /= curr
        for row in range(self.m_sup):
            for col in range(self.k):
                if sup_labels[row] == col:
                    mat[row + self.m_unsup][col] = self.alpha
        return mat

    def gaussian(self, x: np.ndarray, mu: np.ndarray, cov):
        rv = multivariate_normal(mean = mu, cov = cov)
        return rv.pdf(x)
    
    def predict(self, feature: np.ndarray):
        curr = np.zeros(self.k)
        for j in range(self.k):
            curr[j] = self.gaussian(feature, self.means[j], self.covs[j]) * self.phis[j]
        return curr.argmax()
    
