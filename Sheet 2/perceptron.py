import numpy as np

class Gaussian_Kernel():
    def trainSGD(self, y: np.ndarray, X: np.ndarray, lr: float, gamma: float, iter: int):
        self.training_labels = y
        self.training_features = X
        self.gamma = gamma
        self.m = len(y)
        beta = np.zeros(self.m)
        K = self.construct_kernel_matrix(X, X)
        for _ in range(iter):
            i = np.random.randint(0,self.m)
            beta[i] += lr * (y[i] - self.sign( (K @ beta)[i] ))
        self.beta = beta
        return beta
    def gaussian_kernel(self, x: np.ndarray, z: np.ndarray) -> float:
        return np.exp(-self.gamma * np.linalg.norm(x-z)**2)
    def construct_kernel_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        #If A has shape (a,d) and B has shape (b,d) then the output has shape (a,b)
        a = A.shape[0]
        b = B.shape[0]
        matrix = np.empty((a,b))
        for i in range(0,a):
            for j in range(0,b):
                matrix[i,j] = self.gaussian_kernel(A[i], B[j])
        return matrix
    def sign(self, z:float) -> float:
        if z>=0: return 1
        else: return 0
    def make_prediction(self, x: np.ndarray) -> np.ndarray:
        #set the training features to be B and testing featuers to be A so that
        test_matrix = self.construct_kernel_matrix(x, self.training_features)
        numbers = test_matrix @ self.beta
        return np.array([self.sign(x) for x in numbers])