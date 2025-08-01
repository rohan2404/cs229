import numpy as np
from typing import Tuple

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))

class LogisticRegression:
    def train_model(self, features: np.ndarray, labels: np.ndarray, theta_0: np.ndarray, tol: float = 1e-5, max_iter: int = 100) -> np.ndarray:
        X = np.column_stack([np.ones(len(features)), features])
        y = labels
        theta = theta_0
        for _ in range(max_iter):
            theta, stop = self.newton_update_theta(X, y, theta, tol)
            if stop:
                break
        self.theta = theta
        return theta
        
    def build_hessian_grad(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        #Hessian
        #let's get the diagonal matrix D_ii
        h = sigmoid(np.dot(X, theta))
        D = np.diag(h*(1-h))

        #now let's compute the matrix product for the Hessian
        m = len(X)
        H = 1/m * np.dot(X.T, np.dot(D ,X))

        #Grad
        grad = 1/m * np.dot(X.T, h-y)
        return H, grad
    
    def newton_update_theta(self, X: np.ndarray, y: np.ndarray, theta:np.ndarray, tol: float) -> Tuple[np.ndarray, bool]:
        H, grad = self.build_hessian_grad(theta, X, y)
        #theta_new = theta - np.dot(np.linalg.inv(H), grad)
        theta_new = theta - np.linalg.solve(H, grad)
        converged = np.linalg.norm(theta_new-theta) < tol
        return theta_new, converged
    
    def make_prediction(self, X_test: np.ndarray) -> np.ndarray:
        X_test = np.column_stack([np.ones(len(X_test)),X_test])
        z = np.dot(X_test, self.theta)
        return np.where(z >= 0, 1, 0)

class GDA:
    def train_model(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        X = features
        y = labels
        phi = 1/(len(features)) * np.sum(labels)
        X0, X1 = X[y==0],X[y==1] 
        mu_0, mu_1 = X0.mean(axis=0), X1.mean(axis=0)        
        Xc0, Xc1 = X0 - mu_0, X1 - mu_1 
        sigma = 1/len(X) * (np.dot(Xc0.T,Xc0) + np.dot(Xc1.T,Xc1))
        sigma_inv = np.linalg.inv(sigma)
        theta = np.dot(sigma_inv, mu_1 - mu_0)
        theta_0 = np.array([0.5*mu_0.T @ sigma_inv @ mu_0 - 0.5*mu_1.T @ sigma_inv @ mu_1 - np.log((1-phi)/phi)]) #@ means matrix multiplication
        self.theta = np.concatenate([theta_0, theta])
        return self.theta
    
    def make_prediction(self, X_test: np.ndarray) -> np.ndarray:
        X_test = np.column_stack([np.ones(len(X_test)),X_test])
        z = np.dot(X_test, self.theta)
        return np.where(z >= 0, 1, 0)