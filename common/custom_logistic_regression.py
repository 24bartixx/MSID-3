from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score


class CustomLogisticRegressionBinary(BaseEstimator, RegressorMixin):
   
    def __init__(self, epochs=500, batch_size=None, learning_rate=0.01, C=0, mini_batch=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.C = C
        self.mini_batch = mini_batch
        self.coefficients_ = None
        
    def predict_proba(self, X):
        if self.coefficients_ is None:
            raise NotFittedError("Model is not trained.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if scipy.sparse.issparse(X):
            X = X.toarray()   
            
        X_matrix = np.c_[np.ones((X.shape[0], 1)), X]
        return 1 / (1 + np.exp(- X_matrix @ self.coefficients_))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def fit_init(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values  
        if scipy.sparse.issparse(X):
            X = X.toarray()    
        
        self.X_matrix = np.c_[np.ones(shape=(X.shape[0],1)), X]
            
        if isinstance(y, pd.Series):
            y = y.values
        self.y_matrix = y.reshape(-1, 1)
            
        self.samples_count, feat_count = self.X_matrix.shape
        self.coefficients_ = np.zeros((feat_count, 1))
            
        # return X_matrix, y_matrix
    
    
    def fit_epoch(self):
        if self.mini_batch and self.batch_size:           # mini batch gradient descent
            indices = np.random.permutation(self.samples_count)
            for i in range(0, self.samples_count, self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                X_batch = self.X_matrix[batch_indices]
                y_batch = self.y_matrix[batch_indices]
                
                sigmoid = 1 / (1 + np.exp(-X_batch @ self.coefficients_))
                gradients = X_batch.T @ (sigmoid - y_batch) / len(X_batch)
                
                l2_term = (self.C / len(X_batch)) *self.coefficients_
                l2_term[0] = 0
                
                gradients += l2_term
                self.coefficients_ -= self.learning_rate * gradients
        else:
            
            if self.batch_size:                         # stochastic batch gradient descent
                indices = np.random.choice(self.samples_count, self.batch_size, replace = False)
                X_batch = self.X_matrix[indices]
                y_batch = self.y_matrix[indices]
                factor = 1 / self.batch_size
            else:                                       # simple batch gradient descent
                X_batch = self.X_matrix
                y_batch = self.y_matrix
                factor = 1 / self.samples_count
                
            sigmoid = 1 / (1 + np.exp(-X_batch @ self.coefficients_))
            gradients = factor * X_batch.T @ (sigmoid - y_batch)
            
            l2_term = (self.C / len(X_batch)) *self.coefficients_
            l2_term[0] = 0
            
            gradients += l2_term
            self.coefficients_ -= self.learning_rate * gradients
    
    
    def fit(self, X, y):
        self.fit_init(X, y)
        for _ in range(self.epochs):
            self.fit_epoch()


    def transform(self, X):
            return self.predict(X)
    
    
class CustomLogisticRegressionMulticlass(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=500, batch_size=None, learning_rate=0.01, C=0, mini_batch=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.C = C
        self.mini_batch = mini_batch
        self.binary_models = {}
        self.class_labels = None
        

    def fit(self, X, y, get_convergence_diagram=False, X_test=None, y_test=None):
        
        # if isinstance(y, pd.Series):
        #     y = y.to_numpy()
        
        self.class_labels = np.unique(y)
        for class_label in self.class_labels:
            y_bin = (y == class_label).astype(int)
            binary_model = clone(CustomLogisticRegressionBinary(
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                C=self.C,
                mini_batch=self.mini_batch
            ))
            self.binary_models[class_label] = binary_model
            if get_convergence_diagram:
                binary_model.fit_init(X, y_bin)
            else:
                binary_model.fit(X, y_bin)
            
        if get_convergence_diagram:
            accuracy_train = []
            accuracy_test = [] if X_test is not None and y_test is not None else None
            
            for _ in range(self.epochs):
                for binary_model in self.binary_models.values():
                    binary_model.fit_epoch()
                    
                y_pred_train = self.predict(X)
                accuracy_train.append(accuracy_score(y, y_pred_train))
                
                if X_test is not None and y_test is not None:
                    y_pred_test = self.predict(X_test)
                    accuracy_test.append(accuracy_score(y_test, y_pred_test))
                
                
            if accuracy_test is None:
                plt.figure(figsize=(8, 4))
                plt.plot(accuracy_train, label="Training accuracy", color="blue")
                plt.title("Accuracy during training")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                _, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

                axes[0].plot(accuracy_train, label="Training accuracy", color="blue")
                axes[0].set_title("Training set accuracy")
                axes[0].set_ylabel("Accuracy")
                axes[0].grid(True)
                axes[0].legend()

                axes[1].plot(accuracy_test, label="Test accuracy", color="green")
                axes[1].set_title("Test set accuracy")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Accuracy")
                axes[1].grid(True)
                axes[1].legend()

                plt.tight_layout()
                plt.show()
                
        
        return self
    
    def predict_proba(self, X):
        return np.column_stack([
            self.binary_models[class_label].predict_proba(X).ravel() 
            for class_label in self.class_labels
        ])
        
    def predict(self, X):
        return self.class_labels[
            np.argmax(self.predict_proba(X), axis=1)
        ]
    
    def transform(self, X):
        return self.predict(X)
        
