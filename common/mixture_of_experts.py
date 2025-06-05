import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

class MixtureOfExperts(BaseEstimator):
    def __init__(self, experts):
        self.experts = experts 
        self.gate_model = KMeans(n_clusters = len(experts)) 
        
    def fit(self, X, y):
        
        self.gate_model.fit(X)
        
        clusters_pred = self.gate_model.predict(X)
        
        for i in range(len(self.experts)):
            self.experts[i].fit(X[clusters_pred == i], y[clusters_pred == i]) 
            
            
    def predict(self, X):
        
        clusters_goruping = self.gate_model.predict(X)

        results = np.ones(X.shape[0])
        for i in range(len(self.experts)):
            results[clusters_goruping == i] = self.experts[i].predict(X[clusters_goruping == i])
        
        return results
