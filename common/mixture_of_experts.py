from sklearn.base import BaseEstimator


class MixtureOfExperts(BaseEstimator):
    def __init__(self, experts, gate_model):
        self.experts = experts
        self.gate_model = gate_model
        
    def fit(self, X, y):
        self.gate_model.fit(X, y)
        for expert in self.experts:
            expert.fit(X, y)
            
            
    def predict(self, X):
        
        print(X.shape)
        for x in X:
            print(f"x: {x.shape}")
            print(f"reshaped: {x.reshape(1, -1).shape}")
            print(self.gate_model.predict_proba(x))
            
