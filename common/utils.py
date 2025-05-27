from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from common.consts import STUDENT_DATA_PATH, CATEGORICAL_COLUMN_NAMES, CATEGORY_TRANSLATIONS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression


def get_data(data_path = STUDENT_DATA_PATH, only_numeric=False):
    data = pd.read_csv(data_path, sep=";")
    data.columns = data.columns.str.strip()
    
    if data_path == STUDENT_DATA_PATH:
        for column_name in CATEGORICAL_COLUMN_NAMES:
            data[column_name] = data[column_name].map(lambda field: CATEGORY_TRANSLATIONS.get(column_name, {}).get(field, field))

    if only_numeric:
        data = data.select_dtypes(include=["number"])
        
    return data

def get_numeric_data(data_path = STUDENT_DATA_PATH, sep=";"):  
    data = pd.read_csv(data_path, sep=sep)
    data.columns = data.columns.str.strip()
    if data_path == STUDENT_DATA_PATH:
        data = data.drop(columns=CATEGORICAL_COLUMN_NAMES)
    else:
        data = data.select_dtypes(include=["number"])
    
    return data

def show_polynomial_plts(X, y, degrees, x_title = None, y_title = None):
    
    if isinstance(X, pd.Series):
        X = X.values

    if isinstance(y, pd.Series):
        y = y.values
    
    sort_indexes = np.argsort(X.flatten())
    X_sorted = X[sort_indexes]
    y_sorted = y[sort_indexes]

    plt.figure(figsize=(8, len(degrees)*5))

    for i, degree in enumerate(degrees, 1):
        
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_sorted.reshape(-1, 1))

        model = LinearRegression()
        model.fit(X_poly, y_sorted)

        y_pred = model.predict(X_poly)

        plt.subplot(len(degrees), 1, i)
        plt.scatter(X_sorted, y_sorted, label="Data points")
        plt.plot(X_sorted, y_pred, color='red', label="Polynomial regression curve")
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.title(f"Polynomial Regression (degree {degree})")
        plt.legend()
        
    plt.show()
    
def get_preprocessor(numerical_column_names, categorical_column_names):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_column_names),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_column_names)
        ]
    )
    