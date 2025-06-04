from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from common.consts import STUDENT_DATA_PATH, CATEGORICAL_COLUMN_NAMES, CATEGORY_TRANSLATIONS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from imblearn.over_sampling import SMOTE


X_TRAIN = "X_train"
Y_TRAIN = "y_train"
X_TEST = "X_test"
Y_TEST = "y_test"


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
    
def get_preprocessor(numerical_column_names, categorical_column_names, degree=None, include_bias=False):
    
    if degree is None:
        numerical_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])
    else:
        numerical_pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=include_bias)),
            ("scaler", StandardScaler())
        ])
    
    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_column_names),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_column_names)
        ]
    )
    
    
def train_with_plot(model, X, y, start=20, end=251, step=5, degree=None, include_bias=False, random_state=None):
    
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    numerical_column_names = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_column_names = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = get_preprocessor(numerical_column_names, categorical_column_names, degree=degree, include_bias=include_bias)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    accuracies_train = []
    accuracies_test = []
    iter_values = list(range(start, end, step))
    
    for iter in iter_values:
        
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", model(max_iter=iter))
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        accuracies_train.append(accuracy_score(y_train, y_pred_train))
        accuracies_test.append(accuracy_score(y_test, y_pred_test))
        
    _, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
    axs[0].plot(iter_values, accuracies_train, label="Train Accuracy", color="blue")
    axs[0].set_title("Train Accuracy vs max_iter")
    axs[0].set_ylabel("Accuracy")
    axs[0].grid(True)

    axs[1].plot(iter_values, accuracies_test, label="Test Accuracy", color="green")
    axs[1].set_title("Test Accuracy vs max_iter")
    axs[1].set_xlabel("max_iter")
    axs[1].set_ylabel("Accuracy")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    
    
def get_datasets(n_splits = 3, should_oversample=False):
    data = get_data()
    X = data.drop(columns=["Target"])
    y = data["Target"]

    numerical_column_names = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_column_names = X.select_dtypes(include=["object"]).columns.tolist()

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X[numerical_column_names])

    encoder = OneHotEncoder(handle_unknown="ignore")
    X_cat_encoded = encoder.fit_transform(X[categorical_column_names])

    X = np.hstack([X_num_scaled, X_cat_encoded.toarray()])

    kfold = KFold(n_splits=n_splits, shuffle=True)
    datasets = []

    smote = SMOTE()

    for train_indices, test_indices in kfold.split(X, y):
        
        X_train = X[train_indices]
        y_train =y.iloc[train_indices]

        if should_oversample:
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        datasets.append({
            X_TRAIN: X_train,
            Y_TRAIN: y_train,
            X_TEST: X[test_indices],
            Y_TEST: y.iloc[test_indices]
        })
        
    return datasets
    