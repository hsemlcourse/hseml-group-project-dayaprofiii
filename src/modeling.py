import random
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Any

from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score
)
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def split_data(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
    val_size: float = 0.5,
    random_state: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=random_state
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def evaluate_model(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series
) -> Dict[str, float]:
    y_pred = model.predict(x)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x)[:, 1]
    else:
        y_proba = None
    return {
        'F1': f1_score(y, y_pred),
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'ROC_AUC': roc_auc_score(y, y_proba)
    }


def get_baseline_model() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=500, random_state=SEED))
    ])


def get_all_models() -> Dict[str, Any]:
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=SEED),
        'KNN_k5': KNeighborsClassifier(n_neighbors=5),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            n_jobs=-1
        ),
        'xGBoost': XGBClassifier(
            n_estimators=200,
            random_state=SEED,
            verbosity=0,
            eval_metric='logloss'
        ),
        'GaussianNB': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(
            random_state=SEED
        ),
    }
    return models


def train_models(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    models = get_all_models()
    results = []
    trained_models = {}

    for model_name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.fit(x_train, y_train)
        metrics = evaluate_model(pipe, x_val, y_val)
        metrics['Model'] = model_name
        results.append(metrics)
        trained_models[model_name] = pipe

    results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
    results_df = results_df[['Model', 'F1', 'Accuracy', 'Precision', 'Recall', 'ROC_AUC']]

    return results_df, trained_models


def hyperparameter_tuning(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    model: Any
) -> Pipeline:
    param_grids = {
        'RandomForest': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [10, 15, None],
            'model__min_samples_split': [2, 5]
        },
        'xGBoost': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10],
            'model__learning_rate': [0.05, 0.1]
        },
        'KNN_k5': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance']
        },
        'GaussianNB': {
            'model__var_smoothing': [1e-9, 1e-7]
        },
        'DecisionTree': {
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    }

    if model_name not in param_grids:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.fit(x_train, y_train)
        return pipe
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    param_grid = param_grids[model_name]
    best_score = -np.inf
    best_estimator = None

    for params in tqdm(list(ParameterGrid(param_grid)), desc=f'GridSearch {model_name}', unit='comb'):
        candidate_grid = {k: [v] for k, v in params.items()}
        grid_search = GridSearchCV(
            pipe,
            candidate_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(x_train, y_train)
        score = grid_search.cv_results_['mean_test_score'][0]

        if score > best_score:
            best_score = score
            best_estimator = grid_search.best_estimator_

    return best_estimator


def create_voting_ensemble(
    x_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    ensemble_models = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('lr', LogisticRegression(max_iter=500, random_state=SEED))
    ]

    voting = VotingClassifier(estimators=ensemble_models, voting='soft', n_jobs=-1)

    pipe_ensemble = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', voting)
    ])

    pipe_ensemble.fit(x_train, y_train)
    return pipe_ensemble


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)
    print(f"Модель сохранена по пути {path}")


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def train_and_evaluate_models(models, x_train, y_train, x_val, y_val):
    results = []
    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.fit(x_train, y_train)
        metrics = evaluate_model(pipe, x_val, y_val)
        metrics['model'] = name
        results.append(metrics)
    return pd.DataFrame(results).sort_values('F1', ascending=False)