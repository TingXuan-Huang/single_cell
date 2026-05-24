"""Linear and kNN probes for downstream subclass classification."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_iter: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Standardized multiclass logistic regression."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    return {
        "linear_acc": float(accuracy_score(y_test, y_pred)),
        "linear_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0.0)),
    }


def knn_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    k: int = 15,
) -> dict[str, float]:
    """k-Nearest Neighbors classifier with cosine distance."""
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        f"knn{k}_acc": float(accuracy_score(y_test, y_pred)),
        f"knn{k}_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0.0)),
    }
