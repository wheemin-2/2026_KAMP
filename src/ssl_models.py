"""
모델 정의 모듈: 모델 생성(make_model), 하이퍼파라미터 탐색 공간(PARAM_SPACES)

지원 모델
    svc     : Support Vector Classifier
    rf      : Random Forest
    gnb     : Gaussian Naive Bayes (class_weight 지원 버전)
    dnn     : Keras DNN (sklearn 호환 래퍼)
    logreg  : Logistic Regression   (추가)
    knn     : K-Nearest Neighbors   (추가, class_weight 미지원)
    hgb     : HistGradientBoosting  (추가)

모든 모델은 fit / predict / predict_proba(또는 decision_function) 를 가지는
sklearn 호환 estimator 이므로 SelfTrainingClassifier 등에 그대로 넣을 수 있다.
불균형 처리 중 'class_weight' 는 make_model(class_weight='balanced') 로 전달된다.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

MODEL_NAMES = ["svc", "rf", "gnb", "dnn", "logreg", "knn", "hgb"]


# ---------------------------------------------------------------------------
# class_weight 를 지원하는 Gaussian Naive Bayes
# ---------------------------------------------------------------------------
class BalancedGaussianNB(GaussianNB):
    """GaussianNB 는 class_weight 인자가 없어서, fit 시 sample_weight 로 변환해 지원한다."""

    def __init__(self, *, priors=None, var_smoothing=1e-9, class_weight=None):
        super().__init__(priors=priors, var_smoothing=var_smoothing)
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        y_arr = np.asarray(y)
        if self.class_weight is not None:
            classes, counts = np.unique(y_arr, return_counts=True)
            if isinstance(self.class_weight, dict):
                w = {c: float(self.class_weight.get(c, 1.0)) for c in classes}
            elif self.class_weight == "balanced":
                w = {c: len(y_arr) / (len(classes) * n) for c, n in zip(classes, counts)}
            else:
                raise ValueError("class_weight 는 None, 'balanced' 또는 dict 여야 합니다.")
            sw = np.array([w[v] for v in y_arr], dtype=float)
            sample_weight = sw if sample_weight is None else sw * np.asarray(sample_weight)
        return super().fit(X, y_arr, sample_weight=sample_weight)


# ---------------------------------------------------------------------------
# Keras DNN (sklearn 호환)
# ---------------------------------------------------------------------------
class KerasDNNClassifier(ClassifierMixin, BaseEstimator):
    """
    이진 분류용 Keras DNN 을 sklearn estimator 처럼 쓰기 위한 래퍼.
    (GridSearch/SelfTrainingClassifier/clone 과 호환)

    Parameters
    ----------
    hidden_units : 은닉층 노드 수 튜플. 예) (128, 64, 32)
    dropout      : 은닉층 뒤 Dropout 비율
    l2           : Dense 가중치 L2 규제 강도
    learning_rate, batch_size, epochs : 학습 설정
    patience     : EarlyStopping patience (validation loss 기준, best weights 복원)
    val_fraction : early stopping 용 검증 데이터 비율 (stratified 로 분리)
    class_weight : None | 'balanced' | dict  (Keras fit 의 class_weight 로 전달)
    """

    def __init__(
        self,
        hidden_units=(64, 32),
        dropout=0.3,
        l2=0.0,
        learning_rate=1e-3,
        batch_size=64,
        epochs=150,
        patience=10,
        val_fraction=0.15,
        class_weight=None,
        random_state=42,
        verbose=0,
    ):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.l2 = l2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

    def _build(self, n_features: int):
        import tensorflow as tf

        reg = tf.keras.regularizers.l2(self.l2) if self.l2 else None
        layers = [tf.keras.Input(shape=(n_features,))]
        for units in self.hidden_units:
            layers.append(tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg))
            if self.dropout:
                layers.append(tf.keras.layers.Dropout(self.dropout))
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
        )
        return model

    def fit(self, X, y):
        try:
            import tensorflow as tf
        except ImportError as e:  # pragma: no cover
            raise ImportError("DNN 모델을 쓰려면 tensorflow 가 필요합니다: pip install tensorflow") from e
        from sklearn.model_selection import train_test_split

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("KerasDNNClassifier 는 이진 분류만 지원합니다.")
        self.n_features_in_ = X.shape[1]
        yb = (y == self.classes_[1]).astype("float32")

        tf.keras.utils.set_random_seed(self.random_state)
        tf.keras.backend.clear_session()
        self.model_ = self._build(X.shape[1])

        # class_weight -> {0: w0, 1: w1}
        cw = None
        if self.class_weight == "balanced":
            n = len(yb)
            n1 = float(yb.sum())
            n0 = n - n1
            cw = {0: n / (2 * n0), 1: n / (2 * n1)}
        elif isinstance(self.class_weight, dict):
            cw = {i: float(self.class_weight.get(c, 1.0)) for i, c in enumerate(self.classes_)}

        # early stopping 용 검증셋 (Keras validation_split 은 뒤쪽 샘플을 자르므로
        # SMOTE 로 뒤에 붙은 합성 샘플만 검증에 쓰이는 문제가 있어 직접 stratified 분리)
        X_tr, y_tr, val = X, yb, None
        monitor = "loss"
        if self.val_fraction and min(np.bincount(yb.astype(int))) >= 4:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, yb, test_size=self.val_fraction, stratify=yb, random_state=self.random_state
            )
            val = (X_val, y_val)
            monitor = "val_loss"

        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, patience=self.patience, restore_best_weights=True
            )
        ]
        self.model_.fit(
            X_tr,
            y_tr,
            validation_data=val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=cw,
            callbacks=cb,
            verbose=self.verbose,
            shuffle=True,
        )
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return self.classes_[(p >= 0.5).astype(int)]


# ---------------------------------------------------------------------------
# 모델 생성
# ---------------------------------------------------------------------------
def make_model(
    name: str,
    class_weight: Optional[str] = None,
    random_state: int = 42,
    probability: bool = False,
    **params,
):
    """
    name          : MODEL_NAMES 중 하나
    class_weight  : None | 'balanced'  (imbalance='class_weight' 일 때 'balanced')
    probability   : SVC 확률 추정 여부. self-training 처럼 predict_proba 가 필요할 때만 True
                    (평가용 ROC-AUC 는 decision_function 으로 계산하므로 평소엔 False 가 빠름)
    params        : 하이퍼파라미터 (set_params 로 덮어씀)
    """
    cw = class_weight
    if name == "svc":
        m = SVC(class_weight=cw, probability=probability, random_state=random_state, cache_size=500)
    elif name == "rf":
        m = RandomForestClassifier(class_weight=cw, random_state=random_state, n_jobs=1)
    elif name == "gnb":
        m = BalancedGaussianNB(class_weight=cw)
    elif name == "dnn":
        m = KerasDNNClassifier(class_weight=cw, random_state=random_state)
    elif name == "logreg":
        m = LogisticRegression(class_weight=cw, max_iter=2000, random_state=random_state)
    elif name == "knn":
        m = KNeighborsClassifier()  # class_weight 미지원 (imbalance='class_weight' 무시됨)
    elif name == "hgb":
        m = HistGradientBoostingClassifier(class_weight=cw, random_state=random_state)
    else:
        raise ValueError(f"알 수 없는 모델: {name}. 사용 가능: {MODEL_NAMES}")
    if params:
        m.set_params(**params)
    return m


# ---------------------------------------------------------------------------
# 하이퍼파라미터 탐색 공간
#   - random search 는 이 공간에서 n_iter 개를 샘플링, grid search 는 전부 탐색
#   - 값 목록은 자유롭게 수정 가능 (list of dict 도 허용)
# ---------------------------------------------------------------------------
PARAM_SPACES: Dict[str, object] = {
    "svc": [
        {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
        {"kernel": ["linear"], "C": [0.01, 0.1, 1, 10]},
    ],
    "rf": {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "gnb": {
        "var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
    },
    "dnn": {
        "hidden_units": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
        "dropout": [0.1, 0.3, 0.5],
        "l2": [0.0, 1e-4, 1e-3],
        "learning_rate": [1e-3, 3e-4, 1e-4],
        "batch_size": [32, 64, 128],
    },
    "logreg": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "hgb": {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_iter": [100, 200, 300],
        "max_depth": [None, 3, 5, 8],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
}
