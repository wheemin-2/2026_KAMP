"""
평가/실험 로그 모듈

- compute_metrics : Accuracy, Recall, Precision, F1, ROC-AUC (+ PR-AUC, 혼동행렬)
- ExperimentLog   : 전처리 방식 x 모델 결과를 CSV 로 누적 저장/조회/비교
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLS = ["accuracy", "recall", "precision", "f1", "roc_auc", "pr_auc"]


def infer_pos_label(y) -> int:
    """양성 클래스를 소수 클래스로 자동 결정 (불량 탐지처럼 소수 클래스가 관심 대상인 경우)."""
    vc = pd.Series(np.asarray(y)).value_counts()
    return vc.idxmin()


def get_scores(model, X, pos_label) -> np.ndarray:
    """양성 클래스에 대한 점수(확률 또는 decision_function) 반환. ROC-AUC 계산용."""
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, classes.index(pos_label)]
    s = np.asarray(model.decision_function(X))
    # decision_function 은 classes_[1] 쪽이 양수
    return s if pos_label == classes[1] else -s


def compute_metrics(y_true, y_pred, y_score, pos_label) -> Dict[str, float]:
    """이진 분류 지표. 모든 지표는 pos_label 을 양성으로 계산한다."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    neg = [c for c in labels if c != pos_label]
    neg_label = neg[0] if neg else None

    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
    }
    if len(np.unique(y_true)) == 2 and y_score is not None:
        yt = (y_true == pos_label).astype(int)
        m["roc_auc"] = roc_auc_score(yt, y_score)
        m["pr_auc"] = average_precision_score(yt, y_score)
    else:
        m["roc_auc"] = np.nan
        m["pr_auc"] = np.nan

    if neg_label is not None:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label]).ravel()
    else:
        tn = fp = fn = 0
        tp = int(((y_true == pos_label) & (y_pred == pos_label)).sum())
    m.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return m


def cv_score(metric: str, y_true, y_pred, y_score, pos_label) -> float:
    """하이퍼파라미터 탐색용 단일 점수 (클수록 좋음)."""
    if metric == "roc_auc":
        return roc_auc_score((np.asarray(y_true) == pos_label).astype(int), y_score)
    if metric == "average_precision":
        return average_precision_score((np.asarray(y_true) == pos_label).astype(int), y_score)
    if metric == "f1":
        return f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    if metric == "recall":
        return recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    if metric == "precision":
        return precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "balanced_accuracy":
        from sklearn.metrics import balanced_accuracy_score

        return balanced_accuracy_score(y_true, y_pred)
    raise ValueError(f"지원하지 않는 scoring: {metric}")


CV_METRICS = ["f1", "roc_auc", "average_precision", "recall", "precision", "accuracy", "balanced_accuracy"]


def print_result(row: Dict) -> None:
    """한 실험 결과를 사람이 보기 좋게 출력."""
    head = (
        f"[{row.get('product')}/{row.get('process_type')}] model={row.get('model')} | ssl={row.get('semi_supervised')} | "
        f"scaling={row.get('prep_scaling')} | fs={row.get('prep_feature_selection')} | "
        f"imbalance={row.get('prep_imbalance')} | outlier={row.get('prep_outlier')}"
    )
    print(head)
    print(
        "    Accuracy={accuracy:.4f}  Recall={recall:.4f}  Precision={precision:.4f}  "
        "F1={f1:.4f}  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}".format(**{k: row[k] for k in METRIC_COLS})
    )
    print(
        f"    confusion(tn/fp/fn/tp)={row['tn']}/{row['fp']}/{row['fn']}/{row['tp']}  "
        f"cv_{row.get('cv_scoring')}={row.get('cv_score'):.4f}  "
        f"n_features={row.get('n_features')}  fit_time={row.get('fit_seconds'):.1f}s"
    )
    print(f"    best_params={row.get('best_params')}")


class ExperimentLog:
    """
    실험 결과를 CSV 로 누적 저장한다. (행 1개 = 전처리 설정 x 모델 x 준지도 방식)

    log = ExperimentLog('experiment_log.csv')
    log.append(rows)                       # dict 리스트
    df = log.load()                        # 전체 로그
    log.compare(product='cn7', by='f1')    # 비교표
    """

    def __init__(self, path: str = "experiment_log.csv"):
        self.path = path

    def append(self, rows) -> pd.DataFrame:
        new = pd.DataFrame(rows if isinstance(rows, list) else [rows])
        if os.path.exists(self.path):
            old = pd.read_csv(self.path)
            new = pd.concat([old, new], ignore_index=True, sort=False)
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        new.to_csv(self.path, index=False)
        return new

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        return pd.read_csv(self.path)

    def compare(
        self,
        product: Optional[str] = None,
        by: str = "f1",
        top: Optional[int] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """지표 기준 내림차순 비교표 (전처리 설정 + 모델 + 5대 지표)."""
        df = self.load() if df is None else df
        if product is not None:
            df = df[df["product"] == product]
        cols = [
            "exp_id", "product", "process_type", "model", "semi_supervised",
            "prep_scaling", "prep_feature_selection", "prep_imbalance", "prep_outlier",
            "n_features", "cv_score",
        ] + METRIC_COLS
        cols = [c for c in cols if c in df.columns]
        out = df.sort_values(by, ascending=False)[cols]
        return out.head(top) if top else out
