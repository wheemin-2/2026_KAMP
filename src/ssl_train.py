"""
모델 학습 / 하이퍼파라미터 탐색 / 평가 / 실험 로그 실행 모듈

전체 흐름
    ssl_preprocessing.preprocess_product(...)      # 전처리 (제품별)
        -> tune_model(...)                         # labeled train 에서 Stratified K-Fold CV 로 탐색
        -> train_and_evaluate(...)                 # best 파라미터로 재학습 후 test 평가
        -> ExperimentLog.append(...)               # experiment_log.csv 에 누적

Leakage 를 막기 위한 설계
    * 하이퍼파라미터 탐색은 labeled train 안에서만 CV 를 돌리고, test 는 마지막에 한 번만 사용한다.
    * SMOTE 는 CV 의 fold 마다 "학습 fold 의 labeled 샘플에만" 적용한다.
      (검증 fold 로 합성 샘플이 새는 것을 방지, unlabeled 에는 적용하지 않음)
    * 준지도 학습(self-training)은 fold 마다 [학습 fold labeled + 전체 unlabeled] 로 학습하고
      검증 fold 는 라벨이 있는 샘플만으로 평가한다.

사용 예시
    from ssl_train import run_experiments

    df = run_experiments(
        'cn7',
        prep_grid={
            'scaling': ['standard', 'robust'],
            'feature_selection': [None, ['corr', 'kbest']],
            'imbalance': ['class_weight', 'smote'],
        },
        models=['svc', 'rf', 'gnb', 'dnn'],
        ssl_methods=[None, 'self_training'],   # 지도 학습 baseline vs 준지도 학습 비교
        n_iter={'default': 15, 'dnn': 6},
        log_path='../logs/experiment_log.csv',
    )
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import time
import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, ParameterSampler, StratifiedKFold

from ssl_evaluate import (
    CV_METRICS,
    METRIC_COLS,
    ExperimentLog,
    compute_metrics,
    cv_score,
    get_scores,
    infer_pos_label,
    print_result,
)
from ssl_models import MODEL_NAMES, PARAM_SPACES, make_model
from ssl_preprocessing import (
    DEFAULT_DATA_DIR,
    PreprocessedData,
    _apply_imbalance,
    preprocess_data,
    preprocess_product,
)

DEFAULT_SSL_PARAMS = {"threshold": 0.9, "max_iter": 10}


# ---------------------------------------------------------------------------
# 1. 학습 한 번: (SMOTE) -> (self-training) -> fit
# ---------------------------------------------------------------------------
def fit_estimator(
    name: str,
    params: Dict,
    X_l: np.ndarray,
    y_l: np.ndarray,
    X_u: Optional[np.ndarray],
    *,
    imbalance: Optional[str] = None,
    smote_params: Optional[Dict] = None,
    semi_supervised: Optional[str] = None,
    ssl_params: Optional[Dict] = None,
    random_state: int = 42,
):
    """
    imbalance       : None | 'smote' | 'class_weight'
    semi_supervised : None | 'self_training'
    SMOTE 는 labeled 데이터에만 적용된다. unlabeled 는 라벨 -1 로 self-training 에 함께 투입된다.
    """
    if semi_supervised not in (None, "self_training"):
        raise ValueError("semi_supervised 는 None 또는 'self_training' 이어야 합니다.")
    ssl_on = semi_supervised == "self_training" and X_u is not None and len(X_u) > 0
    cw = "balanced" if imbalance == "class_weight" else None
    model = make_model(name, class_weight=cw, random_state=random_state, probability=ssl_on, **params)

    if imbalance == "smote":
        sp = smote_params or {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xr, yr, _, _ = _apply_imbalance(
                pd.DataFrame(X_l),
                pd.Series(y_l),
                "smote",
                random_state,
                sp.get("k_neighbors", 5),
                sp.get("sampling_strategy", "auto"),
            )
        X_l, y_l = Xr.to_numpy(dtype=float), yr.to_numpy()

    if ssl_on:
        if (np.asarray(y_l) == -1).any():
            raise ValueError("라벨에 -1 이 있으면 self-training 의 unlabeled 표시와 충돌합니다.")
        from sklearn.semi_supervised import SelfTrainingClassifier

        p = {**DEFAULT_SSL_PARAMS, **(ssl_params or {})}
        est = SelfTrainingClassifier(model, criterion="threshold", **p)
        X_all = np.vstack([X_l, X_u])
        y_all = np.concatenate([np.asarray(y_l), -np.ones(len(X_u), dtype=int)])
        est.fit(X_all, y_all)
        return est

    model.fit(X_l, y_l)
    return model


# ---------------------------------------------------------------------------
# 2. 하이퍼파라미터 탐색 (Stratified K-Fold CV)
# ---------------------------------------------------------------------------
def _cv_task(pi, fi, name, params, tr, va, X, y, X_u, pos_label, scoring, fit_kwargs):
    try:
        est = fit_estimator(name, params, X[tr], y[tr], X_u, **fit_kwargs)
        pred = est.predict(X[va])
        score = get_scores(est, X[va], pos_label)
        return pi, fi, cv_score(scoring, y[va], pred, score, pos_label), None
    except Exception as e:  # 특정 조합 실패가 전체 탐색을 멈추지 않도록
        return pi, fi, np.nan, f"{type(e).__name__}: {e}"


def _candidates(name: str, search: str, n_iter: int, random_state: int) -> List[Dict]:
    space = PARAM_SPACES[name]
    if search == "none":
        return [{}]
    size = len(ParameterGrid(space))
    if search == "grid" or n_iter >= size:
        return list(ParameterGrid(space))
    if search == "random":
        return list(ParameterSampler(space, n_iter=n_iter, random_state=random_state))
    raise ValueError("search 는 'random', 'grid', 'none' 중 하나여야 합니다.")


def tune_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    X_u: Optional[np.ndarray],
    *,
    pos_label,
    scoring: str = "f1",
    cv: int = 5,
    search: str = "random",
    n_iter: int = 20,
    imbalance: Optional[str] = None,
    smote_params: Optional[Dict] = None,
    semi_supervised: Optional[str] = None,
    ssl_params: Optional[Dict] = None,
    n_jobs: int = 1,
    random_state: int = 42,
):
    """
    반환: (best_params, best_cv_score, cv_results DataFrame)
    search='none' 이면 기본 파라미터를 그대로 쓰고 CV 는 생략한다.
    """
    if scoring not in CV_METRICS:
        raise ValueError(f"scoring 은 {CV_METRICS} 중 하나여야 합니다.")

    cands = _candidates(name, search, n_iter, random_state)
    n_splits = min(cv, int(pd.Series(y).value_counts().min()))
    if search == "none" or len(cands) == 1 and not cands[0]:
        return {}, np.nan, pd.DataFrame()
    if n_splits < 2:
        warnings.warn("소수 클래스 샘플이 너무 적어 CV 를 할 수 없습니다. 기본 파라미터를 사용합니다.")
        return {}, np.nan, pd.DataFrame()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))
    fit_kwargs = dict(
        imbalance=imbalance,
        smote_params=smote_params,
        semi_supervised=semi_supervised,
        ssl_params=ssl_params,
        random_state=random_state,
    )
    jobs = 1 if name == "dnn" else n_jobs  # TensorFlow 는 프로세스 병렬과 궁합이 나쁨

    res = Parallel(n_jobs=jobs)(
        delayed(_cv_task)(pi, fi, name, p, tr, va, X, y, X_u, pos_label, scoring, fit_kwargs)
        for pi, p in enumerate(cands)
        for fi, (tr, va) in enumerate(folds)
    )

    scores = np.full((len(cands), n_splits), np.nan)
    first_err = None
    for pi, fi, s, err in res:
        scores[pi, fi] = s
        first_err = first_err or err
    mean, std = np.nanmean(scores, axis=1), np.nanstd(scores, axis=1)
    if np.all(np.isnan(mean)):
        raise RuntimeError(f"[{name}] 모든 하이퍼파라미터 조합이 실패했습니다. 첫 오류: {first_err}")

    cv_df = pd.DataFrame(
        {"params": [json.dumps(p, default=str) for p in cands], f"mean_{scoring}": mean, f"std_{scoring}": std}
    ).sort_values(f"mean_{scoring}", ascending=False)
    best = int(np.nanargmax(mean))
    return cands[best], float(mean[best]), cv_df


# ---------------------------------------------------------------------------
# 3. 최종 학습 + test 평가
# ---------------------------------------------------------------------------
def train_and_evaluate(
    out: PreprocessedData,
    model_name: str,
    *,
    semi_supervised: Optional[str] = None,
    scoring: str = "f1",
    cv: int = 5,
    search: str = "random",
    n_iter: int = 20,
    ssl_params: Optional[Dict] = None,
    pos_label=None,
    n_jobs: int = 1,
    random_state: int = 42,
) -> Dict:
    """모델 하나를 튜닝 -> 재학습 -> test 평가하고 결과 dict(로그 한 행)를 반환한다."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"알 수 없는 모델: {model_name}. 사용 가능: {MODEL_NAMES}")

    # SMOTE 이전의 labeled train 사용 (fold 마다 SMOTE 를 다시 적용하기 위해)
    X_df = out.X_train_orig if out.X_train_orig is not None else out.X_train
    y_ser = out.y_train_orig if out.y_train_orig is not None else out.y_train
    X = X_df.to_numpy(dtype=float)
    y = np.asarray(y_ser)
    X_u = out.X_unlabeled.to_numpy(dtype=float) if semi_supervised else None
    X_te = out.X_test.to_numpy(dtype=float)
    y_te = np.asarray(out.y_test)

    if pos_label is None:
        pos_label = infer_pos_label(y)

    common = dict(
        imbalance=out.imbalance,
        smote_params=out.smote_params,
        semi_supervised=semi_supervised,
        ssl_params=ssl_params,
        random_state=random_state,
    )

    t0 = time.time()
    best_params, best_cv, cv_df = tune_model(
        model_name, X, y, X_u,
        pos_label=pos_label, scoring=scoring, cv=cv, search=search, n_iter=n_iter,
        n_jobs=n_jobs, **common,
    )
    est = fit_estimator(model_name, best_params, X, y, X_u, **common)
    fit_seconds = time.time() - t0

    y_pred = est.predict(X_te)
    y_score = get_scores(est, X_te, pos_label)
    metrics = compute_metrics(y_te, y_pred, y_score, pos_label)

    row = {
        "model": model_name,
        "semi_supervised": semi_supervised or "none",
        "pos_label": pos_label,
        "cv_scoring": scoring,
        "cv_score": best_cv,
        "cv_folds": cv,
        "search": search,
        "best_params": json.dumps(best_params, default=str),
        "n_features": len(out.selected_features),
        "selected_features": ";".join(out.selected_features),
        "n_train_labeled": len(X),
        "n_unlabeled": len(out.X_unlabeled),
        "n_test": len(X_te),
        "fit_seconds": fit_seconds,
        **metrics,
    }
    row["_cv_results"] = cv_df  # 로그에는 저장하지 않고 호출자가 필요하면 사용
    return row


# ---------------------------------------------------------------------------
# 4. 실험 실행 (전처리 설정 x 모델 x 준지도 방식)
# ---------------------------------------------------------------------------
_PREP_DEFAULTS = {
    k: v.default
    for k, v in inspect.signature(preprocess_data).parameters.items()
    if v.default is not inspect.Parameter.empty and k != "verbose"
}


def _tag(v) -> str:
    if v is None:
        return "none"
    if isinstance(v, (list, tuple)):
        return "+".join(map(str, v)) if v else "none"
    return str(v)


def _prep_columns(prep_options: Dict) -> Dict:
    p = {**_PREP_DEFAULTS, **prep_options}
    return {
        "prep_scaling": _tag(p["scaling"]),
        "prep_feature_selection": _tag(p["feature_selection"]),
        "prep_imbalance": _tag(p["imbalance"]),
        "prep_outlier": _tag(p["outlier"]),
        "prep_drop_constant": p["drop_constant"],
        "prep_fit_on_unlabeled": p["fit_on_unlabeled"],
        "prep_json": json.dumps(p, default=str, ensure_ascii=False),
    }


def run_experiment(
    product: str,
    prep_options: Optional[Dict] = None,
    models: Sequence[str] = ("svc", "rf", "gnb", "dnn"),
    ssl_methods: Sequence[Optional[str]] = (None,),
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    scoring: str = "f1",
    cv: int = 5,
    search: str = "random",
    n_iter: Union[int, Dict[str, int]] = 20,
    ssl_params: Optional[Dict] = None,
    pos_label=None,
    n_jobs: int = 1,
    random_state: int = 42,
    log: Optional[ExperimentLog] = None,
    skip_on_error: bool = True,
    verbose: bool = True,
    _run_tag: Optional[str] = None,
) -> List[Dict]:
    """전처리 설정 1개에 대해 (모델 x 준지도 방식) 전부를 학습/평가하고 로그에 남긴다."""
    prep_options = dict(prep_options or {})
    prep_options.setdefault("random_state", random_state)
    out = preprocess_product(product, data_dir=data_dir, verbose=False, **prep_options)
    prep_cols = _prep_columns(prep_options)
    run_tag = _run_tag or time.strftime("%Y%m%d-%H%M%S")

    rows = []
    for i, (m, s) in enumerate(itertools.product(models, ssl_methods)):
        n = n_iter.get(m, n_iter.get("default", 20)) if isinstance(n_iter, dict) else n_iter
        try:
            r = train_and_evaluate(
                out, m, semi_supervised=s, scoring=scoring, cv=cv, search=search, n_iter=n,
                ssl_params=ssl_params, pos_label=pos_label, n_jobs=n_jobs, random_state=random_state,
            )
        except Exception as e:
            if not skip_on_error:
                raise
            warnings.warn(f"[{product}/{m}/ssl={s}] 실패하여 건너뜁니다: {type(e).__name__}: {e}")
            continue
        r.pop("_cv_results", None)
        r = {
            "exp_id": f"{run_tag}-{product}-{m}-{_tag(s)}-{hashlib.md5(prep_cols['prep_json'].encode('utf-8')).hexdigest()[:6]}",
            "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "product": product,
            **prep_cols,
            **r,
        }
        rows.append(r)
        if verbose:
            print_result(r)
        if log is not None:
            log.append(r)  # 모델 하나 끝날 때마다 저장 -> 중간에 멈춰도 결과 보존
    return rows


def run_experiments(
    product: str,
    prep_grid: Union[Dict[str, list], List[Dict]],
    models: Sequence[str] = ("svc", "rf", "gnb", "dnn"),
    ssl_methods: Sequence[Optional[str]] = (None,),
    *,
    log_path: Optional[str] = "experiment_log.csv",
    sort_by: str = "f1",
    **kwargs,
) -> pd.DataFrame:
    """
    전처리 조합 전체를 순회하며 실험한다.

    prep_grid : {'scaling': [...], 'feature_selection': [...], 'imbalance': [...]} 처럼
                옵션별 후보 리스트(데카르트 곱)이거나, 설정 dict 의 리스트.
    반환      : 이번 호출에서 수행한 실험의 비교표 (sort_by 기준 내림차순)
    """
    if isinstance(prep_grid, dict):
        keys = list(prep_grid)
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*[prep_grid[k] for k in keys])]
    else:
        combos = list(prep_grid)

    log = ExperimentLog(log_path) if log_path else None
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    all_rows: List[Dict] = []
    for i, combo in enumerate(combos, 1):
        print(f"\n===== [{i}/{len(combos)}] {product} preprocessing: {combo} =====")
        all_rows += run_experiment(
            product, combo, models, ssl_methods, log=log, _run_tag=run_tag, **kwargs
        )

    if not all_rows:
        print("수행된 실험이 없습니다.")
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    cmp_ = ExperimentLog.compare(ExperimentLog(""), product=product, by=sort_by, df=df)
    print("\n===== 이번 실행 결과 비교 (sort_by=%s) =====" % sort_by)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(cmp_.round(4).to_string(index=False))
    return cmp_


if __name__ == "__main__":
    # 실행 예시: 데이터 경로/옵션은 환경에 맞게 수정
    run_experiments(
        "cn7",
        prep_grid={
            "scaling": ["standard", "robust"],
            "feature_selection": [None, ["corr", "kbest"]],
            "imbalance": ["class_weight", "smote"],
        },
        models=["svc", "rf", "gnb", "logreg"],
        ssl_methods=[None, "self_training"],
        n_iter={"default": 10},
        log_path="experiment_log.csv",
    )
