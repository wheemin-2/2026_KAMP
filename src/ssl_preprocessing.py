"""
준지도 학습(Semi-supervised learning)용 데이터 전처리 모듈
- 사출성형기 AI 데이터셋 (제품: CN7, RG3 / labeled, unlabeled)

핵심 원칙 (Data Leakage 방지)
    1) labeled 데이터를 먼저 train/test 로 분리한다.
    2) 통계량을 학습(fit)하는 단계(스케일러, 이상치 경계, 상관 필터 등)는
       [labeled train + unlabeled] 로만 fit 한다. (test 는 절대 fit 에 쓰지 않음)
    3) test 는 transform 만 한다.
    4) 라벨을 쓰는 단계(kbest, model 기반 feature selection, SMOTE, class weight)는
       labeled train 만 사용한다. SMOTE 는 labeled train 에만 적용되고
       unlabeled 는 그대로 유지된다.
    5) 제품(CN7/RG3)별로 분포가 다르므로 전처리는 제품별로 따로 fit 한다.

파이프라인 순서
    상수 피처 제거 -> 결측 대체 -> 이상치 클리핑 -> 스케일링
    -> feature selection -> 불균형 처리(SMOTE / class_weight)

사용 예시
    from ssl_preprocessing import preprocess_product

    out = preprocess_product(
        'cn7',
        data_dir='../data/1. 사출성형기 AI 데이터셋',
        scaling='standard',
        feature_selection=['corr', 'kbest'],
        k_best=15,
        imbalance='class_weight',
    )
    out.X_train, out.y_train          # 지도 학습용 labeled train
    out.X_unlabeled                   # 준지도 학습용 unlabeled
    out.X_test, out.y_test            # 평가용
    out.class_weight                  # keras: model.fit(..., class_weight=out.class_weight)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.class_weight import compute_class_weight

TARGET_COL = "PassOrFail"
DEFAULT_DATA_DIR = "../data/1. 사출성형기 AI 데이터셋"

# ---------------------------------------------------------------------------
# 1. 데이터 로드 (Preprocessing_exp.ipynb 의 DataLoader_cn7 / DataLoader_rg3 통합)
# ---------------------------------------------------------------------------


class DataLoader:
    """
    labeled / unlabeled 를 불러오고 labeled 를 stratified train/test 로 분리한다.
    (노트북의 DataLoader_cn7, DataLoader_rg3 와 동일한 분리 방식)

    Attributes
    ----------
    labeled_train_X, labeled_train_Y : 지도 학습용 train
    labeled_test_X,  labeled_test_Y  : 평가용 test
    unlabeled                        : 라벨이 없는 데이터
    """

    def __init__(
        self,
        product: str,
        data_dir: str = DEFAULT_DATA_DIR,
        test_size: float = 0.3,
        random_state: int = 42,
        target_col: str = TARGET_COL,
    ):
        product = product.lower()
        if product not in ("cn7", "rg3"):
            raise ValueError("product 는 'cn7' 또는 'rg3' 이어야 합니다.")

        labeled = pd.read_csv(
            os.path.join(data_dir, f"moldset_labeled_{product}.csv"), index_col=0
        )
        unlabeled = pd.read_csv(
            os.path.join(data_dir, f"moldset_unlabeled_{product}.csv"), index_col=0
        )

        X = labeled.loc[:, labeled.columns != target_col]
        y = labeled[target_col]

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(sss.split(X, y))

        self.product = product
        self.labeled_train_X = X.iloc[train_idx]
        self.labeled_train_Y = y.iloc[train_idx]
        self.labeled_test_X = X.iloc[test_idx]
        self.labeled_test_Y = y.iloc[test_idx]
        self.unlabeled = unlabeled


# ---------------------------------------------------------------------------
# 2. 결과 컨테이너
# ---------------------------------------------------------------------------


@dataclass
class PreprocessedData:
    X_train: pd.DataFrame  # labeled train (imbalance='smote' 이면 oversampled)
    y_train: pd.Series
    X_unlabeled: pd.DataFrame  # unlabeled (변환만 됨, SMOTE 대상 아님)
    X_test: pd.DataFrame
    y_test: pd.Series
    class_weight: Optional[Dict[int, float]] = None  # imbalance='class_weight' 일 때만
    sample_weight: Optional[np.ndarray] = None  # X_train 과 같은 길이 (sklearn용)
    selected_features: List[str] = field(default_factory=list)
    preprocessor: Optional["SSLPreprocessor"] = None  # 새 데이터 변환에 재사용
    info: Dict = field(default_factory=dict)  # 각 단계에서 무엇이 일어났는지 기록
    # --- 모델 학습(교차검증)용: 불균형 처리 "이전"의 labeled train ---
    # SMOTE 를 CV 밖에서 한 번에 적용하면 검증 fold 에 합성 샘플이 새어 들어가므로,
    # 학습 단계에서는 이 원본에 fold 마다 SMOTE 를 다시 적용한다.
    X_train_orig: Optional[pd.DataFrame] = None
    y_train_orig: Optional[pd.Series] = None
    imbalance: Optional[str] = None
    smote_params: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3. 전처리기 (fit / transform 분리 -> leakage 방지)
# ---------------------------------------------------------------------------

_SCALERS = {
    "standard": lambda: StandardScaler(),
    "minmax": lambda: MinMaxScaler(),
    "robust": lambda: RobustScaler(),  # 이상치에 강함
    "maxabs": lambda: MaxAbsScaler(),
    "power": lambda: PowerTransformer(method="yeo-johnson"),  # 정규분포에 가깝게
    "quantile": lambda: QuantileTransformer(
        output_distribution="normal", n_quantiles=200, random_state=0
    ),
}


class SSLPreprocessor:
    """
    제품 하나에 대한 전처리기. fit() 은 [labeled train (+ unlabeled)] 만 보고,
    transform() 은 어떤 데이터(test 포함)에도 적용 가능하다.

    Parameters
    ----------
    scaling : None | 'standard' | 'minmax' | 'robust' | 'maxabs' | 'power' | 'quantile'
    feature_selection : None | str | list[str]
        'corr'  : 피처끼리 상관이 높은 것 제거 (unsupervised, unlabeled 포함 fit)
        'kbest' : 타깃과의 통계적 관련성 상위 k개 (labeled train 만 사용)
        'model' : RandomForest 중요도 기반 (labeled train 만 사용)
        리스트로 주면 순서대로 적용된다. 예) ['corr', 'kbest']
    k_best : kbest 에서 남길 피처 수
    kbest_score : 'f_classif' | 'mutual_info'
    corr_threshold : corr 필터에서 |상관계수| 가 이 값 이상이면 한쪽 제거
    model_threshold : model 기반 선택 임계값 ('median', 'mean' 또는 float)
    drop_constant : labeled train 에서 분산이 (거의) 0인 피처 제거.
        (예: CN7 labeled 의 Clamp_Open_Position 은 모두 0)
        -> 지도 학습이 아무 정보도 얻지 못하고, unlabeled 와 분포 불일치만 만들기 때문
    constant_tol : 이 값 이하의 분산이면 상수로 간주
    impute : None | 'mean' | 'median' | 'most_frequent'  (현재 데이터는 결측 없음, 안전장치)
    outlier : None | 'iqr' | 'quantile'  (경계를 넘는 값을 경계로 클리핑)
    outlier_iqr_k : iqr 방식의 배수 (기본 1.5)
    outlier_quantile : quantile 방식의 (하한, 상한) (기본 0.01, 0.99)
    fit_on_unlabeled : True 면 비지도 단계(이상치 경계/스케일러/상관 필터)를
        labeled train + unlabeled 로 fit (준지도 학습에서 권장)
    """

    def __init__(
        self,
        scaling: Optional[str] = "standard",
        feature_selection: Union[None, str, Sequence[str]] = None,
        k_best: int = 15,
        kbest_score: str = "f_classif",
        corr_threshold: float = 0.95,
        model_threshold: Union[str, float] = "median",
        drop_constant: bool = True,
        constant_tol: float = 1e-12,
        impute: Optional[str] = "median",
        outlier: Optional[str] = None,
        outlier_iqr_k: float = 1.5,
        outlier_quantile: tuple = (0.01, 0.99),
        fit_on_unlabeled: bool = True,
        random_state: int = 42,
    ):
        if scaling is not None and scaling not in _SCALERS:
            raise ValueError(f"scaling 은 None 또는 {list(_SCALERS)} 중 하나여야 합니다.")
        if outlier not in (None, "iqr", "quantile"):
            raise ValueError("outlier 는 None, 'iqr', 'quantile' 중 하나여야 합니다.")
        if kbest_score not in ("f_classif", "mutual_info"):
            raise ValueError("kbest_score 는 'f_classif' 또는 'mutual_info' 여야 합니다.")

        if feature_selection is None:
            fs = []
        elif isinstance(feature_selection, str):
            fs = [feature_selection]
        else:
            fs = list(feature_selection)
        bad = [m for m in fs if m not in ("corr", "kbest", "model")]
        if bad:
            raise ValueError(f"지원하지 않는 feature_selection: {bad}")

        self.scaling = scaling
        self.feature_selection = fs
        self.k_best = k_best
        self.kbest_score = kbest_score
        self.corr_threshold = corr_threshold
        self.model_threshold = model_threshold
        self.drop_constant = drop_constant
        self.constant_tol = constant_tol
        self.impute = impute
        self.outlier = outlier
        self.outlier_iqr_k = outlier_iqr_k
        self.outlier_quantile = outlier_quantile
        self.fit_on_unlabeled = fit_on_unlabeled
        self.random_state = random_state

        self.info_: Dict = {}

    # -- fit ---------------------------------------------------------------
    def fit(
        self,
        X_labeled_train: pd.DataFrame,
        y_labeled_train: pd.Series,
        X_unlabeled: Optional[pd.DataFrame] = None,
    ) -> "SSLPreprocessor":
        cols = list(X_labeled_train.columns)
        info = self.info_ = {}

        # (1) 상수 피처: labeled train 기준
        if self.drop_constant:
            var = X_labeled_train.var(axis=0, ddof=0)
            const = var.index[var <= self.constant_tol].tolist()
        else:
            const = []
        info["dropped_constant"] = const
        cols = [c for c in cols if c not in const]

        # 비지도 단계에 쓸 풀(pool): labeled train (+ unlabeled)
        if self.fit_on_unlabeled and X_unlabeled is not None:
            pool = pd.concat([X_labeled_train[cols], X_unlabeled[cols]], axis=0)
        else:
            pool = X_labeled_train[cols]

        # (2) 결측 대체
        self._imputer = None
        if self.impute:
            self._imputer = SimpleImputer(strategy=self.impute).fit(pool)
            pool = pd.DataFrame(self._imputer.transform(pool), columns=cols)

        # (3) 이상치 클리핑 경계
        self._lo = self._hi = None
        if self.outlier == "iqr":
            q1, q3 = pool.quantile(0.25), pool.quantile(0.75)
            iqr = q3 - q1
            self._lo = (q1 - self.outlier_iqr_k * iqr).to_numpy()
            self._hi = (q3 + self.outlier_iqr_k * iqr).to_numpy()
        elif self.outlier == "quantile":
            lo_q, hi_q = self.outlier_quantile
            self._lo = pool.quantile(lo_q).to_numpy()
            self._hi = pool.quantile(hi_q).to_numpy()
        if self._lo is not None:
            pool = pd.DataFrame(np.clip(pool.to_numpy(), self._lo, self._hi), columns=cols)

        # (4) 스케일러
        self._scaler = None
        if self.scaling:
            self._scaler = _SCALERS[self.scaling]().fit(pool)
            pool_scaled = pd.DataFrame(self._scaler.transform(pool), columns=cols)
        else:
            pool_scaled = pool

        # 이후 단계에서 쓸 labeled train (같은 변환을 통과시킴)
        lab_scaled = self._apply_base(X_labeled_train, cols)

        # (5) feature selection
        self._cols_before_fs = cols
        selected = list(cols)
        for method in self.feature_selection:
            if method == "corr":
                # 비지도: pool 로 상관 계산
                sub = pool_scaled[selected]
                corr = sub.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                drop = [c for c in upper.columns if (upper[c] >= self.corr_threshold).any()]
                info["dropped_corr"] = drop
                selected = [c for c in selected if c not in drop]
            elif method == "kbest":
                k = min(self.k_best, len(selected))
                score = (
                    f_classif
                    if self.kbest_score == "f_classif"
                    else (lambda X, y: mutual_info_classif(X, y, random_state=self.random_state))
                )
                skb = SelectKBest(score_func=score, k=k).fit(lab_scaled[selected], y_labeled_train)
                keep = np.array(selected)[skb.get_support()].tolist()
                info["kbest_scores"] = dict(zip(selected, np.round(skb.scores_, 4)))
                selected = keep
            elif method == "model":
                rf = RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                sfm = SelectFromModel(rf, threshold=self.model_threshold).fit(
                    lab_scaled[selected], y_labeled_train
                )
                info["rf_importance"] = dict(
                    zip(selected, np.round(sfm.estimator_.feature_importances_, 4))
                )
                selected = np.array(selected)[sfm.get_support()].tolist()

            if len(selected) == 0:
                raise ValueError(f"'{method}' 단계 이후 남은 피처가 없습니다. 임계값을 조정하세요.")

        self.selected_features_ = selected
        info["selected_features"] = selected
        return self

    # -- transform ---------------------------------------------------------
    def _apply_base(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """상수 제거 이후 ~ 스케일링까지 (feature selection 이전)."""
        Z = X[cols]
        arr = Z.to_numpy(dtype=float)
        if self._imputer is not None:
            arr = self._imputer.transform(pd.DataFrame(arr, columns=cols))
        if self._lo is not None:
            arr = np.clip(arr, self._lo, self._hi)
        if self._scaler is not None:
            arr = self._scaler.transform(pd.DataFrame(arr, columns=cols))
        return pd.DataFrame(arr, columns=cols, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._apply_base(X, self._cols_before_fs)
        return out[self.selected_features_]

    def fit_transform(self, X_labeled_train, y_labeled_train, X_unlabeled=None):
        return self.fit(X_labeled_train, y_labeled_train, X_unlabeled).transform(X_labeled_train)


# ---------------------------------------------------------------------------
# 4. 불균형 처리
# ---------------------------------------------------------------------------


def _apply_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: Optional[str],
    random_state: int,
    smote_k_neighbors: int,
    smote_sampling_strategy: Union[str, float, dict],
):
    class_weight = None
    sample_weight = None

    if method is None:
        return X_train, y_train, class_weight, sample_weight

    if method == "class_weight":
        classes = np.unique(y_train)
        w = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(v) for c, v in zip(classes, w)}
        sample_weight = y_train.map(class_weight).to_numpy(dtype=float)
        return X_train, y_train, class_weight, sample_weight

    if method == "smote":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SMOTE 를 쓰려면 imbalanced-learn 이 필요합니다: pip install imbalanced-learn"
            ) from e

        min_count = int(y_train.value_counts().min())
        if min_count < 2:
            raise ValueError("소수 클래스 샘플이 2개 미만이라 SMOTE 를 적용할 수 없습니다.")
        k = min(smote_k_neighbors, min_count - 1)
        if k != smote_k_neighbors:
            warnings.warn(f"소수 클래스 샘플이 적어 SMOTE k_neighbors 를 {k} 로 낮췄습니다.")

        sm = SMOTE(
            sampling_strategy=smote_sampling_strategy,
            k_neighbors=k,
            random_state=random_state,
        )
        Xr, yr = sm.fit_resample(X_train, y_train)
        Xr = pd.DataFrame(Xr, columns=X_train.columns).reset_index(drop=True)
        yr = pd.Series(yr, name=y_train.name).reset_index(drop=True)
        return Xr, yr, class_weight, sample_weight

    raise ValueError("imbalance 는 None, 'smote', 'class_weight' 중 하나여야 합니다.")


# ---------------------------------------------------------------------------
# 5. 메인 함수
# ---------------------------------------------------------------------------


def preprocess_data(
    X_labeled_train: pd.DataFrame,
    y_labeled_train: pd.Series,
    X_unlabeled: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    scaling: Optional[str] = "standard",
    feature_selection: Union[None, str, Sequence[str]] = None,
    imbalance: Optional[str] = None,
    drop_constant: bool = True,
    impute: Optional[str] = "median",
    outlier: Optional[str] = None,
    fit_on_unlabeled: bool = True,
    k_best: int = 15,
    kbest_score: str = "f_classif",
    corr_threshold: float = 0.95,
    model_threshold: Union[str, float] = "median",
    outlier_iqr_k: float = 1.5,
    outlier_quantile: tuple = (0.01, 0.99),
    smote_k_neighbors: int = 5,
    smote_sampling_strategy: Union[str, float, dict] = "auto",
    random_state: int = 42,
    verbose: bool = True,
) -> PreprocessedData:
    """
    준지도 학습용 전처리 메인 함수.

    옵션 요약
    ---------
    scaling           : None / 'standard' / 'minmax' / 'robust' / 'maxabs' / 'power' / 'quantile'
    feature_selection : None / 'corr' / 'kbest' / 'model' / 위 항목의 리스트
    imbalance         : None / 'smote' / 'class_weight'   (두 방법은 동시에 사용하지 않음)
    outlier           : None / 'iqr' / 'quantile'
    drop_constant     : labeled train 에서 상수인 피처 제거 여부
    impute            : None / 'mean' / 'median' / 'most_frequent'
    fit_on_unlabeled  : 비지도 단계를 unlabeled 까지 포함해 fit 할지 여부

    반환
    ----
    PreprocessedData
    """
    X_labeled_train = X_labeled_train.copy()
    X_unlabeled = X_unlabeled.copy()
    X_test = X_test.copy()

    # 컬럼 일관성 검사
    for name, df in (("unlabeled", X_unlabeled), ("test", X_test)):
        missing = set(X_labeled_train.columns) - set(df.columns)
        if missing:
            raise ValueError(f"{name} 데이터에 없는 컬럼: {sorted(missing)}")

    pre = SSLPreprocessor(
        scaling=scaling,
        feature_selection=feature_selection,
        k_best=k_best,
        kbest_score=kbest_score,
        corr_threshold=corr_threshold,
        model_threshold=model_threshold,
        drop_constant=drop_constant,
        impute=impute,
        outlier=outlier,
        outlier_iqr_k=outlier_iqr_k,
        outlier_quantile=outlier_quantile,
        fit_on_unlabeled=fit_on_unlabeled,
        random_state=random_state,
    )
    pre.fit(X_labeled_train, y_labeled_train, X_unlabeled)

    Xtr = pre.transform(X_labeled_train)
    Xul = pre.transform(X_unlabeled)
    Xte = pre.transform(X_test)
    Xtr_orig, ytr_orig = Xtr, y_labeled_train  # 불균형 처리 이전 상태 보관

    # 불균형 처리는 labeled train 에만 적용 (test/unlabeled 는 건드리지 않음)
    Xtr, ytr, cw, sw = _apply_imbalance(
        Xtr,
        y_labeled_train,
        imbalance,
        random_state,
        smote_k_neighbors,
        smote_sampling_strategy,
    )

    info = dict(pre.info_)
    info["class_distribution_before"] = y_labeled_train.value_counts().to_dict()
    info["class_distribution_after"] = ytr.value_counts().to_dict()
    info["shapes"] = {
        "X_train": Xtr.shape,
        "X_unlabeled": Xul.shape,
        "X_test": Xte.shape,
    }

    if verbose:
        print(f"[전처리] 상수 제거: {info['dropped_constant'] or '없음'}")
        if "dropped_corr" in info:
            print(f"[전처리] 상관 필터 제거: {info['dropped_corr'] or '없음'}")
        print(f"[전처리] 최종 피처 수: {len(pre.selected_features_)}")
        print(f"[전처리] 클래스 분포 {info['class_distribution_before']} -> {info['class_distribution_after']}")
        if cw:
            print(f"[전처리] class_weight: {cw}")

    return PreprocessedData(
        X_train=Xtr,
        y_train=ytr,
        X_unlabeled=Xul,
        X_test=Xte,
        y_test=y_test,
        class_weight=cw,
        sample_weight=sw,
        selected_features=list(pre.selected_features_),
        preprocessor=pre,
        info=info,
        X_train_orig=Xtr_orig,
        y_train_orig=ytr_orig,
        imbalance=imbalance,
        smote_params={
            "k_neighbors": smote_k_neighbors,
            "sampling_strategy": smote_sampling_strategy,
        },
    )


def preprocess_product(
    product: str,
    data_dir: str = DEFAULT_DATA_DIR,
    test_size: float = 0.3,
    split_random_state: int = 42,
    **preprocess_kwargs,
) -> PreprocessedData:
    """
    제품('cn7' / 'rg3') 데이터를 불러와 바로 전처리한다.
    CN7 과 RG3 는 분포가 달라 제품별로 따로 호출해야 한다.

    preprocess_kwargs 에는 preprocess_data 의 옵션(scaling, feature_selection,
    imbalance, ...)을 그대로 넘기면 된다.
    """
    d = DataLoader(product, data_dir=data_dir, test_size=test_size, random_state=split_random_state)
    return preprocess_data(
        d.labeled_train_X,
        d.labeled_train_Y,
        d.unlabeled,
        d.labeled_test_X,
        d.labeled_test_Y,
        **preprocess_kwargs,
    )


if __name__ == "__main__":
    # 간단한 실행 예시 (데이터 경로를 환경에 맞게 수정)
    for prod in ("cn7", "rg3"):
        out = preprocess_product(
            prod,
            scaling="standard",
            feature_selection=["corr", "kbest"],
            k_best=15,
            imbalance="class_weight",
        )
        print(prod, out.info["shapes"])
