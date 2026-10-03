# 사출성형기 불량 예측 (준지도 학습)

사출성형기 AI 데이터셋(CN7, RG3)에서 labeled/unlabeled 데이터를 함께 써서
불량(PassOrFail)을 예측하는 준지도 학습 실험 프로젝트.

## 프로젝트 구조
```
.
├── data/ # 데이터
├── notebooks/ # 전처리·EDA 노트북
├── src/ # 전처리, 모델, 학습, 평가 모듈
├── logs/ # 실험 로그 (experiment_log.csv)
└── README.md
```

## 환경 설정
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## 데이터
`data/<폴더명>/` 아래에 `{process_type}_moldset_{labeled|unlabeled}_{product}.csv` 배치.
- process_type : processed 또는 processed2
- processed - 데이터 중복이 해결된 데이터, 제품별로 labeled/unlabeled 존재, 총 4개의 파일
- processed2 - processed 데이터에서 `Clam_Open_Position` 대치가 완료된 데이터, labeled 파일만 존재

## 사용법
```
cd src
nohup python -u run_rg3.py > ../logs/rg3_run.log 2>&1 &
```

## 실험 설정
모델(cn7, rg3), 사전 전처리 타입(processed, processed2), 전처리(scaling, feature selection, imbalance), 모델, 준지도(self-training) 옵션 설정 가능
- 사용 가능한 모델 : SVC, RF, GNB, DNN, Logistic Regression, KNN, HistGradientBoosting