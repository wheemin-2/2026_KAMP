## 사용 데이터셋 설명
서로 다른 두 제품(제품명 : CN7, RG3)의 공정 기록을 포함한 데이터
불량품 여부를 포함한 *_labeled_* 데이터와 불량품 여부 정보가 없는 *_unlabeled_*로 나뉘어있음

### Given data
- moldset_(un)labeled_cn7.csv
- moldset_(un)labeled_rg3.csv

### Processed data (1)
- 데이터 중복 문제 해결
- processed_moldset_(un)labeled_cn7.csv
- processed_moldset_(un)labeled_rg3.csv

#### 데이터 중복 문제
- 오른쪽, 왼쪽 몰드가 동시에 생산되는 것으로 추정됨
- 레이블을 포함한 모든 값이 동일한 경우 : 단순 병합
- 피처 값은 모두 동일하지만, 레이블 값이 다른 경우 : 불량으로 간주하고 불량인 경우만 채택

### Processed data (2)
- 현재 진행 중
- labeled data에서 두 제품의 Clamp_Open_Position이 모두 0인 문제 확인
- unlabeled data에는 정상적으로 기록되어있으므로, 이를 바탕으로 labeled의 COP 변수값을 예측 후 대치함
- CN7과 RG3의 데이터 분포가 다르기 때문에 서로 다른 예측 모형을 적용, 각각 Decision Tree (dt)와 Extra Tree (et) - `pycaret` 활용
