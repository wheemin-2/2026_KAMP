"""
cn7 / processed2 실험 실행 스크립트 (nohup 제출용)
 
사용 (src/ 폴더에서 실행):
    mkdir -p ../logs
    nohup python -u run_cn7.py > ../logs/cn7_run.log 2>&1 &
 
경로 주의: data_dir 기본값('../data/...')과 log_path('../logs/...')는
실행하는 위치(현재 작업 디렉터리) 기준 상대 경로다. src/ 에서 실행하면 프로젝트 루트의
data/, logs/ 를 가리킨다. 위치가 다르면 절대 경로로 지정할 것.
"""
 
import time
 
from ssl_train import run_experiments
 
if __name__ == "__main__":
    t0 = time.time()
    print("START", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
 
    df = run_experiments(
        "cn7",
        "processed2",
        prep_grid={
            "scaling": [None],
            "feature_selection": [None, ["corr", "kbest"]],
            "imbalance": ["class_weight", "smote"],
        },
        models=["svc", "rf", "gnb"],
        ssl_methods=["self_training"],
        n_iter={"default": 15, "dnn": 6},
        log_path="../logs/experiment_log_cn7.csv",
        # n_jobs=4,  # svc/rf/gnb 의 CV 를 병렬화하고 싶으면 주석 해제 (dnn 은 자동으로 1)
    )
 
    print(f"DONE {time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
 