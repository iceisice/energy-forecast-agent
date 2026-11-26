import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def run_cross_validation():
    print("--- [엄격한 검증] 5-Fold 교차 검증 시작 (K=5) ---")
    
    try:
        df = pd.read_csv("energy_data.csv")
    except:
        print("데이터 파일이 없습니다.")
        return

    features = ['temperature', 'humidity', 'hour', 'month', 'is_weekend', 'is_holiday']
    X = df[features]
    y = df['usage']

    # 5-Fold 설정
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    mae_list = []
    accuracy_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        # 데이터 분리
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 모델 학습
        model = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # 예측 및 평가
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        accuracy = 100 - mape

        mae_list.append(mae)
        accuracy_list.append(accuracy)
        print(f"Fold {fold}: MAE = {mae:.2f} MW, Accuracy = {accuracy:.2f} %")

    print("\n[최종 검증 결과 보고]")
    print(f"- 5회 반복 평균 MAE: {np.mean(mae_list):.2f} MW")
    print(f"- 5회 반복 평균 정확도: {np.mean(accuracy_list):.2f} %")
    print(f"- 성적 표준편차: {np.std(accuracy_list):.4f} (낮을수록 모델이 안정적)")

if __name__ == "__main__":
    run_cross_validation()
