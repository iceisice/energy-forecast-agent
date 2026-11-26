import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def test_initial_performance():
    print("--- 초기 모델(기온 + 시간) 성능 시뮬레이션 ---")
    
    # 데이터 로드
    try:
        df = pd.read_csv("energy_data.csv")
    except:
        print("데이터 파일이 없습니다.")
        return

    # 1. 초기 모델 피처: 기온, 시간만 사용
    features_v0 = ['temperature', 'hour']
    X = df[features_v0]
    y = df['usage']

    # 학습 및 예측 (동일 조건)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    y_pred = model.predict(X)

    # 지표 계산
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred) * 100
    accuracy = 100 - mape

    print(f"- 초기 모델 MAE: {mae:.2f} MW")
    print(f"- 초기 모델 정확도: {accuracy:.2f} %")
    print(f"- 현재 모델(718.36 MW) 대비 오차: {mae - 718.36:.2f} MW 더 높음")

if __name__ == "__main__":
    test_initial_performance()
