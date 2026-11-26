import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

def evaluate():
    print("--- AI 모델 성능 평가 시작 ---")
    
    # 1. 모델 및 데이터 로드
    try:
        model = joblib.load("energy_model.pkl")
        df = pd.read_csv("energy_data.csv")
    except FileNotFoundError:
        print("에러: 모델(.pkl)이나 데이터(.csv) 파일이 없습니다. 먼저 학습을 진행하세요.")
        return

    # 2. 특성 변수 설정 (is_holiday 추가)
    features = ['temperature', 'humidity', 'hour', 'month', 'is_weekend', 'is_holiday']
    X = df[features]
    y_true = df['usage']

    # 3. 예측 실행
    y_pred = model.predict(X)

    # 4. 지표 계산
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    max_err = np.max(np.abs(y_true - y_pred))

    # 5. 결과 출력
    print(f"\n[성능 지표 요약]")
    print(f"- MAE (평균 절대 오차): {mae:.2f} MW")
    print(f"- MAPE (평균 오차율): {mape:.2f} %")
    print(f"- R² Score (결정 계수): {r2:.4f}")
    print(f"- Max Error (최대 오차): {max_err:.2f} MW")
    
    accuracy = 100 - mape
    print(f"\n=> 최종 모델 정확도: {accuracy:.2f} %")

    # 6. 심층 분석: 오차가 가장 큰 상위 5개 시점
    df['error'] = np.abs(y_true - y_pred)
    df['error_rate'] = (df['error'] / y_true) * 100
    worst_cases = df.sort_values(by='error', ascending=False).head(5)
    
    print("\n[취약 시점 분석 (Top 5 Error)]")
    for i, row in worst_cases.iterrows():
        print(f"- {row['timestamp']}: 실제 {row['usage']}MW vs 예측 {model.predict([row[features]])[0]:.2f}MW (오차 {row['error']:.2f}MW)")

if __name__ == "__main__":
    evaluate()
