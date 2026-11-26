import joblib
import pandas as pd

def test_diff():
    model = joblib.load("energy_model.pkl")
    
    # 동일 조건: 기온 30도, 습도 60%, 14시, 평일(0)
    conditions = []
    for m in [2, 3, 4, 8]: # 2월, 3월, 4월, 그리고 여름인 8월 비교
        conditions.append({
            'temperature': 30.0,
            'humidity': 60.0,
            'hour': 14,
            'month': m,
            'is_weekend': 0
        })
    
    df_test = pd.DataFrame(conditions)
    predictions = model.predict(df_test)
    
    print("--- [실험 결과] 동일 기온(30도) 시 월별 예측 차이 ---")
    for i, m in enumerate([2, 3, 4, 8]):
        print(f"[{m}월 30도]: 예상 전력 수요 {predictions[i]:.2f} MW")

if __name__ == "__main__":
    test_diff()
