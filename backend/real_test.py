import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def real_validation():
    print("--- [진짜 실력 테스트] 8:2 데이터 분리 검증 시작 ---")
    
    # 데이터 로드
    try:
        df = pd.read_csv("energy_data.csv")
    except:
        print("데이터 파일이 없습니다.")
        return

    # 피처 설정
    features = ['temperature', 'humidity', 'hour', 'month', 'is_weekend', 'is_holiday']
    X = df[features]
    y = df['usage']

    # 1. 데이터를 8:2로 분리 (random_state를 고정하여 결과 재현성 확보)
    # test_size=0.2가 바로 말씀하신 '10개 중 2개를 남겨두는' 설정입니다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"- 학습용 데이터: {len(X_train)}개")
    print(f"- 테스트용 데이터 (처음 보는 문제): {len(X_test)}개")

    # 2. 학습용 데이터(80%)로만 학습
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 3. 한 번도 본 적 없는 테스트 데이터(20%)로 예측
    y_pred = model.predict(X_test)

    # 4. 성능 측정
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    accuracy = 100 - mape

    print(f"\n[진짜 성적표 결과]")
    print(f"- MAE (평균 절대 오차): {mae:.2f} MW")
    print(f"- 진짜 정확도: {accuracy:.2f} %")
    
    # 5. 과적합(Overfitting) 여부 판단
    # 학습 데이터에 대한 성적도 같이 뽑아 비교해봅니다.
    train_pred = model.predict(X_train)
    train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100
    print(f"- (참고) 학습 데이터 정확도: {100 - train_mape:.2f} %")
    
    gap = (100 - train_mape) - accuracy
    print(f"\n=> 성적 차이(Gap): {gap:.2f}%p")
    if gap > 5:
        print("결과: 모델이 데이터를 다소 암기(과적합)하고 있습니다.")
    else:
        print("결과: 모델이 처음 보는 데이터에도 잘 대응하고 있습니다(일반화 성공).")

if __name__ == "__main__":
    real_validation()
