import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def process_and_train():
    # 1. 데이터 로드 (인코딩 주의)
    try:
        p_df = pd.read_csv('한국전력거래소_시간별 전국 전력수요량_20251231.csv', encoding='cp949')
        w_df = pd.read_csv('OBS_ASOS_TIM_20260426195628.csv', encoding='cp949')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. 전력 데이터 재구성 (Wide -> Long)
    p_df.columns = ['date'] + [f'{i}' for i in range(1, 25)]
    p_long = p_df.melt(id_vars=['date'], var_name='hour', value_name='usage')
    p_long['hour'] = p_long['hour'].astype(int) - 1
    
    # 3. 기상 데이터 재구성 (기온 + 습도 추가)
    # 인덱스: 2(일시), 3(기온), 10(습도)
    w_df = w_df.iloc[:, [2, 3, 10]]
    w_df.columns = ['timestamp', 'temperature', 'humidity']
    
    # 날짜 변환 및 결측치 제거
    w_df['timestamp'] = pd.to_datetime(w_df['timestamp'], errors='coerce')
    w_df = w_df.dropna(subset=['timestamp', 'temperature', 'humidity'])
    
    # 시간대별 전국 평균 기온 및 습도 계산
    w_avg = w_df.groupby('timestamp')[['temperature', 'humidity']].mean().reset_index()
    w_avg['date'] = w_avg['timestamp'].dt.strftime('%Y-%m-%d')
    w_avg['hour'] = w_avg['timestamp'].dt.hour
    w_avg['month'] = w_avg['timestamp'].dt.month # 월 정보 추가

    # 4. 데이터 병합
    merged = pd.merge(p_long, w_avg, on=['date', 'hour'])
    
    # 5. 특성 생성 (주말 여부)
    merged['is_weekend'] = pd.to_datetime(merged['timestamp']).dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    
    # 시간 순 정렬
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    merged = merged.dropna()
    
    print(f"Merged data size: {len(merged)} rows")
    
    # 6. AI 모델 학습 (변수 추가: month, humidity)
    # 이제 독립변수는 [temperature, humidity, hour, month, is_weekend] 총 5개입니다.
    features = ['temperature', 'humidity', 'hour', 'month', 'is_weekend']
    X = merged[features]
    y = merged['usage']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    mae = mean_absolute_error(y, model.predict(X))
    print(f"Upgraded Model Trained (with Month & Humidity). MAE: {mae:.2f}")
    
    # 7. 저장
    joblib.dump(model, "energy_model.pkl")
    # 대시보드 시각화용 데이터에도 습도와 월 정보 포함하여 저장
    merged[['timestamp', 'temperature', 'humidity', 'hour', 'month', 'is_weekend', 'usage']].to_csv("energy_data.csv", index=False)
    print("Updated energy_model.pkl and energy_data.csv with new features.")

if __name__ == "__main__":
    process_and_train()
