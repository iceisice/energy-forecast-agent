import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# 한전 가중치 그대로 사용
REGION_WEIGHTS = {
    '경기': 23.4, '충남': 10.2, '서울': 8.9, '경북': 8.2, '경남': 6.5,
    '전남': 6.4, '울산': 5.9, '충북': 5.5, '인천': 4.9, '전북': 4.0,
    '부산': 3.8, '강원': 3.1, '대구:': 2.7, '광주': 1.6, '대전': 1.6,
    '세종': 0.8, '제주': 1.1
}

# 2025년 한국 공휴일 (직접 지정하여 라이브러리 버그 우회)
HOLIDAYS_2025 = [
    '2025-01-01', # 신정
    '2025-01-28', '2025-01-29', '2025-01-30', # 설날 연휴
    '2025-03-01', '2025-03-03', # 삼일절 및 대체공휴일
    '2025-05-05', # 어린이날
    '2025-05-06', # 부처님오신날
    '2025-06-06', # 현충일
    '2025-08-15', # 광복절
    '2025-10-03', # 개천절
    '2025-10-05', '2025-10-06', '2025-10-07', # 추석 연휴
    '2025-10-08', # 대체공휴일
    '2025-10-09', # 한글날
    '2025-12-25'  # 성탄절
]

def get_weight_by_stn(stn_id):
    if stn_id in [108, 111, 116]: return REGION_WEIGHTS['서울']
    if stn_id in [112, 201]: return REGION_WEIGHTS['인천']
    if stn_id in [119, 202, 203, 98, 99]: return REGION_WEIGHTS['경기']
    if stn_id in [133, 177]: return REGION_WEIGHTS['대전']
    if stn_id in [129, 141, 232]: return REGION_WEIGHTS['충남']
    if stn_id in [131, 221, 226]: return REGION_WEIGHTS['충북']
    if stn_id in [159]: return REGION_WEIGHTS['부산']
    if stn_id in [143]: return REGION_WEIGHTS['대구:']
    if stn_id in [152]: return REGION_WEIGHTS['울산']
    if stn_id in [140, 146, 271, 272, 273, 276, 277, 278, 279, 281, 283]: return REGION_WEIGHTS['경북']
    if stn_id in [155, 162, 192, 284, 285, 288, 289, 294, 295]: return REGION_WEIGHTS['경남']
    if stn_id in [156]: return REGION_WEIGHTS['광주']
    if stn_id in [165, 168, 169, 170, 172, 174, 251, 252, 253, 255, 257, 258, 259, 260, 261, 262, 263, 264, 266, 268]: return REGION_WEIGHTS['전남']
    if stn_id in [146, 243, 244, 245, 247, 248, 254]: return REGION_WEIGHTS['전북']
    if stn_id in [90, 93, 95, 100, 101, 104, 105, 106, 114, 121, 150, 211, 212, 216, 217]: return REGION_WEIGHTS['강원']
    if stn_id in [184, 185, 188, 189]: return REGION_WEIGHTS['제주']
    return 1.0

def restore_sensitive():
    print("Starting Training with Manual Holiday Map (Most Stable)...")
    
    p_df = pd.read_csv('한국전력거래소_시간별 전국 전력수요량_20251231.csv', encoding='cp949')
    w_df = pd.read_csv('OBS_ASOS_TIM_20260426195628.csv', encoding='cp949', low_memory=False)

    p_df.columns = ['date'] + [f'{i}' for i in range(1, 25)]
    p_long = p_df.melt(id_vars=['date'], var_name='hour', value_name='usage')
    p_long['date'] = pd.to_datetime(p_long['date']).dt.strftime('%Y-%m-%d')
    p_long['hour'] = p_long['hour'].astype(int) - 1
    p_long['usage'] = p_long['usage'].interpolate().ffill().bfill()

    w_data = w_df.iloc[:, [0, 2, 3, 11]].copy()
    w_data.columns = ['stn_id', 'timestamp', 'temperature', 'humidity']
    w_data['timestamp'] = pd.to_datetime(w_data['timestamp'], errors='coerce')
    w_data = w_data.dropna(subset=['timestamp'])
    
    w_data['weight'] = w_data['stn_id'].apply(get_weight_by_stn)
    w_data['w_temp'] = w_data['temperature'] * w_data['weight']
    w_data['w_hum'] = w_data['humidity'] * w_data['weight']
    
    w_grouped = w_data.groupby('timestamp').agg({'w_temp': 'sum', 'w_hum': 'sum', 'weight': 'sum'}).reset_index()
    w_grouped['temperature'] = w_grouped['w_temp'] / w_grouped['weight']
    w_grouped['humidity'] = w_grouped['w_hum'] / w_grouped['weight']
    w_grouped['date'] = w_grouped['timestamp'].dt.strftime('%Y-%m-%d')
    w_grouped['hour'] = w_grouped['timestamp'].dt.hour
    w_grouped['month'] = w_grouped['timestamp'].dt.month

    merged = pd.merge(p_long, w_grouped[['date', 'hour', 'temperature', 'humidity', 'month']], on=['date', 'hour'], how='left')
    merged = merged.sort_values(['date', 'hour']).reset_index(drop=True)
    merged['temperature'] = merged['temperature'].interpolate().ffill().bfill()
    merged['humidity'] = merged['humidity'].interpolate().ffill().bfill()
    
    # 주말 및 공휴일 피처 추가
    merged['is_weekend'] = pd.to_datetime(merged['date']).dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    merged['is_holiday'] = merged['date'].apply(lambda x: 1 if x in HOLIDAYS_2025 else 0)
    
    merged['timestamp'] = merged.apply(lambda row: f"{row['date']} {int(row['hour']):02d}:00:00", axis=1)

    features = ['temperature', 'humidity', 'hour', 'month', 'is_weekend', 'is_holiday']
    X = merged[features]
    y = merged['usage']
    
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=20, 
        min_samples_leaf=1, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X, y)
    
    joblib.dump(model, "energy_model.pkl")
    merged.to_csv("energy_data.csv", index=False)
    print("AI Model with Manual Holiday Map Ready!")

if __name__ == "__main__":
    restore_sensitive()
