import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_energy_data(days=365):
    start_date = datetime(2025, 1, 1)
    data = []
    
    for day in range(days):
        current_day = start_date + timedelta(days=day)
        is_weekend = 1 if current_day.weekday() >= 5 else 0
        
        # 계절별 기온 시뮬레이션 (대략적인 사인파)
        temp_base = 15 + 15 * np.sin(2 * np.pi * day / 365 - np.pi/2)
        
        for hour in range(24):
            # 시간대별 기본 사용량 (낮에 높고 밤에 낮음)
            hour_factor = 1 + 0.5 * np.sin(2 * np.pi * hour / 24 - np.pi/2)
            
            # 기온에 따른 가중치 (냉난방)
            temp_variation = np.random.normal(temp_base, 2)
            temp_factor = 1 + 0.05 * abs(temp_variation - 20)
            
            # 주말은 사용량 감소
            weekend_factor = 0.7 if is_weekend else 1.0
            
            # 랜덤 노이즈
            noise = np.random.normal(0, 5)
            
            # 최종 전력 사용량 (kWh)
            usage = (50 * hour_factor * temp_factor * weekend_factor) + noise
            
            data.append({
                'timestamp': current_day.replace(hour=hour),
                'temperature': round(temp_variation, 1),
                'hour': hour,
                'is_weekend': is_weekend,
                'usage': round(max(10, usage), 2)
            })
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_energy_data()
    df.to_csv("energy_data.csv", index=False)
    print(f"Generated {len(df)} rows of energy data to energy_data.csv")
