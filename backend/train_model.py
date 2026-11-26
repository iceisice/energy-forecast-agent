import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def train():
    # 데이터 로드
    df = pd.read_csv("energy_data.csv")
    
    # 특성과 타겟 분리
    # timestamp는 직접 사용하지 않고 hour, is_weekend 등을 사용
    X = df[['temperature', 'hour', 'is_weekend']]
    y = df['usage']
    
    # 학습/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 정의 및 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 평가
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model trained. Mean Absolute Error: {mae:.2f} kWh")
    
    # 모델 저장
    joblib.dump(model, "energy_model.pkl")
    print("Model saved to energy_model.pkl")

if __name__ == "__main__":
    train()
