from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
from energy_agent import EnergyAgent

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "energy_model.pkl")
data_path = os.path.join(base_dir, "energy_data.csv")

try:
    model = joblib.load(model_path)
except Exception:
    model = None

try:
    df_history = pd.read_csv(data_path)
except Exception:
    df_history = pd.DataFrame()

agent = None
if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    try:
        agent = EnergyAgent()
    except Exception as e:
        print(f"Failed to initialize Agent: {e}")


class PredictRequest(BaseModel):
    temperature: float
    humidity: float
    hour: int
    month: int
    is_weekend: int
    is_holiday: int


class AgentRequest(BaseModel):
    query: str


@app.get("/history")
def get_history(target_date: str = "2025-08-05"):
    if df_history.empty:
        return []
    ts_col = "timestamp" if "timestamp" in df_history.columns else df_history.columns[0]
    filtered = df_history[df_history[ts_col].astype(str).str.startswith(target_date)].copy()
    if filtered.empty:
        return []

    records = []
    for _, row in filtered.iterrows():
        records.append({
            "timestamp": str(row[ts_col]),
            "usage": float(row.get("usage", row.get("demand", 0))),
            "temperature": float(row.get("temperature", 0)),
            "humidity": float(row.get("humidity", 0)),
            "hour": int(row.get("hour", 0)),
            "month": int(row.get("month", 1)),
        })
    return records


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        return {"prediction": 0}
    try:
        input_data = pd.DataFrame(
            [[request.temperature, request.humidity, request.hour, request.month, request.is_weekend, request.is_holiday]],
            columns=["temperature", "humidity", "hour", "month", "is_weekend", "is_holiday"],
        )
        prediction = model.predict(input_data)[0]
        return {"prediction": round(float(prediction), 2)}
    except Exception:
        return {"prediction": 0}


@app.post("/api/agent/analyze")
def analyze_agent(request: AgentRequest):
    global agent
    if agent is None:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            agent = EnergyAgent()
        else:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
    try:
        return agent.analyze(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
