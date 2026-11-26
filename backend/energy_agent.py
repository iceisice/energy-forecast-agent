import os
import time
import json
import itertools
import joblib
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rag_engine import RAGEngine

load_dotenv()

# 학습 데이터(energy_data.csv) 1% / 99% 백분위수 기반 OOD 임계치 상수
TEMP_MIN = -7.4
TEMP_MAX = 32.5
HUMI_MIN = 27.4
HUMI_MAX = 96.0


class EnergyAgent:
    def __init__(self, model_path="energy_model.pkl"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(base_dir, model_path)
        self.model = joblib.load(full_model_path) if os.path.exists(full_model_path) else None
        self.rag_engine = RAGEngine()

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

        # 무료 티어 할당량 분산용 모델 풀 (요청마다 순환 + 에러 시 자동 폴백)
        self.model_pool = [
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ]
        self._model_cycle = itertools.cycle(self.model_pool)

    def predict_and_detect_ood(
        self,
        temperature: float,
        humidity: float,
        hour: int = 14,
        month: int = 8,
        is_weekend: int = 0,
        is_holiday: int = 0,
    ) -> dict:
        """기상 및 시간 피처를 받아 전력 수요를 예측하고 학습 데이터 분포 이탈(OOD) 여부를 판별합니다."""
        is_ood = False
        reasons = []

        if temperature < TEMP_MIN:
            is_ood = True
            reasons.append(f"기온({temperature}°C)이 학습 데이터 하한({TEMP_MIN}°C) 미만 — 극단적 한파 OOD")
        elif temperature > TEMP_MAX:
            is_ood = True
            reasons.append(f"기온({temperature}°C)이 학습 데이터 상한({TEMP_MAX}°C) 초과 — 극단적 폭염 OOD")

        if humidity < HUMI_MIN or humidity > HUMI_MAX:
            is_ood = True
            reasons.append(f"습도({humidity}%)가 정상 분포 범위({HUMI_MIN}%~{HUMI_MAX}%) 이탈")

        prediction_value = 0.0
        if self.model is not None:
            input_df = pd.DataFrame(
                [[temperature, humidity, hour, month, is_weekend, is_holiday]],
                columns=["temperature", "humidity", "hour", "month", "is_weekend", "is_holiday"],
            )
            prediction_value = round(float(self.model.predict(input_df)[0]), 2)

        return {
            "prediction_value": prediction_value,
            "is_ood": is_ood,
            "ood_reason": "; ".join(reasons) if is_ood else "학습 데이터 정상 분포 범위 내(In-Distribution)",
        }

    def search_manual(self, query: str) -> list[str]:
        """전력수급 경보 단계 매뉴얼 및 이상기후 대응 가이드라인에서 관련 근거 문서를 검색합니다."""
        return self.rag_engine.search(query, k=3)

    def analyze(self, user_query: str) -> dict:
        state = {
            "prediction": 0.0,
            "is_ood": False,
            "ood_reason": "분석 미수행",
            "retrieved_docs": [],
            "report": "",
        }

        def tool_predict_and_detect_ood(
            temperature: float,
            humidity: float,
            hour: int = 14,
            month: int = 8,
            is_weekend: int = 0,
            is_holiday: int = 0,
        ) -> str:
            """전력 수요를 예측하고 기온/습도 OOD 여부를 판별합니다. 자연어에 시간/월 정보가 없으면 피크 시간대(hour=14, 폭염 시 month=8, 한파 시 month=1)를 추론해 넣으세요."""
            res = self.predict_and_detect_ood(temperature, humidity, hour, month, is_weekend, is_holiday)
            state["prediction"] = res["prediction_value"]
            state["is_ood"] = res["is_ood"]
            state["ood_reason"] = res["ood_reason"]
            return json.dumps(res, ensure_ascii=False)

        def tool_search_manual(query: str) -> str:
            """전력거래소 경보 기준 및 폭염/한파 대응 매뉴얼을 검색합니다. is_ood가 True이거나 대응 가이드가 필요할 때 호출하세요."""
            docs = self.search_manual(query)
            state["retrieved_docs"] = docs
            return json.dumps({"retrieved_docs": docs}, ensure_ascii=False)

        system_instruction = (
            "당신은 전력거래소 및 에너지 공기업의 AI 전력수요 리스크 분석 에이전트입니다. "
            "반드시 먼저 tool_predict_and_detect_ood를 호출하여 수요 예측치와 OOD 여부를 확인하세요. "
            "그 후 is_ood가 True이거나 매뉴얼 근거가 필요하면 반드시 tool_search_manual을 호출해 대응 기준을 조회하세요. "
            "최종 답변은 [1. 수요 예측 결과], [2. OOD 분포 이탈 진단], [3. 매뉴얼 기반 대응 권고] 형식으로 명확하고 전문적인 한국어 리포트를 작성하세요."
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[tool_predict_and_detect_ood, tool_search_manual],
            temperature=0.2,
        )

        # 이번 요청에서 시작할 모델부터 풀 전체를 순회하며 자동 폴백
        start_model = next(self._model_cycle)
        start_idx = self.model_pool.index(start_model)
        ordered_models = self.model_pool[start_idx:] + self.model_pool[:start_idx]

        last_error = None
        for model_name in ordered_models:
            try:
                print(f"[EnergyAgent] Attempting model: {model_name}")
                chat = self.client.chats.create(model=model_name, config=config)
                response = chat.send_message(user_query)
                state["report"] = response.text or "리포트 생성 결과가 비어 있습니다."
                return state
            except Exception as e:
                last_error = e
                err_str = str(e)
                print(f"[EnergyAgent] Model {model_name} failed: {err_str[:120]}")
                if any(k in err_str for k in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "404", "NOT_FOUND"]):
                    time.sleep(0.5)
                    continue
                raise e

        raise last_error

