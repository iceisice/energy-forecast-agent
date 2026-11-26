# ⚡ National Energy Demand Intelligence Platform
> **지역별 전력 소비 비중 가중치(GWAT) 기반의 고정밀 전력 수요 예측 및 OOD 탐지·RAG AI 리스크 진단 플랫폼**

한국전력거래소의 시간별 전국 전력수요 데이터(8,760시간)와 기상청 ASOS 전국 관측소 기상 데이터를 결합하여 시간별 전력 수요를 예측하고, 학습 분포를 벗어난 극단적 이상기후(OOD) 입력 시 전력수급 경보 매뉴얼을 기반으로 대응 리포트를 생성하는 풀스택 AI 시스템입니다.

---

## 🏗️ System Architecture

```text
[공공데이터 (전력거래소 수요 + 기상청 ASOS)]
       │
       ▼  (선형 보간 결측치 복원 · 17개 시·도 전력 소비 비중 가중 평균 · 공휴일 매핑)
[RandomForestRegressor (MAE 718.36 MW / 5-Fold CV 96.95%)]
       │
       ├──► FastAPI (/predict, /history) ──► React + TypeScript 관제 대시보드
       │
       └──► Gemini Function Calling Agent (/api/agent/analyze)
              ├─ 1단계: 백분위수(1%·99%) 기반 OOD 분포 이탈 판별
              ├─ 2단계: TF-IDF (char_wb) 경량 로컬 RAG 매뉴얼 검색
              └─ 3단계: 라운드로빈 멀티 모델 풀 & 자동 폴백 리포트 생성



Key Features
1. 도메인 가중치 기반 데이터 엔지니어링 (GWAT)지리적 가중 평균 기온·습도 산출: 전국 관측소 데이터의 단순 산술 평균 한계를 보완하기 위해, 17개 시·도별 실제 전력 소비 비중(경기 23.4%, 충남 10.2%, 서울 8.9% 등)을 관측소 ID에 매핑하여 가중 피처 생성시계열 무결성 및 특수 공휴일 처리: 선형 보간법(Linear Interpolation)으로 기상 결측치를 복원하고, 2025년 법정·대체공휴일 맵(Manual Holiday Map)을 통합하여 연휴 기간 수요 급감 오차 방어

2. OOD(분포 이탈) 탐지 & RAG 기반 AI 리스크 진단 에이전트정량적 OOD 가드레일: 트리 모델의 외삽(Extrapolation) 한계를 보완하기 위해 학습 데이터 1%·99% 백분위수 임계치(기온 -7.4°C ~ 32.5°C, 습도 27.4% ~ 96.0%)를 기준으로 극단적 폭염·한파 입력을 실시간 탐지2단계 Tool-Calling 및 경량 한국어 RAG: tool_predict_and_detect_ood 실행 후 분포 이탈 감지 시 tool_search_manual을 통해 전력거래소 비상 매뉴얼(char_wb N-gram TF-IDF + 코사인 유사도)을 검색하여 근거 기반 대응 리포트 생성고가용성 모델 폴백(Fallback): itertools.cycle 기반 멀티 모델 순환 풀(gemini-3.5-flash-lite, gemini-2.5-flash-lite, gemini-2.5-flash)을 적용해 API Rate Limit(429) 및 일시적 장애(503) 자동 대응

3. 실시간 수요 시뮬레이션 대시보드기온·습도·시간·월·주말·공휴일 조건 조정에 따른 실시간 전력 수요 예측 및 과거 일자별 24시간 수요 추이 시각화자연어 질의 기반 AI 리스크 진단 에이전트 인터페이스 제공📊 Model Performance Evolution버전주요 적용 피처 및 튜닝MAE (평균 절대 오차)정확도 (100 - MAPE)비고v1 (Baseline)단순 전국 평균 기온 + 시간1,992.19 MW96.82%초기 기준 모델v2 (GWAT)17개 시·도 전력 소비 비중 가중 기온 도입987.00 MW98.42%오차 50.5% 감축v3 (Multi-Feature)가중 습도 + 월(계절성) + min_samples_leaf=1855.91 MW98.62%미세 기상 변화 포착v4 (Production)2025년 특수 공휴일 매핑(is_holiday) 통합718.36 MW98.85%초기 대비 오차 64% 감축일반화 검증 (5-Fold Cross-Validation): 전체 데이터 8:2 분할 5회 교차 검증 결과 평균 정확도 96.95% (표준편차 0.0525) 달성

Tech Stack

Backend & ML: Python, FastAPI, Uvicorn, Scikit-learn, Pandas, NumPy, Joblib

AI Agent & RAG: Google GenAI SDK (Automatic Function Calling), Scikit-learn TF-IDF (char_wb)

Frontend: React 18, TypeScript, Vite, Recharts

📁 Project StructurePlaintext├── backend/
│   ├── docs/                   # 전력수급 경보 및 이상기후 대응 매뉴얼 (RAG 지식베이스)
│   ├── main.py                 # FastAPI 추론 및 AI 에이전트 API 서버
│   ├── energy_agent.py         # OOD 탐지 및 Function Calling 에이전트
│   ├── rag_engine.py           # 문자 N-gram TF-IDF 기반 로컬 문서 검색 엔진
│   ├── restore_data.py         # 데이터 전처리, 지역 가중치 계산 및 모델 학습
│   ├── cv_test.py              # 5-Fold 교차 검증 스크립트
│   └── evaluate_model.py       # 최종 모델 성능 지표 평가 스크립트
└── frontend/
    └── src/                    # React + TypeScript 시뮬레이터 및 리스크 분석 대시보드

🚀 Getting Started
1. Environment Variables
backend/.env 파일을 생성하고 Gemini API 키를 설정
GEMINI_API_KEY=your_gemini_api_key_here

2. Run Backend ServerBash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn pandas numpy scikit-learn joblib python-dotenv google-genai
python main.py

3. Run Frontend DashboardBashcd frontend
npm install
npm run dev
