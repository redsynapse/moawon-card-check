"""Laya multilingual decision API for 모두의창업 2차 1R.

ENAI CPU pod에서 실행:
    LAYA_TOKEN=... uvicorn laya_server:app --host 0.0.0.0 --port 8000

Cloudflare Pages Function이 서버 간 호출한다. 브라우저에 직접 노출하지 않는다.
Laya는 비목/집행 가능성만 보조 판정하고, 금액·증빙·사전승인은 규칙엔진이 처리한다.
"""

import hmac
import os
import time
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from laya import Router

TOKEN = os.environ.get("LAYA_TOKEN", "")

RULES = (
    "5개 비목: 재료구입비(재료·부품·소모품), 외주용역비(개발·디자인·마케팅·광고), "
    "문헌구입비(도서·논문·보고서), 지급수수료(구독·AI·소프트웨어·인증·임차·특허출원), "
    "여비·회의비(국내 출장·교통·숙박·회의). "
    "불가: 노트북·모니터·태블릿 등 개인 전자기기, 상품권·귀금속, 인건비, 인테리어·간판, "
    "주차·유류, 국외 여비, 취소 수수료, 특허 성공보수, 부채 상환, 유흥·주류·담배·골프. "
    "50만원 초과는 사전승인. 자산성 물품은 50만원 이하만 구입 가능."
)

QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "이 지출의 비목은 무엇인가? 불가 항목이면 '불가', 애매하면 '모르겠음'.",
        "criteria": {
            "재료구입비": "재료, 부품, 소모품, 데이터셋",
            "외주용역비": "외주, 개발, 디자인, 마케팅, 광고, 영상",
            "문헌구입비": "도서, 논문, 보고서, 유료 자료",
            "지급수수료": "구독, AI, 소프트웨어, 인증, 시험, 임차, 특허 출원",
            "여비회의비": "국내 출장, 교통, 숙박, 회의, 식사",
            "불가": "노트북, 주차, 상품권, 인건비, 해외, 취소 수수료",
            "모르겠음": "판단하기 어려움",
        },
    },
    "eligible": {
        "type": "noul",
        "instructions": "이 지출은 위 5개 비목 중 하나로 집행 가능한가? 불가 항목이면 아니오.",
    },
}

app = FastAPI(title="Laya decision API", version="1.0.0")

# 서버 시작 시 multilingual 체크포인트를 메모리에 올린다.
router = Router(max_loaded=1)
router.predict(
    "워밍업: 전문 도서 4만원",
    {"ok": {"type": "noul", "instructions": "집행 가능한가?"}},
    model="multilingual",
    max_len=512,
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    amount: Optional[int] = None


@app.get("/health")
def health():
    return {"ok": True, "service": "laya-decision", "model": "multilingual"}


@app.post("/predict")
def predict(req: PredictRequest, authorization: Optional[str] = Header(default=None)):
    if TOKEN and not hmac.compare_digest(authorization or "", f"Bearer {TOKEN}"):
        raise HTTPException(status_code=401, detail="unauthorized")

    started = time.perf_counter()
    state = f"{RULES}\n\n[지출]\n{req.text}"
    result = router.predict(state, QUESTIONS, model="multilingual", max_len=512)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    category = result.get("answers", {}).get("category", {})
    choice = category.get("choice")
    probabilities = category.get("probabilities", {})

    return {
        "model": result.get("routing", {}).get("model", "multilingual"),
        "category": choice,
        "category_probability": probabilities.get(choice) if choice else None,
        "category_probabilities": probabilities,
        "eligible": result.get("answers", {}).get("eligible", {}).get("noul"),
        "usage": result.get("usage"),
        "latency_ms": elapsed_ms,
    }
