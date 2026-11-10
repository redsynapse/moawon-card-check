"""Laya multilingual decision API for 모두의창업 2차 1R.

ENAI CPU pod에서 실행:
    LAYA_TOKEN=... uvicorn laya_server:app --host 0.0.0.0 --port 8000

Cloudflare Pages Function이 서버 간 호출한다. 브라우저에 직접 노출하지 않는다.
"""

import hmac
import os
import time
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from laya import Router

TOKEN = os.environ.get("LAYA_TOKEN", "")

RULES = """[모두의창업 2차 1R 창업활동자금 200만원 카드]
5개 비목:
- 재료구입비: 재료, 원료, 부품, 학습·검증용 데이터, 소모품
- 외주용역비: 앱·웹 개발, 디자인, 목업·금형, 마케팅·광고, 영상, 시장조사
- 문헌구입비: 도서, 논문, 보고서, 유료 자료
- 지급수수료: 시험·인증·검사, 기자재 임차·렌탈, 전시·박람회, 소프트웨어·생성형 AI 구독, 특허·상표 출원
- 여비·회의비: 국내 출장 교통·숙박, 멘토·외부인 회의비

불가:
- 개인용 전자기기: 노트북, 데스크톱, 모니터, 태블릿, 스마트워치, 이어폰, 휴대전화
- 환금성: 상품권, 기프트카드, 포인트 충전, 귀금속, 유가증권
- 인건비: 본인·팀원 급여, 활동비, 4대보험
- 공간·환경: 사무실 인테리어, 간판, 도색
- 사적 지출: 경조사, 가족·지인 식사, 개인 소비, 무관한 학원
- 주차료, 주유·유류, 국외 여비·항공권, 취소 수수료
- 특허등록 성공보수, 부채 상환, 다른 정부지원 중복 정산
- 유흥·주류·담배·카지노·골프·총포

금액·증빙:
- 1건 10만원 미만: 가맹점명으로 사용처가 확인되면 증빙 면제
- 1건 10만원 이상: 주문상세내역서·영수증 등 증빙
- 1건 50만원 초과: 멘토기관 사전승인
- 자산성 물품: 50만원 이하만 구입 가능(자산보유서약서, 5년 보관), 50만원 초과는 임차
- PG/온라인 결제: 금액과 무관하게 주문상세내역 증빙
- 숙박 한도: 특별시 10만원, 광역시 8만원, 그 밖의 지역 7만원/박
- 해외 결제 불가(국내 전용 신한 클린 선불카드)
"""

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
    "card_ok": {
        "type": "noul",
        "instructions": "이 지출은 신한 클린 선불카드로 결제 가능한가? 해외·제한 업종이면 아니오.",
    },
    "approval": {
        "type": "noul",
        "instructions": "1건 50만원 초과로 멘토기관 사전승인이 필요한가?",
    },
    "evidence": {
        "type": "noul",
        "instructions": "1건 10만원 이상 또는 PG 결제로 증빙이 필요한가?",
    },
}

app = FastAPI(title="Laya decision API", version="1.0.0")

# 서버 시작 시 multilingual 체크포인트를 메모리에 올린다.
router = Router(max_loaded=1)
router.predict(
    "워밍업: 전문 도서 4만원",
    {"ok": {"type": "noul", "instructions": "집행 가능한가?"}},
    model="multilingual",
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    amount: Optional[int] = None


def _noul(result, name: str) -> Optional[float]:
    try:
        return float(result["answers"][name]["noul"])
    except (KeyError, TypeError, ValueError):
        return None


@app.get("/health")
def health():
    return {"ok": True, "service": "laya-decision", "model": "multilingual"}


@app.post("/predict")
def predict(req: PredictRequest, authorization: Optional[str] = Header(default=None)):
    if TOKEN and not hmac.compare_digest(authorization or "", f"Bearer {TOKEN}"):
        raise HTTPException(status_code=401, detail="unauthorized")

    started = time.perf_counter()
    state = f"{RULES}\n\n[지출]\n{req.text}"
    result = router.predict(state, QUESTIONS, model="multilingual")
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    category = result.get("answers", {}).get("category", {})
    choice = category.get("choice")
    probabilities = category.get("probabilities", {})

    return {
        "model": result.get("routing", {}).get("model", "multilingual"),
        "category": choice,
        "category_probability": probabilities.get(choice) if choice else None,
        "category_probabilities": probabilities,
        "eligible": _noul(result, "eligible"),
        "card_ok": _noul(result, "card_ok"),
        "approval": _noul(result, "approval"),
        "evidence": _noul(result, "evidence"),
        "usage": result.get("usage"),
        "latency_ms": elapsed_ms,
    }
