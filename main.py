"""모두의창업 2차 1R — 200만원 카드 사용 판정 API.

FastAPI + 규칙엔진. 외부 LLM/API 없이 동작하며, 애매한 경우에만
사용자가 비목을 직접 고르도록 안내한다.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="모두의창업 2차 1R 카드 판정", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ExpenseInput(BaseModel):
    description: str = ""
    amount: int = Field(default=0, ge=0)
    category: str = "모르겠음"
    payment: str = "카드 직접 결제"
    relevance: str = ""
    asset: str = "아님"
    evidence: str = "가능"
    region: str = "해당 없음"
    nights: int = Field(default=0, ge=0)


class Decision(BaseModel):
    decision: str  # yes | no | conditional
    title: str
    summary: str
    reasons: List[str]
    evidence: List[str]
    cautions: List[str]
    approval_required: bool


CATEGORIES = {
    "재료구입비",
    "외주용역비(마케팅·광고 포함)",
    "문헌구입비",
    "지급수수료",
    "여비·회의비",
    "모르겠음",
}

CATEGORY_KEYWORDS = [
    (
        "재료구입비",
        [
            "재료", "원료", "부품", "원단", "필라멘트", "전자부품", "데이터셋",
            "소모품", "복사용지", "프린터", "잉크", "3d 프린팅", "3d프린팅",
            "목업", "시제품", "키트", "공구", "케이블", "배터리",
        ],
    ),
    (
        "외주용역비(마케팅·광고 포함)",
        [
            "외주", "개발", "디자인", "로고", "명함", "인쇄", "광고", "마케팅",
            "홍보", "영상", "목업", "금형", "시장조사", "홈페이지", "앱 개발",
            "상세페이지", "브랜딩", "ui", "ux", "번역", "촬영",
        ],
    ),
    (
        "문헌구입비",
        ["도서", "책", "논문", "리포트", "보고서", "유료 자료", "전문 서적", "구독형 도서"],
    ),
    (
        "지급수수료",
        [
            "인증", "시험", "검사", "임차", "렌탈", "대관", "참가비", "수수료",
            "구독", "ai", "소프트웨어", "특허", "상표", "출원", "변리사",
            "공유오피스", "크라우드펀딩", "전시", "박람회", "api", "토큰",
        ],
    ),
    (
        "여비·회의비",
        [
            "출장", "교통", "택시", "버스", "기차", "ktx", "숙박", "회의",
            "미팅", "식사", "식음료", "숙소", "호텔", "모텔",
        ],
    ),
]

PERSONAL_ELECTRONICS = [
    "노트북", "데스크톱", "모니터", "태블릿", "스마트워치", "이어폰",
    "휴대전화", "휴대폰", "아이패드", "맥북", "갤럭시탭", "스마트폰",
]
CASH_EQUIVALENT = [
    "귀금속", "금괴", "은괴", "상품권", "기프트카드", "포인트 충전",
    "유가증권", "전당포",
]
LABOR = ["인건비", "급여", "알바비", "아르바이트", "4대보험", "활동비", "팀원 급여"]
SPACE = ["인테리어", "간판", "도색", "사무실 공사", "환경개선", "사무실 인테리어"]
PRIVATE = ["경조사", "가족 식사", "지인 식사", "개인 소비", "학원 수강", "개인 학원"]
PARKING_FUEL = ["주차", "주유", "유류", "기름값", "주차료"]
DEBT_DUPLICATE = ["부채 상환", "대출 상환", "중복 정산", "다른 정부지원", "타 정부지원"]
PATENT_SUCCESS_FEE = ["특허 성공보수", "변리사 성공보수", "등록 성공보수"]
OVERSEAS = ["해외", "국외", "항공권", "해외 숙박", "해외 출장"]
CANCELLATION = ["취소 수수료", "예매 수수료", "변경 수수료"]
RESTRICTED = ["주류", "담배", "노래방", "유흥", "카지노", "복권", "골프", "총포", "사행"]

LODGING_LIMITS = {"특별시": 100_000, "광역시": 80_000, "그 밖의 지역": 70_000}


def _contains(text: str, words: List[str]) -> Optional[str]:
    lowered = text.lower()
    for word in words:
        if word.lower() in lowered:
            return word
    return None


def _infer_category(text: str) -> str:
    for category, words in CATEGORY_KEYWORDS:
        if _contains(text, words):
            return category
    return "모르겠음"


def _normalize_category(category: str, description: str, relevance: str) -> str:
    if category in CATEGORIES and category != "모르겠음":
        return category
    inferred = _infer_category(f"{description} {relevance}")
    return inferred if inferred in {c for c, _ in CATEGORY_KEYWORDS} else "모르겠음"


def _has_any(text: str, words: List[str]) -> bool:
    return _contains(text, words) is not None


def _add_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def evaluate(data: ExpenseInput) -> Decision:
    text = f"{data.description} {data.relevance}".strip()
    category = _normalize_category(data.category, data.description, data.relevance)
    reasons: List[str] = []
    evidence: List[str] = []
    cautions: List[str] = []
    approval_required = data.amount > 500_000

    if not data.description.strip():
        return Decision(
            decision="conditional",
            title="조건부 가능",
            summary="지출 내용을 먼저 입력해 주세요.",
            reasons=["지출 내용이 없으면 비목과 적정성을 판단할 수 없습니다."],
            evidence=[],
            cautions=["무엇을, 얼마에, 왜 결제하는지 적어주세요."],
            approval_required=False,
        )

    if data.amount <= 0:
        return Decision(
            decision="conditional",
            title="조건부 가능",
            summary="금액을 입력해 주세요.",
            reasons=["금액에 따라 증빙 기준(10만원)과 사전승인 기준(50만원)이 달라집니다."],
            evidence=[],
            cautions=["카드 승인 예정 금액을 입력하세요."],
            approval_required=False,
        )

    # 1) 결제 수단 자체가 불가한 경우
    if data.payment == "해외 결제":
        return Decision(
            decision="no",
            title="결제 불가",
            summary="이 카드는 국내 전용 신한 클린 선불카드입니다. 해외 가맹점·외화 결제는 승인되지 않습니다.",
            reasons=["해외 결제 및 외화 결제는 카드 이용약관상 불가합니다."],
            evidence=[],
            cautions=[
                "해외 사업자라도 국내 원화 결제 경로(국내 인앱결제, 국내 PG, 국내 총판·리셀러)를 이용하면 가능할 수 있습니다.",
                "최종 결제 가능 여부는 가맹점 업종 등록과 카드사 승인 결과에 따라 달라집니다.",
            ],
            approval_required=False,
        )

    if data.payment == "계좌이체/현금":
        return Decision(
            decision="no",
            title="결제 불가",
            summary="창업활동자금은 발급받은 전용 선불카드로만 결제해야 합니다. 계좌이체·현금 결제는 별도 지급되지 않습니다.",
            reasons=["전용카드 외 결제 수단(개인카드·계좌이체·현금)은 인정되지 않습니다."],
            evidence=[],
            cautions=["부득이한 경우에도 사전에 멘토기관에 문의하세요."],
            approval_required=False,
        )

    # 2) 환금성·개인용 전자기기·인건비 등 명백한 배제 항목
    matched = _contains(text, PERSONAL_ELECTRONICS)
    if matched:
        return Decision(
            decision="no",
            title="결제 불가",
            summary=f"‘{matched}’은(는) 개인용 범용 전자기기로 분류되어 금액과 관계없이 구입할 수 없습니다.",
            reasons=[
                "노트북·데스크톱·모니터·태블릿·스마트워치·이어폰·휴대전화 등은 아이템 개발 목적이어도 인정되지 않습니다.",
            ],
            evidence=[],
            cautions=["장비가 필요하면 지급수수료 비목의 기자재 임차료(렌탈)로 진행하세요."],
            approval_required=False,
        )

    matched = _contains(text, CASH_EQUIVALENT)
    if matched:
        return Decision(
            decision="no",
            title="결제 불가",
            summary=f"‘{matched}’은(는) 환금성 물품으로 분류되어 사용할 수 없습니다.",
            reasons=["상품권·기프트카드·포인트 충전·귀금속·유가증권 등 환금성 물품은 집행 불가입니다."],
            evidence=[],
            cautions=["설문 참여자 사례비를 상품권으로 지급하는 것도 인정되지 않습니다."],
            approval_required=False,
        )

    matched = _contains(text, LABOR)
    if matched:
        return Decision(
            decision="no",
            title="결제 불가",
            summary=f"‘{matched}’은(는) 인건비 성격으로 창업활동자금에서 집행할 수 없습니다.",
            reasons=["진출자 본인·팀원의 인건비·활동비·4대보험료는 집행 불가입니다."],
            evidence=[],
            cautions=["외주용역비는 외부 전문업체에 위탁하는 비용으로, 팀원 인건비와 구분해야 합니다."],
            approval_required=False,
        )

    matched = _contains(text, SPACE)
    if matched:
        return Decision(
            decision="no",
            title="결제 불가",
            summary=f"‘{matched}’은(는) 공간·환경 개선비로 집행할 수 없습니다.",
            reasons=["사무실 인테리어·간판·도색 등 환경개선비는 사용 불가입니다."],
            evidence=[],
            cautions=["공유오피스 임차료는 지급수수료 비목으로 가능할 수 있습니다."],
            approval_required=False,
        )

    matched = _contains(text, PRIVATE)
    if matched:
        return Decision(
            decision="no",
            title="결제 불가",
            summary=f"‘{matched}’은(는) 사적 지출로 인정되지 않습니다.",
            reasons=["경조사비, 가족·지인과의 식사, 개인 소비재, 아이템과 무관한 학원 수강료는 집행 불가입니다."],
            evidence=[],
            cautions=["창업 아이템과의 직접적인 연관성을 소명할 수 있어야 합니다."],
            approval_required=False,
        )

    matched = _contains(text, PARKING_FUEL)
    if matched:
        return Decision(
            decision="no",
            title="결제 불가",
            summary=f"‘{matched}’은(는) 여비·회의비에서도 집행할 수 없습니다.",
            reasons=["주차료·주유비·유류비는 집행 불가입니다."],
            evidence=[],
            cautions=["대중교통·택시(현장 카드 단말기 결제)는 가능합니다."],
            approval_required=False,
        )

    matched = _contains(text, DEBT_DUPLICATE)
    if matched:
        return Decision(
            decision="no",
            title="결제 불가",
            summary=f"‘{matched}’은(는) 집행할 수 없습니다.",
            reasons=["부채 상환, 다른 정부지원사업에서 이미 정산받은 비용(중복 정산)은 사용 불가입니다."],
            evidence=[],
            cautions=["다른 지원사업과 중복되지 않는지 반드시 확인하세요."],
            approval_required=False,
        )

    matched = _contains(text, PATENT_SUCCESS_FEE)
    if matched:
        return Decision(
            decision="no",
            title="결제 불가",
            summary=f"‘{matched}’은(는) 집행할 수 없습니다.",
            reasons=["특허등록 성공보수는 지급수수료로 인정되지 않습니다."],
            evidence=[],
            cautions=["특허·상표 출원 관납료와 변리사 수수료(성공보수 제외)는 가능합니다."],
            approval_required=False,
        )

    matched = _contains(text, RESTRICTED)
    if matched:
        return Decision(
            decision="no",
            title="결제 불가",
            summary=f"‘{matched}’ 관련 업종은 클린카드 자동 차단 대상일 수 있습니다.",
            reasons=["유흥·향락, 사행성, 주류·담배, 환금성, 부동산·금융, 골프장 등은 제한 업종입니다."],
            evidence=[],
            cautions=["같은 품목이라도 가맹점의 업종 등록에 따라 승인 여부가 달라질 수 있습니다."],
            approval_required=False,
        )

    # 3) 여비·회의비 특례
    if category == "여비·회의비":
        if _has_any(text, OVERSEAS):
            return Decision(
                decision="no",
                title="결제 불가",
                summary="국외 여비(항공편 포함)는 집행할 수 없습니다.",
                reasons=["국외 출장·해외 숙박·항공권은 사용 불가입니다."],
                evidence=[],
                cautions=["국내 출장비만 인정됩니다."],
                approval_required=False,
            )
        if _has_any(text, CANCELLATION):
            return Decision(
                decision="no",
                title="결제 불가",
                summary="취소 수수료가 발생하는 예매·숙박 결제는 카드 이용약관상 승인되지 않습니다.",
                reasons=["철도·고속버스·항공·숙박 등 취소 수수료가 부과되는 예매 건은 결제 제한 대상입니다."],
                evidence=[],
                cautions=["취소 수수료가 없는 현장 카드 단말기 직접 결제(택시 등)는 가능합니다."],
                approval_required=False,
            )

        reasons.append("여비·회의비 비목에 해당합니다.")
        if data.region in LODGING_LIMITS and data.nights > 0:
            limit = LODGING_LIMITS[data.region] * data.nights
            if data.amount > limit:
                return Decision(
                    decision="conditional",
                    title="조건부 가능",
                    summary=f"숙박 한도({data.region} {LODGING_LIMITS[data.region]:,}원/박 × {data.nights}박 = {limit:,}원)를 초과했습니다. 초과분은 자부담입니다.",
                    reasons=[
                        "숙박비는 한도 내 실비로만 정산하며, 한도 초과분은 자부담입니다.",
                        f"현재 입력 금액 {data.amount:,}원 / 한도 {limit:,}원",
                    ],
                    evidence=["일자·목적·장소를 기재한 간이 증빙"],
                    cautions=["한도 내 금액만 카드로 결제하고 초과분은 별도 결제하세요."],
                    approval_required=approval_required,
                )
        evidence.append("일자·목적·장소(회의비는 동석자)를 기재한 간이 증빙")
        if approval_required:
            reasons.append("1건 50만원 초과로 멘토기관 사전승인이 필요합니다.")
            cautions.append("사전승인 없이 결제하면 사후 정산이 어려울 수 있습니다.")
        return Decision(
            decision="conditional" if approval_required else "yes",
            title="조건부 가능" if approval_required else "결제 가능",
            summary="국내 여비·회의비로 볼 수 있습니다. 증빙과 한도 기준을 확인하세요.",
            reasons=reasons,
            evidence=evidence,
            cautions=cautions or ["주차비·유류비는 불가하며, 숙박 한도와 동석자 정보를 확인하세요."],
            approval_required=approval_required,
        )

    # 4) 비목 미정
    if category == "모르겠음":
        return Decision(
            decision="conditional",
            title="조건부 가능",
            summary="5개 비목 중 어디에 해당하는지 먼저 정해야 합니다. 아래 비목을 보고 가장 가까운 항목을 선택하세요.",
            reasons=[
                "비목이 정해지지 않으면 적정·부적정 판단을 할 수 없습니다.",
                "창업 아이템과의 직접적인 연관성도 함께 확인됩니다.",
            ],
            evidence=["비목 선택 후 증빙 기준 확인"],
            cautions=["애매하면 결제 전에 멘토기관에 문의하세요."],
            approval_required=approval_required,
        )

    # 5) 자산성 물품
    if data.asset == "50만원 초과":
        return Decision(
            decision="no",
            title="결제 불가",
            summary="자산성 물품은 50만원 이하만 구입할 수 있습니다. 50만원 초과 물품은 구입 불가이며, 필요하면 임차(렌탈)로 진행하세요.",
            reasons=[
                "50만원 초과 자산성 물품은 구입 불가입니다.",
                "지급수수료 비목의 기자재 임차료(렌탈)로 대체할 수 있습니다.",
            ],
            evidence=[],
            cautions=["자산성 여부가 애매하면 결제 전에 멘토기관에 문의하세요."],
            approval_required=False,
        )

    if data.asset == "50만원 이하":
        reasons.append("자산성 물품이지만 50만원 이하이고 창업활동과의 연관성이 인정되면 구입 가능합니다.")
        evidence.append("자산보유서약서 제출, 취득일로부터 5년간 보유·관리")
        cautions.append("자산보유서약서를 제출하지 않으면 정산이 어려울 수 있습니다.")

    if data.asset == "모르겠음":
        reasons.append("자산성 물품 여부가 불분명합니다. 50만원 이하인지, 5년 보관 의무가 있는지 확인하세요.")
        cautions.append("모호하면 집행 전에 멘토기관에 문의하세요.")

    # 6) 공통 금액·증빙 기준
    if data.amount >= 100_000:
        evidence.append("1건 10만원 이상: 주문상세내역서·영수증 등 증빙 제출")
    else:
        evidence.append("1건 10만원 미만: 가맹점명으로 사용처가 확인되면 증빙 면제")

    if data.payment == "온라인/PG 결제":
        evidence.append(
            "PG/온라인 결제: 금액과 무관하게 주문상세내역서(판매자 상호·사업자등록번호·주문번호·품목·결제일·금액·수단) 캡처"
        )
        cautions.append("PG 결제는 승인내역에 결제대행사명만 표시될 수 있어 증빙이 필수입니다.")

    if approval_required:
        reasons.append("1건 50만원 초과로 멘토기관 사전승인이 필요합니다.")
        cautions.append("사전승인 없이 결제하면 사후 정산이 어려울 수 있습니다.")

    if data.evidence == "불가능":
        cautions.append("증빙을 만들 수 없다면 해당 지출은 인정되지 않을 수 있습니다. 결제 전에 멘토기관과 상의하세요.")
    elif data.evidence == "모르겠음":
        cautions.append("증빙 가능 여부를 먼저 확인하세요. 주문상세내역을 바로 캡처해 두는 것이 안전합니다.")

    if data.amount > 500_000:
        reasons.append("50만원 초과 결제는 사전승인 대상입니다.")

    # 7) 비목별 기본 적합성
    if category == "지급수수료":
        reasons.append("지급수수료 비목(시험·인증·임차·구독·특허 출원 등)에 해당합니다.")
        if _has_any(text, ["구독", "ai", "소프트웨어", "api", "토큰"]):
            cautions.append(
                "구독·정기결제·선충전 API는 사용 기간 내 결제·충전분만 인정됩니다. 기간 종료 후 자동결제가 발생하지 않도록 설정하세요."
            )
        if _has_any(text, ["특허", "상표", "출원", "변리사"]):
            cautions.append("특허등록 성공보수는 불가하고, 출원 관납료·변리사 수수료만 가능합니다.")
    elif category == "문헌구입비":
        reasons.append("문헌구입비 비목(도서·논문·보고서·유료 자료)에 해당합니다.")
    elif category == "외주용역비(마케팅·광고 포함)":
        reasons.append("외주용역비 비목(개발·디자인·마케팅·광고 등)에 해당합니다.")
        cautions.append("외주 계약서·견적서·세금계산서 등 거래 사실을 확인할 수 있는 자료를 준비하세요.")
    elif category == "재료구입비":
        reasons.append("재료구입비 비목(재료·부품·소모품·데이터셋 등)에 해당합니다.")
    else:
        reasons.append(f"{category} 비목으로 분류했습니다.")

    reasons.append("적정·부적정의 핵심 기준은 ‘창업 아이템과의 직접적인 연관성’입니다.")
    if not data.relevance.strip():
        cautions.append("창업 아이템과의 연관성을 한 줄이라도 구체적으로 적어두세요. 심사에서 핵심 기준입니다.")

    decision = "conditional" if (
        approval_required
        or data.asset in {"50만원 이하", "모르겠음"}
        or data.evidence != "가능"
        or data.payment == "온라인/PG 결제"
    ) else "yes"
    title = "조건부 가능" if decision == "conditional" else "결제 가능"
    summary = (
        "입력한 내용 기준으로 결제 가능성이 있습니다. 아래 증빙과 주의사항을 확인하세요."
        if decision == "yes"
        else "결제 자체는 가능할 수 있지만, 사전승인·증빙·자산 기준 등 추가 조건이 있습니다."
    )
    return Decision(
        decision=decision,
        title=title,
        summary=summary,
        reasons=reasons,
        evidence=evidence,
        cautions=cautions or ["최종 판단은 멘토기관에 확인하세요."],
        approval_required=approval_required,
    )


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/check", response_model=Decision)
def check(data: ExpenseInput):
    return evaluate(data)
