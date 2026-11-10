from laya import Router

router = Router(max_loaded=1)
state = "생성형 AI 구독료 52만원. 창업 아이템 기획에 사용."
questions = {
    "category": {
        "type": "choice",
        "instructions": "이 지출의 비목은 무엇인가?",
        "criteria": {
            "지급수수료": "구독, AI, 소프트웨어, 인증, 시험, 임차, 특허",
            "재료구입비": "재료, 부품, 소모품, 데이터셋",
            "외주용역비": "외주, 개발, 디자인, 마케팅, 광고",
            "문헌구입비": "도서, 논문, 보고서, 유료 자료",
            "여비회의비": "출장, 교통, 숙박, 회의, 식사",
            "불가": "노트북, 주차, 상품권, 인건비, 해외",
            "모르겠음": "판단하기 어려움",
        },
    },
    "eligible": {
        "type": "noul",
        "instructions": "이 지출은 모두의창업 창업활동자금으로 집행 가능한가?",
    },
}
result = router.predict(state, questions, model="multilingual")
print(result)
