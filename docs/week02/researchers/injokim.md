<WeekNavBar :week="2" topic="모델에서 실리콘까지" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송민배)', 'badge': 'PyTorch·XLA·MAX 2부작'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': '다이어그램 & 스택 해부'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': '실행 스택 핵심 요약'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '인터랙티브 HTML & 슬라이드'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': '컴파일 파이프라인 정리'}]" />

> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **injokim (김인조)** (`인터랙티브 HTML & 슬라이드`)
>
> 💡 **핵심 기여 요약**: 정적 문서를 넘어 브라우저에서 직접 클릭하며 탐색할 수 있는 인터랙티브 학습 노트(HTML)와 torch.compile 해부 슬라이드 덱을 제작했습니다.

---

# 2주차 인터랙티브 학습 노트 & 슬라이드 덱 (injokim (김인조))

## AI 실행 스택 인터랙티브 학습 노트 (injokim.html)

김인조 연구원이 제작한 인터랙티브 실행 스택 비교 및 학습 노트입니다.

<HtmlDeckViewer src="/decks/injokim-week02.html" title="AI 실행 스택 인터랙티브 학습 노트 (injokim.html)" author="injokim (김인조)" :height="640" />

## torch.compile 내부 구조 해부 슬라이드 덱

TorchDynamo → AOTAutograd → PrimTorch → TorchInductor → Triton Codegen 전 과정을 담은 인터랙티브 프레젠테이션입니다.

<HtmlDeckViewer src="/decks/torch-compile-anatomy-deck.html" title="torch.compile 내부 구조 해부 슬라이드 덱" author="injokim (김인조)" :height="640" />

