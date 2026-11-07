---
layout: home

hero:
  name: "Tensor2Silicon LAB"
  text: "AI 모델에서 가속기 커널까지"
  tagline: "PyTorch·Transformer 수식 한 줄이 그래프(IR), 컴파일러, 런타임을 거쳐 GPU·TPU·NPU 실리콘 위에서 실행되기까지의 전 과정을 정량적 손계산과 커널 실습으로 파헤치는 오픈 리서치 웹북"
  actions:
    - theme: brand
      text: "📖 웹북 읽기 시작하기 (랩 소개)"
      link: /intro/overview
    - theme: alt
      text: "🚀 2주차: 모델에서 실리콘까지"
      link: /week02/
    - theme: alt
      text: "🔥 7주차: Tiled MatMul & IR 해부"
      link: /week07/

features:
  - icon: 🧮
    title: "정량적 손계산 & 인터랙티브 Roofline"
    details: "막연한 개념 설명 대신 FLOPs, HBM 메모리 이동량(Q), 산술 강도(I = F/Q), Shared Memory 타일 버퍼 크기를 손으로 직접 계산하고 웹북 내장 시뮬레이터로 즉시 실험합니다."
  - icon: ⚙️
    title: "다단계 컴파일러 & IR 실측 해부"
    details: "PyTorch(torch.compile), JAX(OpenXLA), Modular MAX, 그리고 Triton 커널이 TTIR → TTGIR → LLVM IR → PTX → SASS로 구체화되는 과정을 A100 및 RTX 5090 실측 데이터로 검증합니다."
  - icon: 🧑‍🔬
    title: "방식 A: 메인 교안 + 6인 연구원 Deep-Dive"
    details: "체계적인 메인 교과서 흐름을 따라가면서도, 매 주차 6명의 연구원이 각자의 하드웨어와 시각에서 분석한 심화 리포트와 인터랙티브 슬라이드를 탭 한 번으로 함께 열람할 수 있습니다."
  - icon: 🌐
    title: "Legato 오픈소스 저수준 IR 생태계"
    details: "과학기술정보통신부·IITP K-클라우드 기술개발 사업의 일환으로, 국산 AI반도체 및 이기종 가속기를 위한 개방형 저수준 IR(Legato) 커뮤니티의 기술적 토대를 구축합니다."
---

## 🗺️ 커리큘럼 한눈에 바로가기 (Curriculum Roadmap)

| 주차 | 주제 (Topic) | 핵심 내용 및 인터랙티브 도구 | 바로가기 |
| :---: | :--- | :--- | :---: |
| **Intro** | **랩 소개 & 5대 실행 스택** | T2S Lab 운영 원칙, 5대 계층 아키텍처, 연구원 아카이브 매트릭스 | [소개 보기](./intro/overview) |
| **Week 02** | **모델에서 실리콘까지** | PyTorch 2.0, JAX·OpenXLA, Modular MAX·Mojo 비교 + **인터랙티브 HTML 슬라이드 덱** | [2주차 열기](./week02/) |
| **Week 03** | **커널의 시간은 어디에서 소비되는가** | 연산 시간 vs 메모리 이동 시간, SRAM vs DRAM 물리 구조, Vector Add & Fusion | [3주차 열기](./week03/) |
| **Week 04** | **Roofline과 Tiling** | 산술 강도($I = F/Q$), 타일 데이터 재사용, **인터랙티브 Roofline 시뮬레이터** | [4주차 열기](./week04/) |
| **Week 05** | **xPU를 어떻게 비교할 것인가** | GPU(SIMT) vs TPU(Systolic Array) vs NPU/LPU 아키텍처 및 워크로드별 비교 기준 | [5주차 열기](./week05/) |
| **Week 06** | **Python에서 Triton 커널로** | 1D 블록 포인터(`tl.arange`), 마스킹, Fused Online Softmax & RTX 4070 실측 | [6주차 열기](./week06/) |
| **Week 07** | **Tiled MatMul과 컴파일 과정** | 2D 타일 행렬곱, L2 그룹핑, **다단계 IR 탐색기(TTIR→SASS)**, A100 vs RTX 5090 비교 | [7주차 열기](./week07/) |
