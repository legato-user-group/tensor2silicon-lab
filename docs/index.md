---
layout: home

hero:
  name: "Tensor2Silicon LAB"
  text: "AI 모델에서 가속기 커널까지"
  tagline: "PyTorch·Transformer 수식 한 줄이 그래프(IR), 컴파일러, 런타임을 거쳐 GPU·TPU·NPU 실리콘 위에서 실행되기까지의 전 과정을 정량적 손계산·능동 학습 퀘스트·커널 실습으로 체득하는 오픈 리서치 웹북"
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
  - icon: 🎯
    title: "전 주차 5단계 능동 학습 & 완독 패스포트"
    details: "단순히 눈으로만 읽고 지나치지 않도록 매 주차 3분 직관 지도, 오개념 O/X, 손계산 빈칸, 인터랙티브 실습 미션, 파인만 한줄설명 인증서를 제공합니다."
  - icon: 🧮
    title: "정량적 손계산 & 인터랙티브 시뮬레이터"
    details: "FLOPs, HBM 메모리 이동량(Q), 산술 강도(I = F/Q), Shared Memory 타일 버퍼 크기를 손으로 직접 계산하고 웹북 내장 Roofline·IR 탐색기로 즉시 실험합니다."
  - icon: ⚙️
    title: "다단계 컴파일러 & IR 실측 해부"
    details: "PyTorch(torch.compile), JAX(OpenXLA), Modular MAX, 그리고 Triton 커널이 TTIR → TTGIR → LLVM IR → PTX → SASS로 구체화되는 과정을 A100 및 RTX 5090 실측 데이터로 검증합니다."
  - icon: 🧑‍🔬
    title: "방식 A: 메인 교안 + 6인 연구원 Deep-Dive"
    details: "체계적인 메인 교과서 흐름을 따라가면서도, 매 주차 6명의 연구원이 각자의 하드웨어와 시각에서 분석한 심화 리포트와 인터랙티브 슬라이드를 탭 한 번으로 함께 열람할 수 있습니다."
---

## 🗺️ 커리큘럼 한눈에 바로가기 (Curriculum Roadmap)

| 주차 | 주제 (Topic) | 능동 학습 퀘스트 & 인터랙티브 도구 (`5/5` 마스터리) | 바로가기 |
| :---: | :--- | :--- | :---: |
| **Intro** | **랩 소개 & 5대 실행 스택** | T2S Lab 운영 원칙, 5대 계층 아키텍처, 연구원 아카이브 매트릭스 | [소개 보기](./intro/overview) |
| **Week 02** | **모델에서 실리콘까지** | 실행 스택 역할 매칭 미션 + Epilogue Fusion 손계산 + **인터랙티브 HTML 슬라이드 덱** | [2주차 열기](./week02/) |
| **Week 03** | **커널의 시간은 어디에서 소비되는가** | 용량 vs 대역폭 진단 + Fusion 속도 배수 계산 + **병목 전환 타일 탐색 미션** | [3주차 열기](./week03/) |
| **Week 04** | **Roofline과 Tiling** | 산술 강도($I = F/Q$) 예측 + 타일 읽기량 손계산 + **Roofline & SRAM 제약 돌파 미션** | [4주차 열기](./week04/) |
| **Week 05** | **xPU를 어떻게 비교할 것인가** | 스펙시트 Sparsity 함정 진단 + 장치 역전 손계산 + **GPU·TPU & Prefill·Decode 매칭** | [5주차 열기](./week05/) |
| **Week 06** | **Python에서 Triton 커널로** | `-inf` 패딩 버그 찾기 + Fused Softmax `4x` 절감 계산 + **Triton 핵심 구문 매칭** | [6주차 열기](./week06/) |
| **Week 07** | **Tiled MatMul과 컴파일 과정** | L2 그룹핑 진단 + 파이프라인 SRAM 계산 + **다단계 IR(`TTIR→SASS`) 탐정 미션** | [7주차 열기](./week07/) |
