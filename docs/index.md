---
layout: home

hero:
  name: "Tensor2Silicon"
  text: "AI 모델에서 가속기 커널까지"
  tagline: "PyTorch·Transformer 수식 한 줄이 그래프(IR), 컴파일러, 런타임을 거쳐 GPU·TPU·NPU 실리콘 명령어에 도달하기까지의 전 과정을 정량적 손계산·능동 학습 퀘스트·커널 실습으로 완주하는 기술 웹북 (집필: 이영빈 · yblee)"
  actions:
    - theme: brand
      text: "⚡ Chapter 01 · 15분 핵심 압축본부터 시작"
      link: /ch01/guide
    - theme: alt
      text: "📘 Chapter 01 · 전체 심화 본문 읽기"
      link: /ch01/
    - theme: alt
      text: "📖 책 소개 & 5대 계층 아키텍처"
      link: /intro/overview

features:
  - icon: ⚡
    title: "듀얼 트랙: 15분 핵심 압축 교안 + 심화 본문"
    details: "바쁠 때는 장당 15분 분량의 『핵심 압축 교안(Core Digest)』으로 수식·도표·코드를 빠르게 정복하고, 필요할 때 『전체 심화 본문』의 해당 절로 즉시 점프할 수 있습니다."
  - icon: 🎯
    title: "전 6장 5단계 능동 학습 & 완독 패스포트"
    details: "단순히 눈으로만 읽고 지나치지 않도록 모든 장마다 3분 핵심 지도, 오개념 O/X, 손계산 빈칸, 인터랙티브 실습 미션, 파인만 한줄설명 인증서를 제공합니다."
  - icon: 🧮
    title: "정량적 손계산 & 인터랙티브 시뮬레이터"
    details: "FLOPs, HBM 데이터 이동량(Q), 산술 강도(I = F/Q), Shared Memory 타일 버퍼 크기를 손으로 직접 계산하고 웹북 내장 Roofline·IR 탐색기로 즉시 실험합니다."
  - icon: ✍️
    title: "이영빈(yblee) 본문 집필 + 연구진 하드웨어 실측"
    details: "대표 저자 이영빈(yblee)이 설계·집필한 전 6장 메인 교과서와 압축 교안을 중심축으로, 5인의 연구진이 A100·RTX 5090 등에서 교차 검증한 실측 노트를 함께 수록했습니다."
---

<div class="t2s-lead-author-card">
  <div class="t2s-lead-author-left">
    <img class="t2s-lead-author-avatar-lg" src="https://github.com/yblee110.png" alt="이영빈 (Youngbin Lee)" />
    <div class="t2s-lead-author-info">
      <h3>
        <span>✍️ 대표 저자: 이영빈 (Youngbin Lee · <code>yblee</code>)</span>
        <span class="t2s-author-role-badge">🧭 커리큘럼 설계 · 전 6장 본문 집필</span>
      </h3>
      <p>
        본 웹북의 아키텍처 설계와 <strong>전 6장(Chapter 01 ~ Chapter 06) 심화 본문, 15분 핵심 압축 교안, 정량적 손계산 예제 및 연습문제·해설을 직접 집필</strong>했습니다.<br/>
        <small>모두의연구소 커뮤니티팀 팀장 (위탁연구개발기관 책임자) · Google AI GDE · JAX-KR / GDG Incheon 오거나이저</small>
      </p>
    </div>
  </div>
  <div class="t2s-lead-author-actions">
    <a class="t2s-author-gh-btn" href="https://github.com/yblee110" target="_blank" rel="noopener">GitHub @yblee110 ↗</a>
    <a class="t2s-author-gh-btn" href="./intro/contributors">저자 &amp; 연구진 소개 →</a>
  </div>
</div>

## 📚 전체 목차 한눈에 보기 (Complete 6-Chapter Edition)

| 장 (Chapter) | 제목 · 본문 집필: 이영빈 (`yblee`) | 능동 학습 퀘스트 & 인터랙티브 도구 (`5/5` 마스터리) | 듀얼 트랙 바로가기 |
| :---: | :--- | :--- | :---: |
| **Preface** | **책 소개 & 5대 실행 스택** | 집필 원칙, 모델→실리콘 5대 계층 아키텍처, 저자 & 연구진 아카이브 | [서문 보기](./intro/overview) |
| **Chapter 01** | **모델에서 실리콘까지** | 실행 스택 역할 매칭 미션 + Epilogue Fusion 손계산 + **인터랙티브 슬라이드 덱** | [⚡ 15분 압축](./ch01/guide) · [📘 심화 본문](./ch01/) |
| **Chapter 02** | **커널의 시간은 어디에서 소비되는가** | 용량 vs 대역폭 진단 + Fusion 속도 배수 계산 + **병목 전환 타일 탐색 미션** | [⚡ 15분 압축](./ch02/guide) · [📘 심화 본문](./ch02/) |
| **Chapter 03** | **Roofline과 Tiling** | 산술 강도($I = F/Q$) 예측 + 타일 읽기량 손계산 + **Roofline & SRAM 제약 돌파 미션** | [⚡ 15분 압축](./ch03/guide) · [📘 심화 본문](./ch03/) |
| **Chapter 04** | **xPU를 어떻게 비교할 것인가** | 스펙시트 Sparsity 함정 진단 + 장치 역전 손계산 + **GPU·TPU & Prefill·Decode 매칭** | [⚡ 15분 압축](./ch04/guide) · [📘 심화 본문](./ch04/) |
| **Chapter 05** | **Python에서 Triton 커널로** | `-inf` 패딩 버그 찾기 + Fused Softmax `4x` 절감 계산 + **Triton 핵심 구문 매칭** | [⚡ 15분 압축](./ch05/guide) · [📘 심화 본문](./ch05/) |
| **Chapter 06** | **Tiled MatMul과 컴파일 과정** | L2 그룹핑 진단 + 파이프라인 SRAM 계산 + **다단계 IR(`TTIR→SASS`) 탐정 미션** | [⚡ 15분 압축](./ch06/guide) · [📘 심화 본문](./ch06/) |
