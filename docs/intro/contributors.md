# ✍️ 대표 저자 & 연구진 소개

**『Tensor2Silicon: AI 모델에서 가속기 커널까지』**는 **대표 저자 이영빈(`yblee`)이 직접 설계·집필한 전 6장 본문 교과서 및 정량 계산 연습**을 중심축으로, 5인의 연구진이 하드웨어 실측·다이어그램·교차 검증 아티클을 더해 완성한 기술 웹북입니다.

## ✍️ 대표 저자 & 커리큘럼 총괄 (Lead Author)

<div class="t2s-lead-author-card">
  <div class="t2s-lead-author-left">
    <img class="t2s-lead-author-avatar-lg" src="https://github.com/yblee110.png" alt="이영빈 (Youngbin Lee)" />
    <div class="t2s-lead-author-info">
      <h3>
        <span>이영빈 (Youngbin Lee · <code>yblee</code>)</span>
        <span class="t2s-author-role-badge">🧭 대표 저자 · 전 6장 본문 단독 집필</span>
      </h3>
      <p>
        <strong>집필 범위:</strong> 전체 6장 커리큘럼 설계, <code>Chapter 01 ~ Chapter 06</code> 본문 교안 단독 집필, 단원별 핵심 요약 및 정량 계산 예제·연습문제·해설 집필<br/>
        <strong>소속 및 활동:</strong> 모두의연구소 커뮤니티팀 팀장 (위탁연구개발기관 책임자) · Google AI GDE · JAX-KR 오거나이저 · GDG Incheon 오거나이저
      </p>
    </div>
  </div>
  <div class="t2s-lead-author-actions">
    <a class="t2s-author-gh-btn" href="https://github.com/yblee110" target="_blank" rel="noopener">GitHub @yblee110 ↗</a>
  </div>
</div>

## 🧑‍🔬 공동 검증 & 심화 리서치 연구진 (Contributing Researchers)

| 프로필 | 이름 (핸들) | 주요 연구 기여 및 하드웨어 실측 분야 | GitHub |
| :---: | :--- | :--- | :---: |
| <img src="https://github.com/Jake-Song.png" width="56" style="border-radius:50%" /> | **송무복 (`mbsong`)** | 🔬 A100 실측 IR 아티팩트 구축 · GPU/TPU 아키텍처 심층 비교 | [@Jake-Song](https://github.com/Jake-Song) |
| <img src="https://github.com/SinGonKim.png" width="56" style="border-radius:50%" /> | **김신곤 (`singon`)** | 🧮 정량적 손계산 해설 · 아키텍처 시각화 & RTX 4070 실측 | [@SinGonKim](https://github.com/SinGonKim) |
| <img src="https://github.com/currybab.png" width="56" style="border-radius:50%" /> | **박준 (`junpark`)** | 🔥 RTX 5090 (`sm_120`) PTX 실측 · 하드웨어 물리 계층 해부 | [@currybab](https://github.com/currybab) |
| <img src="https://github.com/Kiminjo.png" width="56" style="border-radius:50%" /> | **김인조 (`injokim`)** | 🖥️ 인터랙티브 HTML 시각화 제작 · 수식 유도 & MLIR 파이프라인 정리 | [@Kiminjo](https://github.com/Kiminjo) |
| <img src="https://github.com/eloqlo.png" width="56" style="border-radius:50%" /> | **이재형 (`jaehyeong`)** | 📐 핵심 개념 요약 · 컴파일 파이프라인 & 타일링 구조 정리 | [@eloqlo](https://github.com/eloqlo) |

---

## 📚 장별 전체 수록 문서 매트릭스 (Chapter Matrix)

| 장 (Chapter) | ✍️ 본문 교안 (집필: 이영빈) | 🧭 요약·연습 (집필: 이영빈) | `mbsong` | `singon` | `junpark` | `injokim` | `jaehyeong` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chapter 01**<br/><small>모델에서 실리콘까지</small> | [📘 본문 읽기](/ch01/) | [🧭 요약/연습](/ch01/guide) | [열기](/ch01/researchers/mbsong) | [열기](/ch01/researchers/singon) | [열기](/ch01/researchers/junpark) | [열기](/ch01/researchers/injokim) | [열기](/ch01/researchers/jaehyeong) |
| **Chapter 02**<br/><small>커널의 시간은 어디에서 소비되는가</small> | [📘 본문 읽기](/ch02/) | [🧭 요약/연습](/ch02/guide) | [열기](/ch02/researchers/mbsong) | [열기](/ch02/researchers/singon) | [열기](/ch02/researchers/junpark) | [열기](/ch02/researchers/injokim) | [열기](/ch02/researchers/jaehyeong) |
| **Chapter 03**<br/><small>Roofline과 Tiling</small> | [📘 본문 읽기](/ch03/) | [🧭 요약/연습](/ch03/guide) | [열기](/ch03/researchers/mbsong) | [열기](/ch03/researchers/singon) | [열기](/ch03/researchers/junpark) | [열기](/ch03/researchers/injokim) | [열기](/ch03/researchers/jaehyeong) |
| **Chapter 04**<br/><small>xPU를 어떻게 비교할 것인가</small> | [📘 본문 읽기](/ch04/) | [🧭 요약/연습](/ch04/guide) | [열기](/ch04/researchers/mbsong) | [열기](/ch04/researchers/singon) | [열기](/ch04/researchers/junpark) | - | - |
| **Chapter 05**<br/><small>Python에서 Triton 커널로</small> | [📘 본문 읽기](/ch05/) | [🧭 요약/연습](/ch05/guide) | [열기](/ch05/researchers/mbsong) | [열기](/ch05/researchers/singon) | [열기](/ch05/researchers/junpark) | [열기](/ch05/researchers/kiminjo) | [열기](/ch05/researchers/jaehyeong) |
| **Chapter 06**<br/><small>Tiled MatMul과 컴파일 과정</small> | [📘 본문 읽기](/ch06/) | [🧭 요약/연습](/ch06/guide) | [열기](/ch06/researchers/mbsong) | [열기](/ch06/researchers/singon) | [열기](/ch06/researchers/junpark) | [열기](/ch06/researchers/injokim) | [열기](/ch06/researchers/jaehyeong) |
