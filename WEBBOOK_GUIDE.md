# 📚 Tensor2Silicon LAB (T2S Lab) 오픈 리서치 웹북

**[모두의연구소 T2S Lab (LAB #811)](https://modulabs.co.kr/labs/811)** 및 **[Legato User Group](https://github.com/legato-user-group/tensor2silicon-lab)**의 주차별 메인 교안, 학습 안내·계산 예제, 그리고 6인의 연구원 리서치 노트를 **[방식 A (메인 교안 + 연구원 Deep-Dive 하이라이트 & 원문 탭)]** 구조로 통합한 **VitePress 기반 정적 웹북**입니다.

---

## ✨ 주요 특징

1. **방식 A 하이브리드 편집 구조**
   - 각 주차 상단의 **빠른 탭 내비게이션 바(`WeekNavBar`)**를 통해 `📘 메인 교안`, `🧭 학습 안내 & 계산 예제`, `🧑‍🔬 연구원별 리서치 노트 (mbsong / singon / junpark / injokim / jaehyeong)`를 클릭 한 번으로 오갈 수 있습니다.
   - 메인 교안 도입부에 **연구원별 핵심 기여·실측 요약 카드**가 자동 삽입됩니다.
   - 각 교안 및 참고자료의 **연습문제 해답과 해설**은 스포일러 방지를 위해 접기/펼치기(`<details>`) 토글로 변환되어 있습니다.
2. **인터랙티브 시뮬레이터 & 시각화 내장 (서버 불필요, 100% 브라우저 동작)**
   - **🎛️ Roofline & 2D Tiled MatMul 계산기 (`RooflineCalculator.vue`)**: 3·4·7주차 연동. 행렬 크기, 자료형(`FP32`/`FP16`), 타일 크기(`BM×BN×BK`), 파이프라인 단계(`num_stages`), 하드웨어 프리셋(4주차/7주차 교육용 GPU, A100, RTX 5090, H100, TPU v5e)을 바꾸며 산술 강도($I=F/Q$), HBM 이동량 감소율, Shared Memory 요구량을 실시간 시뮬레이션합니다.
   - **🔍 다단계 컴파일 IR 파이프라인 탐색기 (`IrPipelineViewer.vue`)**: 6·7주차 연동. `Python DSL → TTIR → TTGIR → LLVM IR → PTX → SASS` 6단계 변환 과정과 **A100 (`sm_80`) vs RTX 5090 (`sm_120`)** 실측 포인트를 단계별로 비교합니다.
   - **🖥️ 인터랙티브 HTML 덱 뷰어 (`HtmlDeckViewer.vue`)**: 2주차 `injokim.html`과 `torch-compile-anatomy-deck.html`을 웹북 페이지 내에 반응형 프레임(크게 보기 / 전체화면 지원)으로 내장했습니다.
3. **GitHub Pages 무인 자동 배포 (`Zero-Server`)**
   - 로컬 PC나 개인 서버를 켜둘 필요 없이, GitHub에 푸시하기만 하면 `.github/workflows/deploy.yml`(GitHub Actions)이 자동으로 정적 사이트를 빌드하여 **GitHub Pages**에 배포합니다.

---

## 🚀 GitHub Pages 배포 방법 (최초 1회 설정)

1. 본 폴더(`t2s_gitbook`)의 파일들을 배포할 GitHub 레포지토리(예: `legato-user-group/tensor2silicon-lab` 또는 신규 웹북 레포지토리)의 `main` 브랜치에 푸시합니다.
2. GitHub 레포지토리 상단 메뉴에서 **`Settings` → 좌측 `Pages`**로 이동합니다.
3. **Build and deployment** 아래의 **Source**를 기본값(`Deploy from a branch`)에서 **`GitHub Actions`**로 변경합니다.
4. 변경 즉시 `Actions` 탭에서 `Deploy T2S WebBook to GitHub Pages` 워크플로우가 실행되며, 약 1~2분 후 `https://<조직명>.github.io/<레포지토리명>/` 주소로 웹북이 공개됩니다!

---

## 🛠️ 로컬에서 확인하거나 원본 업데이트 반영하기

```bash
# 1. 상위 폴더(/home/leeyb/t2s) 및 tensor2silicon-lab-main의 마크다운 변경사항 다시 동기화
npm run sync

# 2. 로컬 개발 서버 실행 (실시간 미리보기)
npm run docs:dev

# 3. 정적 웹사이트 빌드 테스트 (docs/.vitepress/dist 생성)
npm run docs:build
```
