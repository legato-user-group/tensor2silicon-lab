<WeekNavBar :week="4" topic="Roofline과 Tiling" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송무복)', 'badge': 'Roofline & 타일 크기별 분석'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': '4×4 & 4096×4096 완전 해설'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': '2D Block Tiling 직관'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '이론 유도 & 연습문제 풀이'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': 'Roofline & Tiling 핵심 노트'}]" />


**[4주차 안내] Roofline과 Tiling**

안녕하세요! 4주차에는 **커널의 성능 한계를 Roofline 그래프로 읽고, Tiling으로 데이터를 재사용하는 방법**을 함께 살펴보려고 합니다.
지난주에 연산 시간과 메모리 시간을 따로 계산했다면, 이번에는 두 시간을 하나의 그래프로 연결하고 데이터를 덜 옮길 방법을 생각해보겠습니다.
**“같은 계산을 하면서 데이터를 덜 읽어오면, 커널은 얼마나 빨라질 수 있을까?”**

이번 주에는 그림과 손계산으로 원리를 익혀보면 좋겠습니다. 전체 GPU 커널을 구현하는 실습은 뒤의 Triton·Tiled MatMul 주차에서 이어갑니다.

**1. 공통으로 읽어오면 좋은 자료**

- [Scaling Book 1장: All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/#where-does-the-time-go)
지난주에 읽은 두 시간 공식을 복습하고, **Arithmetic Intensity 정의부터 Visualizing rooflines까지** 읽어주세요. 가로축·세로축, 기울어진 선과 수평선이 각각 무엇을 뜻하는지 살펴보면 됩니다. 뒤의 다중 장치 통신 부분은 이번 주 범위에 포함하지 않습니다.

- [MLC: GPU Acceleration — Local Blocking](https://book.mlc.ai/chapter_gpu_acceleration/part1.html#local-blocking) · [Shared Memory Blocking](https://book.mlc.ai/chapter_gpu_acceleration/part1.html#shared-memory-blocking)
두 절의 **그림과 첫 설명**을 중심으로 읽어주세요. 출력 일부를 계산하는 동안 어떤 입력값을 여러 번 사용하는지 따라가봅시다. TVM 스케줄 코드 전체를 이해하거나 실행할 필요는 없습니다.

**2. 해당 사항들을 다같이 정리해주세요**

**① Arithmetic Intensity — 읽어온 데이터로 얼마나 계산할까?**
FLOPs가 같아도 메모리에서 읽고 쓰는 양이 달라지면 무엇이 바뀔까?
**산술 강도 = 총 FLOPs ÷ 총 이동 Byte**로 계산하고, 이번에는 HBM 읽기·쓰기를 기준으로 삼아주세요. 지난주 Vector Add와 행렬곱은 데이터 재사용에서 어떻게 다를까요?
공통 자료인 Scaling Book 1장 중심

**② Roofline — 어느 자원이 성능을 제한할까?**
왜 메모리 대역폭의 한계는 기울어진 선이고, 연산 성능의 한계는 수평선일까?
아래 세 식을 그래프와 연결해봐주세요. 대역폭 기호는 행렬 B와 구분하기 위해 `BW`로 씁니다.

```text
산술 강도 I = 총 FLOPs ÷ HBM에서 읽고 쓴 총 Byte
성능 상한 = min(최대 연산 성능 P, HBM 대역폭 BW × I)
두 상한이 만나는 산술 강도 = P ÷ BW
```

이 그래프의 세로축은 시간이 아니라 **초당 연산 처리율(FLOPs/s)**입니다. 선은 주어진 조건의 성능 상한이며, 실제 커널이 그 높이까지 도달한다는 뜻은 아닙니다.
[NVIDIA Nsight Compute: Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts)의 **Overview** 그림도 함께 보면 좋겠습니다.

**③ Tiling — 작은 조각으로 나누면 왜 다시 쓸 수 있을까?**
`C = A @ B`에서 A의 한 원소와 B의 한 원소는 각각 C의 어떤 원소 계산에 쓰일까?
출력 타일 하나를 정하고, 필요한 A·B 조각을 온칩 메모리에 가져와 재사용하는 과정을 그려주세요. 연산 횟수와 HBM 이동량 중 무엇이 줄어드는지 설명해보면 좋겠습니다.
공통 자료인 MLC의 두 Blocking 절 중심

**④ 타일 크기 — 크게 만들수록 좋을까?**
더 큰 타일은 데이터를 많이 재사용할 수 있지만, 입력 조각과 중간 합을 저장할 공간도 더 필요하지 않을까?
Shared Memory·레지스터 사용량과 동시에 실행할 수 있는 작업의 양을 연결해봐주세요. **출력 타일의 크기와 스레드 수는 같은 개념이 아니라는 점**도 함께 짚어주세요.
[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy)의 **Calculating Occupancy**, [Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#motivations)의 첫 블록 알고리즘 중심

**3. 정리할 때 담아오면 좋은 내용**

- 내가 맡은 질문의 설명과 핵심 용어
- 축·단위·분기점을 표시한 Roofline 그림 또는 타일 재사용 그림
- 연산량·이동량·산술 강도·예상 시간을 담은 계산표
- 자료형, 어느 메모리 구간을 셌는지, 재사용과 캐시에 관한 가정
- 참고한 자료의 링크·읽은 절과 함께 이야기하고 싶은 질문

가능하면 **`C = A @ B`라는 같은 행렬곱**을 각자 선택한 관점에서 설명해보면 좋겠습니다.

**4. 함께 계산해볼 내용: 행렬곱의 데이터 재사용**

먼저 **4×4 행렬곱을 2×2 출력 타일로 계산하는 그림**을 그려봅시다. 입력 한 번을 읽어 출력 몇 개의 계산에 사용할 수 있는지 세어보면 됩니다.

이후 A·B·C가 모두 **4096×4096 FP32 행렬**인 경우로 확장합니다. 연습용 가상 하드웨어는 이 FP32 연산에 적용 가능한 성능 `20 × 10¹² FLOPs/s`, HBM 대역폭 `1 × 10¹² Byte/s`로 둡니다. 원소는 4Byte이며, 아래 조건으로 비교합니다.

- **재사용 없음:** 곱셈할 때마다 A·B 원소를 HBM에서 다시 읽습니다.
- **타일 안에서 재사용:** 출력 타일의 한 변을 32·64·128로 바꾸고, 읽어온 A·B 조각을 그 타일 안에서 재사용합니다. K 방향 입력 조각의 폭도 같은 값으로 둡니다.
- 두 경우 모두 입력은 이미 장치에 있고, 중간 합은 온칩에 유지하며 최종 C만 한 번 씁니다. 기존 C는 읽지 않고, 서로 다른 출력 타일 사이의 캐시 재사용은 없다고 가정합니다.

각 경우의 **FLOPs, HBM 이동량, 산술 강도, Roofline 성능 상한과 이상적 실행시간**을 비교해주세요. 필요한 입력 버퍼와 중간 합의 크기도 계산해봅시다. 실제 제품의 측정값을 예측하기 위한 조건이 아니라, 재사용 효과를 구분해서 보기 위한 단순화입니다.

정리한 뒤에는 **Roofline의 두 상한을 설명할 수 있는지**, **Tiling 전후에 같은 연산량과 달라진 이동량을 셀 수 있는지**, **큰 타일의 이점과 비용을 함께 설명할 수 있는지** 서로 확인해보겠습니다.

추가로 공부하면서 도움이 된 그림이나 자료가 있으면, 어떤 질문을 이해하는 데 좋았는지도 함께 공유해주세요!
그럼 다음 모임에서 뵙겠습니다!



---

# 4주차 참고자료 — Roofline과 Tiling

안내문과 함께 사용할 자료입니다. **산술 강도 → Roofline → 타일 안에서의 재사용 → 타일 크기의 비용** 순서로 살펴봅니다. 공통 읽기를 마친 뒤 맡은 질문에 맞는 선택 자료를 골라주세요.

자료 확인일: **2026-09-29**. 아래 수치와 그림은 직접 구성한 **교육용 가상 조건의 계산 결과**입니다. GPU 벤치마크나 특정 제품의 사양이 아닙니다.

## 1. 자료별 읽을 범위와 쓰임

| 구분 | 자료 | 읽을 범위와 질문 |
|---|---|---|
| 공통 | [Scaling Book 1장](https://jax-ml.github.io/scaling-book/roofline/#where-does-the-time-go) | Arithmetic Intensity 정의부터 **Visualizing rooflines**까지. 두 시간의 비교를 그래프로 어떻게 옮길까? |
| 공통 | [MLC GPU Acceleration Part 1](https://book.mlc.ai/chapter_gpu_acceleration/part1.html#local-blocking) | **Local Blocking / Shared Memory Blocking**의 그림과 첫 설명. 같은 입력이 여러 출력 계산에 쓰이는 곳은 어디일까? |
| 설명 보충 | [Modern MLC: 성능 모델](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_performance/index.html#the-roofline-model) | 첫 Roofline 설명과 **GEMM**, 이후 메모리 병목 최적화 절의 타일 재사용 식. 이상적인 전체 재사용과 타일별 재사용은 어떻게 다를까? |
| 행렬곱의 기본 | [NVIDIA Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) | **Math And Memory Bounds / GPU Implementation**. `2MNK`를 세는 법과 출력 타일을 고정하고 K 방향으로 누적하는 그림을 봅니다. |
| GPU 구현의 근거 | [CUDA: 행렬곱에서 Shared Memory 활용](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab) | `C=AB` 절의 타일 그림과 중복 읽기를 줄이는 설명. 코드 전체보다 어떤 데이터를 다시 사용하는지 확인합니다. |
| 타일 크기의 비용 | [CUDA: Calculating Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy) | 레지스터·Shared Memory가 동시에 실행할 작업의 양을 제한하는 설명. 같은 문서의 **Shared Memory** 실행 구성 절도 선택해서 읽습니다. |
| 의사코드 연결 | [Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#motivations) | **Motivations**의 첫 블록 알고리즘까지만. 출력 크기 `BLOCK_SIZE_M/N`과 누적 방향의 폭 `BLOCK_SIZE_K`를 구분합니다. |
| 실측 그래프 읽기 | [Nsight Compute: Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts) | **Overview / Analysis**. 지붕과 측정점의 높이를 구분합니다. 설치·프로파일링은 이번 주 필수가 아닙니다. |
| 원전 읽기 | [Williams·Waterman·Patterson의 Roofline 기술 보고서](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/Archive/EECS-2008-134.pdf) | 2008년 보고서 **3. The Roofline Model**, 본문 1~2쪽(PDF 3~4쪽). 캐시를 거친 뒤 DRAM으로 이동하는 Byte를 세는 이유를 봅니다. |

MLC 구교재의 Blocking 절은 **Part 1**에 있습니다. Part 2의 하드웨어 특화·Tensorization은 후속 주차와 연결하는 편이 좋습니다. 최신 MLC 성능 장의 B200 수치와 CUDA 문서의 V100 측정값을 아래 가상 FP32 조건에 섞어 쓰지 않습니다. Triton의 `/main/` 문서는 바뀔 수 있으므로, 실제 커널을 실행하는 주차에는 설치 버전에 맞춰 다시 확인합니다.

## 2. 이번 주 공동 정리의 뼈대

핵심 질문: **같은 계산을 하면서 데이터를 덜 읽어오면, 커널은 얼마나 빨라질 수 있을까?**

1. 지난주의 `max(연산 시간, 메모리 시간)`을 복습한다.
2. 연산량을 이동량으로 나눈 산술 강도를 정의한다.
3. 가로축·세로축·두 상한·분기점을 표시한 Roofline 그림을 만든다.
4. 작은 행렬곱에서 같은 입력이 여러 출력에 쓰이는 모습을 찾는다.
5. 출력 타일 크기를 바꾸며 이동량과 이상적인 시간을 계산한다.
6. 온칩 저장 공간과 병렬성까지 고려해 계산의 한계를 설명한다.

공동 정리 결과로 **Roofline 그림, 타일 재사용 그림, 조건별 계산표, 연습문제 풀이**를 남기면 좋겠습니다. 다음 xPU 비교 주차에는 “성능 숫자를 비교할 때 연산 종류와 메모리 구간도 같아야 한다”는 관점을 이어갈 수 있습니다.

## 3. 시간 모델을 Roofline으로 바꾸기

| 기호 | 의미 | 단위 |
|---|---|---|
| `F` | 총 부동소수점 연산량 | FLOPs |
| `Q` | HBM에서 읽고 쓴 총 데이터량 | Byte |
| `P` | 해당 연산에 적용 가능한 최대 연산 처리율 | FLOPs/s |
| `BW` | HBM 대역폭 | Byte/s |
| `I = F/Q` | HBM 기준 산술 강도 | FLOP/Byte |
| `R` | 실행 중 달성한 연산 처리율 | FLOPs/s |

행렬 B와 혼동하지 않도록 대역폭에는 `BW`를 사용합니다. FLOPs는 연산 횟수, FLOPs/s는 초당 성능으로 구분합니다.

```text
T_compute = F / P
T_memory  = Q / BW
T_ideal   = max(F/P, Q/BW)

R_roof = F / T_ideal
       = min(P, BW × F/Q)
       = min(P, BW × I)

I_ridge = P / BW
```

시간은 두 값 중 큰 쪽이 기준이지만, **성능 상한은 두 값 중 작은 쪽**입니다. `BW × I`의 단위도 `Byte/s × FLOP/Byte = FLOPs/s`가 됩니다. 이 관계는 [Scaling Book의 시간 모델](https://jax-ml.github.io/scaling-book/roofline/#where-does-the-time-go)과 [Roofline 원 보고서 3절](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/Archive/EECS-2008-134.pdf)을 연결해 정리한 것입니다.

`I < I_ridge`이면 HBM 대역폭 쪽 상한이 낮고, `I > I_ridge`이면 연산 쪽 상한이 낮습니다. 두 값이 같으면 경계입니다. 이는 병목을 예상하는 모델이며 실제 성능이 반드시 그 상한에 도달한다는 뜻은 아닙니다.

### 그래프 읽기

![가상 하드웨어의 Roofline과 재사용 조건별 성능 상한](/assets/week04/roofline.png)

- 가로축은 HBM 기준 산술 강도, 세로축은 초당 연산 처리율입니다. 두 축 모두 로그 눈금입니다.
- 기울어진 구간은 `BW × I`, 수평 구간은 `P`가 정합니다.
- **점들은 각 구현에 대해 계산한 상한입니다. 실측점이 아닙니다.** 입력 재사용이 늘면 같은 F에서 Q가 줄어 오른쪽으로 이동하지만, 연산 상한을 넘을 수는 없습니다.
- 실제 측정점을 그릴 때는 같은 집계 기준의 `I = F/Q`, `R = F/실측시간`을 사용합니다. 상한보다 아래에 있는 정도와 추가 프로파일링을 함께 봐야 원인을 판단할 수 있습니다. [Nsight Compute의 그래프 설명](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts).

## 4. 공통 가정과 지난주 예제의 위치

이 문서에서는 아래 조건을 사용합니다.

- 가상 하드웨어: 적용 가능한 FP32 연산 성능 `P = 20 × 10¹² FLOPs/s`, HBM 대역폭 `BW = 10¹² Byte/s`.
- 입력·출력·누산값은 FP32, 원소당 4Byte입니다. BF16·희소 Tensor Core 피크를 가져오지 않습니다.
- 연산과 이동을 충분히 겹치고 필요한 병렬성을 확보할 수 있다고 가정합니다. 시작 비용·컴파일·CPU↔가속기 복사 시간은 제외합니다.
- 아래에 명시한 모든 HBM 읽기·쓰기만 셉니다. 캐시 적중·추가 전송·레지스터가 부족해 메모리로 내보내는 중간값 등은 제외합니다.
- `GB = 10⁹ Byte`, `TB = 10¹² Byte`, `KiB = 1024 Byte`입니다.

이 조건의 분기점은 `20 × 10¹² ÷ 10¹² = 20 FLOP/Byte`입니다.

지난주 FP32 Vector Add는 `F = N`, `Q = 12N`이므로 `I = 1/12 ≈ 0.08333 FLOP/Byte`입니다. Roofline 상한은 약 **0.08333 TFLOPs/s**로, 분기점 왼쪽에 있습니다. 단순히 처리할 원소를 묶는 것만으로는 입력 하나를 여러 출력에 재사용하게 되지 않으므로 이 산술 강도가 높아지지 않습니다.

아래의 행렬곱에서는 한 입력값이 여러 출력 계산에 쓰인다는 점이 달라집니다.

## 5. 작은 그림으로 Tiling 이해하기

`C = A @ B`에서 A·B·C가 모두 4×4라고 합시다. 먼저 왼쪽 위의 C 2×2 출력 타일을 정합니다.

![4×4 행렬곱에서 2×2 출력 타일을 두 단계로 누적하는 그림](/assets/week04/tiling.png)

```text
누산 타일을 0으로 초기화
첫 단계: A[0:2, 0:2] @ B[0:2, 0:2]를 누적
둘째 단계: A[0:2, 2:4] @ B[2:4, 0:2]를 누적
두 단계가 끝나면 C[0:2, 0:2]에 한 번 저장
```

A 타일의 한 원소는 출력의 두 열에, B 타일의 한 원소는 출력의 두 행에 쓰입니다. 중간 합을 온칩에 남기면 K 방향의 다음 조각을 처리할 때 HBM에 중간 합을 썼다가 다시 읽을 필요도 없습니다. 같은 흐름을 나머지 출력 타일 세 개에도 적용합니다. 블록 알고리즘의 근거는 [Triton의 첫 의사코드](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#motivations)입니다.

### 계산 예제 1 — 읽기와 쓰기를 직접 세기

입력값의 캐시 재사용은 없고, 타일 구현에서는 같은 출력 타일 안에서만 재사용한다고 가정합니다.

| 항목 | 매 곱셈마다 입력을 다시 읽기 | 2×2 출력 타일 안에서 재사용 |
|---|---:|---:|
| A·B 읽기 횟수 | 출력 16개 × K 4회 × 입력 2개 = 128원소 | 출력 타일 4개 × K 단계 2회 × 입력 8원소 = 64원소 |
| 입력 읽기 | 512 Byte | 256 Byte |
| C 쓰기 | 16원소 = 64 Byte | 16원소 = 64 Byte |
| 총 이동량 | **576 Byte** | **320 Byte** |
| 연산량 | **128 FLOPs** | **128 FLOPs** |
| 산술 강도 | `128/576 ≈ 0.2222` | `128/320 = 0.4` |

여기서는 각 출력에 대해 0에서 시작해 K번 곱하고 더하며 **FMA 1회 = 2 FLOPs**로 셉니다. 따라서 `F = 2N³`입니다. 첫 항은 덧셈 없이 대입한다고 세면 수학적으로는 `N²(2N−1)`이지만, GEMM 성능 계산에서 흔히 쓰는 전자의 관례로 통일합니다. [NVIDIA의 행렬곱 연산량 설명](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).

이 작은 예제는 이동량을 세기 위한 것입니다. 크기가 너무 작아 실제 실행에서는 시작 비용 등의 영향이 클 수 있으므로 Byte 비율을 그대로 실측 속도 향상이라고 해석하지 않습니다.

## 6. 계산 예제 2 — 타일 크기를 바꾸어 Roofline에 놓기

이제 `N = 4096`인 정사각 행렬곱을 계산합니다. 출력 타일의 한 변과 K 방향 조각의 폭을 모두 `t`로 두고, `t = 32, 64, 128`을 비교합니다.

각 출력 타일의 누산값은 전체 K를 처리할 때까지 온칩에 유지합니다. A·B는 해당 출력 타일 안에서만 재사용하며 서로 다른 출력 타일 사이의 캐시 재사용은 없습니다. `C = A @ B`이므로 기존 C를 읽지 않고 최종 결과만 씁니다. `N`은 모든 후보 타일 크기로 나누어떨어집니다.

### 연산량은 같고, 읽는 횟수가 달라진다

```text
F = 2N³

재사용 없음:
  Q_read  = 8N³        # 곱셈마다 FP32 입력 두 개
  Q_write = 4N²        # 출력 한 번 저장
  Q       = 8N³ + 4N²

타일 안에서 재사용:
  출력 타일 수       = (N/t)²
  출력 타일당 K 단계 = N/t
  단계당 입력 읽기   = 2 × t² × 4 Byte
  Q_read            = (N/t)² × (N/t) × 8t² = 8N³/t
  Q_write           = 4N²
  Q                 = 8N³/t + 4N²
  I                 = 2N³ / Q = Nt / (4N + 2t)
```

위 식은 이 문서의 재사용 조건에서 직접 센 값입니다. Shared Memory로 중복 입력 읽기를 줄이는 원리는 [CUDA의 행렬곱 사례](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)와 연결됩니다.

모든 경우 `F = 137,438,953,472 FLOPs`, 연산 시간은 약 **6.872 ms**입니다.

| 구현 | 총 HBM 이동량 Q (십진 GB) | I (FLOP/Byte) | 성능 상한 (TFLOPs/s) | 메모리 시간 (ms) | 이상적 시간 (ms) | 이 모델의 제한 자원 |
|---|---:|---:|---:|---:|---:|---|
| 재사용 없음 | 549.823 | 0.2500 | 0.2500 | 549.823 | **549.823** | HBM 대역폭 |
| t=32 | 17.247 | 7.9689 | 7.9689 | 17.247 | **17.247** | HBM 대역폭 |
| t=64 | 8.657 | 15.8760 | 15.8760 | 8.657 | **8.657** | HBM 대역폭 |
| t=128 | 4.362 | 31.5077 | 20.0000 | 4.362 | **6.872** | 연산 처리율 |

표는 반올림한 값입니다. 예를 들어 `t=64`의 정확한 Q는 `8,657,043,456 Byte`이며, 약 `8.657 ms`가 됩니다. `t=128`은 메모리 시간이 연산 시간보다 짧아져, 이동량을 더 줄여도 이 두 항만 있는 모델에서는 6.872 ms 아래로 내려가지 않습니다.

**549.823 GB는 필요한 메모리 용량이 아니라 누적 이동량**입니다. 세 행렬을 저장하는 공간은 `3 × 4N² = 201,326,592 Byte = 192 MiB`입니다. 같은 입력을 다시 읽으면 저장 공간보다 훨씬 많은 Byte가 이동할 수 있습니다.

### 이 표를 실제 GPU에 그대로 대입하면 안 되는 이유

“재사용 없음”은 모든 스칼라 입력 읽기가 HBM까지 도달한다는 비교 기준입니다. 실제 단순 커널도 캐시·브로드캐스트를 활용할 수 있습니다. 타일 구현 역시 L2 캐시, 접근 방식, 동기화, 온칩 자원 제약에 따라 달라집니다. 따라서 표의 차이를 실측 가속 배수로 주장하지 않습니다.

### 선택: 행렬 전체를 한 번씩 읽는 식과 왜 다를까?

모든 A·B를 한 번만 읽고 C를 한 번 쓴다는 이상적인 전체 재사용 가정이면 `Q = 12N²`, `I = N/6 ≈ 682.67`입니다. 하지만 여기서는 한 출력 타일의 계산이 끝난 뒤 다른 출력 타일이 같은 입력을 다시 읽으므로 `Q = 8N³/t + 4N²`입니다.

두 식은 재사용 가정이 다릅니다. 유한한 온칩 공간과 스케줄 때문에 실제 구현에서 전체 입력의 단 한 번 읽기가 가능한지 따져야 합니다. [Scaling Book의 Matrix multiplication](https://jax-ml.github.io/scaling-book/roofline/#matrix-multiplication)에서도 타일로 나누면 입력을 반복해서 읽는 효과를 설명합니다.

## 7. 큰 타일의 비용도 계산하기

위처럼 K 방향 폭도 `t`이고 한 단계의 입력만 보관한다면 다음 공간이 필요합니다.

| 출력 타일 | A·B 입력 버퍼 합 `8t²` | FP32 누산값 `4t²` |
|---|---:|---:|
| 32×32 | 8 KiB | 4 KiB |
| 64×64 | 32 KiB | 16 KiB |
| 128×128 | 128 KiB | 64 KiB |

입력 버퍼는 Shared Memory 등에, 누산값은 레지스터 등에 유지하는 단순한 구현을 생각한 계산입니다. 누산값 열은 **타일 전체의 논리적 저장량**이며 스레드 하나의 레지스터 사용량이 아닙니다. 임시값·주소 계산·추가 파이프라인 버퍼까지 포함한 실제 할당량은 더 클 수 있습니다.

출력 타일이 크면 입력 재사용이 늘어나는 대신 저장 공간을 더 사용하고, 출력 타일의 총수도 줄어듭니다. 동시에 실행할 수 있는 작업이 부족해지거나 레지스터가 부족하면 기대한 성능이 나오지 않을 수 있습니다. [CUDA의 Occupancy 설명](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy), [NVIDIA의 타일 크기와 병렬성 비교](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).

**128×128 출력 타일이 CUDA 스레드 16,384개를 뜻하지는 않습니다.** 여러 출력값을 한 스레드가 맡거나, 여러 스레드가 협력하는 방식은 구현에서 별도로 정합니다. 이번 주에는 출력 타일·데이터 조각·스레드 배치를 구분하는 데까지 집중합니다.

## 8. 연습문제

### 문제 1 — 어느 상한을 바꾸면 빨라질까?

어떤 커널의 연산량이 `10¹² FLOPs`, HBM 이동량이 `100 GB`라고 합시다. 공통 가상 하드웨어에서 산술 강도, Roofline 상한, 이상적 시간을 구하세요.

연산 성능만 두 배로 늘리는 경우와 HBM 대역폭만 두 배로 늘리는 경우를 각각 계산하고, 원래 그래프에서 무엇이 바뀌는지 설명해보세요.

### 문제 2 — 입력 버퍼에 들어가는 타일은?

예제 2의 32·64·128 후보를 비교합니다. 한 작업에 허용되는 **A·B 입력 버퍼 합이 96 KiB**이고, K 방향 폭도 `t`이며 한 단계만 보관한다고 합시다. 누산값은 별도 공간에 둘 수 있다고 가정합니다.

입력 버퍼 조건을 통과하는 가장 큰 후보와 그 후보의 이상적 시간을 구하세요. “조건을 통과한 가장 큰 타일이 실제로도 가장 빠르다”고 결론 내릴 수 있을까요?

### 문제 3 — 선택 확장: K 방향 폭만 늘리면?

예제 2의 행렬 크기는 그대로 두고, 출력 타일은 **64×64로 고정**합니다. K 방향에서 한 번에 읽는 폭 `BK`를 32와 64로 바꿉니다.

두 경우의 K 단계 수, 단계당 입력 버퍼, 전체 HBM 이동량을 구하세요. 입력 버퍼가 두 배가 되면 HBM 산술 강도도 두 배가 될까요?

## 9. 해답과 해설

<details class="t2s-solution-toggle">
<summary>💡 연습문제 해답 및 상세 풀이 펼치기 (스스로 먼저 풀어본 뒤 클릭하세요)</summary>

### 문제 1 해답

```text
I = 10¹² / (100 × 10⁹) = 10 FLOP/Byte
R_roof = min(20 × 10¹², 10¹² × 10) = 10 TFLOPs/s
T_compute = 50 ms
T_memory = 100 ms
T_ideal = 100 ms
```

연산 성능만 두 배이면 연산 시간은 25 ms지만 메모리 시간은 100 ms이므로 결과는 **100 ms**입니다. 수평 상한이 40 TFLOPs/s로 높아지고 분기점은 40으로 이동하지만, `I=10`의 상한은 그대로입니다.

대역폭만 두 배이면 메모리 시간이 50 ms로 줄어 **50 ms**가 됩니다. 기울어진 선이 위로 이동하고 분기점은 10으로 옮겨집니다. 새 상한은 20 TFLOPs/s이며 이 커널은 두 시간이 같은 경계에 놓입니다.

### 문제 2 해답

필요한 입력 버퍼는 각각 **8·32·128 KiB**입니다. 96 KiB 조건을 통과하는 가장 큰 후보는 **64×64**, 이상적 시간은 약 **8.657 ms**입니다.

128×128은 이번 입력 버퍼 조건을 넘습니다. 이 결론은 K 방향 폭도 128로 둔 현재 구성에 관한 것입니다. K 방향 폭이나 버퍼 구성을 바꾸면 다시 계산해야 합니다. 또한 버퍼 조건을 통과하더라도 레지스터 사용량·동기화·병렬성 때문에 실제 최적 타일은 달라질 수 있습니다.

### 문제 3 해답

| 항목 | BK=32 | BK=64 |
|---|---:|---:|
| 출력 타일당 K 단계 수 `4096/BK` | 128 | 64 |
| 단계당 A·B 입력 `4 × (64BK + BK64)` | 16 KiB | 32 KiB |
| 출력 타일당 누산값 | 16 KiB | 16 KiB |
| 전체 HBM 이동량 | 8,657,043,456 Byte | 8,657,043,456 Byte |
| HBM 기준 산술 강도 | 약 15.8760 | 약 15.8760 |
| 이상적 시간 | 약 8.657 ms | 약 8.657 ms |

한 단계에 읽는 양은 두 배지만 단계 수가 절반이므로 전체 이동량은 같습니다. 일반 출력 타일 `BM×BN`에 대해, FP32·나머지 없음·동일한 재사용 조건이면 다음과 같습니다.

```text
전체 입력 읽기
 = (M/BM) × (N/BN) × (K/BK) × 4BK(BM + BN)
 = 4MNK × (1/BN + 1/BM)
전체 출력 쓰기 = 4MN
```

이 식에서는 `BK`가 소거됩니다. **출력 타일을 키우는 것과 K 방향 조각의 폭만 키우는 것은 다릅니다.** 다만 BK는 버퍼 크기·반복 횟수·명령 구성에 영향을 주므로 실제 시간까지 같다고 주장할 수는 없습니다. 세 타일 차원의 구분은 [Triton 블록 알고리즘](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#motivations)과 연결해보세요.

</details>

## 10. 공동 검토와 다음 실습에 남길 질문

- 산술 강도의 분모를 저장 용량이 아니라 **해당 메모리 구간의 읽기·쓰기 이동량**으로 셌는가?
- FLOPs는 같은데 Q가 달라지는 이유를 타일 그림에서 가리킬 수 있는가?
- `max` 시간 모델과 `min` 성능 모델이 같은 뜻임을 설명할 수 있는가?
- 성능 상한, 이상적 시간, 실제 측정값을 구분했는가?
- 같은 입력의 캐시 적중과 출력 타일 사이 재사용을 어떻게 가정했는가?
- 타일 크기·K 방향 폭·스레드 수를 구분했는가?

뒤의 커널 실습에서는 이 계산표 옆에 실제 실행시간과 HBM 트래픽을 나란히 놓고 차이를 설명하면 좋겠습니다. 이번 자료의 수치만으로 특정 GPU에서의 최적 타일 크기나 가속 배수를 확정하지 않습니다.

이 문서의 예제·해답은 독립 계산으로 검산했으며, 그림의 점도 같은 가상 조건으로 생성했습니다. 실제 GPU 커널의 실행·프로파일링은 포함하지 않았습니다.
