<WeekNavBar :week="3" topic="커널의 시간은 어디에서 소비되는가" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송무복)', 'badge': 'GPU·TPU 메모리 계층 비교'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': '손계산 & 병목 판별'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': 'SRAM vs DRAM 물리 구조'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '수식 모델링 & 정량 분석'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': '메모리 계층 & 실전 예제'}]" />

> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **singon (김신곤)** (`손계산 & 병목 판별`)
>
> 💡 **핵심 기여 요약**: Vector Add와 다단계 원소별 연산에서 HBM 읽기/쓰기 바이트 수를 직접 손계산하고 Kernel Fusion의 시간 절감 효과를 도출했습니다.

---

# Week 3. GPU 커널의 실행 시간과 메모리 병목 분석

## 1. 서론

GPU는 많은 연산을 병렬로 처리하지만, 높은 연산 성능이 모든 커널의 빠른 실행을 보장하지는 않는다. 계산을 수행하려면 입력을 가져와야 하고, 계산이 끝나면 결과를 저장해야 한다. 따라서 커널의 실행 시간을 이해하려면 계산 자체에 필요한 시간과 데이터를 읽고 쓰는 데 필요한 시간을 함께 살펴봐야 한다.

본 보고서는 2주차에 다룬 GPU 구조와 하드웨어 제약을 바탕으로, 한 GPU 내부에서 발생하는 연산 병목과 메모리 병목을 분석한다. 먼저 연산량과 처리량의 차이를 통해 실행 시간의 추정 방법을 정리하고, GPU의 실행 구조와 메모리 계층이 이 시간에 어떤 영향을 주는지 설명한다. 마지막으로 FP32 벡터 덧셈(Vector Add)의 연산량과 데이터 이동량을 계산하여, 연산 성능·메모리 대역폭·메모리 용량 중 어떤 요소가 성능 개선에 직접적으로 기여하는지 확인한다.

## 2. Compute-bound와 Communication-bound: 실행 시간은 무엇이 결정하는가?

### 2.1. 연산량과 연산 처리량

- FLOPs: 수행해야 하는 부동소수점 연산의 총횟수 (작업의 크기)
- FLOPs/s: 하드웨어가 초당 처리할 수 있는 연산 횟수 (처리 속도)

두 값을 나누면 해당 작업의 연산 시간을 얻을 수 있다.
총 연산량(Flops)을 $F$, 해당 연산에 대한 하드웨어 처리량 (Accelerator Flops/s)을 $P$라고 할 때, 이상적인 연산 시간은 다음과 같다.

$$
T_{\mathrm{compute} }=\frac{F}{P},
\qquad
\frac{\mathrm{FLOPs} }{\mathrm{FLOPs/s} }=\mathrm{s}
$$

예를 들어 길이가 $N$인 두 벡터를 더하면 각 위치에서 덧셈 한 번을 수행하므로 총 연산량은 $N$ FLOPs이다. 입력을 읽고 결과를 쓰는 동작은 이 덧셈 횟수에 포함하지 않고 별도의 데이터 이동량으로 계산한다. 또한 FP32 덧셈을 분석할 때에는 FP32 덧셈 처리량을 사용해야 한다. 행렬곱을 위한 Tensor Core의 최대 처리량을 적용하면 연산 시간을 잘못 추정하게 된다. [Scaling Book의 시간 모델](https://jax-ml.github.io/scaling-book/roofline/#where-does-the-time-go)과 [NVIDIA의 연산 장치 설명](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)은 이러한 구분의 근거가 된다.

### 2.2. 데이터 이동 시간과 병목의 판별

연산 장치가 입력을 소비하는 속도보다 메모리가 데이터를 공급하는 속도가 느리면, 계산 능력이 충분해도 실행 시간이 길어진다. 총 데이터 이동량을 $Q$, 해당 경로의 대역폭을 $W$라고 할 때 이동 시간은 다음과 같이 추정한다.

$$
T_{\mathrm{memory} }=\frac{Q}{W},
\qquad
\frac{\mathrm{bytes} }{\mathrm{bytes/s} }=\mathrm{s}
$$

여기서 $Q$에는 입력 읽기와 결과 쓰기가 모두 포함된다. Communication은 GPU 사이의 통신까지 포함하는 넓은 개념이지만, 본 보고서는 HBM과 GPU 내부 사이의 데이터 이동에 집중한다. 따라서 이 경로의 대역폭이 성능을 제한하는 경우를 memory-bound, 정확히는 memory-bandwidth-bound라고 부른다.

연산과 데이터 이동을 충분히 겹쳐 실행할 수 있다면, 이상적인 실행 시간은 두 시간 중 큰 값으로 추정한다.

$$
T_{\mathrm{ideal} }\approx
\max\left(T_{\mathrm{compute} },T_{\mathrm{memory} }\right)
$$

$T_{\mathrm{compute} }$가 더 크면 compute-bound이며, 연산량을 줄이거나 연산 처리량을 높이는 것이 주요 개선 방향이다. 반대로 $T_{\mathrm{memory} }$가 더 크면 memory-bound이며, 데이터 이동량을 줄이거나 대역폭을 높이는 것이 중요하다. 이는 [Scaling Book의 Compute-bound / Communication-bound 구분](https://jax-ml.github.io/scaling-book/roofline/#where-does-the-time-go)에 해당한다.

이 추정에 최대 처리량과 최대 대역폭을 사용하면 최소 실행시간의 기준을 얻는다. 실제 커널에는 실행 준비, 동기화, 명령 처리 등의 비용이 있고 최대 성능을 항상 활용하지도 못하므로, 추정값과 측정값을 구분해야 한다. 연산과 이동이 전혀 겹치지 않는 단순 모델에서는 두 시간을 더하지만, 이 합 역시 실제 실행 시간의 절대적인 상한은 아니다.

## 3. GPU Acceleration — GPU Architecture

### 3.1. 병렬 연산을 수행하는 구조

GPU는 여러 SM(Streaming Multiprocessor)에 작업을 배분하여 많은 스레드를 병렬로 실행한다. SM 내부에는 산술 연산을 수행하는 CUDA Core, 행렬 곱셈·누산을 가속하는 Tensor Core, 실행할 명령을 선택하는 Warp Scheduler 등이 있다. CUDA Core와 Tensor Core는 계산을 담당하고, Warp Scheduler는 실행을 조율하는 제어 장치이다. [MLC의 GPU Architecture](https://book.mlc.ai/chapter_gpu_acceleration/part1.html#gpu-architecture)와 [Scaling Book의 GPU 구조 설명](https://jax-ml.github.io/scaling-book/gpus/#what-is-a-gpu)은 이러한 구성요소의 관계를 설명한다.
- 머신러닝 컴파일(MLC)은 머신러닝 실행을 개발 형태에서 배포 형태로 변환하고 최적화하는 프로세스
    - 개발 형식은  머신러닝 모델을 개발할 때 사용하는 요소들의 집합을 의미합니다. 일반적인 개발 형식에는 PyTorch, TensorFlow, JAX와 같은 일반적인 프레임워크로 작성된 모델 설명과 해당 모델에 연결된 가중치가 포함됩니다.
    - 배포 형태는  머신 러닝 애플리케이션을 실행하는 데 필요한 요소들의 집합을 의미합니다. 일반적으로 머신 러닝 모델의 각 단계를 지원하기 위해 생성된 코드, 리소스(예: 메모리)를 관리하는 루틴, 그리고 애플리케이션 개발 환경과의 인터페이스(예: 안드로이드 앱용 Java API) 등이 포함됩니다.

프로그램의 스레드들은 블록으로 묶이며, 각 블록은 하나의 SM에 배치된다. 자원이 허용되면 여러 블록이 같은 SM에 상주할 수 있다. NVIDIA GPU에서는 스레드 32개가 하나의 워프를 구성하며, 프로그램의 스레드가 물리적인 CUDA Core 하나에 고정적으로 대응하는 것은 아니다. 한 워프가 메모리 응답을 기다리는 동안 다른 준비된 워프의 명령을 실행하면 대기 시간의 일부를 숨길 수 있다. [NVIDIA GPU Execution Model](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)과 [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)을 참고하였다.

다만 여러 작업을 동시에 실행하는 것과 메모리 대역폭 자체를 늘리는 것은 다르다. 충분한 병렬성으로 대기 시간을 숨기더라도, HBM에서 공급할 수 있는 초당 데이터의 양에는 한계가 있다. 따라서 GPU의 병렬 구조를 이해하는 것만으로는 커널 성능을 설명할 수 없으며, 데이터 이동도 함께 분석해야 한다.

### 3.2. Vector Add의 읽기·계산·쓰기

MLC의 첫 Vector Add 예제는 다음 연산을 수행한다.

$$
C[i]=A[i]+B[i]\qquad(0\le i<N)
$$

각 출력은 같은 위치의 입력 두 개에만 의존하므로 여러 원소를 독립적으로 처리할 수 있다. 원소 하나를 처리하는 과정은 다음과 같다.

| 단계 | 동작 | FP32 기준 비용 | BF16 기준 비용 |
| --- | --- | --- | --- |
| 읽기 | $A[i]$와 $B[i]$를 가져온다 | $4+4=8$ bytes | $2+2=4$ bytes | 
| 계산 | 두 값을 더한다 | 1 FLOP | 1 FLOP |
| 쓰기 | 결과를 $C[i]$에 저장한다 | 4 bytes |2 bytes |

따라서 이 연산은 원소당 덧셈 한 번에 대해 모두 FP32로 계산할 때 총 12 bytes의 이동이 필요하다. 계산은 단순하지만 입력과 출력의 이동은 반드시 발생한다. 이 특성이 5절에서 메모리 병목이 나타나는 이유를 설명한다. 예제의 연산 정의는 [MLC — GPU Architecture의 Vector Add](https://book.mlc.ai/chapter_gpu_acceleration/part1.html#gpu-architecture)를 따랐다.

- 여러 블록을 계산할 때 T.grid(128, 128, 128)는 TensorIR에서 여러 개의 중첩된 반복자를 작성할 수 있도록 해주는 구문적 편의 기능입니다.

```
# TensorIR
for i, j, k in T.grid(128, 128, 128):

# numpy
for i in range(128):
    for j in range(128):
        for k in range(128):
```

## 4. How to Think About GPUs — Memory

### 4.1. 대용량 메모리와 작은 고속 메모리가 함께 필요한 이유

HBM(High Bandwidth Memory)은 여러 DRAM 다이를 적층하고 넓은 인터페이스로 높은 대역폭을 제공하는 DRAM의 한 종류이다. GPU에서는 큰 텐서를 저장하는 주 메모리로 사용된다. 반면 칩 내부의 캐시와 shared memory에는 빠르게 접근할 수 있는 SRAM이 사용된다. SRAM은 DRAM보다 비트당 면적 부담이 크므로, 모든 데이터를 연산 장치 가까이에 같은 방식으로 저장하기는 어렵다. 이에 따라 GPU는 대용량 저장 공간과 작은 고속 저장 공간을 계층적으로 구성한다. [Micron의 HBM 설명](https://www.micron.com/about/blog/applications/data-center/hbm2e-fastest-memory-modern-data-center)과 [Scaling Book — Memory](https://jax-ml.github.io/scaling-book/gpus/#memory)를 바탕으로 정리하였다.

| 저장 공간 | 위치·접근 범위 | 역할 |
| --- | --- | --- |
| HBM | GPU 연산 칩 외부의 주 메모리 | 가중치·활성화 등 큰 텐서 저장 |
| L2 Cache | 칩 내부, 여러 SM이 공유 | 반복되는 HBM 접근 감소 |
| L1 Cache | 각 SM 내부 | 가까운 위치에서 데이터 접근을 지원하는 하드웨어 관리 캐시 |
| Shared Memory | SM 내부, 보통 같은 블록의 스레드들이 사용 | 프로그램이 명시적으로 관리하는 데이터 공유 공간 |
| Registers | SM 내부, 스레드별 상태 저장 | 연산의 피연산자와 중간 결과 보관 |

아래 그림은 저장 공간의 위치와 관계를 단순화한 것이다. 모든 메모리 접근이 나열된 공간을 순서대로 통과한다는 뜻은 아니다.

~~~text
HBM — 적층 DRAM, 큰 텐서 저장
  ↕
GPU 칩 내부
  L2 Cache — SRAM, 여러 SM이 공유
    ↕
  SM
    ├─ L1 Cache — SRAM, 자동 캐싱
    ├─ Shared Memory — SRAM, 명시적 데이터 공유
    └─ Registers ↔ 연산 장치
~~~

특히 L1 Cache와 shared memory는 구분해야 한다. 일부 GPU에서는 물리적인 저장 자원을 나누어 사용하지만, L1은 하드웨어가 캐시로 관리하고 shared memory는 프로그램이 활용 방식을 정한다. Vector Add가 반드시 shared memory를 거쳐야 하는 것은 아니다. 메모리 공간의 접근 범위와 용도는 [MLC — Memory Spaces](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html#memory-spaces)에서도 확인할 수 있다.

이 계층은 자주 쓰는 데이터를 연산 장치 가까이에 두어 HBM 접근을 줄이는 데 의미가 있다. 다만 이번 Vector Add 분석은 캐시 재사용이 없고 입력·출력이 HBM을 한 번씩 오간다는 조건이므로, HBM 기준 이동량을 그대로 계산한다.

### 4.2. 메모리 용량과 메모리 대역폭의 차이

메모리 용량이 16 GB라는 것은 동시에 저장할 수 있는 데이터의 양을 뜻한다. 반면 대역폭이 1 TB/s라는 것은 초당 이동시킬 수 있는 데이터의 양을 뜻한다. 용량은 작업에 필요한 텐서를 담을 수 있는지 판단할 때 사용하고, 대역폭은 읽고 쓰는 데 걸리는 시간을 계산할 때 사용한다. 이 보고서에서 GB와 TB는 각각 $10^9$, $10^{12}$ bytes이다.

예를 들어 FP32 벡터 A·B·C를 각각 $N$개 원소로 할당하면 필요한 저장 공간은 $4\times3N$ bytes이다. 실제 수용 가능 여부는 여기에 다른 텐서와 임시 버퍼의 공간도 합쳐 판단한다. 한편 커널의 이동량은 각 데이터를 읽고 쓴 횟수로 결정된다. 같은 입력을 여러 번 읽으면 저장 공간은 같아도 총 이동량은 커질 수 있다.

따라서 메모리가 충분한 상황에서 용량만 늘린다고 커널이 빨라지는 것은 아니다. 실행 속도를 논하려면 대역폭과 실제 이동량을 별도로 확인해야 한다. 이러한 구분은 [NVIDIA — GPU Architecture Fundamentals / Understanding Performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)의 성능 분석에 기초한다.

## 5. Vector Add의 실행 시간 분석

### 5.1. 분석 조건

길이 $N=100{,}000{,}000=10^8$인 FP32 벡터 A와 B를 더해 새 벡터 C에 저장한다. 원소당 크기는 4 bytes이며, 가상 하드웨어의 FP32 덧셈 처리량과 HBM 대역폭은 다음과 같다.

$$
P=20\times10^{12}\;\mathrm{FLOPs/s},
\qquad
W=1\times10^{12}\;\mathrm{bytes/s}
$$

입력은 이미 HBM에 저장되어 있고 A와 B를 각각 한 번 읽은 뒤 C를 한 번 쓴다고 가정한다. 캐시 재사용과 추가 데이터 이동은 제외한다. 다음 계산은 이 가상 조건에 대한 손계산이며, 실제 GPU의 측정 결과가 아니다.

### 5.2. 연산량과 데이터 이동량

각 원소에서 덧셈 한 번을 수행하므로 덧셈 횟수와 총 FLOPs는 모두 $10^8$이다. 데이터 이동량에는 두 입력 읽기와 한 출력 쓰기를 포함한다.

| 항목 | 계산 | 결과 |
| --- | --- | ---: |
| A 읽기 | $4\times10^8$ bytes | 400 MB |
| B 읽기 | $4\times10^8$ bytes | 400 MB |
| 입력 읽기 합계 | $8\times10^8$ bytes | 800 MB |
| C 쓰기 | $4\times10^8$ bytes | 400 MB |
| 총 이동량 $Q$ | $12\times10^8$ bytes | 1.2 GB |
| 총 연산량 $F$ | $1\times10^8$ FLOPs | 100,000,000 FLOPs |

이 조건에서 A·B·C가 차지하는 저장 공간도 총 1.2 GB이다. 따라서 사용 가능한 용량이 16 GB라면 세 벡터를 담을 수 있다. 저장 공간과 이동량이 같은 수치인 것은 각 입력을 한 번씩 읽고 출력을 한 번 쓰기 때문이며, 두 개념이 같기 때문은 아니다.

### 5.3. 예상 실행 시간과 병목

연산 시간은 총 덧셈 횟수를 FP32 덧셈 처리량으로 나누어 구한다.

$$
T_{\mathrm{compute} }
=\frac{10^8}{20\times10^{12} }
=5\times10^{-6}\;\mathrm{s}
=0.005\;\mathrm{ms}
$$

메모리 시간은 읽고 쓴 총 데이터 크기를 HBM 대역폭으로 나누어 구한다.

$$
T_{\mathrm{memory} }
=\frac{1.2\times10^9}{1\times10^{12} }
=1.2\times10^{-3}\;\mathrm{s}
=1.2\;\mathrm{ms}
$$

따라서 연산과 데이터 이동이 충분히 겹치는 경우의 이상적인 실행 시간은 다음과 같다.

$$
T_{\mathrm{ideal} }
\approx\max(0.005,\;1.2)\;\mathrm{ms}
=1.2\;\mathrm{ms}
$$

메모리 시간이 연산 시간의 240배이므로 이 커널은 **memory-bandwidth-bound**이다. 이는 GPU에 덧셈을 처리할 능력이 부족해서가 아니라, 필요한 데이터를 읽고 쓰는 시간이 상대적으로 길기 때문이다. 결과 쓰기를 제외했다면 메모리 시간을 0.8 ms로 계산하게 되므로, 출력 이동까지 포함하는 것이 중요하다.

### 5.4. 하드웨어 변경에 따른 성능 비교

동일한 작업에서 연산 처리량, 메모리 대역폭, 메모리 용량을 각각 두 배로 늘리는 경우를 비교하였다. 다른 조건은 그대로이며, 메모리 용량은 처음부터 세 벡터를 담기에 충분하다고 가정한다.

| 변경 조건 | 연산 시간 | 메모리 시간 | 이상적 실행 시간 | 기준 대비 속도 |
| --- | ---: | ---: | ---: | ---: |
| 기준 | 0.005 ms | 1.2 ms | 1.2 ms | 1배 |
| 연산 처리량만 2배 | 0.0025 ms | 1.2 ms | 1.2 ms | 1배 |
| 메모리 대역폭만 2배 | 0.005 ms | 0.6 ms | 0.6 ms | 2배 |
| 메모리 용량만 2배 | 0.005 ms | 1.2 ms | 1.2 ms | 1배 |

연산 처리량을 높여도 더 큰 값인 메모리 시간이 유지되므로 예상 실행 시간은 변하지 않는다. 메모리 대역폭을 두 배로 늘리면 이동 시간이 절반으로 줄고, 여전히 메모리 시간이 연산 시간보다 크므로 전체 시간도 절반이 된다. 용량 증가는 이미 데이터를 담을 수 있는 이번 조건에서 직접적인 속도 향상을 만들지 않는다.

이는 계산량이 적은 커널일수록 연산 장치의 최대 성능보다 데이터 이동 비용이 중요할 수 있음을 보여준다. NVIDIA 역시 요소별 연산과 여러 활성화 연산에서 메모리 대역폭이 주요 제약이 될 수 있다고 설명한다. [Memory-Limited Layers — Activations](https://docs.nvidia.com/deeplearning/performance/dl-performance-memory-limited/index.html)

## 6. 결론

GPU 커널의 실행 시간은 연산량만으로 결정되지 않는다. GPU는 여러 SM과 스레드를 통해 계산을 병렬로 수행하지만, 연산에 필요한 데이터를 공급하고 결과를 저장하는 속도에는 별도의 한계가 있다. HBM과 온칩 메모리 계층은 이 데이터 공급을 담당하며, 메모리 용량과 대역폭은 각각 작업의 수용 가능성과 데이터 이동 속도를 결정한다.

본 보고서의 Vector Add는 총 $10^8$ FLOPs와 1.2 GB의 데이터 이동을 요구한다. 주어진 조건에서 연산 시간은 0.005 ms, 메모리 시간은 1.2 ms이며, 이상적인 실행 시간은 1.2 ms로 추정된다. 이 결과는 연산 처리량만 높여서는 성능이 개선되지 않고, 메모리 대역폭을 높여야 직접적인 개선이 나타나는 이유를 설명한다.

따라서 커널의 성능을 분석할 때에는 먼저 연산 횟수와 읽고 쓰는 데이터의 양을 각각 계산하고, 두 시간 중 어느 쪽이 큰지 확인해야 한다. 이 과정을 통해 병목에 맞는 개선 방향을 선택할 수 있으며, 추정에 사용한 가정과 실제 측정 조건을 비교하여 분석을 구체화할 수 있다.
