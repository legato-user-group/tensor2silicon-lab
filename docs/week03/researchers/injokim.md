<WeekNavBar :week="3" topic="커널의 시간은 어디에서 소비되는가" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송민배)', 'badge': 'GPU·TPU 메모리 계층 비교'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': '손계산 & 병목 판별'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': 'SRAM vs DRAM 물리 구조'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '수식 모델링 & 정량 분석'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': '메모리 계층 & 실전 예제'}]" />

> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **injokim (김인조)** (`수식 모델링 & 정량 분석`)
>
> 💡 **핵심 기여 요약**: 커널 실행 시간 분해 모델(Launch Overhead, Compute Time, Memory Time)과 실효 대역폭(Effective Bandwidth) 수식을 체계적으로 정립했습니다.

---

# 커널의 시간은 어디에서 소비되는가

2026년 9월 29일 화요일

커널의 성능을 이해하기 위해서는 크게 세 가지를 구분해서 볼 필요가 있습니다.

1. **얼마나 많이 계산해야 하는가?**
2. **얼마나 많은 데이터를 옮겨야 하는가?**
3. **그 데이터를 저장할 공간이 충분한가?**

이를 Workload/Algorithm, 실제 실행, Hardware limit 관점으로 나누면 아래와 같습니다.

| 구분 | Workload / Algorithm | 실제 실행 | HW Limit |
| --- | --- | --- | --- |
| 연산량 관련 | FLOPs | Achieved FLOP/s | Peak FLOP/s |
| 메모리 관련 | Algorithmic Bytes | Actual Traffic, Achieved/Actual Memory Throughput | Peak Bandwidth |
| HBM 용량 관련 | Memory Footprint | Allocated Memory | Memory Capacity |

---

## 1. FLOPs, FLOP/s, TOPS

FLOPs는 GPU가 수행해야 하는 **연산량**을 나타내는 수치입니다. `Floating Point Operations`의 약자로, 부동소수점 연산이 몇 번 발생하는지를 나타냅니다.

예를 들어 `(3, 2)` 크기의 A 행렬과 `(2, 3)` 크기의 B 행렬을 곱한다고 가정하겠습니다.

`Output[0][0]`을 계산하기 위해서는

```text
A[0][0] * B[0][0] + A[0][1] * B[1][0]
```

을 계산해야 합니다.

출력 하나를 만들기 위해 실제 scalar operation 기준으로 곱셈 2번과 덧셈 1번이 사용되므로, 출력 원소 9개를 모두 계산하면 27번의 연산이 발생합니다.

다만 GPU 성능 분석에서는 일반적으로 GEMM의 연산량을

$$
\mathrm{FLOPs} = 2 \times M \times N \times K
$$

로 계산합니다.

이는 곱셈과 덧셈으로 이루어진 **FMA(Fused Multiply-Add)** 한 번을 2 FLOPs로 계산하기 때문입니다.

따라서 위 예제에서는

$$
2 \times 3 \times 3 \times 2 = 36\ \mathrm{FLOPs}
$$

가 됩니다.

FLOPs는 알고리즘 관점에서 **해야 할 계산의 양**이고, FLOP/s는 **1초 동안 얼마나 많은 부동소수점 연산을 처리할 수 있는가**를 나타냅니다.

- **Peak FLOP/s**: 하드웨어가 이론적으로 제공할 수 있는 최대 연산 처리량
- **Achieved FLOP/s**: 실제 커널이 달성한 연산 처리량

NPU에서는 이와 비슷하게 TOPS(Tera Operations Per Second)를 자주 사용합니다. TOPS 자체는 특정 datatype을 뜻하지 않으므로 일반적으로 INT8 TOPS, INT4 TOPS와 같이 datatype을 함께 봅니다.

Mixed Precision workload라면 INT8 연산은 INT8 처리량 기준으로, FP16이나 FP32 연산은 해당 floating-point throughput 기준으로 따로 보는 것이 정확합니다.

---

## 2. Algorithmic Bytes와 Actual Memory Traffic

GPU가 연산을 처리하기 위해서는 데이터를 메모리에서 읽어오고, 계산 결과를 다시 써야 합니다.

GPU의 memory hierarchy는 단순화하면 다음과 같이 볼 수 있습니다.

```text
Register
   ↕
L1 / Shared Memory
   ↕
L2 Cache
   ↕
HBM / DRAM
```

알고리즘만 보고 **이 연산을 수행하기 위해 논리적으로 필요한 데이터 이동량**을 계산할 수 있는데, 이를 여기서는 **Algorithmic Bytes**라고 부르겠습니다.

예를 들어

```python
C = A + B
```

에서 A, B, C가 각각 FP32 원소 100개를 가진다면

```text
A Read  = 100 × 4 = 400 Bytes
B Read  = 100 × 4 = 400 Bytes
C Write = 100 × 4 = 400 Bytes
```

이므로

$$
\text{Algorithmic Bytes} = 1200\ \mathrm{Bytes}
$$

입니다.

반면 실제 GPU에서는 cache line, memory transaction, cache miss, 중복 load 등으로 인해 실제 이동량이 더 커질 수 있습니다.

예를 들어

```text
Algorithmic Bytes   = 1200 Bytes
Actual DRAM Traffic = 1500 Bytes
```

가 될 수 있습니다.

즉,

- **Algorithmic Bytes**: 원래 옮겨야 하는 데이터의 양
- **Actual Traffic**: 실제 하드웨어에서 이동한 데이터의 양

입니다.

---

## 3. Effective Bandwidth, Actual Throughput, Peak Bandwidth

Bandwidth는 기본적으로

$$
\text{Bandwidth} = \frac{\mathrm{Bytes} }{\text{Execution Time} }
$$

입니다.

다만 어떤 Bytes를 넣느냐에 따라 의미가 달라집니다.

### Effective Bandwidth

$$
\text{Effective Bandwidth}
=
\frac{\text{Algorithmic Bytes} }{\text{Execution Time} }
$$

즉,

> 내가 원래 처리하려던 유용한 데이터를 얼마나 빠르게 처리했는가?

를 나타냅니다.

### Actual / Achieved Memory Throughput

$$
\text{Actual Memory Throughput}
=
\frac{\text{Actual Traffic} }{\text{Execution Time} }
$$

즉,

> 실제 memory interface를 통해 데이터가 얼마나 빠르게 이동했는가?

를 나타냅니다.

예를 들어

```text
Algorithmic Bytes   = 500 GB
Actual DRAM Traffic = 800 GB
Execution Time      = 1 sec
Peak DRAM Bandwidth = 1000 GB/s
```

라면

```text
Effective Bandwidth      = 500 GB/s
Actual Memory Throughput = 800 GB/s
```

입니다.

GPU의 DRAM은 Peak 대비 80% 수준으로 동작하고 있지만, 실제 유용한 데이터는 500GB뿐입니다. 따라서 실제 throughput이 높다고 해서 무조건 잘 최적화된 것은 아닙니다.

Peak Bandwidth는 하드웨어가 이론적으로 제공할 수 있는 최대 데이터 전송률입니다.

예를 들어

```text
DRAM Traffic        = 100 GB
Kernel Time         = 0.1 sec
Peak DRAM Bandwidth = 1.5 TB/s
```

이면

$$
\text{Achieved DRAM Bandwidth} = 1\ \mathrm{TB/s}
$$

이고 Peak 대비 약 66.7%입니다.

다만 이것만 보고 개선 여지가 33%라고 단정할 수는 없습니다. 커널이 Compute-bound이거나 latency 등 다른 요인에 의해 제한될 수도 있기 때문입니다.

---

## 4. Memory Level과 HBM / DRAM / SRAM

Memory Traffic과 Bandwidth는 항상 **어느 memory level의 값인지**를 확인해야 합니다.

예를 들어 다음 값은 서로 다를 수 있습니다.

```text
L1 Traffic
L2 Traffic
DRAM Traffic
L1 Throughput
L2 Throughput
DRAM Throughput
```

Cache hit가 발생하면 L1이나 L2에서 처리되고 HBM까지 내려가지 않기 때문입니다.

또한 HBM은 **DRAM의 한 종류**입니다.

GPU 내부의 Cache, Shared Memory 등은 SRAM 기반으로 구현되는 경우가 많습니다. SRAM은 빠르지만 면적이 크고 비싸기 때문에 용량을 크게 만들기 어렵습니다.

따라서 GPU는 다음과 같이 서로 다른 특성을 가진 메모리를 hierarchy로 사용합니다.

```text
작고 빠름
↑
Register
L1 / Shared Memory
L2 Cache
HBM
↓
크고 느림
```

---

## 5. Memory Capacity와 Memory Footprint

Memory Capacity는 GPU가 제공하는 최대 저장 용량입니다.

예를 들어

```text
HBM Capacity = 16 GB
```

라는 것은 GPU가 약 16GB의 데이터를 저장할 수 있다는 의미입니다.

반면 Memory Footprint는 workload 실행에 실제로 필요한 메모리의 양입니다.

여기에는 상황에 따라

- Model parameters
- Activations
- Gradients
- Optimizer states
- Temporary workspace

등이 포함될 수 있습니다.

따라서 Capacity와 Bandwidth는 완전히 다른 개념입니다.

```text
16 GB  → 얼마나 많이 저장할 수 있는가?
1 TB/s → 1초에 얼마나 많이 옮길 수 있는가?
```

---

## 6. Compute Time과 Memory Time

컴파일러의 퍼포먼스를 개선하기 위해서는 먼저 병목 원인을 파악해야 합니다.

원인은 크게 두 가지로 나눌 수 있습니다.

- **Compute-bound**: GPU의 연산 처리 성능이 병목
- **Memory-bound**: 데이터를 읽고 쓰는 메모리 대역폭이 병목

이를 단순한 모델로 분석하면 다음과 같습니다.

$$
T_{compute}
=
\frac{\mathrm{FLOPs} }{\text{Compute Throughput} }
$$

$$
T_{memory}
=
\frac{\mathrm{Bytes} }{\text{Memory Bandwidth} }
$$

단위는 각각

$$
\frac{\mathrm{FLOPs} }{\mathrm{FLOPs/s} } = \mathrm{s}
$$

$$
\frac{\mathrm{Bytes} }{\mathrm{Bytes/s} } = \mathrm{s}
$$

이므로 시간입니다.

병목을 추정할 때는 보통 실제 Achieved FLOP/s나 Achieved Bandwidth가 아니라 **Peak 또는 attainable throughput**을 사용합니다.

$$
T_{compute}
=
\frac{\mathrm{FLOPs} }{\text{Peak Compute Throughput} }
$$

$$
T_{memory}
=
\frac{\text{Algorithmic Bytes} }{\text{Peak Memory Bandwidth} }
$$

실제 Achieved 값은 이미 Kernel Time을 포함하고 있기 때문에 이를 다시 나누면 결국 Kernel Time이 나오기 때문입니다.

연산과 데이터 이동이 충분히 overlap된다고 가정하면

$$
T_{kernel}
\approx
\max(T_{compute}, T_{memory})
$$

로 볼 수 있습니다.

따라서

```text
T_compute > T_memory → Compute-bound
T_memory > T_compute → Memory-bound
```

입니다.

다만 이 값은 이상적인 최소 실행시간의 기준입니다. 실제 실행에는 latency, dependency, synchronization, occupancy, pipeline stall 등이 존재하므로

$$
T_{actual}
\ge
\max(T_{compute},T_{memory})
$$

라고 보는 것이 더 정확합니다.

---

## 7. Vector Add 손계산

길이가

$$
N=100,000,000
$$

인 FP32 Vector A, B를 더해 C에 저장한다고 가정합니다.

```python
C[i] = A[i] + B[i]
```

FP32 하나는 4 Byte이고, 가상 하드웨어의 성능을 다음과 같이 두겠습니다.

```text
FP32 Compute Throughput = 20 × 10^12 FLOP/s
HBM Bandwidth           = 1 × 10^12 Byte/s
```

입력 A, B는 HBM에서 한 번씩 읽고, C는 한 번 쓴다고 가정합니다.

### 데이터 이동량

Vector 하나의 크기는

$$
100,000,000 \times 4
=
400,000,000 Byte
$$

이므로

```text
A Read  = 400 MB
B Read  = 400 MB
C Write = 400 MB
```

총 Algorithmic Bytes는

$$
1.2\ \mathrm{GB}
$$

입니다.

### FLOPs

각 원소마다 덧셈 한 번이 발생하므로

$$
\text{Total FLOPs} = 100{,}000{,}000 = 1 \times 10^8
$$

입니다.

### Compute Time

$$
T_{compute}
=
\frac{1\times10^8}
{20\times10^{12} }
=
5 \times 10^{-6}\ \mathrm{s}
=
0.005\ \mathrm{ms}
$$

### Memory Time

$$
T_{memory}
=
\frac{1.2\times10^9}
{1\times10^{12} }
=
1.2 \times 10^{-3}\ \mathrm{s}
=
1.2\ \mathrm{ms}
$$

따라서

```text
Compute Time = 0.005 ms
Memory Time  = 1.2 ms
```

이므로 이 Vector Add는 **Memory-bound**입니다.

예상 최소 실행시간은 단순 모델에서 약

$$
1.2\ \mathrm{ms}
$$

입니다.

---

## 8. 어떤 성능을 개선해야 하는가?

현재 Vector Add는 Memory-bound입니다.

Compute Throughput을 두 배 늘려도

```text
0.005 ms → 0.0025 ms
```

로만 줄어들고, Memory Time은 여전히 1.2ms이므로 전체 성능에는 거의 영향이 없습니다.

반대로 HBM Bandwidth를

```text
1 TB/s → 2 TB/s
```

로 늘리면

$$
T_{memory}
=
\frac{1.2\ \mathrm{GB} }{2\ \mathrm{TB/s} }
=
0.6\ \mathrm{ms}
$$

로 줄어듭니다.

즉,

> **현재 병목이 어디인지 먼저 확인한 뒤 그 자원을 개선해야 한다.**

는 것이 핵심입니다.

Memory Capacity 역시 이미 workload가 메모리에 들어간다면 용량을 늘린다고 실행속도가 빨라지지는 않습니다.

```text
Capacity  → 들어가느냐?
Bandwidth → 얼마나 빨리 옮기느냐?
```

를 구분해야 합니다.

---

## 9. 핵심 정리

### FLOPs
해야 할 계산의 양

### Peak FLOP/s
하드웨어가 1초에 수행할 수 있는 최대 연산량

### Algorithmic Bytes
알고리즘 관점에서 원래 옮겨야 하는 데이터의 양

### Actual Memory Traffic
실제 하드웨어에서 이동한 데이터의 양

### Effective Bandwidth

$$
\frac{\text{Algorithmic Bytes} }{\text{Execution Time} }
$$

### Actual / Achieved Memory Throughput

$$
\frac{\text{Actual Traffic} }{\text{Execution Time} }
$$

### Peak Bandwidth
하드웨어가 제공할 수 있는 최대 데이터 전송률

### Memory Footprint
workload 실행에 필요한 실제 메모리 양

### Memory Capacity
하드웨어가 저장할 수 있는 최대 메모리 양

### Compute / Memory Bound

$$
T_{compute}
=
\frac{\mathrm{FLOPs} }{\text{Peak FLOP/s} }
$$

$$
T_{memory}
=
\frac{\text{Algorithmic Bytes} }{\text{Peak Bandwidth} }
$$

```text
T_compute > T_memory → Compute-bound
T_memory > T_compute → Memory-bound
```

다음 단계에서는 이 FLOPs와 Algorithmic Bytes를 이용해 **Arithmetic Intensity와 Roofline**으로 이어갈 수 있습니다.
