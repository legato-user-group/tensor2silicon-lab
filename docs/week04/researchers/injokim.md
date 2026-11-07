<WeekNavBar :week="4" topic="Roofline과 Tiling" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송민배)', 'badge': 'Roofline & 타일 크기별 분석'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': '4×4 & 4096×4096 완전 해설'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': '2D Block Tiling 직관'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '이론 유도 & 연습문제 풀이'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': 'Roofline & Tiling 핵심 노트'}]" />

> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **injokim (김인조)** (`이론 유도 & 연습문제 풀이`)
>
> 💡 **핵심 기여 요약**: 산술 강도 정의, 하드웨어 분기점(Ridge Point) 이동 조건, 타일 크기 증가에 따른 온칩 SRAM 제약의 트레이드오프를 수식으로 정리했습니다.

---

# Arithmetic Intensity란

2026년 10월 5일 월요일 오후 10:07

커널 연산에는 메모리에서 데이터를 가져오는 시간과 가져온 데이터로 계산하는 시간이 필요합니다. 데이터 이동이 성능을 제한하는 경우를 Memory-bound, 계산 처리 능력이 성능을 제한하는 경우를 Compute-bound라고 부릅니다. 추론 속도를 개선하려면 현재 연산에서 어느 쪽이 병목인지 파악해야 합니다. 이를 판단하는 데 사용하는 방법 중 하나가 Arithmetic Intensity를 활용한 Roofline Modeling입니다.

## 1. Arithmetic Intensity

Arithmetic Intensity(산술 강도)는 데이터를 옮기는 양에 비해 얼마나 많은 연산을 하는지 나타내는 지표입니다. 커널의 산술 강도를 HW의 최대 연산 성능과 메모리 대역폭의 비율에 비교하면 계산과 데이터 이동 중 어느 쪽이 성능을 제한하는지 추정할 수 있습니다.

$$
A.I. = \frac{\text{총 FLOPs} }{\text{총 Memory Traffic (Byte)} }
$$

여기서 FLOPs는 전체 연산 횟수이고, Memory Traffic은 메모리에서 읽고 쓴 데이터의 총량입니다. 이 문서에서는 HBM을 기준으로 이동량을 셉니다. 단위는 FLOPs/Byte이며, HBM에서 1Byte를 옮길 때 평균적으로 몇 번의 부동소수점 연산을 하는지 나타냅니다.

같은 실행시간으로 측정한 FLOP/s와 Memory Throughput으로도 표현할 수 있습니다.

$$
A.I.
= \frac{\text{Achieved FLOPs/s} }{\text{Achieved Memory Throughput (Byte/s)} }
= \frac{F/T}{Q/T}
= \frac{F}{Q}
$$

분자와 분모의 시간이 약분되므로 두 식은 같은 값이 됩니다. 다만 커널의 산술 강도를 구하려면 같은 커널에서 측정한 연산 처리율과 HBM 처리율을 사용해야 합니다. HW의 최대 연산 성능과 최대 대역폭을 나눈 값은 뒤에서 설명할 Ridge Point의 산술 강도입니다.

### 연산량과 데이터 이동량

산술 강도는 연산량이 같아도 데이터 이동량에 따라 달라집니다. 같은 입력을 여러 계산에 다시 사용하면 HBM에서 읽는 양이 줄고 산술 강도는 높아집니다.

이동량과 저장 공간도 구분해야 합니다. Memory Footprint는 필요한 저장 공간의 크기입니다. 같은 입력을 HBM에서 반복해서 읽으면 저장 공간은 그대로지만 이동량은 커집니다. 산술 강도에는 이 반복 이동까지 반영합니다.

계산에 사용할 기호는 아래와 같습니다. 행렬 B와 구분하기 위해 대역폭은 BW로 씁니다.

| 기호 | 의미 | 단위 |
| --- | --- | --- |
| F | 커널의 총 연산량 | FLOPs |
| Q | HBM에서 읽고 쓴 총 데이터 이동량 | Byte |
| P | 해당 자료형·연산에 적용 가능한 최대 연산 처리율 | FLOPs/s |
| BW | HBM의 최대 대역폭 | Byte/s |
| I | HBM 기준 산술 강도 | FLOPs/Byte |

### Compute Time과 Memory Time

계산 시간은 연산량을 연산 처리율로 나눈 값이고, 메모리 시간은 이동량을 메모리 대역폭으로 나눈 값입니다. HW의 최대 성능을 기준으로 계산하면 각 작업에 필요한 최소 시간을 구할 수 있습니다.

$$
T_{compute} = \frac{F}{P}, \qquad T_{memory} = \frac{Q}{BW}
$$

T_compute가 더 크면 계산 처리 능력이, T_memory가 더 크면 HBM 대역폭이 성능을 제한합니다. 계산과 데이터 이동이 충분히 겹쳐 진행된다고 가정하면 이상적 실행시간은 두 시간 중 큰 값입니다.

$$
T_{ideal} = \max(T_{compute}, T_{memory})
$$

실제 실행에는 동기화, 명령어 의존성, 지연시간 등도 영향을 주므로 이 값은 실행시간의 하한입니다.

여기서 측정한 Achieved FLOPs/s를 P 대신 넣으면 이미 측정한 실행시간이 그대로 나옵니다. 아직 측정하지 않은 커널의 병목을 추정할 때는 HW의 최대 연산 성능과 대역폭으로 두 시간을 비교해야 합니다. 산술 강도와 시간의 관계는 [Scaling Book의 Arithmetic Intensity 설명](https://jax-ml.github.io/scaling-book/roofline/#where-does-the-time-go)을 참고했습니다.

### Vector Add와 행렬곱의 차이

FP32 Vector Add에서 원소 하나를 계산할 때의 연산량과 이동량을 세어보겠습니다.

```text
C[i] = A[i] + B[i]
출력 원소 하나당:
연산량 = 덧셈 1번 = 1 FLOP
이동량 = A 읽기 4Byte + B 읽기 4Byte + C 쓰기 4Byte = 12Byte
```

길이가 N이면 연산량은 N, 이동량은 12N입니다.

$$
I_{VectorAdd} = \frac{N}{12N} = \frac{1}{12}
\approx 0.0833\ \mathrm{FLOPs/Byte}
$$

Vector Add에서는 입력 원소 하나가 출력 원소 하나를 계산하는 데만 사용됩니다. 입력을 작은 조각으로 나누더라도 같은 값을 여러 출력에 사용할 수는 없습니다.

행렬곱에서는 같은 A 원소가 여러 출력 열에, 같은 B 원소가 여러 출력 행에 사용됩니다. 이 값을 매번 HBM에서 읽을지, 온칩에 보관해서 다시 사용할지에 따라 데이터 이동량과 산술 강도가 달라집니다.

## 2. Roofline Modeling

Roofline은 커널의 산술 강도와 HW의 성능 한계를 하나의 그래프에서 비교하는 모델입니다. 가로축에는 Arithmetic Intensity를, 세로축에는 FLOP/s를 표시합니다. 메모리 대역폭과 최대 연산 성능으로 두 상한을 구하고, 그중 낮은 값을 커널의 성능 상한으로 사용합니다.

### 메모리 대역폭에 따른 성능 상한

HBM에서 1초 동안 최대 BW Byte를 옮길 수 있고 1Byte당 I FLOPs를 처리한다면 메모리가 공급할 수 있는 연산 처리율은 BW × I입니다.

$$
R_{memory} = BW \times I
$$

대역폭의 단위 Byte/s에 산술 강도의 단위 FLOPs/Byte를 곱하면 연산 처리율인 FLOPs/s가 됩니다.

$$
\frac{\mathrm{Byte} }{\mathrm{s} }
\times \frac{\mathrm{FLOPs} }{\mathrm{Byte} }
= \frac{\mathrm{FLOPs} }{\mathrm{s} }
$$

같은 하드웨어에서는 BW가 고정되어 있으므로 I가 커질수록 이 상한은 비례해서 올라갑니다. 선형 축에서는 `y = BW × x`인 직선이고 기울기는 BW입니다. 실제 Roofline에서 흔히 쓰는 로그–로그 축에서는 `log R = log BW + log I`이므로 기울기는 1이며 BW가 바뀌면 선의 높이가 달라집니다.

### 최대 연산 성능에 따른 성능 상한

데이터를 아무리 효율적으로 공급해도 하드웨어의 최대 연산 처리율 P를 넘을 수는 없습니다. 이 상한은 산술 강도와 관계없이 수평선입니다.

커널은 두 제한을 동시에 받으므로 Roofline 성능 상한은 더 작은 쪽입니다.

$$
R_{roof} = \min(P, BW \times I)
$$

세로축은 실행시간이 아니라 초당 연산 처리율입니다. 같은 F를 처리할 때 그래프의 높이가 높아지면 이상적 실행시간은 짧아집니다. 축과 두 상한의 의미는 [Nsight Compute의 Roofline Charts — Overview](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)를 참고했습니다.

### Ridge Point

두 상한이 만나는 지점에서는 `P = BW × I`입니다.

$$
I_{ridge} = \frac{P}{BW}
$$

Ridge Point의 좌표는 (P/BW, P)입니다. P는 세로축의 연산 성능 상한이고 P/BW는 가로축의 분기점입니다. 서로 단위가 다릅니다.

두 시간 공식을 정리해도 같은 분기점이 나옵니다.

$$
T_{compute} > T_{memory}
\iff \frac{F}{P} > \frac{Q}{BW}
\iff \frac{F}{Q} > \frac{P}{BW}
\iff I > I_{ridge}
$$

| 산술 강도 조건 | 두 시간의 관계 | 이 모델의 제한 자원 |
| --- | --- | --- |
| I < P/BW | T_memory > T_compute | HBM 대역폭: Memory-bound |
| I = P/BW | T_memory = T_compute | 두 상한이 만나는 지점 |
| I > P/BW | T_compute > T_memory | 연산 성능: Compute-bound |

I가 커졌다는 사실만으로 Compute-bound라고 판단할 수는 없습니다. 하드웨어의 분기점 P/BW를 넘는지 비교해야 합니다.

연산 성능이 20 TFLOPs/s이고 HBM 대역폭이 1 TB/s인 가상 하드웨어를 가정하겠습니다.

```text
P  = 20 × 10^12 FLOPs/s = 20 TFLOPs/s
BW =  1 × 10^12 Byte/s  =  1 TB/s
I_ridge = 20 FLOPs/Byte
```

```text
연산 처리율 R (TFLOPs/s)
  ↑
20│                  ●──────────────────  연산 성능 상한 P
  │                ／│
  │              ／  │
  │            ／    │
  │          ／      │
  │        ／        │  기울어진 부분: BW × I
  │      ／          │
  │    ／            │
  │  ／              │
  └──────────────────┼─────────────────→ I (FLOPs/Byte)
  0                 20
     Memory-bound         Compute-bound
  선형 축의 개념도. ● = Ridge Point (20 FLOPs/Byte, 20 TFLOPs/s)
```

Roofline의 선은 주어진 조건에서 가능한 성능 상한을 나타냅니다. 실제 커널은 동기화나 데이터 공급 지연 등으로 이보다 낮은 처리율을 보일 수 있습니다. 산술 강도가 Ridge Point 오른쪽에 있더라도 최대 연산 성능에 도달했는지는 측정으로 확인해야 합니다.

## 3. Tiling과 데이터 재사용

Tiling은 계산할 출력을 작은 조각으로 나누고, 그 조각에 필요한 입력을 가져와 재사용하는 방법입니다. 같은 연산량을 더 적은 HBM 이동량으로 처리하면 산술 강도가 높아집니다.

### 행렬곱의 입력 재사용

`C = A @ B`에서 A의 크기를 M×K, B의 크기를 K×N이라고 하면 C는 M×N입니다.

$$
C[i,j] = \sum_{k=0}^{K-1} A[i,k]B[k,j]
$$

`A[i,k]`는 i와 k를 고정한 채 j를 바꾸면 C의 i번째 행에 있는 N개 출력에 사용됩니다. `B[k,j]`는 k와 j를 고정한 채 i를 바꾸면 C의 j번째 열에 있는 M개 출력에 사용됩니다.

출력 하나씩 따로 계산하며 매번 HBM에서 입력을 읽으면 같은 값이 반복해서 이동합니다. 여러 출력을 묶어서 계산하고 필요한 입력을 온칩에 유지하면 그 반복 읽기를 줄일 수 있습니다.

### 4×4 행렬곱을 2×2 출력 타일로 계산하기

먼저 왼쪽 위 출력 타일 `C[0:2, 0:2]`를 계산하겠습니다. K 방향도 폭 2씩 나눕니다.

```text
A (4×4)                    B (4×4)                    C (4×4)
┌─────────┬─────────┐      ┌─────────┬─────────┐      ┌─────────┬─────────┐
│ A00 A01 │ A02 A03 │      │ B00 B01 │ B02 B03 │      │ C00 C01 │ C02 C03 │
│ A10 A11 │ A12 A13 │      │ B10 B11 │ B12 B13 │      │ C10 C11 │ C12 C13 │
├─────────┼─────────┤      ├─────────┼─────────┤      ├─────────┼─────────┤
│ A20 A21 │ A22 A23 │      │ B20 B21 │ B22 B23 │      │ C20 C21 │ C22 C23 │
│ A30 A31 │ A32 A33 │      │ B30 B31 │ B32 B33 │      │ C30 C31 │ C32 C33 │
└─────────┴─────────┘      └─────────┴─────────┘      └─────────┴─────────┘
K 방향 앞부분 (k = 0, 1):
  A[0:2, 0:2]와 B[0:2, 0:2]를 읽어 2×2 중간 합 acc에 누적
K 방향 뒷부분 (k = 2, 3):
  A[0:2, 2:4]와 B[2:4, 0:2]로 입력을 바꾸어 같은 acc에 누적
전체 K 계산이 끝나면 acc를 C[0:2, 0:2]에 한 번만 쓰기
```

첫 번째 K 구간에서 `A00`과 `B00`은 각각 두 출력의 계산에 쓰입니다.

```text
A00 → C00의 A00×B00과 C01의 A00×B01에 사용
B00 → C00의 A00×B00과 C10의 A10×B00에 사용
```

읽어온 A·B의 각 원소를 2번씩 사용할 수 있습니다. K 구간 하나에서 입력은 총 8개 원소이고 연산량은 `2 × 2 × 2 × 2 = 16 FLOPs`입니다. K 구간을 두 번 진행하므로 출력 타일 하나의 연산량은 32 FLOPs입니다.

FP32를 기준으로 출력 타일 4개의 연산량과 이동량을 세면 아래 표와 같습니다. 서로 다른 출력 타일 사이의 캐시 재사용은 세지 않습니다.

| 방법 | FLOPs | A·B의 HBM 읽기 | C 쓰기 | 총 이동량 | I |
| --- | ---: | ---: | ---: | ---: | ---: |
| 재사용 없음 | 128 | 512 Byte | 64 Byte | 576 Byte | 0.2222 FLOPs/Byte |
| 2×2 타일 안에서 재사용 | 128 | 256 Byte | 64 Byte | 320 Byte | 0.4000 FLOPs/Byte |

출력 타일 하나에서는 입력 16개 원소를 읽고 출력 4개 원소를 씁니다. 4개 타일의 총 이동량은 `(16 + 4) × 4Byte × 4 = 320Byte`입니다.

위 비교에서는 Tiling을 적용해도 행렬곱의 연산량은 같습니다. 달라지는 부분은 HBM에서 읽는 양입니다. 입력과 중간 합을 보관하면서 여러 출력의 계산에 사용하기 때문에 반복 읽기가 줄어듭니다. 계산을 작게 나누는 것만으로는 이 효과를 얻을 수 없습니다.

### Local Blocking과 Shared Memory Blocking의 차이

데이터를 스레드 하나 안에서 재사용하는 방식은 Local Blocking, 같은 스레드 블록 안에서 공유하는 방식은 Shared Memory Blocking입니다.

| 방법 | 재사용 범위 | 보관 장소의 대표적인 예 | 핵심 아이디어 |
| --- | --- | --- | --- |
| Local Blocking | 스레드 하나 안 | 레지스터 | 스레드가 여러 출력을 담당하면서 입력과 중간 합을 재사용 |
| Shared Memory Blocking | 같은 스레드 블록 안 | Shared Memory | 여러 스레드에 필요한 공통 입력을 협력해서 읽고 공유 |

두 방법은 함께 사용할 수 있습니다. 블록이 공유하는 A·B 입력 타일을 Shared Memory에 놓고 각 스레드가 담당하는 출력의 중간 합을 레지스터에 유지하는 식입니다. Shared Memory를 공유할 때는 필요한 읽기가 끝난 뒤 계산해야 합니다. 다음 입력으로 덮어쓰기 전에도 사용이 끝났음을 보장해야 합니다.

이 차이는 [MLC — Local Blocking / Shared Memory Blocking](https://book.mlc.ai/chapter_gpu_acceleration/part1.html#local-blocking)의 그림과 설명을 참고했습니다. MLC의 `local` 저장 범위와 CUDA의 local memory는 구분해야 합니다. CUDA의 local memory가 항상 빠른 온칩 공간인 것은 아닙니다. 레지스터에 담지 못해 spill한 값은 디바이스 메모리 접근을 일으킬 수 있습니다.

## 4. 4096×4096 FP32 행렬의 계산 예시

재사용에 따른 차이를 비교하기 위해 같은 크기의 행렬곱을 네 가지 방식으로 계산하겠습니다. 입력을 매번 읽는 경우와 출력 타일의 한 변을 32·64·128로 두는 경우입니다. 아래 결과는 가상 하드웨어의 조건으로 계산한 값이며 실제 GPU의 측정값은 아닙니다.

### 계산 조건

```text
C = A @ B
A·B·C: 모두 4096×4096, FP32 (원소 하나 = 4Byte)
P:  이 FP32 연산에 적용 가능한 20 × 10^12 FLOPs/s
BW: HBM 대역폭 1 × 10^12 Byte/s
```

- 입력은 이미 장치에 있으며 CPU에서 옮기는 시간은 포함하지 않습니다.

- 행렬곱 연산량은 `2 × M × N × K`로 계산하고 FMA를 2 FLOPs로 셉니다.

- 중간 합은 온칩에 유지하고 최종 C만 한 번 씁니다. 기존 C는 읽지 않습니다.

- 재사용 없음에서는 곱셈할 때마다 A·B 원소를 HBM에서 읽습니다.

- 타일 안에서 재사용할 때는 출력 타일이 t×t이고 K 방향 입력 조각의 폭도 t입니다.

- 서로 다른 출력 타일 사이의 캐시 재사용은 없고 각 입력 타일의 읽기는 모두 HBM까지 도달한다고 가정합니다.

- GB·TB는 10진 단위, KiB는 2진 단위입니다. 아래 표는 온칩 공간에 필요한 데이터가 들어간다고 가정한 이상적 모델입니다.

### 연산량과 계산 시간

행렬의 한 변을 n이라고 쓰겠습니다.

$$
F = 2n^3 = 2 \times 4096^3
= 137{,}438{,}953{,}472\ \mathrm{FLOPs}
$$

$$
T_{compute} = \frac{137{,}438{,}953{,}472}{20 \times 10^{12} }
\approx 6.871948\ \mathrm{ms}
$$

### 재사용이 없을 때의 이동량

곱셈은 n³번입니다. 각 곱셈마다 A·B 원소를 하나씩 읽으므로 입력은 8n³ Byte이고 최종 C 쓰기는 4n² Byte입니다.

$$
Q_{no\ reuse} = 8n^3 + 4n^2
= 549{,}822{,}922{,}752\ \mathrm{Byte}
$$

$$
I_{no\ reuse} = \frac{2n^3}{8n^3 + 4n^2}
\approx 0.249969\ \mathrm{FLOPs/Byte}
$$

### 타일 안에서 재사용할 때의 이동량

출력 타일 하나를 기준으로 K 방향 한 구간에서 읽는 입력은 A의 t×t 조각과 B의 t×t 조각, 총 2t²개 원소입니다. K 방향에는 n/t개 구간이 있고 전체 출력 타일 수는 (n/t)²개입니다.

$$
Q_{input}
= \left(\frac{n}{t}\right)^2
\times \frac{n}{t}
\times 2t^2 \times 4
= \frac{8n^3}{t}\ \mathrm{Byte}
$$

여기에 C 쓰기 4n² Byte를 더합니다.

$$
Q_{tile} = \frac{8n^3}{t} + 4n^2
$$

$$
I_{tile} = \frac{2n^3}{8n^3/t + 4n^2}
= \frac{nt}{4n + 2t}
$$

입력 이동량은 재사용이 없을 때의 1/t로 줄어듭니다. C 쓰기는 그대로이므로 총 이동량 전체가 정확히 1/t로 줄어드는 것은 아닙니다.

### 계산 결과

모든 경우의 연산량은 137,438,953,472 FLOPs입니다.

| 방법 | HBM 총 이동량 Q (Byte) | Q (GB) | I (FLOPs/Byte) | Roofline 상한 (TFLOPs/s) | 모델의 제한 자원 |
| --- | ---: | ---: | ---: | ---: | --- |
| 재사용 없음 | 549,822,922,752 | 549.822923 | 0.249969 | 0.249969 | Memory-bound |
| t = 32 | 17,246,978,048 | 17.246978 | 7.968872 | 7.968872 | Memory-bound |
| t = 64 | 8,657,043,456 | 8.657043 | 15.875969 | 15.875969 | Memory-bound |
| t = 128 | 4,362,076,160 | 4.362076 | 31.507692 | 20.000000 | Compute-bound |

| 방법 | T_compute (ms) | T_memory (ms) | 이상적 실행시간 max (ms) |
| --- | ---: | ---: | ---: |
| 재사용 없음 | 6.871948 | 549.822923 | 549.822923 |
| t = 32 | 6.871948 | 17.246978 | 17.246978 |
| t = 64 | 6.871948 | 8.657043 | 8.657043 |
| t = 128 | 6.871948 | 4.362076 | 6.871948 |

타일 크기를 32, 64로 키우면 계산 시간은 그대로이고 메모리 시간은 짧아집니다. 크기 128에서는 산술 강도가 분기점인 20 FLOPs/Byte를 넘습니다. 계산 시간이 메모리 시간보다 길어져 모델에서의 병목이 바뀝니다.

이 지점에서는 HBM 이동량을 더 줄여도 이 모델의 실행시간은 6.871948ms보다 짧아지지 않습니다. 재사용으로 얻는 성능 이득에도 연산 성능 P가 정하는 한계가 있습니다.

## 5. 타일 크기와 저장 공간

타일을 키우면 입력을 더 많이 재사용할 수 있지만, 입력 조각과 중간 합을 보관할 공간도 커집니다. HBM 이동량만 줄었다고 해서 실제 실행도 항상 빨라지는 것은 아닙니다.

### 입력 버퍼와 중간 합의 크기

출력 타일 하나를 계산하는 동안 K 구간마다 A·B 입력 조각은 바뀌지만 출력의 중간 합은 전체 K 계산이 끝날 때까지 유지합니다. FP32 단일 버퍼에서는 입력에 8t² Byte, 중간 합에 4t² Byte가 필요합니다.

$$
S_{input} = (t^2 + t^2) \times 4 = 8t^2\ \mathrm{Byte}
$$

$$
S_{acc} = t^2 \times 4 = 4t^2\ \mathrm{Byte}
$$

| 출력 타일 | A 입력 | B 입력 | 입력 합계 | 중간 합 | 합계 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32×32 | 4 KiB | 4 KiB | 8 KiB | 4 KiB | 12 KiB |
| 64×64 | 16 KiB | 16 KiB | 32 KiB | 16 KiB | 48 KiB |
| 128×128 | 64 KiB | 64 KiB | 128 KiB | 64 KiB | 192 KiB |

위 표는 입력과 중간 합 자체의 데이터 크기를 계산한 것입니다. 입력은 Shared Memory, 중간 합은 레지스터에 놓는다면 각 자원을 따로 확인해야 합니다. 합계 192 KiB만으로는 어느 자원이 부족한지 알 수 없습니다. 실제 구현에는 주소 계산 등을 위한 추가 레지스터나 padding도 필요할 수 있습니다. 입력을 double buffering한다면 입력 버퍼 부분은 2배가 됩니다.

### 레지스터·Shared Memory와 Occupancy

Occupancy는 SM에 상주하는 활성 warp 수를 해당 SM에 상주할 수 있는 최대 warp 수로 나눈 비율입니다. 블록 하나가 Shared Memory나 레지스터를 많이 사용하면 같은 SM에 상주할 수 있는 블록이나 warp 수가 줄어들 수 있습니다. [CUDA Best Practices — Calculating Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy)는 이러한 자원 제약을 설명합니다.

타일을 키우면 입력의 재사용 횟수는 늘어납니다. 저장량도 t²에 비례해서 늘어납니다. 상주할 수 있는 작업이 줄면 메모리 대기 등을 숨기기 어려워질 수 있습니다. 레지스터에 담지 못해 spill하거나, 필요한 Shared Memory를 확보하지 못할 수도 있습니다.

Occupancy만으로 속도를 판단하기는 어렵습니다. 재사용이 늘어나는 이점과 동시에 상주할 수 있는 작업이 줄어드는 비용을 함께 봐야 합니다. 가상 하드웨어의 조건에는 Shared Memory와 레지스터 용량이 없으므로 t=128을 실제로 사용할 수 있는지 또는 가장 빠른지는 판단할 수 없습니다.

### 출력 타일 크기와 스레드 수

128×128 출력 타일은 16,384개 출력 원소를 묶어서 계산합니다. 필요한 스레드 수는 이와 별개입니다. 스레드 하나가 여러 출력이나 부분 계산을 담당할 수 있습니다.

출력 타일의 세로·가로 크기와 K 방향 폭도 일반적으로 따로 선택할 수 있습니다. 출력을 B_M×B_N, K 방향 폭을 B_K라고 하면 입력과 중간 합의 단일 버퍼 크기는 각각 아래와 같습니다.

$$
S_{input} = 4(B_M B_K + B_K B_N), \qquad
S_{acc} = 4B_M B_N
$$

위 계산에서는 세 값을 모두 t로 두었습니다. [Triton Matrix Multiplication의 첫 블록 알고리즘](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#motivations)은 세 값을 따로 다루며 출력의 중간 합을 유지한 채 K 방향으로 계산을 진행합니다.

## 6. 성능 상한과 실행시간

Roofline 성능 상한으로 연산량 F를 나누면 계산 시간과 메모리 시간 중 큰 값이 나옵니다.

$$
\frac{F}{\min(P, BW \times I)}
= \max\left(\frac{F}{P}, \frac{Q}{BW}\right)
$$

같은 행렬곱이라도 매번 HBM에서 입력을 읽을 때와 온칩에서 재사용할 때는 이동량이 다릅니다. Tiling은 그 재사용을 구현하는 방법이며 연산량을 바꾸지 않고 산술 강도를 높일 수 있습니다. 다만 큰 타일을 유지하려면 더 많은 저장 공간이 필요합니다.

### 추가로 확인할 내용

1. Local Blocking과 Shared Memory Blocking을 각각 어느 크기의 출력에 적용할까? 스레드 안의 재사용과 블록 안의 재사용을 같은 그림에서 비교할 수 있을까요?

2. t=128이 Roofline에서는 유리해도 저장 공간 제약을 넣으면 어떻게 달라질까? 용량 조건이 없으므로 실제로 구현할 수 있는지는 별도로 확인해야 합니다.

3. 출력 타일 사이에서 L2 캐시가 작동하면 Q는 어떻게 달라질까? 같은 입력을 다른 출력 타일에서도 재사용하면 HBM 기준 I는 위 계산보다 커질 수 있습니다.

4. I가 분기점을 넘었는데도 실제 성능이 낮다면 무엇을 확인해야 할까? 동기화, 레지스터 사용량, Occupancy, 입력 공급 중 어디서 지연이 생기는지 확인할 필요가 있습니다.

### 참고 자료

- [Scaling Book 1장 — All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/): Arithmetic Intensity 정의부터 Visualizing rooflines까지.

- [MLC — GPU Acceleration, Part 1](https://book.mlc.ai/chapter_gpu_acceleration/part1.html): Local Blocking과 Shared Memory Blocking의 그림과 설명.

- [NVIDIA Nsight Compute — Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts): Overview의 축, 두 상한, Ridge Point.

- [CUDA Best Practices — Calculating Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy): 레지스터와 Shared Memory 제약, 상주 가능한 블록 수와의 관계.

- [Triton — Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#motivations): 첫 블록 알고리즘과 K 방향 누적.
