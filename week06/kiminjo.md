# Triton 기반 고성능 GPU 커널 최적화 및 Fused Softmax

## 1. 블록 단위 프로그래밍 모델: 스레드 중심(CUDA)에서 타일/블록 중심(Triton)으로의 패러다임 전환

고성능 GPU 컴퓨팅 영역에서 전통적인 SIMT(Single-Instruction Multiple-Threads) 기반 CUDA C++ 모델은 개발자에게 저레벨 레지스터 및 메모리 접근에 대한 미시적 제어권을 제공함으로써 극대의 유연성을 보장해 왔다. 그러나 스레드 레벨 인덱스 산출(`threadIdx`, `blockIdx`, `blockDim`), 워프 간 분기 다이버전스(Warp Divergence) 관리, 공유 메모리(Shared Memory) 뱅크 충돌 회피, 메모리 접근 병합(Memory Coalescing), 명시적 동기화 배리어(Synchronization Barrier) 수동 구현 등 고도의 하드웨어 밀착형 최적화 부담은 개발 생산성을 저해하고 코드의 유지보수성을 악화시킨다. OpenAI Triton은 실행 범위를 스칼라 스레드 인덱싱에서 2차원 블록-타일(Block-Tile) 추상화 수준으로 한 단계 끌어올림으로써 이러한 하드웨어 물리 제어 부담을 수동 엔지니어링에서 컴파일러 백엔드로 완전 이관한다. Triton 프로그래밍 모델 하에서 단일 프로그램 인스턴스는 다차원 데이터 타일 전체를 일괄 로드, 변환, 저장하도록 정의되며, 물리적 스레드 레이아웃 인코딩, 워프 레벨 스위즐링(Swizzling), 병합 DRAM 접근 연산은 Triton MLIR 컴파일러의 TTGIR 최적화 패스에 의해 자동 생성된다. 이는 고성능 AI 가속 커널 개발 생산성을 극대화함과 동시에 하드웨어 구조 독립적인 시스템 최적화를 달성하는 전략적 패러다임 전환이다.

### 핵심 시스템 용어 정의 및 메커니즘

- `Program Instance (pid)`: Triton 커널의 실행 그리드(Grid) 상에서 독립적으로 스케줄링되는 단일 실행 인스턴스 고유 식별자. `tl.program_id(axis)` 인터페이스를 통해 획득되며, 하드웨어 스레드 블록의 논리적 추상화 단위로 기능함.
- `BLOCK_SIZE`: 단일 프로그램 인스턴스가 온칩 SRAM/Register 계층으로 로드하여 일괄 처리하도록 정적으로 할당된 데이터 타일의 원소 개수. 컴파일 시점 상수로 확정됨.
- `Grid Dimensioning (triton.cdiv)`: 전체 텐서 원소 수 N을 타일 크기 `BLOCK_SIZE`로 나누어 올림 처리하는 그리드 차원 계산 메커니즘. $\text{grid} = (\text{triton.cdiv}(N, \text{BLOCK\_SIZE}), )$ 정수를 도출하여 시스템 실행 차원을 정의함.

### CUDA vs. Triton 제어 구조 심층 비교

CUDA 하드웨어 프로그래밍 모델은 내장 변수를 통한 스레드 개별 오프셋 산출에 의존한다. 개발자는 32개 연속 스레드로 구성된 워프(Warp)가 글로벌 메모리 128바이트 트랜잭션 경계에 맞추어 DRAM 접근을 단일 연산으로 결합(Coalescing)하도록 메모리 레이아웃을 물리적으로 배치해야 한다. 반면 Triton 모델은 `tl.program_id`와 `BLOCK_SIZE`를 조합하여 블록 전체의 데이터 공간을 타일 단위로 참조한다. 스레드 격자 스케줄링 및 32개 스레드 워프 매핑, 물리적 메모리 접근 병합 처리는 백엔드의 `TTGIR Coalesce Pass`가 하드웨어 토폴로지를 인지하여 자동 실행한다.

| 비교 분석 항목 | CUDA C++ 프로그래밍 모델 | OpenAI Triton 프로그래밍 모델 |
| --- | --- | --- |
| **제어 추상화 단위** | 개별 스레드 레벨 스칼라 제어 (`threadIdx`) | 타일/블록 레벨 벡터·행렬 제어 (`tl.program_id`) |
| **메모리 접근 병합** | 개발자가 워프 내 스레드 간 연속 접근 수동 제어 | `TTGIR Coalesce Pass`에 의한 하드웨어 자동 매핑 |
| **동기화 및 뱅크 충돌** | `__syncthreads()`, shared memory swizzling 수동 구현 | 컴파일러 레이아웃 인코딩(Shared Layout) 자동 최적화 |
| **코드 개발 생산성** | 저레벨 제어권 우수, 오버헤드 및 정교한 복잡성 증가 | 정적 타일 추상화를 통한 고성능 커널의 빠른 구현 |

### 1차원 벡터 분할 텍스트 아키텍처 구조

길이가 N인 1차원 데이터 텐서를 `BLOCK_SIZE` 단위로 분할할 때의 실행 그리드 차원 산출식은 다음과 같다. $\text{grid} = (\text{triton.cdiv}(N, \text{BLOCK\_SIZE}), )$

각 프로그램 인스턴스(`pid = 0, 1, 2, ...`)가 담당하는 데이터 구간 및 포인터 범위를 시각화한 구조도는 다음과 같다.

```
[전체 데이터 텐서 (총 N개 원소)]
|-----------------------|-----------------------|-----------------------|-------------------|
|<--- pid = 0 담당 구간 --->|<--- pid = 1 담당 구간 --->|<--- pid = 2 담당 구간 --->| Out-of-Bounds 영역|
| [0 ~ BLOCK_SIZE-1]    | [BLOCK_SIZE ~ 2*BS-1] | [2*BS ~ 3*BS-1]       | (N 이상, 마스킹)  |
|-----------------------|-----------------------|-----------------------|-------------------|
 ^                       ^                       ^                       ^
 ptr + 0*BLOCK_SIZE      ptr + 1*BLOCK_SIZE      ptr + 2*BLOCK_SIZE      ptr + N (경계)

```
이러한 블록 단위 프로그래밍 모델은 메모리 오프셋 벡터 생성 및 경계 조건 제어의 기틀이 된다. 컴파일 시점 타일 크기와 실제 데이터 크기 간 불일치를 안전하게 제어하는 포인터 마스킹 및 정적 검증 메커니즘을 상세히 분석한다.

## 2. 포인터 연산, 마스킹 및 컴파일 시점 상수의 정적 검증 메커니즘

실무 딥러닝 워크로드의 텐서 차원 크기 N은 컴파일 시점 타일 크기 `BLOCK_SIZE`의 정확한 배수가 아닌 경우가 일반적이다. 타일 기반 연산 시 경계 영역을 벗어난 메모리 읽기/쓰기 접근(Out-of-Bounds Memory Access)은 메모리 오염이나 Segmentation Fault를 유발할 수 있다. Triton은 포인터 오프셋 벡터 생성과 비트 레벨 마스킹(Masking)을 결합하여 메모리 안전성(Memory Safety)을 보장하며, 컴파일 시점 상수를 활용해 SASS 기계어 수준의 정적 최적화를 달성한다.

### 핵심 시스템 용어 정의 및 메커니즘

- `Pointer Vector (offsets)`: 각 프로그램 인스턴스가 담당하는 데이터 블록 내 원소들의 메모리 상대 주소를 나타내는 연속된 오프셋 벡터.
- `Boundary Masking (mask)`: 메모리 접근 시 경계 범위를 초과하는 인덱스에 대해 로드/스토어 명령의 하드웨어 실행을 차단하는 불리언(Boolean) 비트 마스크 벡터.
- `Compile-time Constant (tl.constexpr)`: 컴파일 단계에서 상수로 정적 바인딩되어 IR 로워링 시 루프 전개, 레지스터 할당, SASS 루프 구조 최적화를 결정짓는 핵심 매개변수.

### Python/PyTorch-Triton 커널 인자 전달 하부 메커니즘

PyTorch의 `torch.Tensor`가 `@triton.jit` 커널 함수로 전달될 때, 텐서 객체 자체는 C++ 백엔드로 직접 인입되지 않는다. 텐서는 시작 메모리 주소 포인터(`x_ptr`), 차원 크기 정보(N), 차원별 스트라이드(Stride) 값, 데이터 타입(DataType) 속성 형태로 완전히 해체(Unpack)되어 CUBIN 바이너리의 런타임 인자로 전달된다.

### 포인터 오프셋 벡터 생성 및 마스킹 원리

프로그램 인스턴스는 자신이 담당할 데이터 오프셋 벡터를 아래 산식으로 계산한다. $\text{offsets} = \text{pid} \times \text{BLOCK\_SIZE} + \text{tl.arange}(0, \text{BLOCK\_SIZE})$

- `pid * BLOCK_SIZE` 항목은 해당 프로그램 인스턴스가 접근할 데이터 타일의 기준 시작 주소를 결정한다.
- `tl.arange(0, BLOCK_SIZE)` 연산은 0부터 $\text{BLOCK\_SIZE}-1$까지의 연속된 정수 벡터를 생성하여 인스턴스 내 상대 오프셋을 부여한다.

경계 조건 차단을 위한 마스크 불리언 배열은 아래 조건식으로 산출된다. $\text{mask} = \text{offsets} < N$ `tl.load(x_ptr + offsets, mask=mask)` 또는 `tl.store` 수행 시, 생성된 마스크 벡터가 하드웨어 레벨로 전달된다. `False`로 판정된 인덱스 슬롯은 DRAM 읽기/쓰기 트랜잭션 시 하드웨어 가스켓 레벨에서 실행 무효화 처리되어 Out-of-Bounds 접근을 차단한다.

### `BLOCK_SIZE: tl.constexpr` 컴파일 상수의 정적 검증 필수성

`BLOCK_SIZE`가 `tl.constexpr`로 선언되어 컴파일 시점 정적 상수로 확정되면, LLVM IR 로워링 및 TTGIR 변환 과정에서 텐서의 정적 타일 형상이 확정된다. 컴파일러 백엔드는 이를 기반으로 루프 완전 전개(Full Loop Unrolling), SM 레지스터 및 Shared Memory static allocation size 산출, SASS 명령어 수준의 명시적 루프 카운터 제거, 메모리 정렬 인코딩을 적용하여 실행 오버헤드를 극적으로 제거한다.

### 손계산 실습: 1차원 벡터 인덱싱 및 마스크 표 (N=10, $\text{BLOCK\_SIZE}=4$)

- **실행 그리드 계산:** $\text{grid} = (\text{triton.cdiv}(10, 4), ) = (3, )$ (총 3개의 프로그램 인스턴스 생성)

| Program Instance (`pid`) | 생성되는 오프셋 벡터 (`offsets`) | 마스크 조건식 (`offsets < 10`) | 유효 접근 원소 개수 |
| --- | --- | --- | --- |
| `pid = 0` | `[0, 1, 2, 3]` | `[True, True, True, True]` | 4개 (전원 유효) |
| `pid = 1` | `[4, 5, 6, 7]` | `[True, True, True, True]` | 4개 (전원 유효) |
| `pid = 2` | `[8, 9, 10, 11]` | `[True, True, False, False]` | 2개 (2개 유효, 2개 마스킹 차단) |

단일 블록 포인터 마스킹이 정밀한 온칩 로드를 보장하는 메커니즘을 입증하였다. 온칩 로드 작업이 단일 연산을 넘어 연산자 결합(Kernel Fusion)으로 확장될 때 발생하는 HBM 대역폭 트래픽 절감 효과를 수치 분석한다.

## 3. 커널 결합(Kernel Fusion)과 HBM 트래픽 간의 정량적 계량 모델

메모리 대역폭 제한적(Memory-bound) 연산 구역에서 GPU 연산 성능의 병목은 ALU 연산력이 아닌 글로벌 메모리(HBM/DRAM)와 온칩 메모리(SRAM/Register) 간의 데이터 복사 트래픽에서 발생한다. 단일 element-wise 연산과 달리, Softmax와 같은 복합 리덕션 연산 체인에서 커널 결합(Kernel Fusion)은 HBM 왕복 트래픽을 정량적으로 감축하여 성능을 극대화한다.

### 핵심 시스템 용어 정의 및 메커니즘

- `Kernel Fusion`: 복수의 연산 과정을 단일 GPU 커널 내로 통합하여 중간 연산 결과를 HBM에 방출하지 않고 온칩 SRAM/Register 공간 내에서 직접 소비하는 컴파일러 최적화 기법.
- `HBM Round-trip Traffic`: GPU 글로벌 메모리(HBM)와 온칩 SM 간 오가는 총 데이터 읽기/쓰기 메모리 트랜잭션 원소 수 및 바이트 수.
- `Arithmetic Intensity (I = F/Q)`: 산술 강도. 메모리 접근 트래픽 단위 바이트(Q)당 수행되는 부동소수점 연산 수(F). 단위는 $\text{FLOP/Byte}$.

### Vector Add vs. Softmax 커널 결합 효과의 비대칭성 원인 분석

- **Vector Add (x + y)**: 2MN FLOPs 연산을 위해 2MN 원소를 읽고 1MN 원소를 쓴다. 중간 생체 텐서가 존재하지 않는 단단계 연산으로 결합 시 절감 가능한 HBM 트래픽이 없다.
- **Softmax**: Max Reduction, Subtraction, Exponential, Sum Reduction, Division으로 이어지는 다단계 연산 체인을 형성한다. 연산 단계별로 비결합 실행 시 중간 텐서들이 HBM으로 방출되었다가 재로드되므로, 결합 시 HBM 트래픽 절감 효과가 극대화된다.

### `naive_softmax` 5단계 연산별 텐서 읽기/쓰기 원소 계량화 분석

입력 텐서 X $\in \mathbb{R}^{M \times N}$에 대해 비결합 `naive_softmax`가 수행하는 5단계 메모리 접근 트랜잭션 추적 결과는 다음과 같다.

1. **Step 1:** `x_max = x.max(dim=1)`
   - HBM 읽기: X (MN 원소) | HBM 쓰기: $X_{\max}$ (M 원소)
2. **Step 2:** `z = x - x_max`
   - HBM 읽기: X (MN 원소) + $X_{\max}$ (M 원소) | HBM 쓰기: Z (MN 원소)
3. **Step 3:** `exp_z = exp(z)`
   - HBM 읽기: Z (MN 원소) | HBM 쓰기: $\exp(Z)$ (MN 원소)
4. **Step 4:** `sum_exp = exp_z.sum(dim=1)`
   - HBM 읽기: $\exp(Z)$ (MN 원소) | HBM 쓰기: Sum (M 원소)
5. **Step 5:** `out = exp_z / sum_exp`
   - HBM 읽기: $\exp(Z)$ (MN 원소) + Sum (M 원소) | HBM 쓰기: Output (MN 원소)

- **트랜잭션 총계:** 텐서 읽기 연산 7회 (5MN + 2M 원소), 텐서 쓰기 연산 5회 (3MN + 2M 원소). 총 12회 메모리 접근을 통과하며 이동하는 총 원소 이동량 $Q_{\text{elem}} = 8MN + 4M \approx 8MN$.

### HBM 데이터 이동 정량 대조 표

| 비교 분석 항목 | 단계별 비결합 실행 (`naive_softmax`) | Triton Fused 커널 (`softmax_kernel`) |
| --- | --- | --- |
| **메모리 접근 횟수** | 총 12회 Access (읽기 7회, 쓰기 5회) | 총 2회 Access (읽기 1회, 쓰기 1회) |
| **총 원소 이동량 ($Q_{\text{elem}}$)** | 8MN + 4M 원소 (근사치 8MN) | 2MN 원소 (입력 읽기 MN + 출력 쓰기 MN) |
| **중간 텐서 상주 위치** | 불가능 (HBM 글로벌 메모리로 방출) | 가능 (On-Chip SRAM 및 Register 상주) |
| **산술 강도 (I = F/Q)** | $I \approx \frac{5MN}{8MN \times 4\text{B}} = 0.156 \text{ FLOP/B}$ (극심한 HBM 병목) | $I = \frac{5MN}{2MN \times 4\text{B}} = 0.625 \text{ FLOP/B}$ (대역폭 효율 4배 증가) |

### Triton 컴파일러의 다단계 MLIR 변환 및 중간 표현 인스펙션 메커니즘

Triton 컴파일러 백엔드는 단순 생성을 넘어 다단계 MLIR Dialect 변환 파이프라인을 거쳐 CUBIN 바이너리를 빌드한다.

$$
\text{Python AST} \longrightarrow \text{TTIR} \longrightarrow \text{TTGIR} \longrightarrow \text{LLVM IR} \longrightarrow \text{PTX} \longrightarrow \text{CUBIN}
$$

- **TTIR (Triton IR)**: 타깃 독립적 블록 레벨 IR. 공통식 제거(CSE), 루프 불변 코드 이동(LICM), 데드 코드 제거(DCE)를 수행함.
- **TTGIR (Triton GPU IR)**: 타깃 하드웨어 인프라가 주입된 IR. 스레드 그리드 배치를 위한 `Blocked Layout`, Tensor Core 행렬 연산 매핑을 위한 `MMA Layout`, Shared Memory 뱅크 충돌 방지용 패딩이 적용된 `Shared Layout`, `DotOperand Layout` 속성을 부여함. `Coalesce Pass`, `Software Pipelining Pass`, `Warp Specialization Pass`가 작동함.
- **LLVM IR / PTX / CUBIN**: LLVM 백엔드를 거쳐 가상 어셈블리(PTX) 생성 후, `ptxas` 또는 CUDA 드라이버 API를 통해 SM에서 직접 실행되는 바이너리(CUBIN)로 완결됨.

엔지니어는 파이썬 런타임에서 `compiled_kernel.asm` 딕셔너리 프로퍼티를 호출하여 컴파일 중간 아티팩트를 인스펙션할 수 있다.

```python
# 컴파일러 중간 IR 및 바이너리 추출 인터페이스
ttir_code  = compiled_kernel.asm["ttir"]   # High-level Target-independent MLIR
ttgir_code = compiled_kernel.asm["ttgir"]  # Target-aware GPU Layout MLIR
llir_code  = compiled_kernel.asm["llir"]   # LLVM IR
ptx_code   = compiled_kernel.asm["ptx"]    # PTX Assembly
cubin_bin  = compiled_kernel.asm["cubin"]  # Executable CUBIN Binary

```
### 데이터 이동 흐름 텍스트 아키텍처 시각화

```
[ Un-fused Naive Softmax 데이터 흐름 (12회 접근, 총 8MN+4M 원소 트래픽) ]
HBM (DRAM) <== (Read 5MN+2M / Write 3MN+2M) ==> On-Chip Register / Shared Memory
  ├── 1. Max Reduction ===== (Write X_max) ====> HBM
  ├── 2. Subtract Max  <==== (Read X_max)  ====> HBM (Write Z)
  ├── 3. Exponential   <==== (Read Z)      ====> HBM (Write Exp_Z)
  ├── 4. Sum Reduction <==== (Read Exp_Z)  ====> HBM (Write Sum)
  └── 5. Division      <==== (Read Sum)    ====> HBM (Write Output)

[ Triton Fused Softmax 데이터 흐름 (2회 접근, 총 2MN 원소 트래픽) ]
HBM (DRAM) ─── Read X (MN 원소) ───> [ SRAM / Register 공간 내 상주 ] ─── Write Y (MN 원소) ───> HBM (DRAM)
                                      └── Max -> Sub -> Exp -> Sum -> Div 연산 온칩 일괄 처리

```
커널 결합이 HBM 대역폭 트래픽을 극적으로 절감함을 수치적으로 검증하였다. 2차원 리덕션 실행 시 안전하게 블록 크기를 할당하고 패딩을 제어하는 실무 최적화 기법으로 전환한다.

## 4. 2차원 행 단위 리덕션, 패딩 제어 및 온칩 메모리 한계 최적화

2차원 행렬 연산에서 행(Row) 단위 Softmax 리덕션을 수행할 때는 메모리 스트라이드 계산, 타일 크기의 2의 거듭제곱 정렬, 패딩 영역에 대한 수학적 중립 원소(Neutral Element) 주입이 필수적이다. 패딩값 선택의 정밀성은 연산 결과의 수치적 정확도와 직결된다.

### 핵심 시스템 용어 정의 및 메커니즘

- `Power-of-Two Alignment (next_power_of_2)`: 타일 크기를 입력 차원 크기 이상의 가장 가까운 2^k 거듭제곱 값으로 정렬하는 하드웨어 및 컴파일러 제약 제어 기법.
- `Row Stride Indexing`: 2차원 메모리 레이아웃에서 특정 행의 시작 포인터 위치를 정밀 산출하는 오프셋 연산 (`row_idx * input_row_stride`).
- `Neutral Element Padding (other=-inf)`: 마스킹 처리되는 무효 메모리 영역에 축약 연산 결과를 왜곡하지 않는 수학적 중립 원소($-\infty$)를 주입하는 기법.

### `BLOCK_SIZE = triton.next_power_of_2(N)` 정렬 필수성

GPU 내부의 텐서 코어(Tensor Core) 및 블록 축약 알고리즘(예: Balanced Tree Reduction)은 2^k 크기의 타일에 최적화되어 있다. 임의의 행 길이 N을 2의 거듭제곱 크기로 상향 정렬해야만 컴파일러의 레이아웃 인코딩(`TTGIR Blocked Layout`) 및 파이프라이닝이 올바르게 동작하며, 워프 내 스레드 간 축약 연산 시 인덱싱 다이버전스가 발생하지 않는다.

### 행 시작 주소 계산 포인터 연산

2차원 메모리 레이아웃에서 `row_idx`번째 행의 시작 메모리 포인터 주소 산식은 다음과 같다. $\text{row\_start\_ptr} = \text{input\_ptr} + \text{row\_idx} \times \text{input\_row\_stride}$

### `tl.load(..., other=-float('inf'))` 패딩값 수학적 당위성 증명

- **손계산 실습 조건:** 행 길이 N = 781인 텐서를 $\text{BLOCK\_SIZE} = 1024$로 읽어들임 (243개의 패딩 원소 발생). 행의 모든 실제 원소가 -2.0으로 동일하다고 가정.

#### 케이스 A: 패딩 공간에 잘못된 중립 원소 `other = 0.0` 주입 시 수치 파괴

1. **Max Reduction:** $\max(\underbrace{-2.0, \dots, -2.0}_{781\text{개}}, \underbrace{0.0, \dots, 0.0}_{243\text{개}}) = 0.0 \quad (\text{실제 최댓값 } -2.0\text{이 아닌 } 0.0\text{으로 변질})$
2. **Exponential 계산:**
   - 실제 유효 원소 연산: $\exp(-2.0 - 0.0) = \exp(-2.0) \approx 0.1353$
   - 무효 패딩 원소 연산: $\exp(0.0 - 0.0) = \exp(0.0) = 1.0$
3. **Sum Reduction:** $\sum \exp(\cdot) = (781 \times 0.1353) + (243 \times 1.0) = 105.67 + 243.0 = 348.67$ 패딩 원소들로 인해 243.0이라는 거대한 연산 왜곡이 분모에 주입되어 Softmax 확률 분포가 완전 파괴됨.

#### 케이스 B: 패딩 공간에 올바른 중립 원소 `other = -inf` 주입 시 수학적 정밀성 입증

1. **Max Reduction:** $\max(\underbrace{-2.0, \dots, -2.0}_{781\text{개}}, \underbrace{-\infty, \dots, -\infty}_{243\text{개}}) = -2.0 \quad (\text{진짜 최댓값 } -2.0\text{ 보존})$
2. **Exponential 계산 (중간 단계 명시):**
   - 실제 유효 원소 연산: $x_i - x_{\max} = -2.0 - (-2.0) = 0.0 \implies \exp(0.0) = 1.0$
   - 무효 패딩 원소 연산: $-\infty - (-2.0) = -\infty \implies \exp(-\infty) = 0.0$
3. **Sum Reduction:** $\sum \exp(\cdot) = (781 \times \exp(0.0)) + (243 \times \exp(-\infty)) = (781 \times 1.0) + (243 \times 0.0) = 781.0 + 0.0 = 781.0 \exp(-\infty) = 0.0$ 소멸 특성에 의해 무효 영역의 영향력이 완전 차단되어 수치적 완벽성이 입증됨.

### 대규모 N의 온칩 메모리(SRAM/Register) 한계점 및 대응

행 길이 N이 매우 커서 단일 타일 크기가 SM의 온칩 Shared Memory capacity 및 레지스터 파일 크기를 초과할 경우, 레지스터가 HBM으로 방출되는 Register Spilling이 발생하거나 SRAM 용량 초과로 커널 실행이 실패(`Launch Failure`)한다. 이 경우 단일 블록 일괄 로드가 불가능하므로, 차원을 분할하여 축약하는 Split-K 기법이나 루프 기반 타일링(Loop-based Tiling) 모듈을 도입해야 한다.

## 5. 정밀도 검증, 비동기 벤치마킹 및 가상 하드웨어 성능 정량 평가

GPU 커널의 비동기 실행(Asynchronous Execution) 특성과 JIT 컴파일 지연 시간은 성능 측정 시 왜곡을 유발한다. 이를 분리하고 Roofline Model의 물리적 한계선에 근거하여 커널의 실효 대역폭(Effective Bandwidth)과 수치 정밀도를 정량적으로 평가한다.

### 핵심 시스템 용어 정의 및 메커니즘

- `JIT Warmup Split`: 최초 실행 시 발생하는 JIT 컴파일($\text{AST} \to \text{TTIR} \to \text{TTGIR} \to \text{PTX} \to \text{CUBIN}$) 지연을 성능 측정 타임라인에서 엄격히 격리 분리하는 워밍업 기법.
- `Effective Bandwidth (GB/s)`: 커널이 물리적으로 달성해야 하는 최소 입출력 데이터 트래픽을 순수 실행 시간으로 나누어 산출한 실효 대역폭 평가지표.
- `torch.allclose Accuracy Validation`: 참조 baseline(PyTorch Native)과 개발 커널(Triton) 간의 연산 결과 오차 범위를 엄격 비교하는 정밀도 검증 인터페이스.

### GPU 비동기 실행 오버헤드 분리 및 정밀 측정

GPU 커널 호출은 비동기 제어로 작동하여 Python 제어권이 즉시 반환되므로, 연산 완료 시점을 동기화하기 위해 `cudaDeviceSynchronize()` 또는 CUDA Event(`torch.cuda.Event`) 수신기를 적용해야 한다. 사전 웜업(Warmup) 루프를 실행하여 JIT 컴파일 및 GPU 파워 스테이트 전이 지연을 완전히 제거한 후 본 측정을 진행한다.

### `triton.testing.do_bench` 내장 알고리즘

`triton.testing.do_bench` 메커니즘은 내부적으로 사전 웜업을 실행한 후, 반복 측정 구간 동안 CUDA Event timing을 통해 중앙값(Median) 및 20%/80% Quantile 신뢰 구간을 산출하여 벤치마킹 노이즈를 배제한다.

### 가상 하드웨어 기반 성능 정량 분석 및 Roofline Model 물리적 한계 입증

#### 1. 평가 환경 및 가상 하드웨어 스펙 설정

- **입력 데이터 텐서:** M = 4096, N = 1024, FP32 (원소당 4 Bytes).
  - 단일 텐서 용량: 4096 $\times 1024 \times 4\text{ Bytes} = 16,777,216\text{ Bytes} \approx 16.78\text{ MB}$.
- **가상 GPU 하드웨어 스펙:**
  - FP32 연산 성능: $20\text{ TFLOPS} (20 \times 10^{12}\text{ FLOPs/s})$.
  - HBM 메모리 대역폭: $1\text{ TB/s} (1 \times 10^{12}\text{ Bytes/s})$.

#### 2. 메모리 이동 트래픽 및 소요 시간 계산

1. `naive_softmax` **메모리 이동량 및 소요 시간 (8MN + 4M 원소):** $Q_{\text{naive}} = 8 \times 4096 \times 1024 \times 4\text{B} = 134,217,728\text{ Bytes} \approx 134.22\text{ MB}\quad T_{\text{naive\_mem}} = \frac{134,217,728\text{ Bytes}}{1 \times 10^{12}\text{ Bytes/s}} = 134.22\ \mu\text{s} \quad (0.13422\text{ ms})$
2. `Triton softmax` **메모리 이동량 및 소요 시간 (2MN 원소):** $Q_{\text{triton}} = 2 \times 4096 \times 1024 \times 4\text{B} = 33,554,432\text{ Bytes} \approx 33.55\text{ MB}\quad T_{\text{triton\_mem}} = \frac{33,554,432\text{ Bytes}}{1 \times 10^{12}\text{ Bytes/s}} = 33.55\ \mu\text{s} \quad (0.03355\text{ ms})$
3. **이론적 HBM 대역폭 기준 속도 향상 배수:** $\text{Theoretical Speedup} = \frac{134.22\ \mu\text{s}}{33.55\ \mu\text{s}} \approx 4.0\text{배}$

#### 3. Roofline Model 정량 평가 및 연산 시간 수치 입증

Softmax 연산의 총 부동소수점 연산 수(FLOPs) 계산: $\text{FLOPs} = 5MN = 5 \times 4096 \times 1024 = 20,971,520\text{ FLOPs} \approx 20.97\text{ MFLOPs}$

$20\text{ TFLOPS}$ 가상 GPU 상에서의 pure compute execution time ($T_{\text{compute}}$): $T_{\text{compute}} = \frac{20,971,520\text{ FLOPs}}{20 \times 10^{12}\text{ FLOPs/s}} = 1.048576\ \mu\text{s} \approx 1.05\ \mu\text{s}$

Softmax 연산의 산술 강도(Arithmetic Intensity, I) 산출: I = $\frac{\text{총 연산량 (FLOPs)}}{\text{Triton 최소 메모리 이동량 (Bytes)}} = \frac{5MN \text{ FLOPs}}{2MN \times 4 \text{ Bytes}} = \frac{5}{8} = 0.625\text{ FLOP/Byte}$

가상 하드웨어의 Operational Intensity Elbow Point (변곡점): $\text{Elbow Point} = \frac{\text{Peak Compute Performance}}{\text{Peak Memory Bandwidth}} = \frac{20 \times 10^{12}\text{ FLOPs/s}}{1 \times 10^{12}\text{ Bytes/s}} = 20\text{ FLOPs/Byte}$

- **물리적 결론:** Softmax의 산술 강도 I = $0.625\text{ FLOP/Byte}$는 하드웨어 변곡점인 $20\text{ FLOPs/Byte}$ 대비 32배 아래에 위치한다. 연산 시간($1.05\ \mu\text{s}$)에 비해 Triton의 HBM 이동 시간($33.55\ \mu\text{s}$)이 약 32배 압도적으로 크므로, Softmax 연산은 엄격한 **메모리 바운드(Memory-bound)** 영역에 종속된다. 따라서 메모리 트래픽을 4배 절감하는 커널 결합이 전체 execution time을 정확히 4.0배 단축시킴이 정량 입증된다.

#### 4. 실효 대역폭 분자 표준 산식 설정 당위성

$\text{Effective Bandwidth (GB/s)} = \frac{2MN \times \text{element\_size (Bytes)}}{\text{Measured Execution Time (Seconds)} \times 10^9}$ 실효 대역폭 계산 시 분자에 비효율적인 커널의 실제 메모리 이동량($8MN \times 4\text{B}$)을 대입하지 않고, 알고리즘이 달성해야 하는 이론적 최저 입출력 이동량인 **$2MN \times 4\text{Bytes}$ (읽기 1회 + 쓰기 1회)**를 일관되게 적용한다. 이는 불필요한 HBM 방출 및 재로드라는 커널의 구조적 비효율성을 대역폭 수치상의 감점으로 공정하게 반영하기 위함이다.

### `torch.compile` (TorchInductor) 최적화와의 비교

PyTorch의 TorchInductor 백엔드는 FX 그래프 분석 후 Triton DSL 코드를 동적 자동 생성하여 연산자를 결합한다. 수동으로 개발한 Triton 커널과 TorchInductor 자동 생성 커널은 동일한 수준의 실효 대역폭 보장 및 Roofline 상한선 접근성을 나타낸다.

## 6. 결론 및 세션 공유 가이드라인 (실습 기록, 참고 문헌 및 디버깅 종합)

본 보고서는 스레드 중심 CUDA 모델에서 타일 중심 Triton 모델로의 패러다임 전환, 포인터 연산 및 마스킹 메커니즘, 커널 결합의 정량적 HBM 트래픽 절감 모델, 2차원 리덕션 및 패딩 제어, Roofline Model 기반 정량 평가를 다루었다. 각 엔지니어링 팀이 자사의 GPU 환경에서 실험을 재현하고 검증할 수 있도록 실무 가이드라인을 제공한다.

### 실험 환경 프로파일링 기록 가이드라인

- 타 장치 결과 수치와의 단순 혼용을 엄격히 금지하며, 벤치마킹 데이터 기록 시 아래 환경 프로필을 표준 명시해야 함:
  - GPU 모델명 (예: NVIDIA A100-SXM4-80GB, H100-SXM5-80GB, RTX 4090 등)
  - PyTorch 및 Triton 컴파일러 버전 프로파일
  - 측정 대상 텐서 형상(Tensor Shape: M, N) 및 데이터 타입(FP32, FP16, BF16)

### PyTorch 및 Triton 디버깅 환경 변수 플래그

코드 디버깅 및 생성된 커널 추적 시 다음 환경 변수를 활용한다.

- `TORCH_COMPILE_DEBUG=1`: TorchInductor의 그래프 캡처 과정, FX 그래프 변환 및 동적 생성된 Triton DSL 코드 파일(`output_code.py`)을 디버그 폴더에 출력 기록함.
- `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`: 생성되는 Triton 커널 함수명에 원본 ATen 연산자 정보(예: `triton_poi_fused_softmax_0`)를 부여하여 프로파일링 추적성을 제공함.
- `TORCHINDUCTOR_BENCHMARK_KERNEL=1`: 생성된 개별 Triton 커널을 독립 격리하여 단독 벤치마킹하는 단독 하네스 코드를 자동 생성함.

### CPU 해석기 모드 (`TRITON_INTERPRET=1`) 활용 및 기술적 제약 사항

GPU가 부재한 환경이거나 인덱싱 및 마스크 불리언 배열의 동작을 단계를 밟아가며 추적하고자 할 때 `TRITON_INTERPRET=1` 환경 변수를 적용한다.

```bash
# CPU Interpreter 모드 실행 및 pdb 디버깅 예시
TRITON_INTERPRET=1 python3 test_softmax.py

```
- **작동 원리:** Triton JIT 컴파일을 우회하고 NumPy 기반으로 CPU 상에서 커널 프로그램 인스턴스를 순차 실행함. 커널 내부에서 `print(tensor)` 또는 `pdb.set_trace()`를 직접 삽입하여 intermediate tensor 상태 인스펙션이 가능함.
- **명시적 기술 제약 사항:**
  1. `bfloat16` 데이터 타입을 네이티브하게 지원하지 않음. CPU 인터프리터 디버깅 시에는 `tl.cast(tensor, tl.float32)`를 통해 `float32`로 변환하여 테스트해야 함.
  2. 간접 메모리 접근 패턴(Indirect Memory Access, 예: `ptr = tl.load(ptr); x = tl.load(ptr)`)은 CPU 인터프리터 상에서 실행이 불가함.

### 참고 문헌 및 핵심 소스 체계

1. **NVIDIA CUDA Programming Guide (Section 1.2 Programming Model)**: Thread Blocks, Grids, Warps, SIMT 및 Tile Programming Abstraction 절.
2. **Deep Dive into Triton Internals (Kapil Sharma)**: TTIR, TTGIR, LLVM IR, PTX, CUBIN 다단계 MLIR 로워링 파이프라인 분석 절.
3. **Debugging Triton Documentation**: Interpreter Mode, Debugging Operations 및 Limitations 절.

### 팀 내 기술 토론 및 검증 질문 리스트

1. `pid`와 `BLOCK_SIZE`가 결합되어 오프셋 벡터 `offsets`를 도출할 때, 마스크 배열 `mask = offsets < N`이 Out-of-Bounds 접근을 차단하는 하드웨어 가스켓 메커니즘을 설명할 수 있는가?
2. `naive_softmax` 실행 시 발생하는 12회 메모리 접근 트랜잭션(8MN + 4M 원소) 대비 Triton 결합 커널이 2MN 원소로 메모리 트래픽을 절감하는 정량적 계량 모델을 제시할 수 있는가?
3. 2차원 행 리덕션 시 무효 패딩 공간에 `other = 0.0`을 주입하는 경우와 `other = -inf`를 주입하는 경우, $\max(\cdot)$ 및 $\sum \exp(\cdot)$ 연산 결과에서 수치 왜곡이 발생하는 수학적 원인을 손계산 결과로 증명할 수 있는가?
4. Softmax 연산의 산술 강도(I = $0.625\text{ FLOP/B}$)와 가상 하드웨어 변곡점($20\text{ FLOPs/B}$)을 대조하여, 이 연산이 Compute-bound가 아닌 Memory-bound 구역에 속함을 Roofline Model로 정량 입증할 수 있는가?
