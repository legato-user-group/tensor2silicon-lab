<WeekNavBar :week="2" topic="모델에서 실리콘까지" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송민배)', 'badge': 'PyTorch·XLA·MAX 2부작'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': '다이어그램 & 스택 해부'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': '실행 스택 핵심 요약'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '인터랙티브 HTML & 슬라이드'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': '컴파일 파이프라인 정리'}]" />

> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **mbsong (송민배)** (`PyTorch·XLA·MAX 2부작`)
>
> 💡 **핵심 기여 요약**: PyTorch Eager/Compile부터 JAX·OpenXLA, Modular MAX·Mojo까지 실행 스택의 IR 변환과 런타임 디스패치 과정을 2부작으로 정밀 비교했습니다.

---

# 컴파일러 최적화: 데이터 이동과 연산 배치

> **한 줄 답.** 컴파일러는 데이터 이동, 메모리 배치, 연산 스케줄링을 최적화하여 주어진 하드웨어의 실제 성능을 Roofline ceiling에 가깝게 끌어올린다.

Roofline 모델에서 컴파일러 최적화는 데이터 이동량과 연산기 활용률을 개선하는 역할을 한다.

컴파일러의 핵심 역할은 다음과 같다.

$$
\boxed{\text{같은 모델을 하드웨어에서 더 적게 움직이고, 더 잘 재사용하고, 더 효율적으로 계산하게 만드는 것} }
$$

즉 컴파일러는 하드웨어의 **HBM bandwidth나 Tensor Core 개수 자체를 늘리지는 못하지만**, 주어진 하드웨어에서 실제 성능을 Roofline ceiling에 더 가깝게 끌어올린다.

---

## 전체 그림

| 병목                      | 컴파일러가 최적화하는 문제      | 대표 방법                                                 |
| ----------------------- | ------------------- | ----------------------------------------------------- |
| Memory Bound            | 불필요한 memory traffic | Fusion, tiling, layout, buffer reuse                  |
| Compute Bound           | 연산기 utilization 부족  | Kernel selection, vectorization, tensorization        |
| Communication Bound     | GPU/NPU 간 통신 과다     | communication fusion, overlap, partitioning           |
| Memory Capacity         | tensor/KV가 너무 큼     | buffer planning, recomputation, quantization lowering |
| Launch/Control overhead | kernel이 너무 잘게 나뉨    | graph fusion, static scheduling                       |

---

## 1. Memory Bound 최적화: Operator Fusion

메모리 이동량 감소는 컴파일러의 핵심 최적화 영역 중 하나다.

Roofline에서

$$
AI=\frac{FLOPs}{Bytes}
$$

이므로 컴파일러는 주로 **Bytes를 줄인다.**

Fusion 적용 전:

```text
MatMul
 ↓
HBM write
 ↓
Bias
 ↓
HBM write
 ↓
Activation
 ↓
HBM write
```

Fusion 적용 후:

```text
MatMul → Bias → Activation
```

중간 결과를 register/SRAM에 유지한다.

따라서

$$
Memory\ traffic \downarrow
$$

대표적인 fusion 대상은 다음과 같다.

- MatMul + Bias
- MatMul + Activation
- RMSNorm + residual
- QKV projection
- RoPE
- Attention subgraph
- SwiGLU

---

## 2. Tiling

큰 tensor를 그대로 처리하면 SRAM에 들어가지 않는다.

예를 들어

$$
C=A\times B
$$

를 전체 matrix 단위로 계산하는 대신:

```text
A tile
B tile
  ↓
SRAM
  ↓
MAC array
```

처럼 조각내서 계산한다.

컴파일러는 타일 크기와 실행 순서를 결정한다.

예:

$$
M \times N \times K
$$

에 대해

```text
tile_m = 128
tile_n = 128
tile_k = 32
```

같은 tile size를 선택한다.

좋은 tiling은:

$$
Data\ reuse \uparrow
$$

$$
HBM\ access \downarrow
$$

를 만든다.

타일링은 특히 NPU 컴파일러에서 중요하다.

---

## 3. Memory Layout 최적화

같은 tensor라도 memory layout에 따라 성능이 크게 달라진다.

예:

```text
NCHW
NHWC
blocked layout
```

또는:

```text
[row][column]

vs

[tile][tile]
```

컴파일러가 하드웨어에 맞는 layout으로 바꿀 수 있다.

예를 들어 Tensor Core가 특정 형태를 선호한다면:

$$
Tensor \rightarrow TensorCore-friendly\ layout
$$

로 변환한다.

주요 목표는 다음과 같다.

- contiguous access
- coalesced access
- bank conflict 감소
- vector load 가능

---

## 4. Buffer Allocation / Memory Planning

컴파일러는 tensor를 어디에 둘지도 정할 수 있다.

예:

```text
HBM
L2
SRAM
Register
```

예를 들어:

```text
Tensor A → SRAM
Tensor B → SRAM
Intermediate → Register
Output → HBM
```

처럼 memory hierarchy에 배치한다.

SRAM 용량이 제한되므로, 유지할 데이터와 내보낼 데이터를 결정해야 한다.

이는 NPU 컴파일러의 주요 스케줄링 문제다.

---

## 5. Buffer Reuse

예를 들어:

```text
Tensor A
 ↓
사용 완료

Tensor B
```

라면 A가 쓰던 memory를 B가 재사용할 수 있다.

즉:

```text
buffer 1 → Tensor A
        → Tensor B
        → Tensor C
```

처럼 재사용한다.

그 결과 최대 메모리 사용량이 감소한다.

$$
Peak\ Memory\ Usage \downarrow
$$

LLM에서는 activation이나 intermediate buffer 관리에 중요하다.

---

## 6. Prefetch / Double Buffering

Prefetch와 double buffering의 실행 방식은 다음과 같다.

```text
Compute tile N

동시에

Load tile N+1
```

이 실행 방식도 컴파일러 스케줄링의 대상이다.

컴파일러가 DMA와 compute를 배치해서:

```text
Load A
Compute A + Load B
Compute B + Load C
Compute C
```

처럼 schedule할 수 있다.

즉:

$$
Memory\ latency
$$

를

$$
Compute
$$

뒤에 숨긴다.

이 기법은 NPU 컴파일러에서 특히 중요하다.

---

## 7. Compute Bound 최적화: Kernel Selection

Compute-bound 최적화에서는 FLOPs와 함께 연산기 활용률을 확인해야 한다.

같은 연산도 여러 kernel이 있을 수 있다.

예:

```text
MatMul
```

에 대해:

```text
kernel A → small batch
kernel B → large batch
kernel C → FP8
kernel D → INT8
```

컴파일러가 shape과 hardware를 보고 가장 적절한 kernel을 고른다.

---

## 8. Vectorization

예를 들어 CPU/NPU가 한 번에 8개의 값을 계산할 수 있다면:

기존:

```text
a0*b0
a1*b1
a2*b2
...
```

컴파일러가:

```text
[a0..a7] × [b0..b7]
```

로 바꾼다.

즉 SIMD/vector instruction을 사용한다.

$$
Compute\ utilization \uparrow
$$

---

## 9. Tensorization

AI accelerator에서는 vectorization보다 더 큰 단위가 있다.

예를 들어 Tensor Core가:

$$
16\times16\times16
$$

matrix multiply를 한 instruction으로 처리한다면, 컴파일러는 일반적인 loop:

```text
for i
 for j
  for k
```

를 찾아서

```text
TensorCore MMA
```

instruction으로 바꾼다.

이를 **tensorization**이라 한다.

TVM 같은 compiler에서 매우 중요한 개념이다.

---

## 10. Loop Transformation

컴파일러가 loop 순서를 바꾸는 것도 중요하다.

예:

```text
for i
 for j
  for k
```

를

```text
for i_tile
 for k_tile
  for j_tile
```

처럼 변경한다.

대표적인 기법은 다음과 같다.

- loop tiling
- loop interchange
- loop unrolling
- vectorization
- parallelization

목표는 캐시 지역성과 연산기 활용률을 높이는 것이다.

$$
Cache\ locality \uparrow
$$

$$
Compute\ utilization \uparrow
$$

---

## 11. Quantization lowering

Quantization은 모델 알고리즘 영역처럼 보이지만 compiler 역할도 크다.

예를 들어 모델이:

```text
FP16 MatMul
```

이라도 compiler가:

```text
INT8 MatMul
+
scale
+
dequant
```

으로 lower할 수 있다.

또는:

```text
INT4 weights
→ packed representation
→ hardware INT4 instruction
```

으로 변환한다.

즉 compiler가:

$$
Model\ representation
\rightarrow
Hardware\ instruction
$$

을 연결한다.

---

## 12. Sparsity 활용

모델 weight에 sparsity가 있더라도 hardware가 자동으로 빠르게 계산하는 것은 아니다.

Compiler가:

```text
dense matmul
```

을

```text
sparse matmul
```

로 바꾸거나, hardware sparse instruction을 사용하도록 내려야 한다.

예:

$$
2:4\ structured\ sparsity
$$

를 지원하는 accelerator라면 compiler가 이를 검출하고 sparse Tensor Core instruction을 사용한다.

---

## 13. Kernel Fusion vs Graph Fusion

Graph fusion과 kernel fusion은 적용 수준이 다르다.

### Graph-level

```text
MatMul
 ↓
Bias
 ↓
ReLU
```

를 하나의 graph node로 묶는다.

### Kernel-level

Kernel-level fusion은 실제로 하나의 커널을 생성한다.

```text
fused_matmul_bias_relu()
```

즉 compiler pipeline은 대략:

```text
Graph IR
   ↓
Fusion
   ↓
Tensor IR
   ↓
Tiling
   ↓
Scheduling
   ↓
Kernel IR
   ↓
Machine code
```

---

## 14. Communication Bound 최적화

Multi-GPU/NPU에서는 compiler가:

```text
MatMul
 ↓
AllReduce
 ↓
MatMul
```

을 그대로 실행하는 대신:

```text
MatMul chunk 1
    ↓
AllReduce chunk 1

동시에

MatMul chunk 2
```

처럼 communication과 compute를 overlap할 수 있다.

즉:

$$
T_{total}
\neq
T_{compute}+T_{communication}
$$

이며, 이상적으로는

$$
T_{total}
\approx
\max(T_{compute},T_{communication})
$$

에 가깝게 만든다.

---

## 15. 병렬화 분할 (Parallelism Partitioning)

모델을 여러 device에 나눌 때:

```text
Tensor Parallel
Pipeline Parallel
Expert Parallel
```

을 어떻게 배치할지도 compiler/runtime가 담당할 수 있다.

예:

```text
Layer 0-10 → GPU0
Layer 11-20 → GPU1
```

또는:

```text
MatMul shard → GPU0/GPU1/GPU2/GPU3
```

이 과정은:

$$
Compute + Memory + Communication
$$

세 가지를 동시에 최적화해야 하는 문제이다.

---

## 16. 컴파일러 최적화의 한계

컴파일러는:

$$
HBM\ bandwidth
$$

자체를 늘리지 못한다.

또한:

$$
TensorCore\ count
$$

도 늘릴 수 없다.

예를 들어 hardware가:

```text
1 TB/s HBM
100 TFLOPS
```

라면 compiler가 이를

```text
2 TB/s
200 TFLOPS
```

로 만들 수는 없다.

대신 실제 성능이:

```text
20 TFLOPS
```

밖에 안 나오고 있었다면 compiler가:

```text
70~90 TFLOPS
```

에 가깝게 만드는 역할을 한다.

즉:

$$
\boxed{
Compiler = hardware utilization optimizer
}
$$

로 요약할 수 있다.

---

## 17. Roofline 관점의 최적화 목표

### Memory-bound workload

컴파일러 목표:

$$
Bytes \downarrow
$$

대표적인 방법은 다음과 같다.

- Fusion
- Tiling
- Buffer reuse
- Layout optimization
- Prefetch
- Double buffering

결과:

$$
Arithmetic\ Intensity \uparrow
$$

즉 Roofline에서 **오른쪽으로 이동**한다.

---

### Compute-bound workload

컴파일러 목표:

$$
Actual\ FLOPS \rightarrow Peak\ FLOPS
$$

대표적인 방법은 다음과 같다.

- Tensorization
- Vectorization
- Kernel selection
- Loop optimization
- Instruction scheduling

즉 Roofline의 위쪽 ceiling에 더 가까이 간다.

---

### Communication-bound workload

컴파일러 목표:

$$
Communication \downarrow
$$

또는:

$$
Communication \parallel Compute
$$

대표적인 방법은 다음과 같다.

- Partitioning
- Collective fusion
- Communication overlap
- Placement

---

## 18. NPU 컴파일러의 최적화 파이프라인

```text
               AI Model
                  │
                  ▼
              Graph IR
                  │
       ┌──────────┴───────────┐
       │                      │
      Fusion              Quantization
       │                      │
       └──────────┬───────────┘
                  ▼
              Tensor IR
                  │
       ┌──────────┼─────────────┐
       │          │             │
     Tiling     Layout       Buffering
       │          │             │
       └──────────┼─────────────┘
                  ▼
              Schedule
                  │
         ┌────────┴─────────┐
         │                  │
     Tensorization      Prefetch/DMA
         │                  │
         └────────┬─────────┘
                  ▼
             Machine Code
                  │
                  ▼
           NPU / GPU Hardware
```

컴파일러의 역할은 다음과 같이 요약된다.

$$
\boxed{
\text{Compiler는 데이터 이동, 연산 배치, 메모리 배치, instruction mapping을 최적화한다.}
}
$$

**Tiling → Buffer allocation → Operator fusion → Tensorization → DMA/compute scheduling**은 NPU 컴파일러의 주요 학습 순서다. 이 다섯 영역에서 하드웨어 구조와 컴파일러 최적화가 직접 연결된다.


---

# 예제로 따라가기: Y = ReLU(XW + b)가 코드에서 커널까지 내려가는 길

> 식 하나를 JAX와 PyTorch 양쪽에서 실제로 컴파일해 각 단계의 출력을 그대로 실었다. 사용한 shape은 `X[16,8]`, `W[8,4]`, `b[4]`, 결과 `Y[16,4]`이고, 실행 환경은 **CPU 백엔드**(JAX 0.11.1, PyTorch 2.14 CPU)다. GPU/TPU였다면 달라지는 지점은 각 단계 끝에 따로 적었다.

```
                JAX                                        PyTorch 2
   ────────────────────────────────          ─────────────────────────────────────
   jax.nn.relu(x @ w + b)                    torch.relu(x @ w + b)
      │ ① jax.jit 추적                          │ ① TorchDynamo 바이트코드 해석
      ▼                                        ▼
   jaxpr  (dot_general, add, max)            FX 그래프 (@, +, torch.relu) + guard
      │ ② lowering                             │ ② AOT Autograd
      ▼                                        ▼
   StableHLO  (dot_general, add, maximum)    ATen forward 그래프 / backward 그래프
      │ ③ XLA 최적화                            │ ③ TorchInductor
      ▼                                        ▼
   HLO: fusion 2개                           addmm(외부 커널) + 퓨전 커널 1개 + 래퍼
      │ ④ 코드 생성                             │ ④ 코드 생성
      ▼                                        ▼
   CPU: LLVM → 네이티브 / TPU: LLO            CPU: C++ 벡터 코드 / GPU: Triton
      │ ⑤ PjRt 디스패치                          │ ⑤ 래퍼가 커널 순서대로 런치
      ▼                                        ▼
   jax.Array (future)                        torch.Tensor (핸들, grad_fn 연결)
```

각 단계의 일반 원리는 [② 연산 → 그래프·IR](./02-ops-to-graph-ir.md), [③ IR → 실행 코드](./03-ir-to-executable.md), [④ 실행 요청 → 커널 실행](./04-dispatch-to-kernel.md) 참고.

---

## A. JAX 경로

### A-0. 코드

```python
import jax, jax.numpy as jnp

def f(x, w, b):
    return jax.nn.relu(x @ w + b)

x = jnp.ones((16, 8)); w = jnp.ones((8, 4)); b = jnp.ones((4,))
y = jax.jit(f)(x, w, b)
```

### A-1. 추적 → jaxpr

`jax.jit(f)`를 처음 호출하면 실제 값 대신 Tracer(shape·dtype만 있는 추상값)로 `f`를 한 번 실행한다. `x @ w`는 `dot_general`, `+`는 `add`, `relu`는 `max(·, 0)`로 기록된다.

```python
print(jax.make_jaxpr(f)(x, w, b))
```

```
{ lambda ; a:f32[16,8] b:f32[8,4] c:f32[4]. let
    d:f32[16,4] = dot_general[
      dimension_numbers=(([1], [0]), ([], []))
      preferred_element_type=float32
    ] a b
    e:f32[1,4] = broadcast_in_dim[broadcast_dimensions=(1,)] c
    f:f32[16,4] = add d e
    g:f32[16,4] = custom_jvp_call[
      name=relu
      call_jaxpr={ lambda ; h:f32[16,4]. let
          i:f32[16,4] = jit[
            name=relu
            jaxpr={ lambda ; h:f32[16,4]. let
                i:f32[16,4] = max h 0.0:f32[]
              in (i,) }
          ] h
        in (i,) }
      jvp=jvp
      symbolic_zeros=False
    ] f
  in (g,) }
```

읽을 점:

- `dot_general`의 `dimension_numbers=(([1],[0]),([],[]))`: X의 축 1과 W의 축 0을 contracting, batching 축은 없음. FLOPs = 2·16·8·4 = 1024.
- `b[4]`를 `[16,4]`에 더하기 위해 `broadcast_in_dim`이 **명시적 연산으로** 들어갔다. NumPy 브로드캐스팅이 IR에서는 실제 노드다.
- `relu`는 `custom_jvp_call`로 감싸여 있다. `jax.nn.relu`가 0에서의 미분 규칙을 직접 정의해 두었기 때문이다. `grad`를 적용하면 이 규칙이 쓰인다. 안쪽은 결국 `max(h, 0.0)`이다.

### A-2. lowering → StableHLO

jaxpr의 primitive마다 lowering 규칙이 있어서 StableHLO(MLIR) 연산으로 1:1에 가깝게 바뀐다.

```python
print(jax.jit(f).lower(x, w, b).as_text())
```

```mlir
module @jit_f {
  func.func public @main(%arg0: tensor<16x8xf32>, %arg1: tensor<8x4xf32>, %arg2: tensor<4xf32>)
      -> tensor<16x4xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
         : (tensor<16x8xf32>, tensor<8x4xf32>) -> tensor<16x4xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<16x4xf32>
    %3 = stablehlo.add %0, %2 : tensor<16x4xf32>
    %4 = call @relu(%3) : (tensor<16x4xf32>) -> tensor<16x4xf32>
    return %4 : tensor<16x4xf32>
  }
  func.func private @relu(%arg0: tensor<16x4xf32>) -> tensor<16x4xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x4xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x4xf32>
    return %1 : tensor<16x4xf32>
  }
}
```

읽을 점:

- `custom_jvp_call`이 사라지고 `relu`가 **일반 함수 호출**이 됐다. 미분 규칙은 jaxpr 수준의 일이고, 백엔드는 알 필요가 없다.
- 브로드캐스트가 두 단계(`[4]→[1,4]→[16,4]`)로 펼쳐졌다. 아직 아무것도 최적화되지 않은 상태다.
- **여기서 JAX의 역할이 끝난다.** 이 텍스트가 XLA로 넘어간다.

### A-3. XLA 최적화 → HLO

```python
print(jax.jit(f).lower(x, w, b).compile().as_text())
```

메타데이터를 걷어낸 핵심 부분:

```
HloModule jit_f, is_scheduled=true,
  entry_computation_layout={(f32[16,8]{1,0}, f32[8,4]{1,0}, f32[4]{0})->f32[16,4]{1,0}}

%fused_computation (param_0: f32[16,8], param_1: f32[8,4]) -> f32[16,4] {
  %param_0 = f32[16,8]{1,0} parameter(0)
  %param_1 = f32[8,4]{1,0} parameter(1)
  ROOT %dot_general.0 = f32[16,4]{1,0} dot(%param_0, %param_1),
       lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

%fused_computation.1 (param_0.2: f32[16,4], param_1.4: f32[4]) -> f32[16,4] {
  %param_0.2 = f32[16,4]{1,0} parameter(0)
  %param_1.4 = f32[4]{0} parameter(1)
  %add.1 = f32[16,4]{1,0} broadcast(%param_1.4), dimensions={1}
  %add.0 = f32[16,4]{1,0} add(%param_0.2, %add.1)
  %constant.2 = f32[] constant(0)
  %max.5 = f32[16,4]{1,0} broadcast(%constant.2), dimensions={}
  ROOT %max.4 = f32[16,4]{1,0} maximum(%add.0, %max.5)
}

ENTRY %main.2 (x.1: f32[16,8], w.1: f32[8,4], b.1: f32[4]) -> f32[16,4] {
  %x.1 = f32[16,8]{1,0} parameter(0)
  %w.1 = f32[8,4]{1,0} parameter(1)
  %b.1 = f32[4]{0} parameter(2)
  %ynn_fusion = f32[16,4]{1,0} fusion(%x.1, %w.1), kind=kCustom, calls=%fused_computation
  ROOT %broadcast_maximum_fusion = f32[16,4]{1,0} fusion(%ynn_fusion, %b.1),
       kind=kLoop, calls=%fused_computation.1
}
```

XLA가 결정한 것:

| 결정 | 이 예제에서 |
|---|---|
| **퓨전 경계** | 연산 6개(dot, broadcast×2, add, constant, maximum)가 **fusion 2개**로 묶였다. `kCustom` fusion은 matmul을 CPU 행렬곱 라이브러리(YNN)로 보내는 것이고, `kLoop` fusion은 `broadcast + add + maximum`을 **루프 하나**로 합친 것이다. 중간 텐서 `x@w+b`는 HBM(CPU에서는 RAM)에 내려가지 않는다. |
| **레이아웃** | 모든 텐서에 `{1,0}`(row-major)이 붙었다. StableHLO에는 없던 정보다. |
| **브로드캐스트 단순화** | `[4]→[1,4]→[16,4]` 두 단계가 `broadcast(dimensions={1})` 한 번으로 접혔다. |
| **함수 인라인** | `@relu` 호출이 사라지고 `maximum`이 fusion 안으로 들어갔다. |
| **스케줄** | `is_scheduled=true`. ENTRY의 두 fusion 실행 순서가 확정됐다. |
| **버퍼** | 파라미터 3개, 중간 버퍼 1개(`%ynn_fusion` 결과), 출력 1개. |

**TPU였다면**: dot과 add·maximum이 **하나의 fusion**(matmul + epilogue)으로 합쳐질 가능성이 높다. 또 `[16,8]`, `[8,4]` 같은 작은 shape은 MXU 타일(f32 8×128)에 맞춰 **패딩**되고, HBM→VMEM DMA가 이 시점에 정적으로 스케줄된다. 최적화된 HLO 텍스트 자체는 같은 명령으로 확인할 수 있다.

### A-4. 코드 생성

- **CPU**: 각 fusion이 LLVM IR로 내려가고 LLVM이 x86 벡터 명령으로 컴파일한다. matmul은 라이브러리 커널을 호출한다.
- **TPU**: HLO → LLO. `kLoop` fusion은 VPU 명령(8×128 타일 단위 add, max)과 DMA 명령이 VLIW 번들로 묶이고, dot fusion은 MXU 명령이 된다. 결과 바이너리가 TPU IMEM에 로드된다. 이 단계는 비공개라 텍스트로 볼 수 없고 프로파일러로만 관찰한다.

### A-5. 디스패치

```python
y = jax.jit(f)(x, w, b)   # 즉시 반환. type(y) = jaxlib._jax.ArrayImpl, y.shape = (16, 4)
y.block_until_ready()     # 여기서 실제 계산 완료를 기다림
```

두 번째 호출부터는 A-1~A-4를 전부 건너뛰고, `(f32[16,8], f32[8,4], f32[4])` 캐시 키로 컴파일된 실행 파일을 찾아 PjRt 큐에 넣는다.

PjRt 큐는 컴파일된 XLA executable과 디바이스 버퍼를 받아 GPU/TPU 같은 장치에 비동기 실행을 제출하는 런타임 내부의 작업 흐름이며, 실제 CUDA stream이나 TPU command queue는 각 PjRt 플러그인이 구현합니다.

---

## B. PyTorch 경로

### B-0. 코드

```python
import torch

def f(x, w, b):
    return torch.relu(x @ w + b)

x = torch.ones(16, 8)
w = torch.ones(8, 4, requires_grad=True)
b = torch.ones(4, requires_grad=True)
y = torch.compile(f)(x, w, b)
```

`w`, `b`에 `requires_grad=True`를 준 것이 JAX 예제와의 차이다. PyTorch는 이 플래그 때문에 **backward까지 컴파일 시점에 만든다.**

### B-1. TorchDynamo → FX 그래프

`TORCH_LOGS="graph_code"`:

```python
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[16, 8][8, 1]cpu", L_w_: "f32[8, 4][4, 1]cpu", L_b_: "f32[4][1]cpu"):
        l_x_ = L_x_
        l_w_ = L_w_
        l_b_ = L_b_
        matmul: "f32[16, 4][4, 1]cpu" = l_x_ @ l_w_;  l_x_ = l_w_ = None
        add: "f32[16, 4][4, 1]cpu" = matmul + l_b_;  matmul = l_b_ = None
        relu: "f32[16, 4][4, 1]cpu" = torch.relu(add);  add = None
        return (relu,)
```

읽을 점:

- Dynamo는 `f`를 실행하지 않고 바이트코드를 해석해 이 그래프를 만들었다. 노드의 op는 아직 **사용자 수준**(`@`, `+`, `torch.relu`)이다.
- 타입 주석 `f32[16, 8][8, 1]cpu`는 shape, **stride**, device다. 이것들이 그대로 **guard**가 된다. 다음 호출에서 `x`의 shape이나 stride가 다르면 재컴파일한다.
- jaxpr과 달리 브로드캐스트 노드가 없다. `matmul + l_b_`가 그냥 Python `+`다. 브로드캐스트는 다음 단계에서 ATen 의미론으로 처리된다.
- 이 그래프는 `GraphModule`이므로 그대로 Python으로 실행할 수도 있다.

### B-2. AOT Autograd → ATen forward / backward 그래프

`TORCH_LOGS="aot_graphs"`:

```python
# ===== Forward graph 0 =====
def forward(self, primals_1: "f32[16, 8]", primals_2: "f32[8, 4]", primals_3: "f32[4]"):
    mm:   "f32[16, 4]" = torch.ops.aten.mm.default(primals_1, primals_2)
    add:  "f32[16, 4]" = torch.ops.aten.add.Tensor(mm, primals_3)
    relu: "f32[16, 4]" = torch.ops.aten.relu.default(add)
    # Backward of forward node:
    le:      "b8[16, 4]"  = torch.ops.aten.le.Scalar(relu, 0)
    permute: "f32[8, 16]" = torch.ops.aten.permute.default(primals_1, [1, 0])
    return (relu, le, permute)

# ===== Backward graph 0 =====
def forward(self, le: "b8[16, 4]", permute: "f32[8, 16]", tangents_1: "f32[16, 4]"):
    full_default: "f32[]"    = torch.ops.aten.full.default([], 0.0, ...)
    where: "f32[16, 4]"      = torch.ops.aten.where.self(le, full_default, tangents_1)
    sum_1: "f32[1, 4]"       = torch.ops.aten.sum.dim_IntList(where, [0], True)
    view:  "f32[4]"          = torch.ops.aten.view.default(sum_1, [4])
    mm_1:  "f32[8, 4]"       = torch.ops.aten.mm.default(permute, where)
    return (None, mm_1, view)
```

읽을 점:

- op가 전부 `torch.ops.aten.*`로 내려갔다. `@` → `aten.mm`, `+` → `aten.add.Tensor`, `torch.relu` → `aten.relu`.
- forward 그래프가 **backward용 값을 미리 계산해서 함께 반환**한다. `le = (relu <= 0)`는 ReLU의 미분 마스크이고, `permute = Xᵀ`는 `∂L/∂W = Xᵀ·(∂L/∂Y)`에 쓰인다. 어떤 값을 저장할지 정한 것이 파티셔너다.
- backward 그래프는 [①](./01-model-to-ops.md)의 공식 그대로다. `where`가 ReLU 미분(0 이하면 0, 아니면 gradient 통과), `sum(where, dim=0)`이 `∂L/∂b`, `mm(Xᵀ, where)`가 `∂L/∂W`. `x`는 `requires_grad=False`라 첫 반환값이 `None`이다.
- 여기까지 **PyTorch 프레임워크의 역할**이고, 두 그래프가 Inductor로 넘어간다. JAX의 StableHLO에 해당한다.

### B-3. TorchInductor → 커널 + 래퍼

`TORCH_LOGS="output_code"`. forward에 대해 Inductor가 만든 코드는 **C++ 커널 하나**와 **Python 래퍼**다.

**퓨전 커널(CPU, C++)**:

```cpp
// cpp_fused_relu_threshold_backward_0
extern "C" void kernel(float* in_out_ptr0, bool* out_ptr0)
{
    for (int64_t x0 = 0; x0 < 64; x0 += 8)          // 16*4 = 64 원소, 8개씩 SIMD
    {
        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + x0, 8);
        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));   // relu
        auto tmp3 = at::vec::Vectorized<float>(0.0f);
        auto tmp4 = at::vec::VecMask<float,1>(tmp1 <= tmp3);       // le (backward 마스크)
        tmp1.store(in_out_ptr0 + x0);                               // in-place로 relu 결과 저장
        tmp4.store(out_ptr0 + x0, 8);                               // 마스크 저장
    }
}
```

**래퍼(Python)**:

```python
def call(args):
    primals_1, primals_2, primals_3 = args
    assert_size_stride_grouped((primals_3, primals_1, primals_2),
                               ((4,), (16, 8), (8, 4)), ((1,), (8, 1), (4, 1)), 'input')
    buf0 = empty_strided_cpu((16, 4), (4, 1), torch.float32)
    # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
    extern_kernels.addmm(primals_3, primals_1, primals_2, alpha=1, beta=1, out=buf0)
    buf1 = buf0; del buf0  # reuse
    buf2 = empty_strided_cpu((16, 4), (4, 1), torch.bool)
    cpp_fused_relu_threshold_backward_0(buf1, buf2)
    return (buf1, buf2, reinterpret_tensor(primals_1, (8, 16), (1, 8), 0), )
```

Inductor가 결정한 것:

| 결정 | 이 예제에서 |
|---|---|
| **matmul은 외부 커널로** | `aten.mm` + `aten.add(bias)`를 패턴 매칭해서 **`addmm` 하나**로 바꿨다. 즉 bias 덧셈이 라이브러리 matmul의 epilogue로 들어갔다. `extern_kernels.addmm`은 CPU에서는 MKL/oneDNN, GPU에서는 cuBLAS를 부른다. |
| **elementwise 퓨전** | `relu`와 backward용 `le` 마스크 계산이 **커널 하나**로 합쳐졌다(이름 `fused_relu_threshold_backward`). 데이터를 한 번 읽어 두 결과를 쓴다. |
| **버퍼 재사용** | `buf1 = buf0` 주석 `# reuse`. relu가 addmm 결과 위에 **in-place**로 써서 중간 버퍼를 하나 아꼈다. AOT 단계에서 functionalize된 그래프를 Inductor가 다시 in-place로 되돌린 것이다. |
| **뷰는 계산하지 않음** | `permute(Xᵀ)`는 실제 전치 커널을 만들지 않고 `reinterpret_tensor(primals_1, (8,16), (1,8))`로 **stride만 바꾼 뷰**를 반환했다. |
| **guard 재확인** | 래퍼 첫 줄 `assert_size_stride_grouped`가 입력 shape/stride를 다시 검사한다. |
| **벡터화** | CPU 코드 생성기가 64개 원소를 8개짜리 SIMD 벡터로 처리하는 루프를 만들었다. |

**GPU였다면** 같은 퓨전 커널이 Triton으로 생성된다. 아래는 [pytorch-compile.md](./pytorch-compile.md)의 gelu 예제와 같은 형태로 예상한 모양이며, 이 환경에서 실행한 결과는 아니다.

```python
@triton.jit
def triton_poi_fused_relu_threshold_backward_0(in_out_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_out_ptr0 + xindex, xmask)
    tmp1 = triton_helpers.maximum(tmp0, 0.0)      # relu
    tmp2 = tmp1 <= 0.0                              # le
    tl.store(in_out_ptr0 + xindex, tmp1, xmask)
    tl.store(out_ptr0 + xindex, tmp2, xmask)
```

[④](./04-dispatch-to-kernel.md)의 벡터 덧셈 커널과 구조가 같다. `program_id`로 블록을 고르고, 마스크로 경계를 막고, 로드-계산-스토어한다. 래퍼에서는 `extern_kernels.addmm`이 cuBLAS 호출로, `cpp_fused_...(buf1, buf2)`가 `triton_poi_fused_...[grid(64)](buf1, buf2, 64, XBLOCK=64)` 런치로 바뀐다.

### B-4. 코드 생성

- **CPU**: 위 C++ 소스를 시스템 컴파일러(g++/clang)로 공유 라이브러리로 빌드해 Python에서 `cpp_pybinding`으로 부른다.
- **GPU**: Triton 소스를 Triton 컴파일러가 Triton IR → LLVM IR → PTX → SASS로 내린다.

결과는 디스크 캐시(`/tmp/torchinductor_<user>/`)에 남아 다음 프로세스에서 재사용된다.

### B-5. 디스패치

```python
y = torch.compile(f)(x, w, b)
# type(y) = torch.Tensor, y.shape = (16, 4), y.grad_fn = <CompiledFunctionBackward>
```

- Dynamo가 심은 guard 검사 → 통과하면 위 `call(args)` 실행.
- 래퍼가 `addmm`, 퓨전 커널 순서로 런치한다. GPU라면 두 런치 모두 CUDA 스트림에 들어가고 `y`는 즉시 반환된다.
- `y.grad_fn`이 `CompiledFunctionBackward`다. eager autograd 그래프에 **컴파일된 backward 그래프 전체가 노드 하나**로 걸려 있어서, `y.sum().backward()`를 부르면 B-2의 backward 그래프(역시 Inductor로 컴파일됨)가 실행된다.

---

## C. 단계별 대응표

| 단계 | JAX | PyTorch 2 | 이 예제에서 눈에 띄는 차이 |
|---|---|---|---|
| 그래프 포착 | Tracer 실행 → jaxpr | Dynamo 바이트코드 해석 → FX 그래프 | jaxpr은 브로드캐스트를 명시적 노드로 기록. FX는 Python `+` 그대로 |
| 미분 | `grad` 시 jaxpr 변환. `custom_jvp`로 relu 규칙 지정 | AOT Autograd가 컴파일 시점에 backward 그래프 생성, 저장할 값(`le`, `permute`) 결정 | PyTorch는 forward 그래프에 backward용 출력이 섞여 있음 |
| 백엔드 입력 IR | StableHLO (relu는 함수 호출) | ATen FX 그래프 (relu는 `aten.relu`) | |
| matmul 처리 | XLA가 `kCustom` fusion으로 라이브러리 호출 | Inductor가 `addmm` 외부 커널 호출, bias까지 흡수 | 둘 다 작은 matmul은 직접 생성하지 않음 |
| elementwise 퓨전 | `broadcast + add + maximum` → `kLoop` fusion 1개 | `relu + le` → C++/Triton 커널 1개 | 퓨전 경계가 다름. XLA는 add를 relu와 묶고, Inductor는 add를 matmul과 묶음 |
| 레이아웃 | HLO에 `{1,0}` 명시 | FX 타입 주석에 stride `[4, 1]` 명시, guard로 고정 | |
| 버퍼 | XLA 버퍼 할당 (텍스트에는 안 보임) | 래퍼에 `empty_strided`, `# reuse`, `reinterpret_tensor`로 드러남 | Inductor 쪽이 결정을 읽기 쉬움 |
| 실행 파일 | executable 하나 | 커널 여러 개 + 순서를 담은 Python 래퍼 | |
| 반환값 | `jax.Array` future | `torch.Tensor` 핸들 + `grad_fn` | |
| 재컴파일 조건 | 입력 shape/dtype 캐시 키 | guard(shape, stride, dtype, device, requires_grad) | |

## D. 직접 재현하기

```bash
uv venv venv && uv pip install --python venv/bin/python jax torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

```python
# jax_demo.py
import jax, jax.numpy as jnp
def f(x, w, b): return jax.nn.relu(x @ w + b)
x, w, b = jnp.ones((16, 8)), jnp.ones((8, 4)), jnp.ones((4,))
print(jax.make_jaxpr(f)(x, w, b))                       # A-1
print(jax.jit(f).lower(x, w, b).as_text())              # A-2
print(jax.jit(f).lower(x, w, b).compile().as_text())    # A-3
```

```bash
# torch_demo.py: torch.compile(f)(x, w, b) 한 줄
TORCH_LOGS="graph_code,aot_graphs,output_code" python torch_demo.py   # B-1, B-2, B-3
```
