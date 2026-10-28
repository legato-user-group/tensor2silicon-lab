## Python에서 Triton 커널로 — 블록 인덱싱, Kernel Fusion, 정확도와 벤치마크

> **핵심 질문**  
> Python으로 작성한 Triton 커널은 데이터를 어떻게 블록 단위로 처리하며, 여러 연산을 한 커널로 묶으면 왜 빨라질까?

---

## 목차

1. [이번 주에 알아야 할 큰 그림](#1-이번-주에-알아야-할-큰-그림)
2. [공통 읽기와 다섯 가지 질문](#2-공통-읽기와-다섯-가지-질문)
3. [블록 단위 프로그래밍 모델](#3-블록-단위-프로그래밍-모델)
4. [Vector Add — 포인터와 마스크를 코드로 읽기](#4-vector-add--포인터와-마스크를-코드로-읽기)
5. [Softmax — 다섯 연산을 한 커널로 묶기](#5-softmax--다섯-연산을-한-커널로-묶기)
6. [행 단위 Triton 커널 — 줄마다 따라 읽기](#6-행-단위-triton-커널--줄마다-따라-읽기)
7. [패딩과 other — 왜 음의 무한대인가](#7-패딩과-other--왜-음의-무한대인가)
8. [HBM 이동량과 Roofline 손계산](#8-hbm-이동량과-roofline-손계산)
9. [NVIDIA GPU 실습 — 정확도부터 확인하기](#9-nvidia-gpu-실습--정확도부터-확인하기)
10. [벤치마크 — 어떤 시간을 재는가](#10-벤치마크--어떤-시간을-재는가)
11. [자주 생기는 오해와 문제 찾기](#11-자주-생기는-오해와-문제-찾기)
12. [연습문제와 해설](#12-연습문제와-해설)


---

## 1. 이번 주에 알아야 할 큰 그림

### 1.1 지난주에서 이번 주로

**Triton에서는 데이터 묶음의 주소와 연산을 표현한다. 커널 결합은 중간값을 온칩에 유지하여 장치 메모리를 오가는 양을 줄인다.**


이번 주의 흐름은 **1차원 Vector Add → 2차원 행별 Softmax → 정확도 검증 → 성능 측정**이다. (2차원 타일을 `tl.dot`으로 누적하는 Tiled MatMul과 TTIR → TTGIR → LLVM IR → PTX 컴파일 과정은 7주차로 이어간다.)

### 1.2 Python 문법이지만 실행 장소가 다르다

`@triton.jit` 함수 안에서는 일반 Python 라이브러리를 자유롭게 호출하는 것이 아니다. Triton 언어로 표현한 연산이 GPU 코드로 컴파일된다. 텐서 인자는 데이터 포인터와 자료형을 통해 커널에 전달되므로, 커널에서 사용할 길이·stride는 별도 인자로 넘긴다. [triton.jit API](https://triton-lang.org/main/python-api/generated/triton.jit.html)

### 1.3 이번 문서의 코드 사용법

**4.1 → 4.2 → 5.1 → 6.1 → 6.3 → 9.2 → 10.3의 Python 블록**을 순서대로 같은 노트북에서 실행하거나 하나의 `.py` 파일로 합치면 NVIDIA GPU 실습이 된다. 나머지 블록은 별도 설명·선택 실습이다.

코드는 FP32, 연속적인 입력, 순전파만을 대상으로 한다. 교육용으로 읽기 쉽게 작성했으며, 모든 shape에서 최고 성능을 내는 라이브러리 구현이나 autograd용 사용자 정의 연산은 아니다. 이 문서의 9.3·10.5~10.8절은 2026-10-28에 RTX 4070에서 실제 실행한 결과로 채웠다. [실험 코드와 재현 방법](experiments/README.md), [측정 원본 JSON](experiments/results/results.json), [CSV](experiments/results/benchmarks.csv)를 함께 남겼다.

---

## 2. 공통 읽기와 다섯 가지 질문

### 2.1 먼저 읽어올 자료

| 자료 | 읽을 절 | 읽으면서 표시할 코드 |
|---|---|---|
| [Programming Guide 1장](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html#motivations) | Motivations, Challenges는 가볍게 | CUDA와 Triton의 프로그램 모델 비교 |
| [Tutorial 01: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#compute-kernel) | Compute Kernel, Benchmark | `add_kernel`, `add`, grid와 mask |
| [Tutorial 02: Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#motivations) | Motivations, Compute Kernel | `naive_softmax`, `softmax_kernel` |

Tutorial 02 래퍼의 레지스터·Shared Memory 기반 occupancy 계산과 드라이버 warmup 코드는 이번 주에 외우지 않아도 된다. 먼저 **어떤 주소를 읽고, 중간값을 어디에서 계산하며, 어디에 저장하는가**를 찾는다.

### 2.2 코드로 먼저 이해하는 다섯 가지 핵심 개념

각 개념을 **뜻 → 작은 코드 → 코드가 하는 일** 순서로 살펴본다. 아래 코드는 개념을 보여주는 조각이다. `tl.*`가 있는 코드는 `@triton.jit` 커널 내부에 들어가며, 포인터·입력·import 등의 준비를 생략했다. 실행 가능한 전체 예제는 뒤의 해당 절에서 확인한다.

#### ① 블록 단위 프로그래밍 — 한 프로그램이 원소 묶음을 담당한다

**핵심 개념:** 프로그램 인스턴스는 자기 번호인 `pid`를 가지고, 그 번호에 해당하는 데이터 블록을 처리한다. `BLOCK_SIZE`는 한 프로그램이 다룰 논리적 원소 수이고, `grid`는 실행할 프로그램 개수를 정한다.

길이 10인 벡터를 4개씩 나누면 3개 프로그램이 필요하다. 마지막 프로그램에는 유효한 원소가 2개만 남는다.

```python
# CPU의 호출 준비 코드: ceil(10 / 4) = 3개 프로그램을 실행한다.
grid = (triton.cdiv(10, 4),)  # (3,)

# GPU 커널 내부 코드: 자기 프로그램이 맡은 원소 번호들을 만든다.
pid = tl.program_id(axis=0)
offsets = pid * 4 + tl.arange(0, 4)
# pid=0 → [0, 1, 2, 3]
# pid=1 → [4, 5, 6, 7]
# pid=2 → [8, 9, 10, 11]
```

`offsets`는 원소 번호 하나가 아니라 번호들의 묶음이다. 이 묶음을 실제 스레드·레지스터에 나누는 일은 컴파일러가 처리한다. 따라서 `BLOCK_SIZE=4`를 스레드 4개라는 뜻으로 읽으면 안 된다. 자세한 내용은 **3·4절**에서 다룬다.

#### ② 포인터와 마스킹 — 실제로 존재하는 원소만 읽고 쓴다

**핵심 개념:** 포인터는 데이터가 저장된 주소를 가리킨다. 시작 포인터에 원소 단위 오프셋을 더하면 읽을 주소 묶음이 된다. 마스크는 각 주소에 접근해도 되는지를 나타내는 True/False 묶음이다.

길이가 10이면 유효한 원소 번호는 0부터 9까지다. 마지막 블록에서 만든 10·11번 주소는 읽거나 쓰면 안 된다.

```python
# GPU 커널 내부: pid=2인 마지막 블록을 예로 든다.
offsets = tl.program_id(0) * 4 + tl.arange(0, 4)  # [8, 9, 10, 11]
mask = offsets < 10                             # [True, True, False, False]

# 유효 주소만 읽는다. 읽지 않은 두 자리의 계산값은 0으로 채운다.
values = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# 저장할 때도 같은 마스크를 적용해 범위 밖 쓰기를 막는다.
tl.store(out_ptr + offsets, values + 1.0, mask=mask)
```

FP32 포인터의 `+1`은 4 Byte 크기인 원소 하나만큼 이동한다. `mask`는 메모리 접근 여부를, `other`는 읽지 않은 자리의 계산값을 정한다. 전체 코드에서는 블록 형태를 컴파일할 때 알 수 있도록 `BLOCK_SIZE: tl.constexpr`를 사용한다. 자세한 내용은 **4절**에서 다룬다.

#### ③ 커널 결합 — 중간값을 저장하고 다시 읽는 일을 줄인다

**핵심 개념:** 여러 연산을 한 커널에서 이어서 수행하면 중간 결과를 HBM에 썼다가 다음 커널에서 다시 읽는 일을 줄일 수 있다. 이를 커널 결합(Kernel Fusion)이라고 한다.

먼저 Softmax보다 간단한 `out = (x + 1) * 2`를 생각해보자.

```python
# PyTorch eager 실행: 두 연산 사이에 중간 텐서 tmp를 만든다.
tmp = x + 1.0
out = tmp * 2.0
```

```python
# GPU의 결합 커널 내부: offsets와 mask는 앞에서 만들었다고 가정한다.
values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
shifted = values + 1.0  # 중간값을 커널 안에서 사용한다.
result = shifted * 2.0
# 중간값 shifted를 HBM에 저장하지 않고 최종 결과만 저장한다.
tl.store(out_ptr + offsets, result, mask=mask)
```

원소 수가 N이고 FP32라고 하자. 캐시를 무시한 단순 모델에서 단계별 실행은 입력 읽기·tmp 쓰기·tmp 읽기·출력 쓰기로 **4N개 원소**를 이동한다. 결합은 입력 읽기·출력 쓰기로 **2N개 원소**만 이동한다. 같은 유효 연산량 F에 대해 이동량 Q가 줄면 산술 강도 `I = F/Q`는 커진다. 이 원리를 다섯 단계의 Softmax에 적용하는 과정은 **5·8절**에서 다룬다.

#### ④ 행 단위 리덕션과 패딩 — 여러 값을 하나로 줄인다

**핵심 개념:** 리덕션은 여러 원소에서 최댓값이나 합계 같은 값 하나를 구하는 연산이다. 행별 Softmax는 한 행의 최댓값과 지수 합계를 구한다. 행 길이를 블록 크기에 맞추려고 추가한 자리는 계산을 바꾸지 않는 값으로 채워야 한다.

```python
# GPU 커널 내부: 한 행에 실제 원소는 3개, 논리적 블록 크기는 4다.
cols = tl.arange(0, 4)
mask = cols < 3
row_ptr = input_ptr + tl.program_id(0) * input_row_stride
row = tl.load(row_ptr + cols, mask=mask, other=-float("inf"))
# 실제 행이 [-2, -1, -3]이면 row는 [-2, -1, -3, -inf].

row_max = tl.max(row, axis=0)             # -1
numerator = tl.exp(row - row_max)         # [exp(-1), 1, exp(-2), 0]
denominator = tl.sum(numerator, axis=0)   # 패딩의 기여는 0
result = numerator / denominator
```

`-inf`는 최댓값을 바꾸지 않고, 유한한 최댓값을 뺀 뒤 지수를 취하면 0이 되어 합계도 바꾸지 않는다. 반대로 패딩을 0으로 채우면 위 예제의 max가 0으로 바뀌고, 가짜 원소의 지수값까지 분모에 들어간다. 실제 `N=781`에서는 `BLOCK_SIZE=1024`로 잡고 이 원리를 적용한다. 자세한 내용은 **6·7절**에서 다룬다.

#### ⑤ 정확도 검증과 벤치마크 — 맞는 결과인지 확인한 뒤 시간을 잰다

**핵심 개념:** GPU 실행 요청은 비동기이므로 Python 함수가 반환됐다고 GPU 계산까지 끝난 것은 아니다. 먼저 오차 범위 안에서 결과가 맞는지 확인하고, 최초 컴파일을 마친 뒤 GPU 실행을 고려하는 도구로 반복 측정한다.

```python
# CPU의 검증·측정 코드: softmax는 뒤에서 정의할 Triton 호출 래퍼다.
x = torch.randn(1823, 781, device="cuda", dtype=torch.float32)
expected = torch.softmax(x, dim=-1)
actual = softmax(x)  # 첫 호출에는 JIT 컴파일이 포함될 수 있다.
assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8)
torch.cuda.synchronize()  # 사전 실행이 끝난 뒤 측정을 시작한다.

ms = triton.testing.do_bench(
    lambda: softmax(x), warmup=25, rep=100, return_mode="median"
)
# 필수 작업량 = 입력 한 번 읽기 + 출력 한 번 쓰기.
useful_bytes = 2 * x.numel() * x.element_size()
gbps = useful_bytes / (ms * 1e-3) / 1e9
```

`allclose`는 결과의 비트 단위 일치가 아니라 지정한 허용 오차를 검사한다. `warmup`과 `rep`의 단위는 횟수가 아닌 ms이며, 위 GB/s는 필수 입력·출력량을 기준으로 한 **실효 대역폭**이다. 실제 HBM 트래픽을 직접 측정한 값은 아니다. 이 짧은 예제는 하나의 shape로 검증과 측정 흐름을 보여주며, **9·10절**의 전체 실습에서는 정확도용 `1823×781`과 성능 비교용 `4096×1024`를 나누어 사용한다.

---

## 3. 블록 단위 프로그래밍 모델

### 3.1 스레드 하나의 일에서 데이터 묶음의 일로

단순한 CUDA Vector Add는 다음처럼 스레드 하나가 맡을 인덱스를 직접 정한다. CUDA에서도 스레드당 여러 원소를 처리할 수 있지만, 여기서는 두 모델을 비교하기 위한 가장 작은 예를 사용한다.

```cpp
// CUDA 설명용 조각: i는 이 스레드가 처리할 원소 하나의 번호다.
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    out[i] = x[i] + y[i];
}
```

Triton에서는 프로그램 인스턴스가 맡을 **인덱스 벡터**를 만든다.

```python
# Triton 커널 내부 설명용 조각
pid = tl.program_id(axis=0)                       # 프로그램 번호 하나
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 여러 원소의 번호
```

위 조각에서 `offsets`는 스칼라가 아니라 길이 `BLOCK_SIZE`인 값의 묶음이다. 실제 실행 가능한 전체 코드는 4절에 있다.

| 구분 | 단순 CUDA 예제 | Triton 예제 |
|---|---|---|
| 코드가 표현하는 단위 | 한 스레드의 원소 하나 | 한 프로그램의 원소 묶음 |
| 식에서 보이는 인덱스 | `blockIdx`, `threadIdx` | `tl.program_id`, `tl.arange` |
| 조건 처리 | `if (i < n)` | 원소별 `mask` |
| 내부 스레드의 역할 | 프로그래머가 스레드별 일을 표현 | 컴파일러가 블록 연산을 스레드·레지스터에 배치 |

공식 가이드의 **Scalar Program, Blocked Threads**와 **Blocked Program, Scalar Threads**는 코드 표현의 관점 차이다. Triton이 GPU 스레드를 없애거나, 프로그램 인스턴스 하나가 스레드 하나가 된다는 뜻은 아니다. [Programming Guide — Motivations](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html#motivations)

### 3.2 BLOCK_SIZE는 스레드 수가 아니다

`BLOCK_SIZE=1024`는 이 코드가 다루는 **논리적 원소 수**다. `num_warps=4`라면 일반적인 NVIDIA 실행에서 4개 warp, 즉 128개 스레드가 협력해 이 묶음을 처리한다. 한 스레드가 여러 값을 맡을 수 있다.

연속 주소 접근은 인접 스레드의 메모리 요청을 합치기 좋은 형태다. 이것을 **coalescing**이라고 한다. 컴파일러는 논리적 블록을 하드웨어에 배치하고, 하드웨어는 실제 메모리 요청을 처리한다. 다만 프로그래머가 불연속 stride나 나쁜 접근 패턴을 주어도 자동으로 모두 해결되는 것은 아니다. 블록 크기·주소 패턴·`num_warps` 선택은 여전히 중요하다.

### 3.3 N = 10, BLOCK_SIZE = 4를 손으로 펼치기

$$
\mathrm{grid}=\left(\left\lceil\frac{10}{4}\right\rceil,\right)=(3,)
$$

Python에서 `(3,)`는 원소 하나짜리 튜플이다. `grid`는 프로그램을 몇 개 실행할지 정하며, 세 프로그램이 반드시 동시에 실행된다는 뜻은 아니다.

```text
입력의 원소 번호:   0  1  2  3 | 4  5  6  7 | 8  9
프로그램:          └─ pid 0 ─┘ └─ pid 1 ─┘ └─ pid 2 ─────┐
논리적 offsets:    [0,1,2,3]   [4,5,6,7]   [8,9,10,11]  │
마지막 mask:                                 T T  F  F  │
실제 메모리:                                 읽음  접근 안 함
```

| pid | `pid * 4` | `offsets` | `offsets < 10` |
|---:|---:|---|---|
| 0 | 0 | `[0, 1, 2, 3]` | `[True, True, True, True]` |
| 1 | 4 | `[4, 5, 6, 7]` | `[True, True, True, True]` |
| 2 | 8 | `[8, 9, 10, 11]` | `[True, True, False, False]` |

10·11이라는 오프셋을 **계산하는 것**과 그 주소의 메모리를 **읽는 것**은 구분해야 한다. 실제 접근은 `tl.load`·`tl.store`의 마스크로 막는다.

---

## 4. Vector Add — 포인터와 마스크를 코드로 읽기

### 4.1 공통 import

```python
import os
import platform

import torch
import triton
import triton.language as tl
import triton.testing
```

### 4.2 주석을 따라 읽는 전체 코드

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # ① 이 프로그램이 맡을 블록 번호. grid가 (3,)이면 pid는 0, 1, 2.
    pid = tl.program_id(axis=0)

    # ② 블록 내부의 상대 위치: [0, 1, ..., BLOCK_SIZE - 1].
    local_offsets = tl.arange(0, BLOCK_SIZE)

    # ③ 입력 전체에서의 위치. 예: pid=2, BLOCK_SIZE=4 → [8, 9, 10, 11].
    offsets = pid * BLOCK_SIZE + local_offsets

    # ④ 유효한 원소인지 원소별로 판정한다. Python의 단일 bool이 아니다.
    mask = offsets < n_elements

    # ⑤ 포인터 + 원소 오프셋으로 주소 묶음을 만든 뒤 읽는다.
    # mask=False인 위치는 메모리를 읽지 않고 계산용 값 0을 사용한다.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # ⑥ 블록의 각 위치에서 덧셈한다. PyTorch Tensor 연산 호출이 아니다.
    result = x + y

    # ⑦ load뿐 아니라 store도 마스킹해야 범위 밖 쓰기를 막을 수 있다.
    tl.store(out_ptr + offsets, result, mask=mask)


def add(x, y, block_size=256):
    # 이 래퍼는 CPU에서 실행된다. 입력 조건을 좁혀 주소 계산을 단순화한다.
    assert x.is_cuda and y.is_cuda
    assert x.device == y.device
    assert x.dtype == y.dtype == torch.float32
    assert x.ndim == y.ndim == 1 and x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()
    assert block_size > 0 and (block_size & (block_size - 1)) == 0

    out = torch.empty_like(x)  # 아직 값이 채워지지 않은 출력 공간
    n = x.numel()
    if n == 0:
        return out            # 빈 입력에서는 grid가 0인 커널을 실행하지 않는다.

    grid = (triton.cdiv(n, block_size),)  # 올림 나눗셈으로 마지막 조각까지 포함
    with torch.cuda.device(x.device):
        # []는 실행 grid, ()는 커널 인자와 컴파일/실행 설정이다.
        add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size, num_warps=4)
    return out                # 반환 자체가 GPU 작업 완료를 보장하지는 않는다.
```

### 4.3 한 줄을 네 단계로 나눠 보기

`tl.load(x_ptr + offsets, mask=mask, other=0.0)`를 다음처럼 읽는다.

| 표현 | 뜻 | 마지막 블록의 예 |
|---|---|---|
| `x_ptr` | 입력 첫 원소를 가리키는 포인터 | `x[0]`의 주소 |
| `offsets` | 첫 원소에서 몇 원소 떨어졌는가 | `[8, 9, 10, 11]` |
| `x_ptr + offsets` | 원소별 주소 묶음 | `x[8]`, `x[9]`, 범위 밖 주소 둘 |
| `mask`, `other` | 유효 주소만 읽고 나머지 계산값 지정 | `[x[8], x[9], 0, 0]` |

**FP32 포인터에서 `+1`은 1 Byte가 아니라 FP32 원소 하나, 즉 4 Byte만큼 이동한다.** 따라서 주소 식에 다시 `*4`를 붙이면 틀린 위치를 읽는다.

`mask=False`는 읽기를 생략한다. `other`는 생략된 자리에 사용할 값을 정한다. `other`를 지정하지 않은 masked load의 값은 정의되지 않으므로, 리덕션에 들어가는 값은 특히 명시해야 한다. `tl.store`의 마스크는 쓰기 자체를 생략한다. [tl.load](https://triton-lang.org/main/python-api/generated/triton.language.load.html), [tl.store](https://triton-lang.org/main/python-api/generated/triton.language.store.html)

### 4.4 왜 BLOCK_SIZE: tl.constexpr인가?

`tl.arange(0, BLOCK_SIZE)`가 만드는 블록의 형태는 컴파일할 때 정해져야 한다. `tl.constexpr`는 이 인자를 컴파일 시점 상수로 다루도록 한다. `BLOCK_SIZE=256`과 `1024`는 서로 다른 특수화 커널이 생길 수 있다.

반면 `n_elements`는 실제 입력 길이이며 마스크 비교에 사용한다. 마지막 블록만 일부 원소를 가질 수 있으므로 `n_elements`와 `BLOCK_SIZE`는 다른 개념이다. 이 예제의 `tl.arange(0, BLOCK_SIZE)`에는 2의 거듭제곱 크기를 사용한다. `tl.arange`는 Python 리스트를 순차 생성하는 루프가 아니라 Triton 블록 값을 만든다. [tl.arange API](https://triton-lang.org/main/python-api/generated/triton.language.arange.html)

---

## 5. Softmax — 다섯 연산을 한 커널로 묶기

### 5.1 수식과 PyTorch 단계별 코드

입력 $X$의 shape은 $[M,N]$이다. 행마다 독립적으로 다음을 구한다.

$$
m_i=\max_j x_{ij},\qquad
s_i=\sum_j \exp(x_{ij}-m_i),\qquad
y_{ij}=\frac{\exp(x_{ij}-m_i)}{s_i}
$$

최댓값을 빼도 모든 항에 같은 배율이 적용되어 수학적인 Softmax 결과는 같다. 지수 함수에 들어가는 값의 최댓값을 0으로 만들면 큰 양수의 지수에 의한 overflow를 피할 수 있다.

```python
def naive_softmax(x):
    # x: [M, N]. dim=1은 각 행 안에서 열 방향으로 줄인다는 뜻이다.
    # max는 (최댓값, 인덱스)를 반환하므로 [0]으로 값만 선택한다.
    x_max = x.max(dim=1)[0]                # [M]

    # [M]을 [M, 1]로 바꿔 각 행의 모든 열에 같은 최댓값을 뺀다.
    # [:, None]은 차원을 추가하는 view이며 [M, N] 복사본을 만들지 않는다.
    z = x - x_max[:, None]                # [M, N]
    numerator = torch.exp(z)             # [M, N]
    denominator = numerator.sum(dim=1)   # [M]
    return numerator / denominator[:, None]  # [M, N]
```

`naive`는 계산식이 틀렸다는 뜻이 아니다. 여러 기본 연산을 eager 모드로 실행해 중간 텐서를 만드는 비교 기준이다. `torch.softmax`는 이 함수를 그대로 다섯 번 실행하는 구현으로 가정하면 안 된다. 이미 최적화된 구현을 사용할 수 있다.

### 5.2 결합하면 없어지는 것은 중간값의 왕복이다

![단계별 Softmax와 결합 Softmax의 메모리 이동 구조](<img width="1359" height="572" alt="image" src="https://github.com/user-attachments/assets/5a6e9164-df71-4f1d-ac6c-82991f585b05" />
)

**Figure 1.** 단계별 실행은 중간 텐서를 장치 메모리에 두지만, 행 단위 결합은 한 번 읽은 값으로 리덕션과 정규화를 이어서 수행한다. 아래 이동량은 8절의 무캐시 값 텐서 모델이며, 하드웨어 카운터 실측과 구분한다.

```text
단계별 실행: 서로 다른 커널 사이에서 중간 텐서를 장치 메모리에 둔다.

HBM의 X ──max──> x_max
   │              │
   └──── 뺄셈 ────┘ ──> z ──exp──> numerator ──sum──> denominator
                                       │                  │
                                       └──── 나눗셈 ──────┘ ──> Y

결합 실행: 한 행의 계산을 한 프로그램 안에서 이어간다.

HBM의 X 한 행 ──load──> [온칩: max → 뺄셈 → exp → sum → 나눗셈]
                                                       │
                                      HBM의 Y 한 행 <──store
```

단일 `x+y`는 이미 입력 두 개를 읽고 출력 하나를 쓰는 단순한 연산이다. Triton으로 바꾸어도 그 필수 이동량은 거의 줄지 않는다. Softmax의 단계별 실행에는 없앨 수 있는 중간값 왕복이 많다. 런치 횟수 감소도 도움이 될 수 있다.

다만 **Triton이면 항상 PyTorch보다 빠르다**는 결론은 아니다. 비교 대상이 eager `naive_softmax`인지, 최적화된 `torch.softmax`인지, 컴파일된 함수인지 구분한다. 메모리 병목에서 데이터 이동을 줄이는 원리는 [Modern GPU Programming — Optimizing Memory-Bound Kernels](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_performance/index.html#optimizing-memory-bound-kernels)와 연결된다.

---

## 6. 행 단위 Triton 커널 — 줄마다 따라 읽기

### 6.1 한 프로그램이 한 행을 맡는 교육용 커널

```python
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # ① grid=(M,)이므로 프로그램 번호가 곧 행 번호다.
    row_idx = tl.program_id(axis=0)

    # ② 다음 행으로 이동하는 거리는 '원소 수' 단위 stride다.
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # ③ 한 행 안의 열 번호. N=781이면 BLOCK_SIZE=1024로 만든다.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    input_ptrs = row_start_ptr + col_offsets

    # ④ 실제 열만 읽고 나머지를 -inf로 채운 논리적 블록을 만든다.
    # 원본 행렬을 [M, BLOCK_SIZE]로 복사하거나 확장하는 코드가 아니다.
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # ⑤ row는 1차원 블록이므로 axis=0이 이 행 전체를 줄이는 축이다.
    row_max = tl.max(row, axis=0)
    shifted = row - row_max

    # ⑥ 유효 원소의 exp를 구한다. 패딩은 exp(-inf)=0이 된다.
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)

    # ⑦ 행마다 분모 하나를 모든 열에 적용한다.
    result = numerator / denominator

    # ⑧ 출력도 자기 stride로 행 시작 주소를 구하고, 유효 열만 저장한다.
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, result, mask=mask)
```

### 6.2 shape를 같이 쓰면 리덕션이 보인다

| 변수 | 논리적 shape | `N=781`일 때 뜻 |
|---|---|---|
| `row_idx` | 스칼라 | 현재 행 번호 |
| `col_offsets`, `mask` | `[1024]` | 열 번호와 유효 여부 |
| `row`, `shifted`, `numerator` | `[1024]` | 유효 값 781개와 패딩 243개 |
| `row_max`, `denominator` | 스칼라 | 이 행의 최댓값과 지수 합 |
| `result` | `[1024]` | 실제 저장은 앞의 781개만 |

PyTorch의 `x.max(dim=1)`은 `[M,N]` 전체를 바라본다. 이 커널의 `tl.max(row, axis=0)`은 이미 뽑은 **한 행의 1차원 블록**을 바라본다. 같은 행별 최댓값인데 축 번호가 다른 이유다. 스칼라와 블록의 산술 연산에는 broadcasting이 적용된다. [Triton Semantics — Broadcasting](https://triton-lang.org/main/python-api/triton-semantics.html#broadcasting)

### 6.3 호출 래퍼

```python
def softmax(x):
    assert x.is_cuda
    assert x.dtype == torch.float32 and x.ndim == 2
    # 열 stride=1을 전제로 주소를 계산한다. 전치 입력은 이 래퍼에서 거부한다.
    assert x.is_contiguous(), "이 실습은 contiguous FP32 입력을 사용합니다."
    n_rows, n_cols = x.shape
    assert n_rows > 0 and n_cols > 0

    block_size = triton.next_power_of_2(n_cols)
    # 수업의 두 shape는 모두 이 범위 안이다. 범용 하드웨어 한도가 아니다.
    assert block_size <= 1024, "교육용 래퍼는 열 수 1024 이하만 다룹니다."
    out = torch.empty_like(x)

    with torch.cuda.device(x.device):
        softmax_kernel[(n_rows,)](
            x, out,
            x.stride(0), out.stride(0),  # Byte가 아니라 원소 단위
            n_cols,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return out
```

예를 들어 contiguous `[1823,781]`의 `x.stride(0)`은 781이다. 두 번째 행은 `input_ptr + 781`에서 시작한다. **BLOCK_SIZE가 1024라고 다음 행 주소에 1024를 더하면 안 된다.** 패딩은 커널 내부의 논리적 값일 뿐 원본 행렬의 행 간격을 바꾸지 않는다.

### 6.4 공식 튜토리얼과 다른 점

공식 [Tutorial 02 — Compute Kernel](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#compute-kernel)은 프로그램 수를 자원 사용량에 맞춰 정하고, 각 프로그램이 `row_start, row_start + row_step, ...`의 여러 행을 순회한다. 따라서 공식 코드에서는 `pid`가 첫 행 번호이고, 항상 그 행 하나만 처리하는 것은 아니다.

여기서는 인덱싱을 먼저 배우기 위해 `grid=(M,)`, 프로그램당 한 행으로 단순화했다. **행 안의 계산 원리는 같지만 스케줄링과 성능이 같은 구현은 아니다.** 공식 벤치마크 숫자를 이 코드의 예상 실측값으로 옮겨 적지 않는다.

온칩은 레지스터와 Shared Memory 등을 묶어 설명하는 말이다. `tl.load`가 명시적으로 Shared Memory 배열을 만드는 것은 아니다. 중간값 배치와 리덕션 구현은 컴파일러가 정한다.

---

## 7. 패딩과 other — 왜 음의 무한대인가

### 7.1 load 마스크와 패딩값의 역할은 서로 다르다

- `mask`: 범위 밖 메모리를 읽지 않게 한다.
- `other`: 읽지 않은 자리의 값이 이후 계산에 미칠 영향을 정한다.
- store의 `mask`: 범위 밖 출력 저장을 막는다. 이미 틀어진 분모를 고치지는 못한다.

이 커널에서 `BLOCK_SIZE=next_power_of_2(N)`은 한 행을 담는 2의 거듭제곱 블록을 만들기 위한 선택이다. `N=781`이면 1024, `N=1024`이면 그대로 1024다. 무조건 다음의 더 큰 수를 택하는 함수는 아니다.

### 7.2 781개가 모두 -1인 행으로 직접 계산

패딩 자리 수는 $1024-781=243$이다. 입력은 모두 유한한 수이며 `N>0`이라고 가정한다.

| 계산 | 올바른 `other=-inf` | 잘못된 `other=0` |
|---|---|---|
| 로드한 블록 | `[-1 × 781, -inf × 243]` | `[-1 × 781, 0 × 243]` |
| `tl.max(row)` | -1 | 0 |
| 최댓값 뺀 뒤 | `[0 × 781, -inf × 243]` | `[-1 × 781, 0 × 243]` |
| `tl.exp(...)` | `[1 × 781, 0 × 243]` | `[exp(-1) × 781, 1 × 243]` |
| 분모 | 781 | $781e^{-1}+243\approx530.313844$ |
| 유효 출력 하나 | $1/781\approx0.001280410$ | $e^{-1}/(781e^{-1}+243)\approx0.000693701$ |
| 저장된 781개 출력의 합 | 1 | 약 0.541781 |

중요한 점은 두 가지다. 첫째, 음수만 있는 행에서는 패딩 0이 최댓값을 바꾼다. 둘째, 더 직접적인 오답 원인은 **가짜 패딩 원소의 지수가 분모에 더해진다는 것**이다. 공통 상수로 최댓값을 다르게 빼는 일만으로 수학적인 Softmax 비율이 달라지지는 않는다. 패딩을 분모에서 제외하지 않은 것이 문제다.

실제 최댓값이 양수인 행에서도 `other=0`이면 패딩의 지수 기여가 남으므로 일반적으로 틀린다. `-inf`는 max를 바꾸지 않고, 유효한 최댓값을 뺀 뒤에도 `-inf`이며, 지수를 취하면 0이 되어 sum을 바꾸지 않는다.

![패딩값에 따라 달라지는 실제 출력 확률 합](<img width="1366" height="525" alt="image" src="https://github.com/user-attachments/assets/f9657be7-1d3d-4c40-be6b-20078a7db270" />
)

**Figure 2. GPU 반례 재현.** `[3,781]`의 입력을 모두 -1로 설정하고 `BLOCK_SIZE=1024`로 실행했다. `other=-inf`일 때 세 행의 출력 합은 모두 1.0이었다. 의도적으로 `other=0`으로 바꾼 커널에서는 첫 행의 합이 **0.541780770**이었고, 손계산 **0.541780772**와 일치하는 수준의 결과가 나왔다. 출력 한 원소는 각각 **0.001280409750**, **0.000693701382**였다. 두 커널 모두 load·store 마스크를 사용하므로, 메모리 경계가 안전하다는 사실만으로 Softmax 계산의 정확성이 보장되지 않음을 확인했다. 반례 커널은 [실험 코드](experiments/run_experiments.py)의 `wrong_padding_kernel`로 남겼고 성능 비교에서는 제외했다.

### 7.3 N이 아주 커지면?

한 행의 입력 데이터만 최소 $4N$ Byte가 필요하다. `N=781`에서는 실제 입력이 3,124 Byte이고, 논리적인 1024개 FP32 블록은 4,096 Byte 규모다. 그러나 **이 숫자를 커널의 실제 레지스터·Shared Memory 사용량이라고 단정할 수는 없다.** 중간값, 리덕션 임시 공간, 스레드 배치, 값의 생존 구간이 자원 사용에 영향을 준다.

블록을 키우면 레지스터 압박, spill, 동시에 상주할 수 있는 프로그램 감소, 자원 초과가 생길 수 있다. 긴 행은 여러 조각의 부분 최댓값·부분합을 결합하거나 여러 단계로 나누는 알고리즘이 필요할 수 있다. 그때에는 HBM 이동량이 다시 늘 수 있으므로 ‘모든 Softmax는 항상 2MN 원소만 이동한다’고 일반화하지 않는다.

---

## 8. HBM 이동량과 Roofline 손계산

### 8.1 비교 조건 고정

| 항목 | 조건 |
|---|---|
| 입력 | $M=4096$, $N=1024$, FP32, 원소당 4 Byte |
| 행렬 원소 수 | $MN=4,194,304$ |
| 행렬 한 개 크기 | 16,777,216 Byte = 16 MiB ≈ 16.78 MB |
| 가상 연산 처리율 P | $20\times10^{12}$ FLOPs/s |
| 가상 HBM 대역폭 BW | $1\times10^{12}$ Byte/s |
| 측정 경계 | 입력 X는 이미 장치 메모리에 있음. 호스트 전송 제외 |
| 손계산 가정 | 커널 사이 캐시 적중 없음. 각 단계의 입력을 논리적으로 한 번 읽음 |

여기서 HBM은 3·4주차의 가상 장치 메모리 경계다. 실제 GPU의 외부 메모리가 GDDR 계열일 수도 있으므로 실측 환경의 메모리 종류와 이 가정은 구별한다.

### 8.2 단계별 읽기·쓰기 표

아래는 **값 텐서만 세는 튜토리얼의 단순화 모델**이다. 길이 M 벡터를 방송할 때 한 단계에서 M개를 읽는다고 세고, 리덕션은 입력을 한 번 읽는다고 가정한다.

| 단계 | 읽는 값 | 읽기 원소 수 | 쓰는 값 | 쓰기 원소 수 | 읽기+쓰기 Byte |
|---|---|---:|---|---:|---:|
| 행별 max | X | MN | x_max | M | 16,793,600 |
| 뺄셈 | X, x_max | MN + M | z | MN | 33,570,816 |
| exp | z | MN | numerator | MN | 33,554,432 |
| 행별 sum | numerator | MN | denominator | M | 16,793,600 |
| 나눗셈 | numerator, denominator | MN + M | Y | MN | 33,570,816 |
| **단계별 합계** | | **5MN + 2M** | | **3MN + 2M** | **134,283,264** |
| **결합 커널** | X | **MN** | Y | **MN** | **33,554,432** |

원소 수와 Byte를 혼동하지 않는다.

$$
Q_{\mathrm{naive}}=4(8MN+4M)=134,283,264\ \mathrm{Byte}
$$

$$
Q_{\mathrm{naive,approx}}=4(8MN)=134,217,728\ \mathrm{Byte},\qquad
Q_{\mathrm{fused}}=4(2MN)=33,554,432\ \mathrm{Byte}
$$

`8MN`은 **FP32 원소 이동 횟수의 합**이고, 바이트 수로 바꾸려면 다시 4를 곱한다. 근사에서 생략한 $4M$ 원소는 65,536 Byte다.

**실제 코드와 모델 사이의 차이:** `x.max(dim=1)`은 값뿐 아니라 argmax 인덱스도 반환한다. `[0]`으로 값만 선택해도 eager 실행에서 인덱스 출력 비용이 생길 수 있으며 위 표는 그 비용을 제외한다. 실제 HBM 트래픽은 캐시·방송 재사용·리덕션 구현·임시 버퍼에도 영향을 받는다. 따라서 `8MN+4M`은 이 수업의 가정 아래 정확한 합계이지 하드웨어 카운터의 실측값은 아니다. 공식 계산 기준은 [Tutorial 02 — Motivations](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#motivations)에 있다.

### 8.3 중간 텐서별로 다시 세어 보기

| 텐서 | 쓰기 횟수 | 이후 읽기 횟수 | 값 원소 이동량 |
|---|---:|---:|---:|
| X | 경계 밖에서 이미 준비 | max, 뺄셈에서 2회 | 2MN |
| x_max | 1 | 뺄셈에서 1 | 2M |
| z | 1 | exp에서 1 | 2MN |
| numerator | 1 | sum, 나눗셈에서 2 | 3MN |
| denominator | 1 | 나눗셈에서 1 | 2M |
| Y | 1 | 이 작업 안에서 없음 | MN |

이 표를 더해도 $8MN+4M$이다. 특히 numerator는 한 번 쓰고 **두 번 읽는다.** 결합 커널에서는 X 읽기와 Y 쓰기만 이 경계를 넘고 중간값은 온칩에서 사용한다는 이상적 모델이다.

### 8.4 한 행의 최소 데이터 크기

$$
N\times4=1024\times4=4096\ \mathrm{Byte}=4\ \mathrm{KiB}
$$

행렬 전체가 아니라 행 하나를 먼저 생각하면 온칩 재사용의 단위가 보인다. 이 값은 입력 한 행의 크기이며 출력·임시값·리덕션 통신을 모두 포함한 프로그램당 자원량은 아니다.

### 8.5 메모리 시간과 이론적 속도 향상

$$
t_{\mathrm{mem}}=\frac{Q}{BW}
$$

| 경로 | Q (Byte) | 메모리 시간 (ms) | 메모리 시간 (μs) |
|---|---:|---:|---:|
| 단계별, M 벡터 포함 | 134,283,264 | 0.134283264 | 134.283264 |
| 단계별, M 벡터 생략 | 134,217,728 | 0.134217728 | 134.217728 |
| 결합 | 33,554,432 | 0.033554432 | 33.554432 |

$$
\frac{t_{\mathrm{naive}}}{t_{\mathrm{fused}}}
=\frac{8MN+4M}{2MN}
=4+\frac{2}{N}
=4.001953125\approx4
$$

이는 **동일 대역폭을 달성하고 메모리 시간이 지배할 때의 이론적인 비율**이다. 커널 런치, 리덕션 통신, 지수 함수, 점유율 저하 등을 포함한 실제 속도비를 보장하지 않는다.

### 8.6 산술 강도와 20 TFLOPs/s는 어떻게 연결할까?

$$
I=\frac{F}{Q},\qquad
P_{\mathrm{roof}}=\min(P,BW\times I),\qquad
I_{\mathrm{ridge}}=\frac{P}{BW}=20\ \mathrm{FLOPs/Byte}
$$

같은 유효 연산량 F를 기준으로 두면 Q가 약 1/4이므로 결합 후 I는 약 4배가 된다. 결합의 이점은 수학적인 연산 수를 1/4로 만드는 것이 아니다.

Softmax의 exp·max·나눗셈을 단순한 FP32 덧셈이나 FMA와 같은 비용으로 취급하면 정밀한 연산 시간 모델이 되지 않는다. 연습용으로만 비교 1회·exp 1회도 각각 1개의 연산으로 센다면 행당 횟수는 다음과 같다.

$$
(N-1)_{\max}+N_{\mathrm{sub}}+N_{\exp}+(N-1)_{\mathrm{sum}}+N_{\mathrm{div}}
=5N-2
$$

이 **설명용 연산수**를 F 대신 대입하면 $F^*=20,963,328$, $F^*/P=1.0481664\ \mu s$, $I^*_{naive}\approx0.1561$, $I^*_{fused}\approx0.6248$이다. 이 장난감 모델에서는 둘 다 메모리 쪽 시간이 훨씬 크다. 하지만 이것은 특수 함수 처리율까지 반영한 실제 FLOPs 측정이 아니다. 앞의 메모리 시간 계산은 이 임의의 연산수 정의 없이도 성립한다.

---

## 9. NVIDIA GPU 실습 — 정확도부터 확인하기

### 9.1 실행 환경

Linux의 NVIDIA CUDA 환경에서 PyTorch와 호환되는 Triton 설치를 사용한다. 노트북에서는 GPU 런타임을 선택한 뒤 다음 코드를 실행한다. 설치된 패키지를 무조건 최신 버전으로 덮어쓰기보다 먼저 버전을 기록한다. CPU나 Apple GPU만 있는 환경에서는 11절로 진행한다.

이 실습에서는 **유한한 FP32 입력**을 사용한다. NaN, 무한대만 있는 행, 빈 행의 정책까지 다루는 범용 Softmax 구현은 별도 과제다.

### 9.2 환경 기록과 정확도 확인

```python
if os.environ.get("TRITON_INTERPRET") == "1":
    raise RuntimeError("GPU 실습에서는 TRITON_INTERPRET를 해제하고 새 프로세스로 실행하세요.")
if not torch.cuda.is_available() or torch.version.cuda is None:
    raise RuntimeError("이 실습에는 NVIDIA CUDA GPU가 필요합니다. GPU가 없으면 11절을 보세요.")

DEVICE = torch.device("cuda", torch.cuda.current_device())
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(DEVICE))
print("Compute capability:", torch.cuda.get_device_capability(DEVICE))
print("Device:", DEVICE)
print("dtype: float32; num_warps: 4; forward only")
# 드라이버 버전은 별도로 터미널의 nvidia-smi 출력과 함께 기록한다.

torch.manual_seed(0)
RTOL, ATOL = 1e-5, 1e-8


def check_close(name, actual, expected):
    # allclose는 모든 원소에서 |actual-expected| <= atol + rtol*|expected|인지 본다.
    ok = torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)
    max_abs = (actual - expected).abs().max().item()
    print(f"{name}: allclose={ok}, max_abs_error={max_abs:.3e}")
    assert ok, f"{name} 정확도 실패: 마스크·stride·분모·오차 분포부터 확인하세요."


with torch.no_grad():
    # 작은 예제: 마지막 블록에서 두 원소가 마스킹되는지 실제로 확인한다.
    a = torch.arange(10, device=DEVICE, dtype=torch.float32)
    b = torch.ones_like(a)
    check_close("add N=10, BLOCK_SIZE=4", add(a, b, block_size=4), a + b)

    # 랜덤 781열: 1024 블록의 마지막 243개 위치가 패딩이다.
    x_test = torch.randn(1823, 781, device=DEVICE, dtype=torch.float32)
    reference = torch.softmax(x_test, dim=-1)
    y_naive = naive_softmax(x_test)
    y_triton = softmax(x_test)
    print("Correctness shape:", tuple(x_test.shape), "BLOCK_SIZE:", 1024)
    check_close("naive vs torch", y_naive, reference)
    check_close("triton vs torch", y_triton, reference)
    check_close("triton vs naive", y_triton, y_naive)
    check_close("row sums", y_triton.sum(dim=1), torch.ones_like(reference[:, 0]))

    # 랜덤 입력만으로 지나칠 수 있는 other=0 버그를 드러내는 회귀 사례.
    x_negative = torch.full((3, 781), -1.0, device=DEVICE, dtype=torch.float32)
    check_close("all-negative rows", softmax(x_negative), torch.softmax(x_negative, dim=-1))
```

`1823×781`은 불규칙한 크기에서 주소 계산과 열 패딩을 확인하기 위한 입력이다. 이 교육용 구현은 정확히 M개 프로그램을 실행하므로 행 방향에는 별도 패딩이 없다. `4096×1024`만 검사하면 모든 열의 마스크가 True라서 마지막 블록의 버그를 놓칠 수 있다.

`tl.exp`의 근사와 리덕션 순서 때문에 부동소수점 결과가 비트 단위로 같을 필요는 없다. 다만 검증이 실패했을 때 허용 오차부터 크게 늘려서는 안 된다. 유효 출력의 행별 합, 가장 큰 오차의 위치, 경계 열, 음수 행을 확인한다. `allclose`와 `.item()`으로 결과를 확인하는 작업은 타이밍 구간 밖에 둔다. [Tutorial 02 — Unit Test](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#unit-test)

### 9.3 실제 GPU 정확도 결과 — 2026-10-28

`torch.manual_seed(0)`, FP32, `rtol=1e-5`, `atol=1e-8`을 그대로 사용했다. 허용 오차를 완화하지 않았으며 **정확도 검사 41건이 모두 통과**했다. 의도적으로 틀린 패딩 반례는 이 통과 건수에 포함하지 않았다.

| 검사 | 결과 | 최대 절대오차 |
|---|---|---:|
| Vector Add, N=10 / B=4 | PASS | 0.000e+00 |
| naive vs torch, 1823×781 | PASS | 7.451e-09 |
| Triton vs torch, 1823×781 | PASS | 7.451e-09 |
| Triton vs naive, 1823×781 | PASS | 7.451e-09 |
| Triton 행별 합 vs 1, 1823개 행 | PASS | 2.384e-07 |
| 모두 -1인 3×781 입력 | PASS | 0.000e+00 |
| torch.compile vs torch, 1823×781 | PASS | 7.451e-09 |
| Triton vs torch, 4096×1024 | PASS | 1.490e-08 |

Vector Add는 빈 입력과 `N=1,10,255,256,257,98432`를 검사했고 최대 절대오차가 모두 0이었다. Softmax는 추가로 열 수 `1,3,31,32,33,1023,1024`에서 표준정규 입력에 100을 곱한 유한값을 검사했다. 작은 열 수, 2의 거듭제곱 경계, 큰 지수 입력에서 모두 통과했으며 이 추가 입력들의 최대 절대오차는 `2.980e-08`이었다. 성능 측정용 다섯 shape에서도 네 경로를 각각 검증했다. 검사별 shape와 오차는 [원본 JSON](experiments/results/results.json)의 `correctness`에 기록했다.

이 검사는 지정된 유한 FP32 입력에 대한 결과다. NaN·무한대·일반 stride·역전파 지원을 검증한 것은 아니다.

---

## 10. 벤치마크 — 어떤 시간을 재는가

### 10.1 Python 함수가 끝나도 GPU는 실행 중일 수 있다

GPU 작업은 일반적으로 비동기로 요청된다. 따라서 단순히 Python 호출 앞뒤에서 시계를 읽으면 GPU 완료시간 대신 작업을 제출한 시간만 재기 쉽다. 반대로 매 호출마다 무조건 동기화하면 원래 실행 흐름과 다른 오버헤드를 더할 수 있다.

`do_bench`는 GPU 타이밍을 위한 도구다. `warmup=25`, `rep=100`의 단위는 **호출 횟수가 아니라 ms 단위의 시간 예산**이다. 실행시간 추정에 따라 반복 횟수를 결정하므로 전체 벽시계 시간이 정확히 125 ms가 된다는 뜻은 아니다. 아래에서는 기본값에 기대지 않고 `return_mode="median"`을 명시한다. [triton.testing.do_bench API](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)

공식 구현의 흐름은 사전 호출·동기화 → 대략적인 실행시간 추정 → 워밍업·측정 반복 횟수 결정 → 반복별 이벤트 기록 → 완료 동기화·통계 계산이다. 캐시를 비우는 동작도 포함되지만, 함수 호출 사이의 캐시 정리와 `naive_softmax` 내부의 다섯 단계 사이에 캐시 적중이 전혀 없다는 가정은 다르다. 설치 버전에 따른 구현 차이는 [Triton testing.py의 do_bench 구현](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py)에서 확인한다.

### 10.2 실효 대역폭의 분자는 모두 같은 필수 작업량

Softmax의 필수 입력 읽기와 출력 쓰기를 기준으로 다음을 정의한다.

$$
Q_{\mathrm{useful}}=2MN\times4\ \mathrm{Byte}
$$

$$
BW_{\mathrm{effective}}[\mathrm{GB/s}]
=\frac{Q_{\mathrm{useful}}}{t_{\mathrm{ms}}\times10^{-3}}\times10^{-9}
=\frac{Q_{\mathrm{useful}}}{t_{\mathrm{ms}}\times10^6}
$$

예를 들어 **가상의 측정시간**이 0.1 ms라면 $33,554,432/(0.1\times10^6)=335.54432$ GB/s다. 이 숫자는 실제 GPU 실행 결과가 아니다.

단계별 구현에도 같은 `2MN×4`를 사용해야 **동일한 입력→출력 작업을 얼마나 빨리 마쳤는가**를 비교할 수 있다. `8MN×4`를 분자에 쓰면 불필요한 중간값 이동까지 유용한 일로 보상하게 된다. 모델상의 총 이동량/시간을 별도로 계산할 수는 있지만 이는 다른 지표다.

실효 대역폭은 물리적인 HBM 링크에서 실제로 측정한 Byte/s가 아니다. 캐시 적중이나 실행 범위의 차이로 사양표와 직접 비교하기 어려울 수 있다. 실제 트래픽은 프로파일러의 메모리 카운터로 별도 확인해야 한다.

### 10.3 실행시간과 GB/s를 함께 출력하는 코드

```python
# 선택: True로 바꾸면 torch.compile 경로도 검증 후 측정한다.
RUN_COMPILE = False
WARMUP_MS, REP_MS = 25, 100
M, N = 4096, 1024

with torch.no_grad():
    x_bench = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    reference_bench = torch.softmax(x_bench, dim=-1)

    # 각 함수는 같은 입력에 대해 결과 텐서를 반환한다.
    providers = {
        "naive_softmax": naive_softmax,
        "torch.softmax": lambda x: torch.softmax(x, dim=-1),
        "triton.softmax (one row/program)": softmax,
    }

    if RUN_COMPILE:
        # 기본 컴파일 모드. 실제 fusion 여부와 성능은 설치 버전·shape에 달려 있다.
        compiled_softmax = torch.compile(naive_softmax)
        # shape가 바뀌면 다시 컴파일될 수 있어 두 shape를 모두 먼저 실행한다.
        check_close("compiled irregular", compiled_softmax(x_test), reference)
        check_close("compiled negative", compiled_softmax(x_negative), torch.softmax(x_negative, dim=-1))
        providers["torch.compile(naive_softmax)"] = compiled_softmax

    # 측정할 shape로 모든 경로를 먼저 실행·검증한다.
    # JIT 및 torch.compile의 최초 컴파일 비용을 정상 반복 실행과 분리한다.
    for name, fn in providers.items():
        check_close(name + " benchmark shape", fn(x_bench), reference_bench)
    torch.cuda.synchronize(DEVICE)

    useful_bytes = 2 * x_bench.numel() * x_bench.element_size()
    print("\nBenchmark shape:", tuple(x_bench.shape), "dtype:", x_bench.dtype)
    print("BLOCK_SIZE:", triton.next_power_of_2(N), "num_warps:", 4)
    print("warmup_ms:", WARMUP_MS, "rep_ms:", REP_MS, "statistic: median")
    print("useful_bytes:", useful_bytes)
    print(f"{'provider':40s} {'ms':>12s} {'effective GB/s':>18s}")

    results = {}
    for name, fn in providers.items():
        # fn=fn로 지금의 함수를 고정해서 전달한다.
        # 입력 생성, 정확도 검사, print, CPU 전송은 이 호출 안에 넣지 않는다.
        ms = triton.testing.do_bench(
            lambda fn=fn: fn(x_bench),
            warmup=WARMUP_MS,
            rep=REP_MS,
            return_mode="median",
        )
        gbps = useful_bytes / (ms * 1e-3) / 1e9
        results[name] = (ms, gbps)
        print(f"{name:40s} {ms:12.6f} {gbps:18.3f}")

    baseline_ms = results["naive_softmax"][0]
    for name, (ms, _) in results.items():
        print(f"speedup vs naive — {name}: {baseline_ms / ms:.3f}x")
```

`torch.compile`은 Python 연산 그래프를 컴파일러 백엔드로 최적화하는 경로다. 수작업 Triton 코드와 같은 커널 수·배치를 보장하지 않는다. 선택 실행이 실패하면 오류와 버전을 기록하고 해당 경로는 ‘측정 불가’로 남긴다. 다른 함수의 결과를 컴파일 결과로 대체하지 않는다. [PyTorch — torch.compiler](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html)

### 10.4 이 코드가 재는 범위를 정확히 적기

| 구분 | 이번 측정에서의 처리 |
|---|---|
| 입력 생성과 CPU→GPU 전송 | 제외. 입력은 측정 전에 생성 |
| 최초 JIT·그래프 컴파일 | 각 shape로 사전 호출하여 제외 |
| 추가 워밍업 | `do_bench(warmup=25)` |
| 출력 할당·중간 텐서 생성 | 각 함수 호출 경로에 포함. 메모리 allocator 재사용 가능 |
| GPU 실행 | `do_bench`의 GPU 이벤트 기반 타이밍으로 비교 |
| 정확도 검사·출력·CPU 복사 | 제외 |
| 전체 Python 요청 왕복 지연 | 이 측정의 직접 대상이 아님 |
| 캐시 | 도구와 장치의 동작을 따름. 손계산의 무캐시 가정을 실측에서 보장하지 않음 |

이 표의 시간은 할당과 런치를 수행하는 **래퍼 함수를 대상으로 한 반복 GPU 타이밍**이다. 순수한 단일 커널만 격리해 측정한 시간이나 모든 CPU 비용을 포함한 end-to-end 시간과 동일하지 않다. 특히 작은 작업에서는 호스트의 제출 지연이 GPU 이벤트 간 시간에 영향을 줄 수도 있다. 순수 커널 비교를 하려면 출력 사전 할당 등 모든 경로의 측정 경계를 다시 맞춘다.

### 10.5 실제 실행 환경과 측정 방법

| 항목 | 실제 기록 |
|---|---|
| 실행일·플랫폼 | 2026-10-28 23:06 KST, Linux / WSL2 (`6.18.54.1-microsoft-standard-WSL2`) |
| GPU 이름·장치 | NVIDIA GeForce RTX 4070, `cuda:0`, Compute Capability 8.9, SM 46개 |
| GPU 메모리 | `nvidia-smi` 보고값 12,282 MiB |
| 드라이버·PyTorch CUDA build | 610.88 / 13.0 |
| Python·PyTorch·Triton | 3.12.13 / 2.14.1+cu130 / 3.8.0 |
| 사용 환경 | 기존 `../inference/.venv/bin/python`; GPU 패키지 재설치 없이 사용 |
| 정확도 | `[1823,781]` FP32 포함 총 41건 PASS, `rtol=1e-5`, `atol=1e-8` |
| 주 성능 입력 | `[4096,1024]`, FP32, contiguous, seed 0 |
| Triton 설정 | `BLOCK_SIZE=1024`, `num_warps=4`, 행당 프로그램 1개 |
| 측정 설정 | `do_bench`: warmup 25 ms, rep 100 ms, median; 5라운드 |
| 대표값·변동 폭 | 라운드별 중앙값 5개의 중앙값; 범위는 그 5개의 최소~최대 |
| 측정 순서 | 각 shape의 매 라운드마다 경로 순서를 seed 0으로 무작위화 |
| compile 선택 | `torch.compile(naive_softmax)`, 기본 모드, 모든 shape 검증·측정 성공 |
| 최초 준비 | compile 생성·불규칙 입력/음수 입력 검증·동기화까지 약 74.98초, 성능 타이밍에서 제외 |
| GPU 상태 참고 | 전체 실험 전 P8·31°C·21.33 W → 종료 후 P2·41°C·126.01 W |

입력은 GPU에 미리 만들고, 각 shape에서 네 경로를 먼저 실행·검증해 최초 컴파일을 측정에서 제외했다. 출력 할당을 수행하는 래퍼 전체를 `do_bench`에 넘겼다. allocator가 메모리를 재사용할 수 있으며, 순수 커널만의 시간이나 CPU 왕복 지연과 같지 않다. Python·Triton 캐시 위치는 `/tmp`로 지정했다. 설치된 Triton 3.8.0의 [실제 do_bench 소스](experiments/results/do_bench_source.py.txt)도 남겼다.

### 10.6 주 결과 — 4096×1024 FP32

네 경로 모두 필수 I/O **33,554,432 Byte**를 같은 분자로 사용했다. 실효 대역폭과 속도비는 표시용 반올림 전 시간으로 계산했다.

| 경로 | 대표 시간 ms | 라운드 중앙값 범위 ms | 실효 GB/s | naive 대비 속도비 |
|---|---:|---:|---:|---:|
| naive_softmax | 0.171008 | 0.171008–0.172032 | 196.216 | 1.000× |
| torch.softmax | 0.081920 | 0.081920–0.081920 | 409.600 | 2.088× |
| triton.softmax | 0.076800 | 0.076800–0.077824 | 436.907 | 2.227× |
| torch.compile | 0.076800 | 0.075776–0.076800 | 436.907 | 2.227× |

![RTX 4070에서 측정한 Softmax 시간과 실효 대역폭](<img width="1366" height="526" alt="image" src="https://github.com/user-attachments/assets/0d5cf905-54c9-4a99-b64e-a0c0cd1fe8dc" />
)

**Figure 3.** 막대는 5라운드 중앙값의 중앙값이다. 왼쪽 오차 막대는 라운드 중앙값의 최소·최대이며 통계적 신뢰구간이 아니다. 오른쪽 GB/s는 유용한 입력·출력량을 기준으로 계산했다.

교육용 Triton 구현은 naive의 **171.008 μs → 76.800 μs**, 즉 **2.227배**의 속도 향상을 보였다. `torch.softmax`는 81.920 μs로 naive보다 이미 2.088배 빨랐다. Triton은 `torch.softmax` 대비 시간 기준 약 6.25% 짧았지만, 이 작은 차이만으로 모든 입력에서의 우위를 주장할 수는 없다. `torch.compile`은 이 shape에서 Triton과 같은 대표값인 76.800 μs였다. 두 구현의 커널 구성이 같다는 뜻은 아니며, 컴파일 생성 코드의 커널 수는 프로파일링하지 않았다.

8절 모델의 이동량 비율 **4.002배**와 실측 속도비 **2.227배**는 다르다. 모델은 커널 사이 캐시 적중을 배제하고 같은 대역폭을 가정하지만, 실제 실행에는 리덕션·지수 연산·런치·메모리 할당·스케줄링이 포함된다. 캐시도 작동할 수 있다. 따라서 중간값 이동 감소와 런치 감소는 결과를 설명하는 요인이지만, 각각의 기여율이나 실제 메모리 Byte 수는 이번 측정으로 분리할 수 없다. 8절의 가상 1 TB/s 및 20 TFLOPs/s를 RTX 4070의 실제 사양으로 대입하지 않는다.

### 10.7 추가 shape 비교

| 입력 M×N | naive μs | torch.softmax μs | Triton μs | torch.compile μs | Triton의 naive 대비 속도비 |
|---|---:|---:|---:|---:|---:|
| 4096×1024 | 171.008 | 81.920 | 76.800 | 76.800 | 2.227× |
| 1823×781 | 61.440 | 34.816 | 28.672 | 21.696 | 2.143× |
| 128×1024 | 22.528 | 9.216 | 7.168 | 6.144 | 3.143× |
| 4096×256 | 48.592 | 21.504 | 22.528 | 29.696 | 2.157× |
| 4096×781 | 125.456 | 65.536 | 60.416 | 62.464 | 2.077× |

![행 수와 열 수에 따른 Softmax 실행시간 비교](<img width="1362" height="560" alt="image" src="https://github.com/user-attachments/assets/88c5d0af-b336-47ca-a928-44d0b37a6629" />
)

**Figure 4.** 세로축은 μs 단위의 로그 축이다. 모든 shape에서 같은 FP32·warmup·rep·라운드 정책을 사용했다. 781열도 1024 크기의 논리적 블록을 사용하지만 실제 load·store는 781개 열에만 접근한다.

`4096×256`에서는 `torch.softmax`가 21.504 μs로 Triton의 22.528 μs보다 빨랐다. 반면 `1823×781`에서는 compile 21.696 μs, Triton 28.672 μs였고, `4096×781`에서는 Triton 60.416 μs, compile 62.464 μs였다. 입력 크기에 따라 순위가 바뀌므로 결합 효과와 라이브러리 대비 우위를 구분해야 한다. 작은 `128×1024`에서는 전체 시간이 수 μs이므로 제출 지연·측정 해상도에 더 민감하다.

시간들이 약 1 μs 간격으로 반복되는 양상도 보였다. 이는 기록된 GPU 이벤트 시간의 양상이며, 정확한 원인을 별도 검증하지 않았다. 약 1 μs 차이를 정밀한 성능 격차로 해석하지 않는다. 클록·전력 제한 고정, 타 작업 완전 배제, 다중 세션 재현은 수행하지 않았으며 전·후 GPU 상태만 기록했다. 이 표의 변동 폭은 한 세션 내 다섯 번 측정의 범위다.

### 10.8 실행 코드·측정 원본·Figure 재현

- [실험 실행 및 figure 재현 안내](experiments/README.md)
- [문서와 같은 Vector Add·Softmax 커널](experiments/kernels.py)
- [정확도·패딩 반례·반복 GPU 벤치마크 코드](experiments/run_experiments.py)
- [환경·정확도·라운드별 시간·실행 순서 JSON](experiments/results/results.json), [요약 CSV](experiments/results/benchmarks.csv), [실행 로그](experiments/run.log)
- [Figure 생성 코드](experiments/make_figures.py), [Matplotlib 의존성](experiments/requirements-figures.txt)

Figure는 측정 JSON에서 Matplotlib으로 생성했다. 설명 그림의 이동량은 8절의 모델이고, 성능 그림은 실제 실행값이다. 물리 메모리 트래픽·occupancy·spill·실제 커널 수는 별도 프로파일링하지 않았다.

---

## 11. 자주 생기는 오해와 문제 찾기

| 오해·증상 | 확인할 것 |
|---|---|
| BLOCK_SIZE가 1024이므로 스레드도 1024개다 | 논리적 원소 수와 `num_warps`를 구별 |
| `@triton.jit` 함수 안에서 torch 연산을 부르면 된다 | 호스트 Python 래퍼와 GPU Triton 언어를 구별 |
| load만 마스킹하면 된다 | store의 범위 밖 쓰기도 차단 |
| `other=0`인데 출력 mask를 했으니 맞다 | 패딩이 이미 분모에 들어갔는지 확인 |
| 두 번째 행부터 결과가 이상하다 | 행 간격에 BLOCK_SIZE 대신 실제 stride를 썼는지 확인 |
| 전치한 입력에서만 틀린다 | 열 stride=1 가정을 확인. 이 래퍼는 전치를 거부 |
| 4 KiB가 커널의 전체 온칩 사용량이다 | 입력 외 중간값·리덕션 자원·레지스터 배치를 구별 |
| Python 호출이 빠르니 GPU 계산도 빠르다 | 비동기 실행과 타이밍 경계를 확인 |
| 첫 실행만 매우 느리다 | JIT·라이브러리 초기화·compile 비용 분리 |
| Triton이 torch.softmax보다 느리면 결합이 실패했다 | torch.softmax도 최적화되어 있음. eager naive와 별도 비교 |
| 실효 GB/s가 실제 HBM 전송량이다 | 분자는 필수 작업량. 물리 트래픽은 별도 지표 |
| CPU 해석기로 GPU 속도비를 비교한다 | 해석기는 의미·인덱싱 디버깅 전용 |

---

## 12. 연습문제와 해설

### 문제 1 — 블록과 포인터

`N=10`, `BLOCK_SIZE=4`에서 grid와 모든 pid의 offsets·mask를 써보자. FP32 입력 포인터를 기준으로 `pid=2`의 첫 유효 원소는 시작 주소에서 몇 Byte 떨어져 있는가? load만 마스킹하면 안전한가?

**해설:** grid는 `(3,)`, 세 블록은 `[0,1,2,3]`, `[4,5,6,7]`, `[8,9,10,11]`이다. 마지막 마스크는 `[T,T,F,F]`. 첫 원소 `x[8]`은 `8×4=32` Byte 뒤에 있다. store도 마스킹해야 한다.

### 문제 2 — 패딩 버그 찾기

781개 원소가 모두 -1인 행에서 패딩 243개를 0으로 채우면 최댓값과 분모는 얼마인가? 출력의 유효한 원소 합이 1인가? 입력에 양수가 있으면 버그가 없어지는가?

**해설:** 최댓값은 0, 분모는 $781e^{-1}+243$이다. 유효 출력의 합은 약 0.541781이므로 틀린다. 양수가 있어도 패딩의 지수 기여가 분모에 남는다. `other=-inf`라면 패딩의 기여가 0이 된다.

### 문제 3 — 모델과 측정의 분자

`4096×1024` FP32 Softmax의 단계별·결합 Q와 메모리 시간을 구해보자. 단계별 구현의 실측시간이 **가령** 0.2 ms라면 실효 대역폭의 분자에 어떤 값을 넣어야 하는가?

**해설:** 값 텐서 모델에서 Q는 134,283,264 Byte와 33,554,432 Byte다. 가상 대역폭 1 TB/s에서 시간은 0.134283264 ms와 0.033554432 ms이며 비는 약 4.002배다. 실효 대역폭은 두 구현 모두 `2MN×4=33,554,432` Byte를 분자로 사용하므로 가상의 0.2 ms에서는 167.77216 GB/s다. 이 값은 실제 총 트래픽/시간과 다르다.
