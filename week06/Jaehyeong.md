# 6주차 | Triton의 블록 단위 프로그래밍과 Kernel Fusion

**학습 목표:** Triton이 데이터를 Tile 단위로 처리하는 방식과 Kernel Fusion이 메모리 이동량 및 실행 성능에 미치는 영향을 이해하고, 이론적 계산과 실제 GPU 벤치마크를 통해 검증한다.

## 1. CUDA와 Triton의 프로그래밍 모델

**핵심 차이는 병렬 연산을 기술하는 추상화 수준이다.**

| 구분 | CUDA C++ (일반적인 SIMT 구현) | Triton |
| --- | --- | --- |
| 프로그래밍 중심 | Thread | Program Instance / Tile |
| 데이터 처리 | 스레드별 원소 지정 | 프로그램별 원소 블록 지정 |
| 인덱싱 | `threadIdx`, `blockIdx` | `tl.program_id`, `tl.arange` |
| 스레드 배분 | 개발자가 명시적으로 제어 | 컴파일러가 상당 부분 처리 |
| 장점 | 하드웨어 세밀 제어 | 구현 간결화, 자동 최적화 |
- **`Tile`:** Tensor의 일부를 묶은 논리적 데이터 블록
- **`Program Instance`:** 할당된 Tile을 처리하는 실행 인스턴스
- **`Warp`:** NVIDIA GPU에서 32개 Thread를 묶는 실행·스케줄링 단위

Triton은 Tile 단위 연산을 기술하면 <u>컴파일러</u>가 **스레드별 데이터 배분, 메모리 접근 병합(Coalescing) 등을 최적화**한다. 단, CUDA도 Tile 기반 최적화가 가능하며, Triton 역시 Tile 크기와 Warp 수 등의 튜닝은 필요하다.

길이가 N인 벡터를 BLOCK_SIZE단위로 나눌 때 필요한 프로그램 인스턴스 수

![image.png](image.png)

`grid = (triton.cdiv(N, BLOCK_SIZE), )`는 GPU에서 실행할 Triton 프로그램 인스턴스의 개수를 결정하는 코드
****

## 2. 블록 인덱싱과 Mask

**실습 조건:** `N=10`, `BLOCK_SIZE=4`

프로그램 인스턴스 개수: 10

블록 크기: 4

각 프로그램이 담당할 인덱스는 다음과 같다.

```
입력 X:  [0 1 2 3 | 4 5 6 7 | 8 9]
          Tile 0    Tile 1    Tile 2
                              (10,11은 범위 밖)
```

![image.png](image%201.png)

![image.png](image%202.png)

**Triton 코드의 핵심 흐름**

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements

x = tl.load(x_ptr + offsets, mask=mask, other=0)
tl.store(output_ptr + offsets, x, mask=mask)
```

- `tl.program_id(0)`: 현재 프로그램 인스턴스 식별
- `tl.arange()`: Tile 내부 인덱스 벡터 생성
- `x_ptr + offsets`: 텐서 시작 주소에 원소별 오프셋을 더한 포인터 블록 생성
- `mask`: 범위 밖 메모리 접근 방지
- `other`: Mask가 False인 위치에서 반환할 대체값
- `tl.constexpr`: Tile의 크기처럼 컴파일 단계에서 확정해야 하는 값을 지정

**호스트 Python의 `torch.Tensor`는 JIT 커널에 전달될 때 <u>GPU 메모리를 가리키는 포인터</u>로 취급된다.**

**정리:** Grid가 작업 개수를 결정하고, PID와 Offset이 담당 원소를 정하며, Mask가 메모리 접근을 보호한다.

## 3. Kernel Fusion과 HBM 이동량

**Kernel Fusion은 여러 연산을 하나의 GPU 커널로 결합해 중간 데이터의 불필요한 DRAM 왕복을 제거하는 최적화 기법이다.**

Softmax의 연산 순서: `Max → Subtract → Exp → Sum → Divide`

- **Naive Softmax:** 5개 연산을 각각 독립 커널로 실행하며 중간 결과를 DRAM에 저장·재읽기
    
    ![image.png](image%203.png)
    
    입력 행렬 크기가 M,N이면, 각 행의 최댓값과 합계는 원소 M짜리 1D 벡터이다.
    
    따라서 총 데이터 이동량은 
    
    ![image.png](image%204.png)
    
    마지막 4B는 FP32의 자료형 크기.
    
- **Fused Softmax:** 한 행을 온칩으로 읽고 5개 연산을 연속 수행한 뒤 최종 결과만 저장
    
    ![image.png](image%205.png)
    
    총 이동량: (데이터 읽기)MN + (데이터쓰기)MN = 2MN * 4Byte
    
    ![image.png](image%206.png)
    
    결론: Fused Softmax는 Naive보다 약 4배 이상 효율적이다.
    

### **HBM 이동량 손계산**

과제 조건은 다음과 같아.

- (M=4096, N=1024)
- FP32 = 4 Byte

| 항목 | Naive | Fused |
| --- | --- | --- |
| HBM 이동량 | 134.28 MB | 33.55 MB |
| 상대적 이동량 | 100% | 약 25% |
| Kernel 수 | 5 | 1 |

Fused가 Naive보다 데이터 이동량이 75% 적다.

※ 커널 간 캐시 적중이 없고 Fused 연산의 중간 결과가 온칩에 유지된다는 가정이다.

## 4. Reduction과 Fused Softmax의 패딩

**Reduction:** 여러 데이터 원소를 최댓값·합계 등의 집계 결과로 축소하는 연산.

Softmax에서는 두 번 사용한다.

- `tl.max()`: 최댓값을 빼 지수 계산의 Overflow 방지
- `tl.sum()`: 지수값의 합을 계산해 출력 정규화

Triton의 기본 Softmax 구현에서는 다음과 같이 2의 제곱에 맞게 Tile 크기를 지정한다.

```python
BLOCK_SIZE = triton.next_power_of_2(N) 
```

예를 들어 `N=781`이면 Tile 크기는 1024이고, 243개 원소는 패딩 영역이다.

```python
row = tl.load(
    input_ptrs,
    mask=mask,
    other=-float("inf")
)
```

**패딩에 `-inf`를 사용하는 이유**

1. `tl.max()`가 유효 원소의 최댓값을 왜곡하지 않는다.
2. `exp(-inf)=0`이므로 패딩이 지수 합계에 영향을 주지 않는다.
3. `mask`는 별도로 범위 밖 메모리 접근을 방지한다.

`other=0`이면 음수 원소만 있는 행에서 최댓값이 잘못 계산될 수 있고, 패딩 위치의 지수값이 합계에 포함된다.

**온칩 메모리 제약:** 행 길이가 증가하면 레지스터·Shared Memory 요구량이 커진다. Register Spill이나 Occupancy 감소로 성능이 저하될 수 있으므로 Fusion과 Tile 확대가 항상 유리하지는 않다.

## 5. Roofline 관점의 이론적 성능 분석

산술 강도(Arithmetic Intensity):

$$
I=\frac{F}{Q}
$$

- (F): 연산량
- (Q): DRAM 데이터 이동량

Kernel Fusion은 연산량을 크게 변경하지 않으면서 Q를 줄여 산술 강도를 증가시킨다.

**가상 하드웨어 조건**

- FP32 연산 성능: 20 TFLOPS
- HBM 대역폭: 1 TB/s

Memory-bound를 가정하면,

$$
T_{\text{memory}}=\frac{Q}{BW}
$$

| 구분 | 이론적 메모리 시간 |
| --- | --- |
| Naive | 0.1343 ms |
| Fused | 0.0336 ms |

$$
Speedup=\frac{T_{\text{naive}}}{T_{\text{fused}}}
\approx \boxed{4.0\times}
$$

**핵심:** 두 커널이 모두 Memory-bound라면 데이터 이동량을 약 4분의 1로 줄임으로써 이론적으로 약 4배의 성능 향상을 기대할 수 있다.

단, Compute-bound 영역에서는 메모리 이동량 감소가 동일한 비율의 성능 향상으로 이어지지 않는다.

## 6. Google Colab 실습 결과

### 환경

GPU: Tesla T4
PyTorch: 2.11.0+cu130
Triton: 3.6.0

### **1. Fused Softmax 구현 및 정확도 확인**

`입력` 1823 × 781, FP32

```python

import torch
import triton
import triton.language as tl
import triton.testing

def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    return numerator / denominator[:, None]

@triton.jit
def softmax_kernel(X, Y, stride_x: tl.constexpr,
                   stride_y: tl.constexpr,
                   N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols,
                mask=mask, other=-float("inf"))

    e = tl.exp(x - tl.max(x, axis=0))
    y = e / tl.sum(e, axis=0)

    tl.store(Y + row * stride_y + cols, y, mask=mask)

def fused_softmax(x):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    softmax_kernel[(M,)](
        x, y, x.stride(0), y.stride(0), N, BLOCK,
        num_warps=4
    )
    return y

x = torch.randn(1823, 781, device="cuda")
ref = torch.softmax(x, dim=1)

print("Naive:", torch.allclose(naive_softmax(x), ref))
print("Triton:", torch.allclose(fused_softmax(x), ref))
```

```
Naive: True
Triton: True
```

- 기준: `torch.softmax`와 `torch.allclose` 비교

2의 거듭제곱이 아닌 크기를 사용해 패딩 및 Mask 처리의 정확성도 함께 검증하였다.

### **2. 실제 실행시간 및 대역폭 비교**

입력 4096 × 1024, FP32

```python

x = torch.randn(4096, 1024, device="cuda")

methods = {
    "Naive": lambda: naive_softmax(x),
    "PyTorch": lambda: torch.softmax(x, dim=1),
    "Triton": lambda: fused_softmax(x),
}

results = {}

for name, fn in methods.items():
    fn()  # 초기 실행 및 JIT 컴파일
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(fn, warmup=100, rep=300)
    gbps = 2 * x.numel() * x.element_size() / (ms * 1e6)
    results[name] = ms

    print(f"{name:8s} | {ms:.4f} ms | {gbps:.2f} GB/s")

print(f"\nSpeedup (Naive / Triton): "
      f"{results['Naive']/results['Triton']:.2f}x")
      
```

```
Naive    | 0.6334 ms | 52.98 GB/s
PyTorch  | 0.1579 ms | 212.50 GB/s
Triton   | 0.1474 ms | 227.66 GB/s

Speedup (Naive / Triton): 4.30x
```

Kernel Fused Triton softmax가 Naive Softmax보다 4배 이상 빠르다.

### 실효 대역폭 계산

$$
BW_{\text{effective}}
=\frac{2MN\times4}{T_{\text{ms}}\times10^6}
\quad [GB/s]
$$

분자 `2MN`은 입력 한 번 읽기 + 출력 한 번 쓰기인 **최소 필수 데이터 이동량**을 기준으로 한다. 따라서 Naive의 실제 이동량인 `8MN+4M`을 사용하지 않으며, 계산된 GB/s는 실제 DRAM 대역폭 계측값과 다르다.

`do_bench`는 워밍업과 반복 측정을 수행한다. GPU 커널 실행은 비동기적이므로 일반적인 Python 함수 호출 시간만 측정해서는 정확한 GPU 실행시간을 구할 수 없다.

## 7. 최종 결론

1. **Triton은 Tile을 중심으로 병렬 연산을 작성한다.** 스레드별 데이터 배분과 메모리 접근 최적화의 상당 부분을 컴파일러에 위임한다.
2. **Kernel Fusion의 주요 이점은 DRAM 이동량 감소다.** Softmax의 중간 결과를 온칩 자원에서 유지해 데이터 이동량을 약 75% 줄인다.
3. **이론적으로 약 4배, 실측으로 4.30배의 Naive 대비 속도 향상을 확인했다.** 그러나 실제 성능은 메모리 대역폭뿐 아니라 커널 실행 오버헤드와 GPU 자원 활용도에도 영향을 받는다.

### 참고 자료

- [Triton Programming Guide — Introduction, Motivations / Challenges](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)
- [Triton Tutorial 01 — Vector Addition, Compute Kernel / Benchmark](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton Tutorial 02 — Fused Softmax, Motivations / Compute Kernel](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [NVIDIA CUDA — Thread Blocks and Grids](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-blocks-and-grids)
- [Triton API — do_bench](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)