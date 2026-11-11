<WeekNavBar :week="2" topic="모델에서 실리콘까지" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송무복)', 'badge': 'PyTorch·XLA·MAX 2부작'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': '다이어그램 & 스택 해부'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': '실행 스택 핵심 요약'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '인터랙티브 HTML & 슬라이드'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': '컴파일 파이프라인 정리'}]" />

> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **singon (김신곤)** (`다이어그램 & 스택 해부`)
>
> 💡 **핵심 기여 요약**: 풍부한 아키텍처 다이어그램을 바탕으로 모델 수식이 그래프 캡처(Dynamo/FX)와 컴파일러를 거쳐 실리콘 명령어로 내려가는 흐름을 시각화했습니다.

---

**“모델의 연산 하나가 가속기에서 실행되기까지, 어떤 단계를 거치고 각 단계는 무엇을 담당할까?”**

<img width="672" height="722" alt="Image" src="https://github.com/user-attachments/assets/cd3699e6-1fc7-46aa-b63c-d0b7ded5a1ce" />

**그림:** 2절 에 나온 다이어그램으로 TPU가 요소별 곱셈을 수행하는 방식을 보여줍니다. 배열의 크기와 다양한 링크의 대역폭에 따라 연산 병목 현상(하드웨어 연산 용량을 최대한 활용) 또는 메모리 병목 현상(메모리 부하로 인한 병목 현상)이 발생할 수 있습니다.

# **커널 → 칩 내부**연산은 어디서 이루어지고 데이터는 어디에서 이동해올까?

Scaling Book 2장의 What Is a TPU? 또는 12장의 What Is a GPU?와 Memory 중심

## TPU란 무엇인가?

**TPU는 기본적으로 행렬 곱셈에 특화된 컴퓨팅 코어(TensorCore)와 고속 메모리 스택(HBM)이 연결된 구조**

<img width="1260" height="632" alt="image" src="https://github.com/user-attachments/assets/3580ad51-f8d7-4907-973b-7769dc84fbfc" />


<img width="1266" height="478" alt="image" src="https://github.com/user-attachments/assets/dad2631a-c905-482f-8cbe-40c22315cb2a" />


### Tensor Core의 구성요소

- MXU (Matrix Multiply Unit, 행렬 곱셈 장치)
    - TensorCore의 핵심
    - 대부분의 TPU 세대에서 MXU는 하나의 행렬 곱셈 연산을 수행
    - `bf16[8,128] @ bf16[128,128] -> f32[8,128]`
- VPU (Vector Processing Unit, 벡터 처리 장치) 부록 A를 참조하십시오.
    - ReLU 활성화 함수 사용이나 벡터 간의 간단한 pointwise addition 또는 multiplication과 같은 일반적인 수학 연산을 수행
    - Reductions(sums) 연산도 수행
- **VMEM (Vector Memory)**
    - (벡터 메모리)은 TensorCore의 연산 장치 근처에 위치한 온칩 임시 저장 장치
    - HBM(예: TPU v5e의 경우 128MiB)보다 훨씬 작지만 MXU와의 대역폭은 훨씬 높음
    - VMEM은 CPU의 L1/L2 캐시와 유사하게 작동하지만 용량이 훨씬 크고 프로그래머가 제어할 수 있음
    - TensorCore에서 연산을 수행하려면 HBM에 저장된 데이터를 VMEM으로 복사해야 합니다.

## **GPU란 무엇인가요?**

**현재 나오는 GPU (H100, B200 등)는 Streaming Multiprocessors** or **SMs라 불리는 Matrix Multiplication에 특화된 많은 코어들이 HBM에 연결되어있음**

<img width="1268" height="592" alt="image" src="https://github.com/user-attachments/assets/81fa817c-733a-4910-a37f-5c72b5845218" />


- Streaming Multiprocessors (SMs)의 구성
    - Tensor Core (matrix multiplication core)
        - 각 서브파티션에는 TPU MXU와 같은 전용 행렬 곱셈 장치인 텐서 코어가 있습니다. 텐서 코어는 GPU의 FLOPs/s 성능의 대부분을 담당합니다(예: H100에서 bf16 TC는 990 TFLOP/s의 성능을 보이는 반면, CUDA 코어는 66 TFLOP/s에 불과합니다).
    - Warp scheduler (a vector arithmetic unit)
        - **CUDA 코어:** 각 하위 파티션에는 SIMD/SIMT 벡터 연산을 수행하는 CUDA 코어라고 하는 ALU 세트가 포함되어 있습니다. 각 ALU는 일반적으로 사이클당 하나의 산술 연산(예: f32.add)을 수행할 수 있습니다.4각 서브파티션에는 32개의 fp32 코어(및 더 적은 수의 int32 및 fp64 코어)가 포함되어 있으며, 이 코어들은 매 사이클마다 동일한 명령어를 실행합니다. TPU의 VPU와 마찬가지로 CUDA 코어는 ReLU 활성화 함수, 점별 벡터 연산 및 축소(합계) 연산을 담당합니다.
        - VPU와 달리 각 CUDA 코어(CUDA 프로그래밍 모델에서는 "스레드")는 고유한 명령어 포인터를 가지며 독립적으로 *프로그래밍* 할 수 있습
    - a fast on-chip cache(SMEM)
- TPU와 차이점
    - 최대 2개의 독립적인 텐서 코어를 가진 TPU와 달리, 최신 GPU는 100개 이상의 SM(H100은 132개)을 탑재하고 있습니다.
    - GPU의 SMs는 Tensor의 Tensor Core보다 약하지만 대부분 독립적이며 한번에 다양한 프로세스를 돌릴 수 있음 (유연성이 강함)
        - SM은 멀티스레드 CPU와 유사하게 작동하여 여러 프로그램( **워프** )을 동시에(SM당 최대 64개) "스케줄링"할 수 있음
        - 단, 각 *워프 스케줄러는* 클록 사이클마다 단 하나의 프로그램만 실행합니다
    - GPU는 LLM이나 ML 모델만을 위해 설계된 것이 아니라 범용 가속기로 설계
        - GPU는 TPU에 비해 새로운 작업에 적용될 때 훨씬 더 자주 "just work"하며, 우수한 컴파일러에 대한 의존도가 훨씬 낮음
        - 수많은 컴파일러 기능이 병목 현상을 일으킬 수 있어 GPU의 작동 원리를 파악하거나 최대 성능을 끌어내는 것이 훨씬 더 어려워짐
        - 반대로 모듈성 차이로 인해 TPU는 구축 비용이 훨씬 저렴하고 이해하기 쉽지만, 컴파일러가 올바른 처리를 해야 한다는 부담이 더 커짐
        - 대부분 GPU가 성능과 HBM 메모리는 더 뛰어나지만 비쌈. TPU는 많은 것을 연결하는 것에 집중
        - **TPU는 GPU보다 훨씬 더 많은 고속 캐시 메모리를 가지고 있음**
            - SMEM(+TMEM)보다 훨씬 더 많은 VMEM을 보유
            - 이 메모리를 사용하여 가중치와 활성화 값을 매우 빠르게 로드하고 사용할 수 있도록 저장
        
        <img width="1114" height="332" alt="image" src="https://github.com/user-attachments/assets/04e88ebd-9b54-45d4-94eb-221d3b01e32c" />

        
        <img width="660" height="916" alt="image" src="https://github.com/user-attachments/assets/9f5409fe-fbdc-4ef9-aad1-590e4a244409" />

        
        <img width="1248" height="480" alt="image" src="https://github.com/user-attachments/assets/03f97abc-179c-45e2-af2b-fe42ba60d029" />

## 하드웨어에서 알고리즘을 실행할 때, 고려해야할 3가지 제약 조건

- 첫째는 컴퓨터의 연산 속도(초당 연산 횟수)
    - computation
        - Our accelerator speed determines how long these take to compute:
- 둘째는 데이터 전송에 사용할 수 있는 대역폭(초당 바이트 수)
    - HBM 대역폭: *가속기 내부에서* 가속기 메모리(HBM)와 연산 코어 간에 텐서를 전송할 때 속도
        - H100에서는 약 3.35TB/s
        - TPU v6e에서는 약 1.6TB/s
        
        <img width="1130" height="874" alt="image" src="https://github.com/user-attachments/assets/b29c7148-2b71-4f2c-9b9f-57588e61c43c" />


        
    - computation, communication time 측정에 lower bound, upper bound를 두고 계산할 수 있음
        - COmputation time이 크다면 사용 가능한 Flops를 최대한 사용
        - Communication time이 크다면 데이터 통신 병목이 엄청 큰 것, Flops 계산 효율 버리는 것
        
        <img width="962" height="334" alt="image" src="https://github.com/user-attachments/assets/9e2f78bf-4693-44d0-b5e2-f27081119be1" />

        
- 셋째는 데이터를 저장하는 데 사용할 수 있는 총 메모리 용량(바이트 수)

# **연산 → 그래프·IR**Python 코드가 어떻게 컴파일러가 다룰 표현으로 바뀔까?

Scaling Book 9장: How to Profile TPU Code

## **A Thousand-Foot View of the TPU Software Stack**

### **TPU 소프트웨어 스택에 대한 전체적인 개요**

대부분의 프로그래머는 JAX 코드만 사용하여 NumPy 스타일의 추상적인 선형 대수 프로그램을 작성하고 자동으로 컴파일하여 TPU에서 효율적으로 실행

### Jax.jit

```python
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

- **`jax.jit`을 호출하면 JAX는 해당 함수를 추적하여 StableHLO라는 저수준 IR(중간 표현)을 생성**
    - **IR(Intermediate Representation):** 소스 코드와 실제 기계어 사이에 있는 **중간 코드 표현**
    - **StableHLO:**  JAX가 사용하는 여러 IR 중 하나로, **텐서 연산을 표현하는 표준화된 저수준 IR**
    - jax.jit 실행 흐름 예시
        
        ```
        Python / JAX 코드
                ↓
        JAX tracing
                ↓
        JAXPR
                ↓
        StableHLO
                ↓
        XLA 내부 IR
                ↓
        GPU / TPU / CPU 기계어
        ```
        
- **StableHLO는 머신러닝 연산을 위한 플랫폼 독립적 IR**
    - https://openxla.org/stablehlo?hl=ko
    
    !image.png
    
- **이후 XLA 컴파일러에 의해 HLO로 변환(lowering)**

프로그램 실행 속도가 원하는 수준보다 느릴 경우, 주로 JAX 레벨에서 성능 개선 작업을 진행

- HLO의 의미 체계와 TPU에서 코드가 실제로 어떻게 실행되는지 이해필요
- 하위 레벨에서 문제가 발생하면, Pallas 에서 사용자 정의 커널을 작성하는 또 다른 방법을 사용
- 프로그램의 HLO와 실행 통계를 확인하려면 JAX 프로파일러를 사용

제목은 프로파일링이지만 앞부분은 JAX 코드가 TPU 실행 코드로 변환되는 흐름을 설명합니다.

# torch.compiler

• PyTorch: torch.compiler 개요

도입부와 **Dynamo / AOT Autograd / Inductor** 설명을 중심으로, 익숙한 PyTorch 코드가 어떻게 컴파일되는지 살펴봐주세요.

- pytorch 2.X 부터 제공
- PyTorch에서 정확한 그래프 캡처 문제를 해결하고 궁극적으로 소프트웨어 엔지니어가 PyTorch 프로그램을 더 빠르게 실행할 수 있도록 하는 것을 목표
    - 그래프가 무엇인가?
        - **연산들의 의존 관계를 그린 그림**
        - 연산 하나가 **노드** 가 되고, “이 연산의 결과가 저 연산의 입력으로 들어간다"는 관계가 **간선** 이 됩니다. 결과가 자기 자신에게 되돌아오는 일은 없으니 방향이 있고 순환이 없는 그래프, 즉 **방향 비순환 그래프(Directed Acyclic Graph, DAG)** 가 됩니다.
        
        <img width="1116" height="696" alt="image" src="https://github.com/user-attachments/assets/7b01e5e9-3002-4978-bc99-f46ce365cc59" />

        
- python으로 작성 되었으며 pytorch가 C++에서 python으로 코드가 변하게 된 계기

종류

<img width="1262" height="878" alt="image" src="https://github.com/user-attachments/assets/a410b344-8345-4971-9e07-d03251a5bf45" />


- **TorchDynamo(torch._dynamo)**
    - CPython feature를 사용한 internal API
    - 캡처된 그래프를 빠른 기계어로 변환하는 백엔드
- **TorchInductor**
    - `torch.compile` 의 딥러닝 컴파일러의 default
    - multiple accelerators and backends를 위한 빠른 코드 생성
- **AOT Autograd**
    - 사용자 수준 코드뿐만 아니라 역전파 과정까지 캡처하므로 역방향 전달 과정을 "미리" 캡처할 수 있습니다. 이를 통해 TorchInductor를 사용하여 순방향 및 역방향 전달 모두를 가속화할 수 있습니다.
    - 의미를 보존하면서 표현을 단순하게 만드는 일을 함
        - 그래프 다듬는 두 가지 종류
            - **functionalization**
                - **기존 메모리를 덮어쓰는 연산** 을 없애는 작업입니다. `x.add_(1)` 은 새 텐서를 만드는 대신 `x` 자체를 고치고, `view` 로 만든 텐서는 원본과 메모리를 공유합니다. 이러면 컴파일러가 연산을 옮기거나 합칠 때마다 “지금 이 `x` 가 언제 시점의 값인지"를 일일이 따져야 합니다. 그래서 **한 번 만들어진 값은 변하지 않도록** 그래프를 고쳐 둡니다.
            - **decomposition**
                - 수천 개의 연산을 **core ATen IR** 이라는 수백 개짜리 부분집합으로 낮추는 작업입니다. 백엔드를 만드는 사람 입장에서 이건 결정적입니다. 구현해야 할 연산의 개수가 한 자릿수 배 줄어듭니다. 새로운 하드웨어를 PyTorch에 붙일 때 이 성질이 왜 중요한지는 다음 편에서 다시 이야기하겠습니다.
    - 같은 연산을 부르는 여러 방법을 하나로 통일하고, 부작용을 걷어내고, 연산의 종류를 줄임
        - 예시
            
            <img width="1050" height="564" alt="image" src="https://github.com/user-attachments/assets/aa1fe510-f4be-45d1-9a05-5b1c891ff195" />

            
- 가장 많이 사용하는 백엔드
    
    <img width="1180" height="632" alt="image" src="https://github.com/user-attachments/assets/ad571712-4953-4394-8271-ade474eba7b5" />

    

# Sharded

<img width="1260" height="648" alt="image" src="https://github.com/user-attachments/assets/3b663471-890d-478c-ab14-9f3dead7836d" />


**AllGather란 무엇일까요?**

- 특정축을 따라 sharding을 제거하고 합침(reaseemble)
    - 모든 축의 sharding을 제거하지 않아도 AllGather을 의미할 수 있음
    - 결국 행렬 연산을 위해 그에 맞는 축을 모두 concat하기 위해 사용
- Model의 Loss와 같은 것을 AllGather할 때 모든 파라미터를 한번에 하는 것이 아닌 block단위나 layer 단위로 AllGather 하는 것
    - M_{peak}≈M_{persistent shards}+M_{current full layer}+M_{activation}+M_{temporary buffer}

<img width="1182" height="274" alt="image" src="https://github.com/user-attachments/assets/993f897d-9f1f-4ead-b1bb-ae1d309ba374" />


전부 **여러 GPU가 가지고 있는 텐서를 어떻게 서로 주고받고, 합치고, 다시 나눌지**를 정의하는 collective communication

1. **AllGather:**
2. **ReduceScatter:**
3. **AllReduce (Tensor parallel, TP):**

전문가 혼합 모델(MoE) 및 기타 계산의 경우에 발생하는 또 다른 핵심 통신 기본 요소가 있는데, 바로 **AllToAll** 입니다 .

| 연산 | 여러 GPU 데이터 합치기 | Reduce(sum 등) | 결과가 각 GPU에 전체 복제? | 결과가 다시 shard? |
| --- | --- | --- | --- | --- |
| **AllGather** | O | X | O | X |
| **ReduceScatter** | O | O | X | O |
| **AllReduce** | O | O | O | X |
| **AllToAll** | 재배치 | X | X | O, 다른 축으로 |

**2. 해당 사항들을 다같이 정리해주세요**

**① 모델 → 연산**Transformer를 펼치면 어떤 연산과 텐서가 나올까?

Scaling Book 4장의 Counting Dots, Transformer Accounting 중심

# Transformer Math

## Counting Dots

<img width="480" height="556" alt="image" src="https://github.com/user-attachments/assets/7ad1fc15-2ce4-4506-b473-2579fde098ce" />


- x*y 는 P번 더하거나 곱하는 연산, 2P floating point operational total
- AB = 2*NPM FLOPS

<img width="1222" height="364" alt="image" src="https://github.com/user-attachments/assets/17876755-6f92-42d4-8bc6-9186ed2f5d43" />


- 각 Element에 대해서 곱하고, 나중에 Sum을 하기 때문에 2배가 됨
- 

# OpenXLA: XLA architecture의 Objectives

 **IR → 실행 코드**컴파일러는 무엇을 결정하고 하드웨어별로 무엇이 달라질까?

OpenXLA: XLA architecture의 Objectives, How it works 중심

## XLA이란?

- XLA (Accelerated Linear Algebra)는 선형 대수를 최적화하는 머신러닝 (ML) 컴파일러
- 실행 속도와 메모리 사용량을 개선
- **커스텀 작업에 대한 의존도를 줄임**
- **이동성 개선**

# **실행 요청 → 커널 실행** Python 함수가 반환되면 가속기 계산도 끝난 걸까? 커널은 어떻게 실행할까?

JAX: Asynchronous dispatch와 Triton: Vector Addition의 커널·호출 코드 중심

## Asynchronous dispatch

- Python이 GPU/TPU에게 “이 계산 해”라고 명령만 던지고, 계산이 끝날 때까지 기다리지 않고 다음 Python 코드를 계속 실행하는 방식
    - `jnp.dot(x, x)` 같은 연산을 실행하면, 실제 연산이 끝나기 전에 Python 제어권을 돌려주고 `jax.Array`를 반환한다고 설명합니다. 이 배열은 일종의 **future**로, 값이 “나중에 준비될 예정”인 객체
- 예시
    
    ```python
    import jax
    import jax.numpy as jnp
    y = jnp.dot(x, x)
    z = y + 3
    ```
    
    ```
    Python             GPU
    
    dot 실행 요청  ───────→  dot 계산 시작
       │
       ├─ 바로 돌아옴
       │
    +3 실행 요청   ───────→  dot 끝나면 +3 수행
       │
       ├─ 바로 돌아옴
       │
    다음 Python 코드
    ```
    
    - 즉 Python은 GPU가 계산하는 동안 놀지 않고 계속 다음 작업을 **queue에 넣음**
    - JAX에서는 shape이나 dtype 같은 metadata는 계산 완료를 기다리지 않고 볼 수 있고, 해당 array를 다음 JAX 연산에 그대로 넘길 수도 있다고 설명
    
    ```python
    start = time.time()
    
    y = jnp.dot(x, x)
    y.block_until_ready() # 해당 코드로 연산이 끝날때까지 걸린 시간 측정 가능
    
    end = time.time()
    ```
    
    - `jax.jit`는 **계산 자체를 컴파일/최적화**하는 기능
- JAX의 asynchronous dispatch는 GPU 연산이 끝날 때까지 Python이 기다리지 않고 연산 요청만 계속 device queue에 넣는 구조이며, 실제 값이 host에 필요할 때만 동기화합니다.

## Trition: Vector addition

- Triton도 커널 실행 자체는 비동기적
- 단 Jax: asynchronous dispatch와는 “비동기성이 나타나는 층위”가 조금 다름
- 예시
    
    ```python
    import torch
    
    import triton
    import triton.language as tl
    
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    
    @triton.jit
    def add_kernel(x_ptr,  # *Pointer* to first input vector.
                   y_ptr,  # *Pointer* to second input vector.
                   output_ptr,  # *Pointer* to output vector.
                   n_elements,  # Size of the vector.
                   BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                   # NOTE: `constexpr` so it can be used as a shape value.
                   ):
    ```
    
    - add_kernel 함수 하나가 사실상 GPU 커널 launch
    
    ```
    Python
      ↓
    @triton.jit 함수 정의
      ↓
    add_kernelgrid
      ↓
    Triton compiler가 kernel compile
      ↓
    GPU kernel launch
      ↓
    GPU에서 여러 program instance 실행
      ↓
    output tensor에 결과 저장
    ```
    
    - `@triton.jit`는 무엇을 하는가
        - 일반 Python 함수처럼 CPU에서 한 줄씩 실행되는 함수가 아님
        - Triton은 이 함수를 GPU에서 실행할 **custom GPU kernel**로 컴파일합니다. 공식 튜토리얼도 `triton.jit`을 Triton kernel을 정의하는 데 사용하는 decorator라고 설명합니다.
    - Triton에서는 CUDA thread를 직접 하나씩 다루기보다는 더 큰 **block of elements** 단위로 프로그램
- Vector add 워크플로우
    
    ```
    CPU / Python                 GPU
    
    add_kernel...
          │
          ├──── kernel launch ─────→ Vector Add 실행
          │                            █████████████
          │
          ↓
    return output
          ↓
    다음 Python 코드
    ```
    
- Jax와 비교
    - JAX 공식 문서는 `jax.Array`를 **future**, 즉 accelerator에서 향후 만들어질 값으로 설명합니다.
    - Triton 쪽 역시 `torch.Tensor`를 반환하지만 실제 Triton kernel이 GPU에서 아직 실행 중일 수 있습니다.
    - **JAX는 계산 그래프 수준**, **Triton은 GPU kernel 수준**에서 작업하는 도구
    
    |  | JAX async dispatch | Triton kernel launch |
    | --- | --- | --- |
    | 사용자가 작성하는 것 | tensor computation | GPU kernel |
    | 예시 | `jnp.dot(x, x)` | `add_kernelgrid` |
    | 비동기 대상 | JAX computation | 특정 GPU kernel launch |
    | compiler | JAX + XLA | Triton compiler |
    | 반환 | `jax.Array` | 보통 `torch.Tensor` |
    | GPU synchronization | `block_until_ready()` | `torch.cuda.synchronize()` 등 |
    | abstraction | ML framework 수준 | GPU kernel 수준 |
