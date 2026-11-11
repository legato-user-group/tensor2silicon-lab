<WeekNavBar :week="2" topic="모델에서 실리콘까지" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송무복)', 'badge': 'PyTorch·XLA·MAX 2부작'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': '다이어그램 & 스택 해부'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': '실행 스택 핵심 요약'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '인터랙티브 HTML & 슬라이드'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': '컴파일 파이프라인 정리'}]" />

> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **jaehyeong (이재형)** (`컴파일 파이프라인 정리`)
>
> 💡 **핵심 기여 요약**: 프레임워크별 그래프 캡처 방식, IR 계층의 존재 이유, 커널 퓨전(Fusion)이 메모리 왕복을 줄이는 원리를 체계적으로 정리했습니다.

---

# 2주차: 모델에서 실리콘까지

### 2주차 안내

**[2주차 안내] 모델에서 실리콘까지**

안녕하세요! 2주차에는 **AI 모델이 실제 가속기에서 실행되기까지 어떤 과정을 거치는지** 각자 정리하고 함께 이야기해보려고 합니다.

주제가 넓은 만큼, 이번에는 모든 내용을 깊게 파기보다 아래 질문을 중심으로 전체 흐름을 잡아보면 좋겠습니다.

**“모델의 연산 하나가 가속기에서 실행되기까지, 어떤 단계를 거치고 각 단계는 무엇을 담당할까?”**

**1. 공통으로 읽어오면 좋은 자료**

- [Scaling Book 9장: How to Profile TPU Code](https://jax-ml.github.io/scaling-book/profiling)

첫 절인 **A Thousand-Foot View of the TPU Software Stack**만 읽어주세요. 제목은 프로파일링이지만 앞부분은 JAX 코드가 TPU 실행 코드로 변환되는 흐름을 설명합니다.

- [PyTorch: torch.compiler 개요](https://docs.pytorch.org/docs/stable/torch.compiler.html)

도입부와 **Dynamo / AOT Autograd / Inductor** 설명을 중심으로, 익숙한 PyTorch 코드가 어떻게 컴파일되는지 살펴봐주세요.

**2. 해당 사항들을 다같이 정리해주세요**

**① 모델 → 연산**

Transformer를 펼치면 어떤 연산과 텐서가 나올까?

[Scaling Book 4장](https://jax-ml.github.io/scaling-book/transformers)의 Counting Dots, Transformer Accounting 중심

**② 연산 → 그래프·IR**

Python 코드가 어떻게 컴파일러가 다룰 표현으로 바뀔까?

공통 자료인 PyTorch 개요와 Scaling Book 9장 첫 절 중심

**③ IR → 실행 코드**

컴파일러는 무엇을 결정하고 하드웨어별로 무엇이 달라질까?

[OpenXLA: XLA architecture](https://openxla.org/xla/architecture)의 Objectives, How it works 중심

**④ 실행 요청 → 커널 실행**

Python 함수가 반환되면 가속기 계산도 끝난 걸까? 커널은 어떻게 실행할까?

[JAX: Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)와 [Triton: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)의 커널·호출 코드 중심

**⑤ 커널 → 칩 내부**

연산은 어디서 이루어지고 데이터는 어디에서 이동해올까?

[Scaling Book 2장](https://jax-ml.github.io/scaling-book/tpus)의 What Is a TPU? 또는 [12장](https://jax-ml.github.io/scaling-book/gpus)의 What Is a GPU?와 Memory 중심

**3. 정리할 때 담아오면 좋은 내용**

- 내가 맡은 단계의 역할과 입력·출력
- 핵심 용어 3개
- 앞뒤 단계와 연결되는 그림 1개
- 이해가 안 됐거나 함께 이야기하고 싶은 질문

가능하면 **Y = ReLU(XW + b)**라는 같은 예시를 각자 선택한 관점에서 설명해보면 좋겠습니다.

추가로 공부하면서 이것도 같이 보면 좋겠다는 것도 같이 찾아서 공유해주세요!

그럼 다음주 화요일에 뵙겠습니다!

- **1.① 모델 → 연산**
    - 변환 주체: -
    - 변환 대상: 상위 라이브러리로 짜여진 ML/DL 모델(Computation Graph). 이를 펼쳐보면 수많은 MatMul, Addition, Softmax 등의 연산들로 분해된다.
    - 변환 결과: -
    
    Transformer를 펼치면 어떤 연산과 텐서가 나올까?
    [Scaling Book 4장](https://jax-ml.github.io/scaling-book/transformers)의 Counting Dots, Transformer Accounting 중심
    
    → Transformer는 엄청난 양의 MatMul로 이루어져있다.
    
    ---
    
    - Kernel은 컴파일된 AI모델에서 특정 HW가 실제로 실행하는 HW-specific 실행 단위이다.
        - 예를 들어, y = relu(wx+b)라는 모델이 있다. 이를 NPU A에 맞게 컴파일했다고 해보자.
        Kernel A : Matmul + Add
        Kernel B : ReLU
        이렇게 나올 수 있다. 핵심은 반드시 저렇게 나뉘는게 아니라, 컴파일러가 Fusion을 하면 Kernel A : Matmul + Add + ReLU 이렇게 하나가 될 수 도 있다.
        - 핵심은 Op(ReLU등 모델 구성 함수)는 모델 관점의 연산 단위이고, Kernel은 HW에서 실제 실행되는 단위이다. 그리고 Kernel의 범위는 compiler + HW 특성에 따라 결정된다.
    - **Ch4. Counting Dots**(FLOPs 계산방법)
        - vector끼리의 multiply FLOPs 계산법
        - Matrix 끼리 multiply FLOPs 계산법(BATCHING)
            - Transformer의 절대 다수를 구성하고있는 `MatMul layer` 은 
            Forward 에서 2MLKN FLOPs가 소요되고, 
            Backward에서  weight를 위한 grad 계산에 2MLKN FLOPs가, 뒷 레이어에 넘길 activation에 대한 gradient 계산을 위해 다시 2MLKN FLOPs 가 소요된다.
    - **Ch4. Transformer Accounting**
        - Transformer decoder architecture
            
            > 🖼️ *[도식 참고: transformer-diagram.png]*
            
            - input sequence의 길이가 hidden dimesion크기의 8배를 넘기는 순간부터, attention연산에 소요되는 FLOPs가 모델 전체의 MatMul을 넘어서게 된다.
            - 그 이유는 self-attention layer 중간에 QK^T연산 때문이다. 그 결과는 내가 kaggle 하면서 GPU가 다운됐었던 [seq_len * seq_len] 크기의 matrix 때문이다.
            
- **1.② 연산 → 그래프·IR**
    - 변환 주체: `jax.jit()`, `torch.compile()`
    - 변환 대상: `torch.nn.functional.layer_norm()` 같은 연산 레이어
    - 변환 결과: 행렬 곱, Mean, Variance, Element-wise Sub/Mul/Add 연산 등 원시 IR 단위
    
    Python 코드가 어떻게 컴파일러가 다룰 표현으로 바뀔까?
    공통 자료인 PyTorch 개요와 Scaling Book 9장 첫 절 중심
    
    - jax란 google에서 만든 XLA(선형대수 전문 컴파일러) 기반의 머신러닝 라이브러리이다.
        - 기존 pytorch/Tensorflow 는 GPU 연산 하나 수행 시 결과를 메모리에 저장하고 다음 연산에서 다시 읽어온다. 다시말해 메모리 R/W 연산이 빈번히 일어나 이로 인한 병목이 생기는 경우가 많다.
            - **장점**: 파이썬 인터프리터 위에서 돌기 때문에 직관적이고 편리하다.
            - **단점**: python 인터프리터의 오버헤드가 존재한다. 연산 최적화가 자동으로 일어나지 않아 jax 대비 비효율적이다.
            - `Note`: pytorch 2.0부터는 torch.compile 기능이 생겨, 학습 코드를 컴파일해서 효율적으로 학습하는 기능이 추가되었다.
        - jax는 컴파일러인 XLA를 기반으로 학습코드를 계산 최적화된 실행파일로 만들어 인터프리터의 병목을 개선한 머신러닝 개발 언어이다.
            - **장점**: HW 효율적이기에 실행속도가 매우 빠르다.
            - **단점**: 컴파일 시간이 걸린다. 컴파일 타임에 결정되어야하는 요소들이 있다(배열 크기, dtype 등)
    
    ---
    
    #### 1. JAX
    
    **`핵심 질문`**: JAX library 로 작성된 python 코드가 어떻게 TPU에서 실행되는 기계어까지 변환될까?
    
    1. python Level의 JAX코드
        
        ```jsx
        import jax
        import jax.numpy as jnp
        
        def multiply(x, y):
          return jnp.einsum('bf,fd->db', x, y)
        
        y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
        ```
        
        `jax.jit()` 는, 대상 함수를 trace하여 `StableHLO`라는 저수준 IR(Intermediate Representation) 으로 변환한다.
        
        이는 다시 XLA 컴파일러에 의해 `HLO` 라는 저수준 패키지로 변환된다.
        
    2. HLO 변환 결과
        
        ```jsx
        ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
          %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
          %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
          %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
          ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
        }
        ```
        
        이 HLO(High Level Optimizer)은 XLA Compiler에 의해 다시 한번 LLO(Low-Level Optimizer)로 변환된다.
        
        LLO는 메모리간 데이터 복사 스케쥴링, TPU 메모리의 DMA를 스케쥴링하는 수준의 여러 로우레벨 연산들을 포함한다.
        
        LLO로 낮춰진 코드는 최종 기계어로 컴파일되어 TPU IMEM(Code 영역)에 로드되고 실행된다.
        
    
    #### 2. torch.compile()
    
    [torch.compile 해부 1편: TorchDynamo, AOTAutograd, TorchInductor](https://hyper-accel.github.io/posts/torch-compile-anatomy/)
    
    `핵심 질문`: torch.compile() 과 jax를 비교하면, 누가 어떤 상황에서 더 왜 유리할까?
    
    - torch.compile 의 3대 요소
        
        > 🖼️ *[도식 참고: part1-pipeline.png]*
        
        1. TorchDynamo: 그래프를 추출
        2. AOTAutograd: 그래프 다듬기
        3. TorchInductor: 커널 만들기
        
        (추후 더 자세히 알아보기, 웬만하면 실습으로)
        
    
- **1.③ IR → 실행 코드**
    
    **컴파일러는 무엇을 결정하고 하드웨어별로 무엇이 달라질까?**
    [OpenXLA: XLA architecture](https://openxla.org/xla/architecture)의 Objectives, How it works 중심
    
    ---
    
    컴파일러는
    
    1. IR에 표현된 여러 연산을 fusion하여 더 효율적인 파이프라인으로 변환해준다.
        
        예를 들어, XLA 컴파일러는 여러 Op를 하나의 kernel로 묶어서 중간 tensor을 HBM에 R/W하는 비용을 줄인다. 
        또, buffer analysis로 불필요한 중간 buffer도 줄여준다.
        
    2. IR를 target HW에 가장 적합한 기계어로 변환해준다.
        
        결국 딥러닝 HW(GPU, NPU, , TPU, CPU 까지)에서 연산이 수행되려면 해당 HW를 동작시키는 명령이 내려져야한다.
        
        따라서 컴파일러는 해당 IR을 target HW에서 가장 최적으로 동작시키기 위한 명령어로 변환해준다.
        
    
    결론 → 컴파일러는 IR을 target HW에서 가장 효율적으로 실행할 수 있는 실행코드(바이너리) 묶음으로 바꿔준다.
    
- Q. GPU/NPU는 내부가 어떻게 구성되어있고, 데이터와 연산 명령이 들어오면 어떤 일이 일어날까?
    
    [생성형 AI 개발을 위한 NVIDIA GPU 아키텍처의 이해 | 인사이트리포트 | 삼성SDS](https://www.samsungsds.com/kr/insights/nvidia-gpu-architecture.html)
    
    > 🖼️ *[도식 참고: 144개의 Streaming Multiprocessor을 가진 Nvidia H100 GPU]*
    
    144개의 Streaming Multiprocessor을 가진 Nvidia H100 GPU
    
    nvidia GPU인 H100은 144개의 Streaming Multiprocessor로 구성되어있다고 한다. 듣던대로 무언가 비슷한 연산장치들이 빼곡한 것 같다.
    
    > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
    
    저 하나를 구성하는 Streming Multiprocessor은 무엇일까? Streaming Multiprocessor은 내부에 고유 메모리와, 캐시, 컴퓨팅 코어를 가지고있는 연산 모듈이다.
    
    세부 구성은 아래 7가지가 있다고 한다.
    
    1. Tensor Core
        
        행렬의 곱셈과 덧셈 연산을 한번에 수행하는 MatMul, Conv 전용 코어이다.
        
        mixed precision 연산을 지원하여 정확도를 유지하며 계산량을 높일 수 있다.
        
        > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
        
        예를 들어, A와 B는 FP16으로 곱셈을 진행한 결과를 FP32로 더하기 연산을 수행한다. 곱셈 연산에는 FP16을 써서 빠르고, 더하기에는 FP32를 수행하여 정확도를 유지하는 mixed-precision 전략이다.
        
    2. L1 Data Cache와 shared memory의 통합
        
        L1 data cache와 shared memory를 하나의 메모리 블록으로 톡합하여 두 메모리 모두 성능이 향상되었다.
        
        왜냐하면 L1 캐시가 더 많이 필요할 경우 공유 메모리를 유연하게 희생하고, 반대로도 메모리 크기를 조정함으로써 전체적인 캐시 크기를 더 큰것처럼 사용할 수 있게 하여 연산 성능이 올라갔다고 한다.
        
    3. Thread BLock Clusters
        
        > 🖼️ *[도식 참고: nvidia-gpu-architecture-img04.jpg]*
        
        고성능 병렬 프로그래밍을 수행하기 위해 중요한 두 가지는 데이터 지역성과 비동기적 수행이다.
        
        - data를 수행하려는 unit 근처에 놓아서 지연시간과 더 큰 대역폭을 달성하는 방법이다.
        - 비동기적 수행으로 메모리와 데이터 전송과 계산 시간이 겹치도록 하여 GPU 사용률을 최대한 높일 수 있다.
        
        해당 block은 이를 달성하기 위해 고안되었다.
        
    4. Distributed Shared Memory
    5. Asyncronous Execution
    6. Tensor Memory Accelerator
    7. Asyncronous Transaction Barrier
    
- **2.④ 실행 요청 → 커널 실행**
    
    Python 함수가 반환되면 가속기 계산도 끝난 걸까? 커널은 어떻게 실행할까?
    [JAX: Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)와 [Triton: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)의 커널·호출 코드 중심
    
    ---
    
    1. jax의 asyncronous dispath: 
        
        jax는 python overheads를 감추기 위해 계산이 끝나지 않은 메모리 공간을 반환받는 트릭을 사용한다. 해당 공간에 대한 shape를 찍어보거나 심지어 다른 jax computation으로 넘길 수 도 있다.
        
        실제 연산은 61ms걸리지만, 연산 호출에는 300us밖에 걸리지 않는다.
        
        이 방식의 장점은 무엇일까? 어디서 유용하게 사용될 수 있을까?
        
    2. [Triton: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#compute-kernel)(커널은 어떻게 실행될까?)
        - Triton은 병렬 프로그래밍을 위한 언어이자 컴파일러이다.
        - python 환경에서 커스텀 DNN compute kernel을 만들라고 만든 컴파일러/언어이다.
        - GPU에서 실행되는 kernel을 직접 디자인할 수 있다.
        
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
            # There are multiple 'programs' processing different data. We identify which program
            # we are here:
            pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
            # This program will process inputs that are offset from the initial data.
            # For instance, if you had a vector of length 256 and block_size of 64, the programs
            # would each access the elements [0:64, 64:128, 128:192, 192:256].
            # Note that offsets is a list of pointers:
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # Create a mask to guard memory operations against out-of-bounds accesses.
            mask = offsets < n_elements
            # Load x and y from DRAM, masking out any extra elements in case the input is not a
            # multiple of the block size.
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            # Write x + y back to DRAM.
            tl.store(output_ptr + offsets, output, mask=mask)
        ```
        
        GPU는 큰 데이터를 한번에 처리하기 위해, 전체 작업을 작은 단위(블록/Program)으로 나누어 병렬로 실행한다. 이 때 각 작업 블록에 0,1,2,3 같은 논리 번호가 붙는데 이것이 지금 코드의 `pid`이다.
        
        만일 전체 데이터 개수가 256개고, 한번에 64개씩 처리하기로 BLOCK_SIZE = 64로 했다고 하자. 그러면 Triton은 전체 작업 처리를 위해 총 4개의 블록(Program)을 동시에 생성한다.
        
        - 0번 실행 블록: pid = 0/ 0~63 offset 처리
        - 1번 실행 블록: pid = 1/ 63~127 offset 처리
        - 2번 실행 블록: pid = 2/ 128~xx offset처리
        - 3번 실행 블록: pid = 3/ xx~255 offset 처리
        
        `tl.load()`는 DRAM(GPU 전역 메모리)에 있는 데이터를 읽어와 레지스터(SRAM)로 가져오는 함수이다. 메모리 포인터를 참조하여 실제 연산에 사용할 텐서를 로드할 때 사용한다.
        
        `tl.store()`은 GPU 레지스터(SRAM)에서 연산 완료된 결과를 DRAM으로 다시 기록하는 함수이다. GPU local memory에 저장된 output을 output_ptr + offsets 주소에 저장한다는 의미이다.
        
        ```python
        def add(x: torch.Tensor, y: torch.Tensor):
            # We need to preallocate the output.
            output = torch.empty_like(x)
            assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
            n_elements = output.numel()
            # The SPMD launch grid denotes the number of kernel instances that run in parallel.
            # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
            # In this case, we use a 1D grid where the size is the number of blocks:
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
            # NOTE:
            #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
            #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
            #  - Don't forget to pass meta-parameters as keywords arguments.
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
            # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
            # running asynchronously at this point.
            return output
        
        ```
        
        이렇게 torch의 wrapper function으로 사용 가능하다.
        
        ```python
        torch.manual_seed(0)
        size = 98432
        x = torch.rand(size, device=DEVICE)
        y = torch.rand(size, device=DEVICE)
        output_torch = x + y
        output_triton = add(x, y)
        print(output_torch)
        print(output_triton)
        print(f'The maximum difference between torch and triton is '
              f'{torch.max(torch.abs(output_torch - output_triton))}')
        
        ```
        
        ```
        tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
        tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
        The maximum difference between torch and triton is 0.0
        ```
        
        torch와 triton kernel의 연산 결과는 같다.
        
- **2.⑤ 커널 → 칩 내부**
    
    연산은 어디서 이루어지고 데이터는 어디에서 이동해올까?
    [Scaling Book 2장](https://jax-ml.github.io/scaling-book/tpus)의 What Is a TPU? 또는 [12장](https://jax-ml.github.io/scaling-book/gpus)의 What Is a GPU?와 Memory 중심
    
    ---
    
    ### Part 12. How to Think about GPUs
    
    - how each chip works, how they’re networked together, what that means for LLMs compared to TPUs.
    - Focused on NVIDIA GPUs
    
    H100같은 현대 GPU는 사실상 HBM 메모리랑 연결된 matmul 특화 compute core(SM - **Streaming Multiprocessors**) 묶음 덩어리이다.
    
    #### GPU HW Diagram
    
    > 🖼️ *[도식 참고: gpu-diagram.png]*
    
    GPU는 연산장치 SM들이 L2 Cache와 HBM에 연결된 형태로 설계되어있다.
    
    - SM 구성 요소:
        - `Tensor Core`: MatMul 전용 연산장치(TPU의 MXU와 동일 개념)
        - `vector arithmetic unit(Warp Scheduler)`: CUDA Cores라고 불린다. SIMD(Single Instruction Multiple Data)로 martix의 모든 lane(vector)은 하나의 cycle에 같은 Op를 수행한다.
        - `fast cache mem(SMEM)`: 빠른 L1 Cache(256KB in H100)
        
        동일 사양 FLOPs TPU와 GPU를 비교하면, 2개의 Tensor Core을 가진 TPU에 비해, GPU는 100+개의 Tensor Core을 가진다. 따라서 GPU의 연산 할당이 더 유연하다.
        
    - HBM의 용도
        - model parameters, activations, optimizer state, etc 를 저장하는 모델이 올라가는 공간으로 쓰인다.
        - 한마디로, 큼직한 메모리들이 올라간다.
    
    #### SM Details
    
    > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
    
    1. **SM의 구조:**
        
        위에서 보던 숫자 그대로, 하나의 SM은 4개의 동일한 SM Subpartition으로 나뉜다. 각 subpartition에는: Tensor Core, Register, SIMD/SIMT 벡터 연산 장치, CUDA Core(ALU)가 있다. 이 중  FLOPs/s 계산의 대부분은 Tensor Core가 담당한다.
        
        - **CUDA Cores**:
            
            각 Subpartition이 가지는 ALU 묶음이다. SIMD/SIMT는 명령어 하나가 들어오면, Vector lane에 대해서 같은 연산을 동일하게 파바박 수행한다는 의미이다.
            
            각 ALU는 일반적으로 cycle마다 1개의 산술 명령을 처리할 수 있다(e.g. f32.add). 그리고 각 subpartition에는 cycle마다 같은 Op가 수행되는 “fp32 cores 32개, int32 cores 32보다 조금, fp64 cores 32보다 조금”을 가지고있다.
            
            TPU의 VPU처럼, CUDA Cores는 ReLU나 pointwise vector Op, reductions(SUMs)를 처리한다.
            
        - **Tensor Cores(TC)**:
            
            각 subpartition은 고유의 TC를 가진다. 
            
            - TPU처럼 GPU도 lower precision matmuls at higher throughput이 가능하다.
            - GPU는 Volta세대 이후 TC size가 계속 커져서, 현재 B200의 TC 수행 능력은 SMEM 수용 용량보다 커지게 되었다. 따라서 B200 은 TMBM이라는 새로운 mem을 도입하였다.
    2. **CUDA Core: SIMT의 장점**
        
        V100부터 CUDA Core은 **SIMT**(*Single Instruction Multiple Threads)* 모델을 사용한다. 같은 warp의 thread들은 기본적으로 같은 명령을 실행하지만, 각 thread는 자기 instruction pointer/state를 가질 수 있어서 서로 다른 코드 경로를 탈 수도 있다.
        
        - Q. thread란?
            
            CUDA에서 thread는 GPU가 실행하는 가장 작은 논리적 작업 단위이다. thread는 자기만의 theradIdx, register state, 연산할 데이터를 갖는다. 그리고 NVIDIA GPU는 보통, 32 threads = 1 warp 로 묶어서 같이 실행한다.
            
            예시: 벡터 덧셈
            
            ```
            A = [1,2,3,4,...]
            B = [10,20,30,40,...]
            C = A + B
            ```
            
            CUDA 에서는 보통 원소 하나당 thread 하나를 맡긴다.
            
            ```
            Thread 0: C[0] = A[0] + B[0]
            Thread 1: C[1] = A[1] + B[1]
            Thread 2: C[2] = A[2] + B[2]
            ...
            ```
            
            그래서 32개 원소를 처리하면 32 threads = 1 warp.
            
            이 32 thread가 같은 ADD 명령을 각자 자기 데이터에 적용한다.
            
            비유하면: thread는 “배열 원소 하나 맡은 작업자 1명”이다.
            
    3. **CUDA Core의 스케쥴링은 매우 유연하다**
        
        SM은 multi-thrad CPU처럼 여러 warp를 동시에 대기시킬 수 있다(SM 하나당 최대 64 warps를 대기시킬 수 있음). 각 warp scheduler은 한 clock에 warp 하나의 명령만 내보내지만, 어떤 warp가 메모리 로드를 기다리면 다른 warp로 바로 갈아탄다.
        
        즉 GPU가 빠른 이유는 한 작업이 기다릴 때 다른 작업을 돌려서 더 빨라질 수 있는것이다.
        
        결론: CPU 파이프라이닝 처럼, warpA를 처리중이면 warpB는 TC 대기, warpC는 memory load 대기 이런…
        
    
    #### 3. Memory
    
    GPU의 메모리 구조는 다음과 같다. HBM이 largest main memory로 동작하고, 다음으로 연속적인 작은 캐시들(L2, L1/SMEM, TMEM, register memory)이 온다.
    
    - **Registers - Subpartition 마다**
        
        각 subpartition은 고유의 register file을 갖는다. 이 register은 H100/B200 에서 SM마다 16,384 32-bit words(256KiB)의 크기이며, CUDA Core에서 접근 가능하다.
        
        각  CUDA Core은 한번에 오직 256 registers에 접근 가능하다.
        
    - SMEM(L1 cache) - SM 마다
        
        SM마다 256KiB의 SMEM이 있다. 사용자 설정에 따라 SM들의 shared memory로 사용될 수 있고, on-chip cache로 쓰일 수 도 있다.  SMEM은 TC matmul을 위한 activations/inputs 저장처로 사용된다.
        
    - **L2 cache - GPU에 하나 공유**
        
        모든 SM들이 공유하는 50MB 수준의 캐시메모리 공간이다. 
        
        full duplex
        
        HBM의 1.6x 속도
        
    - **HBM - GPU에 하나 공유**
        
        model weights, gradients, activations, etc 를 저장.
        
        HBM → CUDA Tensor Core 전송 bandwidth는 HBM Bandwidth, memory bandwidth라고 불리고, H100에서 3.35TB/s이고, B200에서 9TB/s 이다.