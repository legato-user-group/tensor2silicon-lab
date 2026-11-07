<WeekNavBar :week="3" topic="커널의 시간은 어디에서 소비되는가" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송민배)', 'badge': 'GPU·TPU 메모리 계층 비교'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': '손계산 & 병목 판별'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': 'SRAM vs DRAM 물리 구조'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '수식 모델링 & 정량 분석'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': '메모리 계층 & 실전 예제'}]" />

> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **jaehyeong (이재형)** (`메모리 계층 & 실전 예제`)
>
> 💡 **핵심 기여 요약**: GPU 메모리 공간(Global, L2, Shared, Register)의 특성과 Activation 레이어가 Memory-Bound에 걸리는 실제 사례를 분석했습니다.

---

# 3주차: 커널의 시간은 어디에서 소비되는가

### 3주차 안내

**[3주차 안내] 커널의 시간은 어디에서 소비되는가**

안녕하세요! 3주차에는 **커널이 계산하는 시간과 데이터를 옮기는 시간을 나누어 보고, 간단한 연산의 실행시간을 직접 추정**해보려고 합니다.

지난주에 모델에서 실리콘까지의 흐름을 살펴봤다면, 이번에는 그중 커널 하나를 골라 시간이 어디에서 쓰이는지 들여다보겠습니다.

**“같은 연산인데 왜 어떤 커널은 빠르고 어떤 커널은 느린가?”**

이번 주에는 아래 질문과 손계산을 중심으로 기본 감각을 잡아보면 좋겠습니다. Roofline 그래프와 Tiling은 이 계산을 바탕으로 4주차에 이어갑니다.

**1. 공통으로 읽어오면 좋은 자료**

- [Scaling Book 1장: All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/#where-does-the-time-go)
    
    첫 절인 **Where Does the Time Go?**에서 연산 시간·데이터 이동 시간 공식과 **Compute-bound / Communication-bound** 설명까지 읽어주세요. 이번 주에는 한 가속기 안에서의 메모리 이동에 집중합니다. Arithmetic Intensity와 Roofline 그래프는 다음 주에 다룹니다.
    
    - 연산 시간/데이터 이동 시간 공식
        
        #### Computation
        
        T_math: 특정 HW에서 연산에 걸리는 시간
        
        > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
        
        연산 FLOPs를 Accelerator FLOPs/s로 나누어준 값.
        
        #### Communication within a chip
        
        accelerator 내부에서 data가 이동하는데 걸리는 시간(HBM ↔ Compute Cores)
        
        #### Communication between chips
        
        다양한 accelerators에 모델을 분산하면, tensor은 hw들 사이에 자주 전송 되게 된다.
        
        어떤 communication 시간이든 bytes/s로 측정해서 total communication time 을 이렇게 계산한다.
        
        > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
        
        일반적으로, computation within a single chip은 communication time으로 overlapped 된다. NPU/GPU 등은 실행과 데이터 전송이 병렬적으로 가능하기 때문에 둘 중 더 긴 시간이 소요되는 time으로 전체 실행 시간이 overlapped 되는 것이다.
        
    - 용어 정리
        - Communication-bound
            
            데이터를 가져오는 시간이 연산 시간보다 훨씬 길어서, 연산 장치는 계산을 마쳐도 다음 데이터를 기다리며 놀게 됩니다(Stall). 결과적으로 전체 실행 시간은 통신 시간으로 수렴(퉁쳐짐)합니다.
            
        - Compute-bound
            
            반대로 연산에 걸리는 시간이 훨씬 길어, 연산 장치가 열일하는 동안 데이터 전송은 이미 끝나 배경에서 대기 중인 상태입니다. 이때 전체 실행 시간은 연산 시간으로 결정됩니다.
            
        - lower-bound(하한선, 최소한 걸리는 시간)
            
            > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
            
            **의미**: 연산과 통신이 **100% 완벽하게 동시에(Ideal Overlap)** 이루어진다고 가정한 최선의 시나리오
            
            아무리 하드웨어 성능이 뛰어나고 최적화가 잘 되어도, **둘 중 더 오래 걸리는 작업 시간보다 빠르게 끝날 수는 없기 때문에** 이론적 최저 시간(하한선)이 됩니다.
            
        - **상한선 (Upper-bound, 최악의 경우 걸리는 시간)**
            
            > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
            
            **의미**: 연산과 통신이 **전혀 겹치지 않고 순차적으로(Sequential)** 진행되는 최악의 시나리오
            
            데이터를 다 가져온 뒤에야 계산을 시작하고, 계산이 다 끝난 뒤에야 다음 데이터를 가져오는 방식. 어떤 경우에도 **두 시간을 합친 것보다는 오래 걸릴 수 없으므로** 최대 한계 시간(상한선)이 됩니다.
            
        
        - Roofline Model: 컴퓨팅 환경에서 프로그램 성능 한계와 병목 현상을 분석하고 시각화하는 성능 모델
- [MLC: GPU Acceleration — GPU Architecture](https://book.mlc.ai/chapter_gpu_acceleration/part1.html#gpu-architecture)
    
    **GPU 구조 그림과 첫 Vector Add 코드**를 살펴봐주세요. `C[i] = A[i] + B[i]`에서 무엇을 읽고, 무엇을 계산하고, 무엇을 쓰는지 찾으면 됩니다. 뒤의 루프 분할과 스케줄 변환까지 이해하거나 코드를 실행할 필요는 없습니다.
    
    - 이해(Code)
        
        ```python
        @tvm.script.ir_module
        class MyModuleVecAdd:
            @T.prim_func
            def main(A: T.Buffer((1024,), "float32"),
                     B: T.Buffer((1024,), "float32"),
                     C: T.Buffer((1024,), "float32")) -> None:
                T.func_attr({"global_symbol": "main", "tirx.noalias": True})
                for i in T.grid(1024):
                    with T.sblock("C"):
                        vi = T.axis.remap("S", [i])
                        C[vi] = A[vi] + B[vi]
        ```
        
        GPU에서 A+B vector add 연산이 어떻게  thread에 나뉘어 실행되는가?
        
        Thread 0: A[0], B[0] read → 더하고 → C[0]에 write
        
        Thread i: 반복
        
        핵심은, 읽기/ 계산/ 쓰기 작업을 thread들을 한번에 병렬적으로 순차 실행한다는 점.
        
- [Scaling Book 12장: How to Think About GPUs — Memory](https://jax-ml.github.io/scaling-book/gpus/#memory)
    
    **HBM, L2 Cache, L1/Shared Memory, Register**의 위치와 역할을 중심으로 읽어주세요. 하드웨어별 용량 수치를 외우기보다 데이터가 연산 장치에 도착하기까지 어떤 저장 공간을 거치는지 따라가보면 좋겠습니다.
    
    - 이해
        
        #### Top to Bottom memory
        
        `HBM`(large VRAM)에 model weights, gradients, activations, etc.
        
        `L2 Cache`(~50MB)은 HBM 병목 완화를 위해 모든 SM 들이 공유하는 공간이다.
        
        `L1 Cache`(SMEM)은 각 SM에 있는 256KiB 메모리이다.
        
        `Registers`(16,384 32-bit word registers per Subpartition, SM total 4*16384*4 = 256KiB)은 SM내 4개의 subpartition이 보유하는 CUDA Core에서 접근 가능한 공간이다.
        
        - 하드웨어 구조상 SM 1개당 최대 64개의 상주 워프(resident warp)를 스케쥴링 대기 상태로 둘 수 있다.
        - 하지만 각 thread가 접근 가능한 최대 한도인 256개의 registers를 사용할 경우, 1 warp(32thread)는 32*256 = 8,192개의 레지스터를 소비한다.
        - 따라서 1 SM에서 전체 레지스터 개수 65,536개를 생각해보면, 실제로는 최대 8 warp만이 상주 가능하다.
        
        결론은, 스케줄러는 이론상 SM당 64 warps를 상주시킬 수 있지만 thread가 레지스터를 많이 사용할수록 상주 가능한 warps가 떨어질 수 있다는 점을 고려해야한다.
        
        한번에 상주되는 warps가 떨어진다는 것은, 계산 병목으로 이어질 여지가 있어보인다.
        

**2. 해당 사항들을 다같이 정리해주세요**

**① FLOPs와 FLOPs/s**

해야 할 계산의 양과, 하드웨어가 1초에 처리할 수 있는 계산의 양은 어떻게 다를까?

Vector Add의 덧셈 횟수를 세고, `연산량 ÷ 연산 성능`의 단위가 왜 시간이 되는지 설명해봐주세요.

공통 자료인 Scaling Book 1장 중심

**② 메모리 용량과 메모리 대역폭**

메모리가 16GB라는 것과 대역폭이 1TB/s라는 것은 각각 무엇을 뜻할까?

텐서가 메모리에 들어가는지 판단하는 계산과, 텐서를 옮기는 데 걸리는 시간을 구하는 계산을 구분해봐주세요.

[NVIDIA: GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)의 **GPU Architecture Fundamentals / Understanding Performance** 중심

**③ HBM·DRAM·SRAM의 역할**

데이터를 모두 연산 장치 가까이에 둘 수는 없을까? 용량이 큰 메모리와 빠르게 접근할 수 있는 작은 메모리가 왜 함께 필요할까?

[**HBM은 DRAM의 한 종류**](https://www.micron.com/products/memory/hbm)라는 점을 짚고, GPU의 HBM과 SRAM 기반 캐시·Shared Memory를 그림으로 연결해봐주세요.

공통 자료인 Scaling Book 12장과 [MLC: Memory Spaces](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html#memory-spaces)의 첫 설명·표 중심. TPU가 궁금하다면 [Scaling Book 2장: What Is a TPU?](https://jax-ml.github.io/scaling-book/tpus/#what-is-a-tpu)의 HBM·VMEM 설명도 함께 봐주세요.

**④ Compute Time과 Memory Time**

계산량이 적어도 느린 커널이 있을까? 입력만 읽으면 끝일까, 결과를 쓰는 시간도 세어야 할까?

이번 주에는 아래 모델을 공통으로 사용합니다.

```
연산 시간 = 총 FLOPs ÷ 하드웨어 연산 성능(FLOPs/s)
메모리 시간 = 읽고 쓴 총 Byte ÷ 메모리 대역폭(Byte/s)
예상 실행시간 ≈ max(연산 시간, 메모리 시간)
```

이 값은 **연산과 데이터 이동을 충분히 겹치고, 주어진 성능을 활용한다고 가정한 이상적 시간 추정치**입니다. 최대 성능을 대입하면 최소 실행시간의 기준이 되며, 실제 측정값과는 차이가 날 수 있습니다. 이번에는 입력이 이미 가속기 메모리에 있다고 가정합니다.

**⑤ Compute-bound와 Memory-bound**

연산 성능만 두 배가 되면 Vector Add도 두 배 빨라질까? 메모리 대역폭만 두 배가 되면 어떨까?

두 시간 중 더 큰 값을 근거로 병목을 설명해봐주세요. 여기서 Memory-bound는 **메모리 대역폭이 시간을 제한하는 경우**를 뜻합니다.

[NVIDIA: Memory-Limited Layers](https://docs.nvidia.com/deeplearning/performance/dl-performance-memory-limited/index.html)의 **Memory-Limited Layers / Activations**도 참고할 수 있습니다.

**3. 함께 계산해볼 내용: Vector Add**

길이 `N = 100,000,000`인 FP32 벡터 A, B를 더해 새 벡터 C에 저장한다고 가정합니다. FP32 원소 하나는 4Byte입니다.

연습용 가상 하드웨어의 **FP32 덧셈 처리 성능은 20 × 10¹² FLOPs/s**, **HBM 대역폭은 1 × 10¹² Byte/s**로 놓겠습니다. 실제 제품 사양이나 측정값은 아닙니다.

입력은 HBM에서 한 번씩 읽고 결과는 한 번 씁니다. 캐시 재사용과 추가 데이터 이동은 없다고 가정하고, 아래 항목을 각자 계산해와서 비교해주세요.

- A와 B를 읽는 데이터 크기
- C를 쓰는 데이터 크기
- 필요한 덧셈 횟수와 총 FLOPs
- 연산 시간과 메모리 시간
- 더 큰 시간과, 그 값으로 설명할 수 있는 병목

계산이 끝나면 **연산 성능·메모리 대역폭·메모리 용량 중 무엇을 늘렸을 때 이 커널이 빨라질지** 이야기해보겠습니다.

[Triton: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)의 `tl.load`, 덧셈, `tl.store`를 찾아 손계산과 연결해보는 것도 좋습니다.

정리한 뒤에는 **연산 성능과 메모리 대역폭을 구분할 수 있는지**, **간단한 커널의 최소 실행시간을 가정과 함께 추정할 수 있는지**, **어떤 조건에서 Memory-bound인지 설명할 수 있는지** 서로 확인해보겠습니다.

추가로 공부하면서 이것도 같이 보면 좋겠다는 자료가 있으면, 어떤 질문에 도움이 되는지도 함께 공유해주세요!

그럼 다음 모임에서 뵙겠습니다!

- **① FLOPs와 FLOPs/s**
    
    해야 할 계산의 양과, 하드웨어가 1초에 처리할 수 있는 계산의 양은 어떻게 다를까? Vector Add의 덧셈 횟수를 세고, `연산량 ÷ 연산 성능`의 단위가 왜 시간이 되는지 설명해봐주세요.공통 자료인 Scaling Book 1장 중심
    
    ---
    
    ### Vector Add의 덧셈 횟수
    
    - FLOPs 정의
        
        Floating Point Operations Per Seconds.
        
        1초 동안 수행할 수 있는 부동소수점 연산 횟수를 뜻하는 성능 측정 단위.
        
    
    Vector Add `C[i] = A[i] + B[i]`는 vector length가 N이라면, 각 원소마다 덧셈이 1번씩 수행되므로 총 연산량은 **N FLOPs**이다.
    
    ### 연산량/연산성은 단위가 time인 이유
    
    연산량의 단위는 FLOPs, 연산 성능의 단위는 FLOPs/s 이므로,  둘을 나누면 second 단위가 도출된다.
    
    즉, ‘해야 할 총 계산량’을 ‘초당 처리 가능한 계산량’으로 나누었기에 ‘계산에 필요한 시간’이 도출되는 것.
    
- **② 메모리 용량과 메모리 대역폭**
    
    메모리가 16GB라는 것과 대역폭이 1TB/s라는 것은 각각 무엇을 뜻할까?
    
    텐서가 메모리에 들어가는지 판단하는 계산과, 텐서를 옮기는 데 걸리는 시간을 구하는 계산을 구분해봐주세요.
    [NVIDIA: GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)의 **GPU Architecture Fundamentals / Understanding Performance** 중심
    
    ---
    
    ## 메모리 용량
    
    메모리 16GB는 GPU가 한 번에 저장할 수 있는 데이터의 총량(HBM 용량, H100은 80GB)이다. 따라서 모델의 weight, activation, intermediate tensor 등이 이 용량 안에 들어가는지를 판단할 때 사용한다. 예를 들어 어떤 tensor의 크기는 `원소 개수 × dtype 크기(byte)`로 계산할 수 있다.
    
    ## 메모리 대역폭(memory bandwidth)
    
    메모리 대역폭 **1TB/s**는 메모리와 연산 장치 사이에 1초에 최대 약 1TB의 데이터를 이동할 수 있다는 뜻이다. tensor을 메모리에서 연산 장치까지 옮긴다고 하면, 걸리는 시간은 
    
    *T_mem = (이동해야 할 데이터 크기, byte)/(메모리 대역폭 byte/s)*
    
    로 계산된다. NVIDIA도 memory time을 `bytes_accessed / memory_bandwidth` 로 설명.
    
    결론적으로 메모리 용량은 model parameter 등이 GPU에 올라가는가를 판단할 때 보는 수치이고, 메모리 대역폭은 얼마나 빨리 옮기느냐를 보는 GPU의 성능을 볼 때 판단하는 수치이다.
    
- **③ HBM·DRAM·SRAM의 역할**
    - 데이터를 모두 연산 장치 가까이에 둘 수는 없을까? → 비용/설계  문제로 비효율
    - 용량이 큰 메모리와 빠르게 접근할 수 있는 작은 메모리가 왜 함께 필요할까? → 데이터 사용 빈도에 따라 효율적 pipelining을 위해
    
    [**HBM은 DRAM의 한 종류](https://www.micron.com/products/memory/hbm)라는 점을 짚고, GPU의 HBM과 SRAM 기반 캐시·Shared Memory를 그림으로 연결해봐주세요.**
    
    공통 자료인 Scaling Book 12장과 [MLC: Memory Spaces](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html#memory-spaces)의 첫 설명·표 중심. TPU가 궁금하다면 [Scaling Book 2장: What Is a TPU?](https://jax-ml.github.io/scaling-book/tpus/#what-is-a-tpu)의 HBM·VMEM 설명도 함께 봐주세요.
    
    ---
    
    ## HBM/DRAM/SRAM의 역할
    
    HBM은 DRAM의 한 종류로, 차이점은 여러 DRAM을 수직으로 쌓아올려서 데이터 R/W 속도를 극대화한 초고속 메모리이다. 일반 서버용 DRAM인 DDR5은 I/O 버스가 32-64bit임에 반해, HBM은 1024-2048bit수준이다. 초당 데이터 대역폭은 DDR5가 30~60GB/s 일 때, HBM은 1.2~3.3TB/s 정도가 나온다. 결론은 HBM은 무지막지하게 빠른 DRAM 이며, GPU의 대용량 주 메모리로 쓰인다. 이 HBM에는 모델 weight, activation, intermediate tensor처럼 많은 데이터를 저장할 수 있지만, 연산 장치와 비교하면 접근 지연이 크다.
    
    반면 GPU 내부의 cache, register, shared memory는 주로 SRAM 기반이다. SRAM은 DRAM보다 훨씬 빠르게 접근할 수 있지만, 면적과 전력 비용, 생산 비용이 커서 용량을 크게 만들기 어렵다.
    
    ## 왜 둘 다 필요한가
    
    모든 데이터를 연산 장치 바로 옆의 빠른 SRAM에 둘 수 있다면 가장 좋지만, SRAM은 너무 비싸고 면적을 많이 차지한다. 그래서 GPU는 당장 자주 쓰이지 않는 큰 데이터는 HBM에 저장하고, 자주 쓰는 일부 데이터만 SRAM 기반의 cache, shared memory, register로 옮겨 사용한다. CPU처럼 말이다.
    
    ## GPU에서 실제로 어떻게 나눠 쓰는가?
    
    > 🖼️ *[도식 출처: [https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html#memory-spaces](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html#memory-spaces)]*
    
    [https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html#memory-spaces](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html#memory-spaces)
    
    `CTA`: Cooperative Thread Array의 약자: CUDA의 thread block. 같은 CTA안의 thread들은 shared memory를 같이 쓰고 서로 syncronize 가능하다.
    
    Global Memory(GMEM)는 GPU 전체가 공유하는 HBM으로, 모델 weight나 tensor처럼 큰 데이터를 저장한다. Shared Memory(SMEM)는 한 CTA가 사용하는 빠른 on-chip 임시 공간으로, HBM에서 가져온 tensor tile을 잠시 보관하고 **재사용**한다. Register File은 각 thread가 사용하는 가장 가까운 저장공간으로, 계산 중 임시값을 저장한다. Blackwell에서는 Tensor Core의 행렬곱 누적값을 위한 TMEM도 추가되었다.
    
    즉 HBM에 전체 데이터를 두고, 계산에 필요한 일부만 SMEM과 Register 같은 가까운 메모리로 가져와 반복 사용한다.
    
    ## 데이터 흐름
    
    > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
    
    위 그림처럼, GPU도 폰노이만 구조 컴퓨터처럼 “HBM → L2/L1 Cache → Registers” 계층 순서로 메모리를 사용하고있다.
    
    데이터의 흐름을 따라가면, 
    
    > **`HBM` → `L2 Cache` → `Shared Mem/ L1 Cache` → `Register` → `CUDA Core/ Tensor Core`**
    > 
    
    main memory 에서 연산장치로 이동한다. 또한, 연산 장치에 가까워질수록 메모리는 작아지지만 R/W가 빨라진다.
    
    ## 핵심 차이
    
    HBM/DRAM은 많은 데이터를 담을 수 있으나 R/W가 조금 느린 메모리, SRAM 기반 메모리는 연산 장치 바로 옆의 R/W이 매우 빠른 작은 메모리에 가깝다. GPU 성능은 필요한 데이터를 HBM에서 계속 가져오기보다, 가까운 메모리에 올려두고 최대한 재사용할수록 좋아질 것이다.
    
- **④ Compute Time과 Memory Time**
    
    계산량이 적어도 느린 커널이 있을까? 입력만 읽으면 끝일까, 결과를 쓰는 시간도 세어야 할까?이번 주에는 아래 모델을 공통으로 사용합니다.
    
    ```
    연산 시간 = 총 FLOPs ÷ 하드웨어 연산 성능(FLOPs/s)
    메모리 시간 = 읽고 쓴 총 Byte ÷ 메모리 대역폭(Byte/s)
    예상 실행시간 ≈ max(연산 시간, 메모리 시간)
    ```
    
    이 값은 **연산과 데이터 이동을 충분히 겹치고, 주어진 성능을 활용한다고 가정한 이상적 시간 추정치**입니다. 최대 성능을 대입하면 최소 실행시간의 기준이 되며, 실제 측정값과는 차이가 날 수 있습니다. 이번에는 입력이 이미 가속기 메모리에 있다고 가정합니다.
    
    ---
    
    ## Compute Time
    
    Compute Time은 커널이 수행해야 하는 총 연산량을 하드웨어의 연산 성능으로 나눈 값이다.
    
    > 연산 시간 = 총 FLOPs ÷ 하드웨어 연산 성능(FLOPs/s)
    > 
    
    즉 계산해야 할 일이 많거나, 하드웨어의 연산 성능이 낮을수록 오래 걸린다.
    
    ## Memory Time
    
    Memory Time은 커널 실행 중 이동해야 하는 전체 데이터 양을 메모리 대역폭으로 나눈 값이다.
    
    > 메모리 시간 = 읽고 쓴 총 Byte ÷ 메모리 대역폭(Byte/s)
    > 
    
    여기서는 입력을 읽는 시간뿐 아니라 결과를 메모리에 다시 쓰는 시간도 포함해야 한다.
    
    ## 예상 실행시간
    
    연산과 메모리 이동이 충분히 겹쳐 수행된다고 가정하면,
    
    > 예상 실행시간 ≈ max(연산 시간, 메모리 시간)
    > 
    
    으로 볼 수 있다.
    
    즉 계산이 더 오래 걸리면 compute-bound, 데이터 이동이 더 오래 걸리면 memory-bound에 가깝다.
    
    ## 핵심
    
    연산량이 적다고 반드시 빠른 커널은 아니다. Vector Add처럼 계산은 단순하지만 데이터를 많이 읽고 써야 하는 연산은 메모리 이동 시간이 병목이 될 수 있다. 이번주에 사용하는 모델은 이상적인 최소 실행시간을 추정하는 모델이며, 실제 실행시간은 더 길어질 수 있다.
    
- **⑤ Compute-bound와 Memory-bound**
    - 연산 성능만 두 배가 되면 Vector Add도 두 배 빨라질까?
    - 메모리 대역폭만 두 배가 되면 어떨까?
    
    두 시간 중 더 큰 값을 근거로 병목을 설명해봐주세요. 여기서 Memory-bound는 **메모리 대역폭이 시간을 제한하는 경우**를 뜻합니다.
    
    [NVIDIA: Memory-Limited Layers](https://docs.nvidia.com/deeplearning/performance/dl-performance-memory-limited/index.html)의 **Memory-Limited Layers / Activations**도 참고할 수 있습니다.
    
    ---
    
    ## **연산 성능이 두 배가 되어도 Vector Add가 빨라지지 않을 수 있다**
    
    Vector Add `C[i] = A[i] + B[i]`는 원소 하나당 덧셈은 1번뿐이지만, `A[i]`, `B[i]`를 읽고 `C[i]`를 다시 써야 한다. 즉 계산량에 비해 메모리 접근량이 크다. NVIDIA도 activation, normalization, pooling처럼 원소당 계산이 적은 연산은 GPU에서 메모리 전송 시간이 성능을 제한하는 경우가 많다고 설명한다. ReLU, sigmoid, tanh 같은 activation도 입력 크기에 비례해 실행시간이 증가하며, 그 이유는 계산보다 메모리 대역폭의 영향을 더 크게 받기 때문이다.
    
    따라서 Vector Add가 이미 memory-bound라면 GPU의 FLOPs/s가 두 배가 되어도 실행시간은 거의 줄지 않을 수 있다.
    
    > 🖼️ *[도식 참고: activation layer의 forward/backward 시간이  input size(N*H*W*C)에 비례해 증가한다. 다시말해 memory-bound에 걸리는 상황.]*
    
    activation layer의 forward/backward 시간이  input size(N*H*W*C)에 비례해 증가한다. 다시말해 memory-bound에 걸리는 상황.
    
    ## 메모리 대역폭이 두 배가 되어도 Vector Add가 항상 두 배 빨라지는 것은 아니다
    
    반대로 메모리 대역폭을 두 배로 높여도 항상 실행시간이 정확히 절반이 되는 것은 아니다. [NVIDIA 자료](https://docs.nvidia.com/deeplearning/performance/dl-performance-memory-limited/index.html#mem-limited)에서도 입력 tensor가 매우 작으면 메모리 대역폭을 충분히 포화시키지 못해서, 데이터 크기가 줄어들거나 대역폭이 커져도 실행시간이 비례해서 줄지 않는 구간이 나타난다고 한다. 메모리 대역폭이 병목의 전부가 아닌 경우에는 대역폭만 두 배로 올려도 전체 실행시간은 두 배 개선되지 않는다.
    
    ## 커널 실행시간 결론
    
    커널 실행시간은 단순히 FLOPs/s나 메모리 대역폭 하나만으로 결정되지 않는다. 먼저 해당 커널이 계산을 많이 하는지, 아니면 데이터를 많이 읽고 쓰는지를 봐야 한다.
    
    NVIDIA 자료의 activation 예시처럼 계산량이 적고 메모리 접근이 많은 연산은 메모리 대역폭이 주된 병목이 된다. 반대로 계산량이 큰 MatMul 같은 연산은 연산 성능의 영향이 더 커질 수 있다.
    
    > 예상 실행시간 ≈ max(연산 시간, 메모리 시간)
    > 
    
    결국 위 모델은 “둘 중 더 오래 걸리는 쪽이 병목을 결정한다”는 뜻이다.
    
- **함께 계산해볼 내용: Vector Add**
    
    길이 `N = 100,000,000`인 FP32 벡터 A, B를 더해 새 벡터 C에 저장한다고 가정합니다. FP32 원소 하나는 4Byte입니다.
    
    연습용 가상 하드웨어의 **FP32 덧셈 처리 성능은 20 × 10¹² FLOPs/s**, **HBM 대역폭은 1 × 10¹² Byte/s**로 놓겠습니다. 실제 제품 사양이나 측정값은 아닙니다.
    
    입력은 HBM에서 한 번씩 읽고 결과는 한 번 씁니다. 캐시 재사용과 추가 데이터 이동은 없다고 가정하고, 아래 항목을 각자 계산해와서 비교해주세요.
    
    - A와 B를 읽는 데이터 크기
    - C를 쓰는 데이터 크기
    - 필요한 덧셈 횟수와 총 FLOPs
    - 연산 시간과 메모리 시간
    - 더 큰 시간과, 그 값으로 설명할 수 있는 병목
    
    ```
    연산 시간 = 총 FLOPs ÷ 하드웨어 연산 성능(FLOPs/s)
    메모리 시간 = 읽고 쓴 총 Byte ÷ 메모리 대역폭(Byte/s)
    예상 실행시간 ≈ max(연산 시간, 메모리 시간)
    ```
    
    계산이 끝나면 **연산 성능·메모리 대역폭·메모리 용량 중 무엇을 늘렸을 때 이 커널이 빨라질지** 이야기해보겠습니다.
    
    [Triton: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)의 `tl.load`, 덧셈, `tl.store`를 찾아 손계산과 연결해보는 것도 좋습니다.
    
    정리한 뒤에는 **연산 성능과 메모리 대역폭을 구분할 수 있는지**, **간단한 커널의 최소 실행시간을 가정과 함께 추정할 수 있는지**, **어떤 조건에서 Memory-bound인지 설명할 수 있는지** 서로 확인해보겠습니다.
    
    ---
    
    1. A,B 읽는 데이터 크기
        - Read A = 400MB
        - Read B = 400MB
        
        총 입력 데이터 읽기량은 800MB.
        
        ```python
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # Create a mask to guard memory operations against out-of-bounds accesses.
        mask = offsets < n_elements
        
        a = tl.load(A, mask=mask)    # 400 MB
        b = tl.load(B, mask=mask)    # 400 MB
        ```
        
    2. C를 쓰는 데이터 크기
        
        C역시 FP32 벡터이므로, 400MB를 HBM에 쓸 것이다.
        
        ```python
        output = a + b
        # Write a + b back to HBM.
        tl.store(output_ptr + offsets, output, mask=mask)
        ```
        
        따라서 총 메모리 이동량은 
        
        > 800MB + 400MB = 1.2 GB
        > 
        
        이다.
        
    3. 필요한 덧셈 횟수와 총 FLOPs
        
        각 FP32 원소마다 `C[i] = A[i] + B[i]` 를 한 번 수행하므로 덧셈은 100,000,000 번 수행된다.
        
        따라서 총 FLOPs 는 10^8 FLOPs이다.
        
    4. 연산 시간과 메모리 시간
        
        연산 성능이 20 × 10^12 FLOPs/s이므로,
        
        > 연산 시간 = 10^8 / (20 x 10^12) = `5us`
        > 
        
        이다.
        
        HBM 대역폭이 10^12 Byte/s 이고, 총 1.2 x 10^9 Byte를 이동하므로
        
        > 메모리 시간 = 1.2 x 10^9/10^12 = 1.2 x 10^-3 = `1.2ms`
        > 
        
        이다.
        
    5. 더 큰 시간과 병목
        
        연산 시간은 5us, 메모리 시간은 1.2ms 이다.
        
        따라서 이상적인 실행시간은 대략
        
        > max(5 μs, 1.2 ms) = 1.2 ms
        > 
        
        이다. 연산과 메모리 이동이 충분히 겹친다는 가정에서, 더 오래 걸리는 메모리 시간이 전체 실행시간을 결정한다. 즉 이 Vector Add는 memory-bound이다.
        
    6. 무엇을 늘리면 빨라질까?
        
        메모리 이동에서 계산보다 240배 더 많은 시간이 소요되고있다.
        
        1. 따라서 연산 성능보다 메모리 대역폭을 높여야 해당 kernel을 더 효율화할 수 있다.
        2. 뒷 절에서 다루겠지만, pipeline 병렬화를 사용해도 좋을 것 같다.
            
            A[i+1], B[i+1]를 읽어올 때, CUDA Core가 연산을 하고, 때에 맞춰 C[i]을 Write하는?