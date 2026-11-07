<WeekNavBar :week="6" topic="Python에서 Triton 커널로" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송민배)', 'badge': 'Vector Add & Fused Softmax'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': 'RTX 4070 실측 & 벤치마크'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': 'Online Softmax & IR 분석'}, {'id': 'kiminjo', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '블록 포인터 & 마스킹 정밀 해부'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': 'Triton 프로그래밍 핵심 요약'}]" />


**[6주차 안내] Python에서 Triton 커널로**

안녕하세요! 6주차에는 **Python 문법으로 작성한 Triton 커널이 어떻게 블록 단위로 데이터를 읽고 계산하는지, 그리고 여러 연산을 한 커널로 묶을 때 왜 빨라지는지** 함께 살펴보려고 합니다.
3·4주차에 손계산으로 익힌 메모리 이동량과 Roofline 관점, 5주차에 비교한 가속기별 프로그래밍 경로와 측정 경계를 이어받아 이번에는 GPU 커널 코드를 직접 읽고 실행해보겠습니다.
**“Python으로 작성한 Triton 커널은 데이터를 어떻게 블록 단위로 처리하며, 여러 연산을 한 커널로 묶으면 왜 빨라질까?”**

이번 주에는 1차원 블록 인덱싱과 마스킹, 그리고 행 단위(row-wise) 연산을 하나로 묶는 **커널 결합(Kernel Fusion)**과 벤치마크 방법에 집중합니다. 2차원 타일을 `tl.dot`으로 누적하는 **Tiled MatMul**과 `TTIR → TTGIR → LLVM IR → PTX`로 내려가는 **컴파일 과정**은 7주차에서 이어갑니다.

공통 과제는 공식 튜토리얼 코드의 인덱싱·마스킹과 HBM 이동량을 손계산으로 따라가는 것입니다. 직접 실행해보고 싶은 분은 Colab·Kaggle 무료 GPU나 CPU 해석기를 활용할 수 있으며 장비가 없어도 코드 분석과 계산으로 참여할 수 있습니다.

**1. 공통으로 읽어오면 좋은 자료**

- [Triton Programming Guide 1장: Introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html#motivations)
**Motivations** 절의 **CUDA Programming Model(Scalar Program, Blocked Threads)**과 **Triton Programming Model(Blocked Program, Scalar Threads)** 비교 코드를 중심으로 읽어주세요. 스레드 하나가 스칼라 원소 하나를 맡는 방식과 프로그램 인스턴스 하나가 데이터 블록 전체를 맡는 방식이 어떻게 다른지 살펴보면 됩니다. 뒤의 **Challenges** 절은 컴파일러가 자동으로 처리하는 역할만 가볍게 훑어주세요.

- [Triton Tutorial 01: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#compute-kernel)
**Compute Kernel**과 **Benchmark** 절 전체를 읽어주세요. 2·3주차에 맛보기로 봤던 `add_kernel`과 호출 함수 `add`를 다시 보면서 `tl.program_id`, `tl.arange`, `mask`, `tl.load`, `tl.store`와 실행 그리드(`grid`)가 어떻게 연결되는지 확인해봅시다.

- [Triton Tutorial 02: Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#motivations)
**Motivations**와 **Compute Kernel** 절에서 `naive_softmax`와 `softmax_kernel` 함수를 중심으로 읽어주세요. PyTorch 기본 연산을 여러 줄로 실행할 때 HBM을 몇 번 왕복하는지 세어 보고 한 행을 온칩에 올려 한 번에 계산하는 이유를 살펴봅니다. 래퍼 함수 `softmax` 안의 레지스터·Shared Memory 기반 `occupancy` 계산과 드라이버 `warmup` 코드는 이번 주에 외우지 않아도 됩니다.

**2. 해당 사항들을 다같이 정리해주세요**

**① 블록 단위 프로그래밍 모델 — 스레드가 아니라 블록으로 생각하기**
CUDA처럼 개별 스레드 인덱스(`threadIdx`)를 직접 다루는 것과 Triton처럼 프로그램 인스턴스 하나(`tl.program_id`)가 원소 묶음(`BLOCK_SIZE`)을 한꺼번에 다루는 것은 무엇이 다를까?
길이 `N`인 벡터를 `BLOCK_SIZE` 단위로 나눌 때 필요한 프로그램 인스턴스 수 `grid = (triton.cdiv(N, BLOCK_SIZE), )`와 각 인스턴스가 맡는 구간을 작은 그림으로 그려봐주세요. 블록 안의 스레드 배치와 메모리 접근 정렬(coalescing)은 누가 맡는지도 함께 정리해봅시다.
공통 자료인 Triton Programming Guide 1장과 [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-blocks-and-grids)의 **Thread Blocks and Grids** 중심

**② 포인터 연산과 마스킹 — 텐서 크기와 블록 크기가 나누어떨어지지 않으면?**
Python에서 `torch.Tensor`를 넘기면 `@triton.jit` 커널은 무엇을 받을까? 텐서 길이 `N`이 `BLOCK_SIZE`의 배수가 아닐 때 마지막 블록에서는 어떤 일이 일어날까?
시작 포인터(`x_ptr`)에 오프셋 벡터 `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`를 더해 **포인터 블록**을 만드는 과정과 `mask = offsets < n_elements`가 범위 밖 메모리 접근을 막는 원리를 설명해봐주세요. `BLOCK_SIZE: tl.constexpr`를 왜 컴파일 시점 상수로 지정해야 하는지도 확인해봅시다.
공통 자료인 Tutorial 01과 [Triton API: `triton.jit`](https://triton-lang.org/main/python-api/generated/triton.jit.html), [`tl.arange`](https://triton-lang.org/main/python-api/generated/triton.language.arange.html), [`tl.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html), [`tl.store`](https://triton-lang.org/main/python-api/generated/triton.language.store.html) 중심

**③ 커널 결합(Kernel Fusion)과 HBM 왕복 — 왜 덧셈보다 Softmax에서 차이가 커질까?**
단일 `x + y`는 Triton으로 작성해도 PyTorch 기본 연산과 속도가 비슷한데 왜 Softmax처럼 여러 단계를 거치는 연산은 한 커널로 묶을 때 크게 빨라질까?
`naive_softmax`의 다섯 단계(`max`, 뺄셈, `exp`, `sum`, 나눗셈)가 각각 별도 커널로 실행될 때 중간 텐서가 HBM에 쓰였다가 다시 읽히는 횟수를 세어 봐주세요. 이를 한 커널로 묶어 온칩(SRAM·레지스터)에서 이어서 계산할 때 **HBM 이동량($Q$)과 산술 강도($I = F/Q$)**가 어떻게 달라지는지 3·4주차 모델로 설명해보면 좋겠습니다.
공통 자료인 Tutorial 02(**Motivations**)와 [Modern GPU Programming: Optimizing Memory-Bound Kernels](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_performance/index.html#optimizing-memory-bound-kernels)의 첫 설명 중심

**④ 행 단위 리덕션과 패딩값(`other`) — 한 행을 블록 하나에 안전하게 담으려면?**
2차원 행렬 `[M, N]`에서 각 행의 최댓값(`tl.max`)과 합계(`tl.sum`)를 구하는 `softmax_kernel`은 왜 `BLOCK_SIZE = triton.next_power_of_2(N)`으로 잡을까?
`row_start_ptr = input_ptr + row_idx * input_row_stride`로 행의 시작 주소를 찾는 방법과 `tl.load(input_ptrs, mask=mask, other=-float('inf'))`에서 빈자리를 `0`이 아니라 `-inf`로 채워야 최댓값과 지수 합계가 왜곡되지 않는 이유를 설명해봐주세요. 행의 길이 `N`이 아주 커져 온칩 메모리 한계를 넘으면 어떤 제약이 생기는지도 짚어봅시다.
공통 자료인 Tutorial 02(**Compute Kernel**)와 [Triton Semantics](https://triton-lang.org/main/python-api/triton-semantics.html#differences-with-numpy) 중심

**⑤ 정확도 검증과 벤치마크 — 무엇을 확인하고 어느 시간을 재야 할까?**
커널 호출 직후 Python 함수가 끝나면 GPU 계산도 끝난 걸까? 첫 실행의 JIT 컴파일 시간을 제외하고 반복 실행의 실효 대역폭(`GB/s`)을 구하려면 어떻게 측정해야 할까?
`1823 × 781`처럼 2의 거듭제곱이 아닌 입력으로 `torch.allclose`를 먼저 확인하는 이유와 `triton.testing.do_bench`가 워밍업(`warmup`)과 반복 측정(`rep`)을 처리하는 방식을 정리해봐주세요. 측정한 커널 시간(`ms`)을 **실효 대역폭(`GB/s = 필수 HBM 이동 Byte ÷ 실행시간`)**으로 환산하는 식도 함께 살펴보고 가능하면 `torch.compile` 결과와도 비교해봅시다.
공통 자료인 Tutorial 01·02의 **Unit Test / Benchmark** 절과 [Triton API: `triton.testing.do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html), [PyTorch: `torch.compiler` 개요](https://docs.pytorch.org/docs/stable/torch.compiler.html) 중심

**3. 정리할 때 담아오면 좋은 내용**

- 내가 맡은 질문에 대한 설명과 핵심 용어 3개
- 블록 인덱싱(`pid`, `offsets`, `mask`) 또는 결합 전후의 HBM↔온칩 데이터 이동을 비교한 작은 그림 1개
- 단계별 실행과 결합 커널의 HBM 읽기·쓰기 횟수 및 이동량(Byte) 비교표
- 계산이나 실습에 사용한 텐서 크기·자료형·`BLOCK_SIZE`와 실제 실행했다면 사용한 GPU 모델·소프트웨어 버전·측정 범위
- 참고한 자료의 링크·읽은 절과 함께 이야기하고 싶은 질문

가능하면 **1차원 `Vector Add`에서 출발해 2차원 행 단위 `Fused Softmax`로 이어지는 흐름**을 각자 맡은 질문과 연결해 설명해보면 좋겠습니다.

**4. 함께 해볼 내용: 블록 인덱싱과 Fused Softmax의 데이터 이동 비교**

먼저 **작은 1차원 벡터로 블록 인덱싱과 마스크를 손으로 확인**해봅시다.
길이 `N = 10`인 벡터를 `BLOCK_SIZE = 4`로 처리할 때 실행 그리드 크기 `grid = (triton.cdiv(10, 4), )`는 얼마인지 구하고 각 `pid = 0, 1, 2`가 만드는 `offsets`와 `mask = offsets < 10`을 표로 적어보세요. 이어서 길이가 `N = 781`인 행 하나를 `BLOCK_SIZE = 1024`로 읽는 Softmax에서 `other = 0.0`을 넣으면 음수 원소만 있는 행의 `tl.max`와 `tl.sum(tl.exp(...))`가 어떻게 달라지는지 확인해봅시다.

그다음 **`M = 4096, N = 1024`인 FP32 2차원 행렬 `X`**(원소당 4Byte, 행렬 1개 크기 `4096 × 1024 × 4 = 16,777,216 Byte ≈ 16.78 MB`)에 대해 각 행의 Softmax를 구하는 두 경로를 비교합니다. 3·4주차와 같이 연습용 가상 하드웨어의 **적용 가능한 FP32 연산 성능은 `20 × 10¹² FLOPs/s`**, **HBM 대역폭은 `1 × 10¹² Byte/s`**로 두겠습니다. 입력 `X`는 이미 장치 메모리에 올라와 있고 커널 사이의 캐시 적중은 없다고 가정합니다.

- **결합 없는 단계별 실행 (`naive_softmax`):** `x.max(dim=1)` → `x - x_max[:, None]` → `torch.exp(z)` → `numerator.sum(dim=1)` → `numerator / denominator[:, None]`의 다섯 단계가 매번 HBM에서 읽고 결과를 HBM에 씁니다.
- **Triton 결합 커널 (`softmax_kernel`):** 각 행(`1024`원소, `4 KiB`)을 HBM에서 온칩으로 한 번 읽고(`tl.load`) 최댓값·뺄셈·지수·합계·나눗셈을 모두 온칩에서 마친 뒤 결과 `Y`를 HBM에 한 번 씁니다(`tl.store`).

각자 아래 항목을 계산해 비교해주세요.

- 두 경로의 단계별 HBM 읽기·쓰기 원소 수 합계(`8MN + 4M` 및 길이 `M` 벡터를 생략한 근사치 `8MN` vs 결합 커널 `2MN`)와 총 이동 Byte
- 한 행을 온칩에 올릴 때 필요한 최소 데이터 크기(`N × 4 Byte`)
- 가상 하드웨어에서 두 경로의 메모리 시간과 이론적 속도 향상 배수
- 튜토리얼 벤치마크 코드가 실효 대역폭을 `2 × M × N × 4 Byte ÷ 실행시간`으로 계산할 때 분자에 `8MN`이 아니라 `2MN`을 쓰는 이유

**5. 실습 참여 방법: 무료 GPU와 CPU 해석기**

직접 코드를 실행해보고 싶은 분은 5주차와 동일하게 [Google Colab](https://colab.research.google.com/) 또는 [Kaggle Notebooks](https://www.kaggle.com/code)의 무료 GPU 런타임을 이용할 수 있습니다. 아래 순서로 진행해보면 좋겠습니다.

1. **정확도 확인:** `1823 × 781` 크기의 FP32 입력에서 `naive_softmax(x)`, `torch.softmax(x, dim=-1)`, Triton `softmax(x)`의 결과가 `torch.allclose`를 통과하는지 확인합니다.
2. **성능과 대역폭 비교:** `M = 4096, N = 1024`에서 `triton.testing.do_bench`로 `naive_softmax`, `torch.softmax`, Triton `softmax`(가능하면 `torch.compile(naive_softmax)` 추가)의 실행시간(`ms`)과 실효 대역폭(`GB/s`)을 측정합니다.
3. **조건 기록:** 배정받은 실제 GPU 이름(예: Tesla T4), PyTorch·Triton 버전, 입력 shape·dtype을 함께 적어주세요. 튜토리얼 문서에 실린 그래프 수치(다른 GPU에서 측정한 값)를 내 실행 결과와 섞어 비교하지 않습니다.

GPU가 없는 환경이라면 손계산과 코드 분석만으로 참여해도 충분합니다. CPU에서 커널 동작과 인덱싱을 눈으로 확인하고 싶다면 [Triton 디버깅 가이드의 CPU 해석기(`TRITON_INTERPRET=1`)](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html#using-the-interpreter)와 `print`·`pdb`를 사용해볼 수 있습니다. 다만 CPU 해석기는 NumPy 연산으로 프로그램 인스턴스를 하나씩 순차 실행하며 `bfloat16` 미지원 등 제약이 있으므로 **코드 동작과 마스크 확인에만 쓰고 실행시간 비교에는 사용하지 않습니다.**

정리한 뒤에는 **프로그램 인스턴스(`pid`)와 오프셋·마스크로 블록의 메모리 접근 범위를 설명할 수 있는지**, **커널 결합 전후의 HBM 읽기·쓰기 횟수와 이동량을 직접 셀 수 있는지**, **초기 컴파일·워밍업을 분리한 벤치마크와 실효 대역폭(`GB/s`)의 의미를 설명할 수 있는지** 서로 확인해보겠습니다.

추가로 공부하면서 도움이 된 Triton 예제나 디버깅 팁이 있으면 어떤 부분을 이해하는 데 좋았는지도 함께 공유해주세요!
그럼 다음 모임에서 뵙겠습니다!
