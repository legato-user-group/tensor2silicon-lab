<WeekNavBar :week="2" topic="모델에서 실리콘까지" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송민배)', 'badge': 'PyTorch·XLA·MAX 2부작'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': '다이어그램 & 스택 해부'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': '실행 스택 핵심 요약'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '인터랙티브 HTML & 슬라이드'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': '컴파일 파이프라인 정리'}]" />


**[2주차 안내] 모델에서 실리콘까지**

안녕하세요! 2주차에는 **AI 모델이 실제 가속기에서 실행되기까지 어떤 과정을 거치는지** 각자 정리하고 함께 이야기해보려고 합니다.
주제가 넓은 만큼, 이번에는 모든 내용을 깊게 파기보다 아래 질문을 중심으로 전체 흐름을 잡아보면 좋겠습니다.
**“모델의 연산 하나가 가속기에서 실행되기까지, 어떤 단계를 거치고 각 단계는 무엇을 담당할까?”**

**1. 공통으로 읽어오면 좋은 자료**
• [Scaling Book 9장: How to Profile TPU Code](https://jax-ml.github.io/scaling-book/profiling)
첫 절인 **A Thousand-Foot View of the TPU Software Stack**만 읽어주세요. 제목은 프로파일링이지만 앞부분은 JAX 코드가 TPU 실행 코드로 변환되는 흐름을 설명합니다.

• [PyTorch: torch.compiler 개요](https://docs.pytorch.org/docs/stable/torch.compiler.html)
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