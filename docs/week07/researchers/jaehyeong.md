<WeekNavBar :week="7" topic="Tiled MatMul과 컴파일 과정" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송무복)', 'badge': 'NVIDIA A100 IR·SASS 실측'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': 'RTX 5090 (sm_120) 실측 해부'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': 'L2 그룹핑 손계산 & IR 대응표'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '알고리즘 & MLIR 하향 변환 종합'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': '핵심 코드 & 컴파일 단계 요약'}]" />

> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **jaehyeong (이재형)** (`핵심 코드 & 컴파일 단계 요약`)
>
> 💡 **핵심 기여 요약**: 프로그램 ID의 2차원 타일 좌표 매핑, K축 포인터 전진, 오토튜닝 메타파라미터와 다단계 IR의 핵심 역할을 깔끔하게 정리했습니다.

---

# 7주차: tiled matmul, triton compile process

# **7주차 | Tiled MatMul과 Triton 컴파일 과정**

# **① 2차원 포인터 블록과 K축 누적**

C = A @ B에서 **프로그램 인스턴스 하나가 BM × BN 크기의 출력 타일 하나를 담당**. 출력 위치 (i, j)는 고정하고, K축을 따라 A는 오른쪽으로, B는 아래로 이동하며 부분 행렬곱 누적.

K축이 두 타일인 경우, 왼쪽 위 출력 타일의 계산식은 다음과 같이 표현 가능. 각 항은 **타일 단위 행렬곱**.

$$
C_{0,0}=A_{0,0}B_{0,0}+A_{0,1}B_{1,0}
$$

```python
# 1. 프로그램 ID → 담당 출력 타일 좌표 (pid_m, pid_n)
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m

# 2. 첫 번째 A·B 타일의 2차원 주소 구성
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)

a_ptrs = a_ptr + (
    offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
)
b_ptrs = b_ptr + (
    offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
)

# 3. K축을 따라 A·B 타일 로드 → 부분 행렬곱 누적 → 다음 타일로 이동
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

    accumulator = tl.dot(a, b, accumulator)

    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk

# 4. 선택적 활성화 함수 적용 후 FP16 변환
if ACTIVATION == "leaky_relu":
    accumulator = leaky_relu(accumulator)
c = accumulator.to(tl.float16)

# 5. 출력 타일의 유효한 위치에만 저장
offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

| **핵심 구문** | **역할** |
| --- | --- |
| pid = tl.program_id(0) | 프로그램 ID 획득 후 출력 타일 좌표 (pid_m, pid_n) 결정 |
| m[:, None] * stride_am + k[None, :] * stride_ak | 행·K 오프셋의 브로드캐스팅으로 A의 [BM, BK] 주소 오프셋 구성 |
| k[:, None] * stride_bk + n[None, :] * stride_bn | K·열 오프셋의 브로드캐스팅으로 B의 [BK, BN] 주소 오프셋 구성 |
| tl.load → tl.dot(a, b, acc) | A·B 타일 로드 후 [BM, BN] 누산기에 부분곱 합산 |
| tl.store | K축 누적 종료 후 유효한 출력 위치에 저장 |
- **포인터 전진** — 반복마다 A는 BK × stride_ak, B는 BK × stride_bk만큼 이동. 마지막 K 구간의 범위 밖 입력은 0으로 마스킹.
- **타일 내부 재사용** — A 원소 하나는 여러 출력 열의 계산에, B 원소 하나는 여러 출력 행의 계산에 사용.
- **정밀도와 연산 결합** — FP16 입력을 FP32로 누적해 반올림 오차 감소. 누산 결과에 leaky_relu 적용 후 FP16으로 변환·저장하면 중간 행렬의 HBM 왕복 생략 가능.

# **② Grouped Ordering과 L2 캐시**

같은 **출력 타일 행**의 프로그램은 A 입력을 공유하고, 같은 **출력 타일 열**의 프로그램은 B 입력을 공유. GROUP_SIZE_M은 가까운 pid에 배정할 출력 좌표를 조정해 **프로그램 간 L2 재사용 기회**를 높이는 설정.

9 × 9 출력 타일, K축 9조각, 처음 9개 pid 기준:

- **행 우선** — 출력 1행 × 9열 담당. A 한 행분과 B 아홉 열분이 필요하므로 고유 입력 타일은 1×9 + 9×9 = 90개.
- **3행 그룹** — 출력 3행 × 3열 담당. A 세 행분과 B 세 열분이 필요하므로 고유 입력 타일은 3×9 + 3×9 = 54개.

두 배치의 연산량은 동일. **90 → 54는 고유 입력 타일 수의 감소**이며, 실제 HBM 읽기 감소량은 L2 적중률에 따라 결정.

# **③ 타일 설정·자동 튜닝·Roofline**

| **설정** | **역할** | **주요 영향** |
| --- | --- | --- |
| BM, BN | 출력 타일의 높이·너비 | 확대 시 입력 재사용 증가, 누산기 크기 증가 |
| BK | 한 번에 처리할 K축 길이 | 확대 시 K 루프 횟수 감소, 단계당 입력 버퍼 증가 |
| GROUP_SIZE_M | 한 그룹에 묶을 출력 타일 행 수 | 프로그램 간 A·B 재사용 패턴 변화 |
| num_warps | 인스턴스당 워프 수 | 타일 작업 분배와 SM 자원 점유 변화 |
| num_stages | 소프트웨어 파이프라인 단계 수 | 다음 타일 로드와 현재 연산의 중첩 가능, 온칩 버퍼 부담 증가 |

@triton.autotune은 여러 Config 후보의 실행시간을 비교해 설정을 선택하는 기능. key=['M','N','K']에서는 행렬 크기와 입력 텐서 dtype에 해당하는 결과가 있으면 재사용, 새로운 조합이면 후보 벤치마크 진행. [Triton autotuner 구현](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py)⁠

**이론 비교 조건** — M=N=K=4096, A·B·C는 FP16, 누산기는 FP32. 가상 GPU 성능 P=80 TFLOP/s, HBM 대역폭 BW=1 TB/s. 출력 타일 간 L2 재사용을 제외한 손계산 기준.

| **항목** | **후보 1** | **후보 2** | **후보 3** |
| --- | --- | --- | --- |
| BM × BN × BK | 64×32×32 | 128×128×32 | 128×256×64 |
| 워프 수 / 단계 수 | 2 / 5 | 4 / 4 | 8 / 3 |
| 프로그램 수 / 타일당 K 반복 | 8,192 / 128 | 1,024 / 128 | 512 / 64 |
| HBM 읽기 + 쓰기 = 총량 (MiB) | 6,144 + 32 = **6,176** | 2,048 + 32 = **2,080** | 1,536 + 32 = **1,568** |
| 산술 강도 (FLOP/B) | 21.22 | 63.02 | 83.59 |
| 이상 시간 / 이론 병목 | 6.476 ms / 대역폭 | 2.181 ms / 대역폭 | 1.718 ms / 연산 |
| 입력 버퍼 1단계 → 전체 | 6 → 30 KiB | 16 → 64 KiB | 48 → 144 KiB |
| FP32 누산기 논리 크기 | 8 KiB | 64 KiB | 128 KiB |
- **HBM 이동량** — BM·BN 확대 시 다른 출력 타일에서 반복해서 읽는 입력량 감소. BK만 확대할 경우 동일한 K축을 더 큰 조각으로 읽으므로 위 모델의 총 입력량은 동일.
- **자원 한계** — 분기점은 P/BW = 80 FLOP/B. 후보 3은 연산 병목 구간에 진입하지만, 단계별 입력을 모두 Shared Memory에 보관하는 모델에서는 **144 KiB** 필요. CTA당 한도 64 KiB 기준으로 후보 3은 초과, 후보 2는 한도와 동일. 누산기의 논리 크기는 별도 항목.
- **성능 지표** — Softmax는 필수 데이터 이동량 기준의 **GB/s**, MatMul은 연산량 2MNK 기준의 **TFLOP/s**로 비교 가능. 실행시간 단위가 ms일 때 TFLOP/s = 2MNK / (실행시간_ms × 10⁹).

# **④ Triton 컴파일 과정**

NVIDIA PTX 백엔드의 변환 경로는 **Python AST → TTIR → TTGIR → LLVM IR → PTX → CUBIN**. 타일 연산의 의미를 먼저 표현하고, 이후 GPU별 데이터 배치와 명령 선택을 구체화하는 구조. [Triton NVIDIA 백엔드 구현](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py)⁠

| **단계** | **표현 대상** | **MatMul 커널에서의 의미** |
| --- | --- | --- |
| **Python AST** | 커널 소스의 문법 구조 | for 루프와 tl.load, tl.dot, tl.store 호출 |
| **TTIR** | 타일 단위 연산과 데이터 흐름 | **어떤 타일을 읽고, 곱하고, 저장할지** 표현 |
| **TTGIR** | GPU의 타일 배치와 실행 방식 | **타일을 스레드·워프에 어떻게 분배할지** 표현. 메모리 레이아웃, dot 피연산자 배치, 파이프라이닝 구체화 |
| **LLVM IR** | 낮은 수준의 주소 계산과 타깃 전용 연산 | GPU 코드 생성에 필요한 메모리 접근과 연산 표현 |
| **PTX** | NVIDIA 가상 명령어 | GPU 세대·자료형·레이아웃에 따라 mma 계열 행렬 명령 등으로 표현 가능 |
| **CUBIN** | 대상 GPU용 바이너리 | 해당 GPU 아키텍처에서 실행 가능한 기계어 포함 |