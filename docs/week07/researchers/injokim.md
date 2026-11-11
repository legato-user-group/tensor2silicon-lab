<WeekNavBar :week="7" topic="Tiled MatMul과 컴파일 과정" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송무복)', 'badge': 'NVIDIA A100 IR·SASS 실측'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': 'RTX 5090 (sm_120) 실측 해부'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': 'L2 그룹핑 손계산 & IR 대응표'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '알고리즘 & MLIR 하향 변환 종합'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': '핵심 코드 & 컴파일 단계 요약'}]" />

> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **injokim (김인조)** (`알고리즘 & MLIR 하향 변환 종합`)
>
> 💡 **핵심 기여 요약**: 2D 브로드캐스팅 주소 수식 유도부터 MLIR Dialect(tt → ttg → llvm) 하향 변환 파이프라인까지 학술 보고서 수준으로 체계화했습니다.

---

# Triton 기반 2차원 타일 행렬곱과 컴파일 파이프라인 심화 정리

본 문서는 Triton 행렬곱($C = A \times B$) 커널의 **알고리즘 및 메모리 재사용 최적화**와 **MLIR 기반 컴파일러 하향 변환(Lowering) 파이프라인**을 통합하여 체계적으로 정리한 종합 보고서입니다.

---

## 1. 서론 및 핵심 관점

Triton 커널에서 출력 행렬 $C \in \mathbb{R}^{M \times N}$는 $BM \times BN$ 크기의 2차원 타일(Tile)로 분할되며, 각 프로그램 인스턴스(`tl.program_id`)가 하나의 출력 타일을 맡아 독립적으로 병렬 처리합니다.

본 보고서는 동일한 행렬곱 커널을 다음 두 가지 핵심 관점에서 다룹니다:
1. **알고리즘 및 메모리 재사용 관점**: 2차원 포인터 블록 생성, L2 캐시 그룹핑(Grouped Ordering), FP32 누산기(Accumulator), 에필로그 결합(Epilogue Fusion), 오토튜닝 메타파라미터.
2. **컴파일러 IR 파이프라인 관점**: Python AST $\rightarrow$ TTIR (`tt dialect`) $\rightarrow$ TTGIR (`ttg dialect`) $\rightarrow$ LLVM IR $\rightarrow$ PTX $\rightarrow$ CUBIN으로 이어지는 다단계 컴파일 및 하드웨어 레이아웃 매핑.

---

## 2. 2차원 포인터 블록과 메모리 브로드캐스팅

### 2.1 1차원 메모리 주소 구조와 Stride
GPU 메인 메모리(HBM)는 연속적인 1차원 주소 공간이므로 2차원 행렬도 행 우선(Row-major) 방식으로 직렬화되어 저장됩니다. 원소 $X[i, j]$의 메모리 주소는 다음 식에 의해 결정됩니다:
$$\text{Address}(X[i, j]) = X_{\text{ptr} } + i \times \text{stride}_0 + j \times \text{stride}_1$$

PyTorch에서 Triton 커널을 호출할 때 텐서 전체 데이터가 아닌 **시작 주소 포인터(`ptr`), 행렬 크기($M, N, K$), 각 축의 보폭(`stride`)**만을 전달합니다.

### 2.2 None 인덱싱과 2차원 주소 표 생성
2차원 타일 데이터는 메모리상에서 단일 연속 구간이 아니라 중간중간 건너뛰며 띄엄띄엄 배치되어 있습니다. 따라서 1차원 오프셋 벡터에 `[:, None]`과 `[None, :]` 차원 확장을 적용한 뒤 브로드캐스팅(Broadcasting)하여 $(BM, BK)$ 및 $(BK, BN)$ 크기의 2차원 포인터 표를 구성합니다:

```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M  # (BM,)
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  # (BN,)
offs_k  = tl.arange(0, BLOCK_SIZE_K)                                # (BK,)

# 브로드캐스팅을 통한 2차원 포인터 표 생성
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # [BM, BK]
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)  # [BK, BN]
```

* `offs_am[:, None]`: (BM,) 벡터를 (BM, 1) 세로 열 벡터로 변환.
* `offs_k[None, :]`: (BK,) 벡터를 (1, BK) 가로 행 벡터로 변환.
* 두 벡터의 더하기 연산 시 Triton 브로드캐스팅에 의해 모든 $(i, k)$ 조합의 메모리 주소를 담은 $(BM, BK)$ 2차원 주소 표가 자동 생성됩니다.

### 2.3 $K$-루프 포인터 전진
$K$축 순회 루프 내부에서는 매번 브로드캐스팅을 재계산하지 않고, 완성된 2차원 주소 표 전체에 스칼라 값을 더하여 K축의 다음 타일 위치로 포인터를 이동시킵니다:
```python
a_ptrs += BLOCK_SIZE_K * stride_ak
b_ptrs += BLOCK_SIZE_K * stride_bk
```

---

## 3. 수치 정밀도 유지 및 에필로그 결합 (Epilogue Fusion)

### 3.1 FP32 누산기(Accumulator) 사용 이유
입력 텐서 $A, B$가 FP16(16비트 부동소수점)이더라도, 누산기(`accumulator`)는 **FP32(32비트 부동소수점)** 정밀도로 초기화합니다.
* **수치 안정성**: FP16은 가수가 10비트에 불과하여 $K$축 방향으로 수십~수백 번 곱셈 결과를 누적하면 정보 손실(Underflow) 또는 값 넘침(Overflow)이 발생합니다.
* **하드웨어 가속 지원**: 현대 GPU의 Tensor Core는 FP16 곱셈과 FP32 누산을 동일한 수치 연산 속도로 수행하도록 설계되어 있어 추가 성능 저하 없이 고정밀도 결과를 얻을 수 있습니다.

### 3.2 Epilogue Fusion과 Leaky ReLU 결합 과정
$K$-루프가 종료된 시점에 연산 결과 타일은 GPU 온칩 레지스터(Register) 공간에 보관되어 있습니다. 

```python
# 에필로그 구간 (Epilogue)
if ACTIVATION == "leaky_relu":
    accumulator = leaky_relu(accumulator)  # 1. 레지스터 상태에서 Leaky ReLU 적용
c = accumulator.to(tl.float16)              # 2. FP32 -> FP16 변환 (Downcasting)
tl.store(c_ptrs, c, mask=c_mask)            # 3. HBM에 단 1회 쓰기
```

#### 메모리 트래픽 최적화 효과
* **분리 실행 시**: 행렬곱 커널 결과 쓰기($2MN$ Byte) $\rightarrow$ 활성화 커널 읽기($2MN$ Byte) $\rightarrow$ 활성화 결과 쓰기($2MN$ Byte)로 총 $4MN$ Byte의 추가 HBM 왕복 트래픽이 발생합니다.
* **Epilogue Fusion 적용 시**: 레지스터에 머물러 있는 동안 Leaky ReLU($x \ge 0 \rightarrow x, x < 0 \rightarrow 0.01x$) 연산과 FP16 캐스팅을 마친 후 HBM에 단 1회만 쓰기를 수행하므로 추가 메모리 왕복을 100% 제거합니다.

---

## 4. L2 캐시 재사용 최적화: 그룹 순서 (Grouped Ordering)

### 4.1 행 우선 실행 vs 그룹 순서 실행
GPU는 번호(`pid`)가 인접한 프로그램 인스턴스들을 동시 실행하며, 이 인스턴스들은 L2 캐시를 공유합니다.

* **행 우선(Row-Major) 배치**: $9 \times 9$ 출력 타일 그리드에서 `pid = 0~8`이 첫 행 $C[0, 0\sim8]$을 계산할 때, $A$ 행렬 1개 행 타일(9개)과 $B$ 행렬 9개 열 타일 전체(81개)를 읽어 **총 90개의 입력 타일**이 필요합니다.
* **그룹 순서(Grouped Ordering, `GROUP_SIZE_M = 3`) 배치**: $3 \times 3$ 정사각 구역으로 배치 순서를 변경하면, $A$ 행렬 3개 행 타일(27개)과 $B$ 행렬 3개 열 타일(27개)을 읽어 **총 54개의 입력 타일**로 감소합니다.

### 4.2 그룹 순서 좌표 계산식
```python
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

# 그룹 내부에서 세로 방향(M축) 우선 배치
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```
동일 면적 대비 정사각 구역이 겉둘레(필요 입력 타일 수)를 최소화하여 L2 적중률을 극대화합니다. 타일 크기 변경 없이 실행 순서 변경만으로 A100 기준 약 10%의 성능 향상(220 TFLOPS $\rightarrow$ 245 TFLOPS)을 달성합니다.

---

## 5. 메타파라미터, 오토튜닝, 소프트웨어 파이프라이닝

### 5.1 `@triton.autotune` 탐색 메커니즘
`triton.Config` 후보 목록을 전달하고 `key=['M', 'N', 'K']`가 바뀔 때 실행 시간을 벤치마크하여 최적 설정을 선택 및 캐싱합니다. 자원 초과 후보는 `OutOfResources`로 자동 건너끕니다.

### 5.2 `num_warps`와 `num_stages` 영향
* **`num_warps`**: 프로그램 인스턴스당 워프(Warp, 32 스레드) 수. 워프 수가 적으면 스레드 당 할당 레지스터 부담이 커져 Register Spill이 발생할 수 있습니다.
* **`num_stages` (소프트웨어 파이프라이닝)**: $K$-루프 순회 시 다음 타일 데이터를 비동기 복사(`cp.async`)로 Shared Memory에 미리 불러와 HBM 대기시간을 감춥니다(Latency Overlap). 
  * stages가 증가하면 대기시간 은닉 효과가 커지지만, Shared Memory 사용량이 단계 수에 비례하여 증가하므로 SM당 Occupancy가 떨어지거나 자원 한계를 초과할 수 있습니다.

---

## 6. Triton MLIR 컴파일 파이프라인 및 IR 계층

Triton은 `@triton.jit` 파이프라인을 다단계 중간 표현(IR)을 거쳐 하향 변환(Lowering)합니다:

$$\text{Python AST} \longrightarrow \text{TTIR (`tt` dialect)} \longrightarrow \text{TTGIR (`ttg` dialect)} \longrightarrow \text{LLVM IR} \longrightarrow \text{PTX} \longrightarrow \text{CUBIN}$$

```
+-------------------------------------------------------------------------------+
| Python AST     : @triton.jit 함수 파싱 및 구문 뼈대 추출                        |
+-------------------------------------------------------------------------------+
       |
       v
+-------------------------------------------------------------------------------+
| TTIR (tt)      : MLIR 기반 블록 텐서 연산 (tt.load, tt.dot). 스레드/배치 정보 없음 |
+-------------------------------------------------------------------------------+
       |  (하드웨어 정보 num_warps, num_stages, GPU 계층 반영)
       v
+-------------------------------------------------------------------------------+
| TTGIR (ttg)    : 스레드/워프 레이아웃(#blocked, #mma), cp.async 파이프라이닝 매핑|
+-------------------------------------------------------------------------------+
       |  (스레드 1개 관점 연산으로 전개)
       v
+-------------------------------------------------------------------------------+
| LLVM IR        : 스레드 단위 일반 IR 코드 및 Shared Memory 주소 공간 배치       |
+-------------------------------------------------------------------------------+
       |  (NVPTX 백엔드)
       v
+-------------------------------------------------------------------------------+
| PTX            : NVIDIA 가상 어셈블리 (mma.sync, wgmma, cp.async 명령어)        |
+-------------------------------------------------------------------------------+
       |  (ptxas 컴파일러)
       v
+-------------------------------------------------------------------------------+
| CUBIN          : 특정 GPU 세대 전용 최종 실행 바이너리 (예: sm_80)             |
+-------------------------------------------------------------------------------+
```

### 6.1 TTIR과 TTGIR의 핵심 차이 및 Layout 구조
* **TTIR (`tt dialect`)**: MLIR 프레임워크 기반의 Triton 전용 추상 연산 집합입니다. 텐서 타입이 `tensor<64x32xf16>`과 같이 블록 모양만 나타내며 스레드 배치는 정해지지 않았습니다.
* **TTGIR (`ttg dialect`)**: TTIR에 GPU 물리 레이아웃 속성이 부여된 저수준 MLIR 단계입니다.
  * `#blocked`: `sizePerThread`, `threadsPerWarp`, `warpsPerCTA`, `order`를 지정하여 메모리 병합(Coalescing) 및 연속 읽기를 구현합니다.
  * `#mma`: Tensor Core 하드웨어 누산기 전용 스레드 배치 레이아웃입니다.
  * `#shared`: Shared Memory 타일 저장용 레이아웃입니다.

---

## 7. 컴파일 산출물 검사 및 Roofline 벤치마크

### 7.1 컴파일 객체 메타데이터 확인
```python
compiled = matmul_kernel[grid](a, b, c, M, N, K, ...)
print(compiled.asm['ttgir'])         # 레이아웃 및 비동기 복사 확인
print(compiled.asm['ptx'])           # mma.sync / wgmma 어셈블리 확인
print(compiled.metadata.shared)      # Shared Memory 사용량 (Bytes)
print(compiled.n_regs, compiled.n_spills) # 레지스터 사용 수 및 Spill 수
```

### 7.2 Roofline 모델: Softmax (GB/s) vs GEMM (TFLOPs/s)
* **Softmax**: 산술 강도가 낮아 메모리 대역폭에 지배받는 커널(Memory-bound)로 **GB/s**로 측정합니다.
* **MatMul**: 연산량이 $F = 2MNK$로 크며, $M=N=K=4096$ 시 산술 강도가 약 1,365 FLOP/Byte에 도달하여 연산 능력에 지배받는 커널(Compute-bound)이므로 **TFLOPs/s** ($\frac{2MNK \times 10^{-12} }{\text{시간(sec)} }$) 단위로 측정합니다.

---

## 8. 결론

Triton 2차원 타일 행렬곱은 **알고리즘적 재사용 최적화**(브로드캐스팅 오프셋, Epilogue Fusion, Grouped Ordering)와 **컴파일러 하향 변환**(TTIR $\rightarrow$ TTGIR의 스레드 레이아웃 할당, 파이프라이닝, Tensor Core 매핑)이 상호 유기적으로 결합되어 GPU 성능을 극대화함을 보여줍니다.
