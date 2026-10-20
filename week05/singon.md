## 실제 가속기로 이어가는 Roofline과 Tiling — GPU·TPU·NPU의 구조와 공정한 성능 비교

> **핵심 질문**  
> 같은 AI 작업에 쓸 가속기를 비교하려면, 어떤 조건부터 맞춰야 할까? 사양표의 연산 성능이 더 큰 장치가 실제 요청도 더 빨리 처리할까?

---

## 목차

1. [이번 주에 알아야 할 큰 그림](#1-이번-주에-알아야-할-큰-그림)
2. [무엇을 비교할 것인가 — 칩과 시스템과 사용 목적](#2-무엇을-비교할-것인가--칩과-시스템과-사용-목적)
3. [하드웨어 구조 — 계산과 데이터 이동의 담당자](#3-하드웨어-구조--계산과-데이터-이동의-담당자)
4. [성능표 읽기 — 정밀도와 희소성과 집계 단위](#4-성능표-읽기--정밀도와-희소성과-집계-단위)
5. [프로그래밍 방식 — 모델에서 장치 실행까지](#5-프로그래밍-방식--모델에서-장치-실행까지)
6. [공통 예제의 수학과 비교 조건](#6-공통-예제의-수학과-비교-조건)
7. [배치 1과 128 — 연산량과 이동량 계산](#7-배치-1과-128--연산량과-이동량-계산)
8. [가상 장치 계산 — 연산 성능이 커도 느릴 수 있다](#8-가상-장치-계산--연산-성능이-커도-느릴-수-있다)
9. [같은 작업의 제품별 비교 조건표](#9-같은-작업의-제품별-비교-조건표)
10. [실제로 빠르다는 뜻 — 시간과 처리량과 에너지](#10-실제로-빠르다는-뜻--시간과-처리량과-에너지)
11. [자주 생기는 오해와 문제 찾기](#11-자주-생기는-오해와-문제-찾기)
12. [연습문제 3개](#12-연습문제-3개)
13. [해답과 해설](#13-해답과-해설)
14. [공식 자료와 추가 읽기](#14-공식-자료와-추가-읽기)

---

## 1. 이번 주에 알아야 할 큰 그림

### 1.1 한 문장으로 정리

**가속기의 성능은 제품 이름이나 TOPS 하나로 정해지지 않는다. 같은 작업·수치 조건·장치 범위·측정 구간을 맞춰야 비교할 수 있다.**
- **TOPS(Tera Operations Per Second)는 초당 연산 횟수를 나타내는 단위**다. Tera는 $10^{12}$, 즉 1조를 뜻하므로 **1 TOPS는 초당 1조 회의 연산**이다.

지난주에서는 같은 행렬곱이라도 데이터 재사용에 따라 메모리 이동량과 실행시간의 하한이 달라짐을 살펴봤다. 이번에는 그 식에 넣을 실제 제품의 수치와 실행 조건을 읽는다.

$$
t_{\mathrm{ideal}}=\max\left(\frac{F}{P},\frac{Q}{BW}\right),\qquad
I=\frac{F}{Q}
$$

이 식의 기호와 단위는 다음과 같다.

| 기호 | 뜻 | 단위 |
|---|---|---|
| $F$ | 한 작업에서 수행하는 부동소수점 연산의 총횟수 | FLOPs, 연산 횟수 |
| $P$ | 해당 자료형·연산 조건의 최대 연산 처리율 | FLOPs/s, 초당 연산 횟수 |
| $Q$ | 정한 메모리 경계를 넘어 읽고 쓰는 데이터의 총량 | Byte |
| $BW$ | 그 메모리 경계에서 초당 옮길 수 있는 데이터량, 즉 대역폭 | Byte/s |
| $t_{\mathrm{ideal}}$ | 연산과 데이터 이동이 충분히 겹친다고 가정한 이상적 시간 하한 | s, 초 |
| $I$ | 산술 강도: 데이터 1 Byte를 옮길 때 수행하는 연산 횟수 | FLOPs/Byte |
| $\max(a,b)$ | 두 값 a와 b 중 더 큰 값을 고르는 함수 | a·b와 같은 단위 |

즉, $F/P$는 ‘총 연산 횟수 ÷ 초당 연산 횟수’로 구한 연산 시간이고, $Q/BW$는 ‘총 이동량 ÷ 초당 이동량’으로 구한 메모리 시간이다. 두 일이 완전히 겹쳐도 더 오래 걸리는 쪽보다 일찍 끝날 수 없으므로 큰 값을 고른다. 추가 실행 비용을 제외한 **하한**이지 실측시간의 예측을 보장하는 식은 아니다.

| 지난주에 배운 것 | 이번 주에 이어갈 질문 |
|---|---|
| 연산 처리율 P | 어떤 자료형·명령·희소성 조건에서 가능한 값인가? |
| 메모리 대역폭 BW | HBM, 시스템 메모리, 호스트 연결 중 어느 구간인가? |
| 이동량 Q | 입력 전송과 중간 텐서, 다른 장치로의 이동까지 세었는가? |
| 타일과 온칩 재사용 | 프로그래머·컴파일러·하드웨어 중 누가 배치와 실행을 결정하는가? |
| 시간 하한 | 초기 컴파일과 요청 대기까지 포함한 실측시간과 얼마나 다른가? |

### 1.2 이번 주의 범위와 자료 상태

사양 조사 대상은 **NVIDIA H100 SXM, Google TPU v5e, Intel Core Ultra 9 288V에 통합된 NPU**다. GPU·TPU·NPU 전체를 대표하는 보편적 순위를 만들지 않는다. TPU도 신경망 가속기의 한 사례이며, 여기서 ‘Intel NPU’는 288V의 특정 장치를 가리킨다.


### 1.3 이번 주에 배울 다섯 가지 — 질문의 뜻부터 이해하기

아래 다섯 가지는 미리 답을 알아야 하는 시험 문제가 아니다. 이번 문서를 읽으며 배울 내용이다. 먼저 쉬운 예시로 뜻을 살펴보고, 뒤에서 제품별 구조와 계산으로 연결한다. 이 절의 성능 숫자는 설명을 위해 만든 가상 값이다.

#### ① 지금 보고 있는 숫자는 어디까지 합친 것일까?

컴퓨터 한 대에 CPU와 가속기 여러 개가 들어갈 수 있다. 따라서 ‘성능이 얼마다’라는 말에는 **어느 부품을 몇 개 포함했는지**가 필요하다.

| 말 | 여기서 뜻하는 범위 | 먼저 확인할 것 |
|---|---|---|
| 가속기 칩 하나 | GPU나 TPU 같은 계산 장치 한 개 | 한 개의 성능·메모리인가? |
| 호스트 하나 | 가속기에 작업을 보내는 CPU와 메모리가 있는 컴퓨터 환경 | 연결된 가속기는 몇 개인가? |
| 전체 시스템 | 실제 작업에 사용한 호스트·가속기·연결 장치를 포함한 구성 | 서버 한 대인가, 여러 대인가? |

예를 들어 가속기 A 한 개의 최대 성능이 100이고, B 여덟 개의 최대 성능 합이 400이라고 하자. 이를 보고 ‘B 칩 하나가 A 칩 하나보다 4배 빠르다’고 말할 수는 없다. B의 400은 여덟 개를 합친 값이기 때문이다. 또한 최대 성능을 합한 값이 실제 작업에서도 그대로 나오는지는 별도로 확인해야 한다.

호스트와 전체 시스템이 항상 서로 다른 크기인 것은 아니다. 호스트 한 대와 그 가속기만으로 작업한다면 그 구성이 전체 시스템일 수 있다.

**이 항목에서 할 일:** 숫자 옆에 ‘가속기 1개 기준’ 또는 ‘서버 1대·가속기 8개 기준’처럼 범위를 적는다. 자세한 내용은 **2절**에서 다룬다.

#### ② 계산하기, 순서 정하기, 데이터 옮기기는 각각 누가 할까?

이번 작업은 `Y = ReLU(XW + b)`다. X는 입력 숫자 묶음, W는 가중치, b는 더해 주는 보정값, Y는 결과다. ReLU는 음수를 0으로 바꾸는 함수다. 이 한 줄을 실행하려면 다음 일들이 필요하다.

| 필요한 일 | 쉬운 뜻 | 이번 작업의 예 |
|---|---|---|
| 행렬곱 | 여러 숫자를 곱해서 더하기 | `XW` 계산 |
| 원소별 연산 | 결과의 각 숫자에 계산 적용하기 | b 더하기, 음수는 0으로 바꾸기 |
| 실행 제어 | 어떤 작업을 언제 실행할지 정하기 | 입력이 준비된 계산을 실행하고 완료 확인 |
| 데이터 이동 | 숫자를 저장한 곳과 계산할 곳 사이에서 옮기기 | 메모리에서 X·W 읽기, Y 저장 |

‘메모리에 숫자가 저장되어 있다’와 ‘계산 장치가 그 숫자를 바로 사용할 수 있다’는 다르다. 필요한 조각을 가까운 저장 공간으로 가져와야 할 수 있다. 지난주의 타일링은 이렇게 가져온 조각을 여러 계산에 다시 쓰는 방법이다.

**이 항목에서 할 일:** 3절의 그림에서 ‘숫자를 보관하는 곳’, ‘곱하고 더하는 곳’, ‘실행을 관리하는 곳’, ‘숫자가 이동하는 경로’를 찾는다. 처음부터 모든 부품 이름을 외울 필요는 없다. 실제로 어느 연산을 어느 부품이 맡는지는 제품과 컴파일 결과에 따라 달라진다.

#### ③ 왜 성능 숫자 두 개를 그냥 나누면 안 될까?

비율이 의미 있으려면 같은 기준으로 센 숫자여야 한다. 아래 조건이 다르면 **비교하는 일 자체가 달라질 수 있다.**

| 조건 | 무슨 뜻일까? | 맞추지 않으면 생기는 문제 |
|---|---|---|
| 정밀도·자료형 | 숫자를 어떤 표현으로 저장하고 계산하는가? 예: INT8, BF16 | 결과 오차와 하드웨어의 처리 가능한 연산 속도가 달라질 수 있음 |
| 희소성 | 특정한 0 배치를 이용해 일부 계산을 생략할 수 있는가? | 계산을 생략할 수 있는 경우의 피크를 일반 행렬에 적용할 수 없음 |
| 장치 수 | 몇 개의 가속기를 합산했는가? | 칩 하나의 성능과 여러 칩의 합계를 혼동함 |

예를 들어 A의 ‘INT8·희소성 적용·8개 합계 800 TOPS’와 B의 ‘BF16·희소성 미적용·1개 100 TFLOPs/s’를 나눈 8은 같은 모델의 속도비가 아니다. 숫자 표현, 계산 생략 조건, 장치 수가 모두 다르다.

**이 항목에서 할 일:** 사양 숫자 옆에 자료형·희소성·장치 수를 적고, 맞지 않는 조건을 표시한다. 그 조건을 맞춰도 실제 실행시간의 비율까지 보장되는 것은 아니다. 자세한 내용은 **4절**에서 다룬다.

#### ④ 같은 계산식을 주면 기계 안에서도 똑같이 계산할까?

두 장치에 모두 `Y = ReLU(XW + b)`를 전달했다고 하자. 사용자가 요구한 계산식은 같지만 실행 방법은 다를 수 있다.

- 한쪽은 행렬곱 결과를 메모리에 저장한 뒤 b를 더하고 ReLU를 실행할 수 있다.
- 다른 쪽은 이 작업들을 결합해 마지막 결과만 저장할 수 있다.
- 입력이 FP32여도 내부 곱셈에는 다른 정밀도를 사용할 수 있다.

프레임워크는 모델의 계산을 표현하고 실행하도록 돕는 소프트웨어이고, 컴파일러는 그 계산을 장치가 실행할 명령으로 바꾼다. 이 과정에서 데이터 배치·연산 결합·계산 정밀도 등이 정해진다.

따라서 **‘같은 모델을 전달했다’는 것은 같은 계산을 요청했다는 뜻**이다. **‘같은 저수준 계산 조건을 맞췄다’는 것은 실제로 곱하는 숫자의 자료형, 중간 합의 정밀도, 출력 반올림 같은 조건까지 확인했다는 뜻**이다. 서로 다른 장치에서 명령어까지 동일하게 만들어야 한다는 뜻은 아니다.

**이 항목에서 할 일:** 모델 입력의 자료형과 실제 곱셈·중간 합의 자료형을 나누어 적는다. 확인할 수 없는 값은 미확인으로 남긴다. 자세한 내용은 **5·6절**에서 다룬다.

#### ⑤ 배치 1과 128에서는 무엇을 적으면 될까?

배치는 **한 번의 실행에 묶어 처리하는 입력 개수**다. 배치 1은 입력 하나, 배치 128은 입력 128개를 한 번에 처리한다. 여러 입력이 같은 W를 사용하면 가중치를 재사용할 기회가 늘어난다. 그렇다고 실행시간이 반드시 같거나 128배가 되는 것은 아니다.

| 비교할 조건 | 배치 1 | 배치 128 |
|---|---|---|
| 한 번에 처리할 입력 수 | 1개 | 128개 |
| 사용할 가중치 W | 같은 W | 같은 W |
| 입력·계산·중간 합의 자료형 | 확인해서 기록 | 왼쪽과 같은 조건인지 확인 |
| 결과가 충분히 정확한가? | 공통 기준 출력과 오차 비교 | 같은 오차 기준 적용 |
| 한 호출의 완료시간 | 측정 전에는 ‘아직 미측정’ | 측정 전에는 ‘아직 미측정’ |
| 초당 완료한 입력 수 | 완료 입력 수 ÷ 측정 시간 | 완료 입력 수 ÷ 측정 시간 |

예를 들어 배치 128의 호출이 2 ms 걸렸다면, 입력 128개의 결과를 받는 데 2 ms가 걸린 것이다. 2 ms를 128로 나눈 값은 입력 하나당 평균 처리 비용이지, 사용자가 결과를 기다리는 시간이 아니다.

---

## 2. 무엇을 비교할 것인가 — 칩과 시스템과 사용 목적

### 2.1 조사 대상을 먼저 고정하기

| 항목 | NVIDIA | Google | Intel |
|---|---|---|---|
| 제품·세대 | H100 SXM, Hopper | TPU v5e | Core Ultra 9 288V, Series 2·Lunar Lake, NPU 4 |
| 이번 사양표의 범위 | GPU 1개 전체, MIG 분할 제외 | TPU 칩 1개 | 프로세서 내부 NPU 1개, CPU·GPU 성능 제외 |
| 가정한 환경 | 호스트 CPU가 연결된 서버 | TPU VM의 단일 칩 실행 | 해당 프로세서를 탑재한 개인 기기 |
| 실제 사용 소프트웨어 버전 | 아직 미확인 | 아직 미확인 | OpenVINO 2026 문서 기준, 설치 버전은 미확인 |

제품 식별과 세대의 근거는 [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/), [Google TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e), [Intel 288V 사양](https://www.intel.com/content/www/us/en/products/sku/240961/intel-core-ultra-9-processor-288v-12m-cache-up-to-5-10-ghz/specifications.html), [Intel Lunar Lake 아키텍처 자료](https://cdrdv2-public.intel.com/824436/2024_Intel_Tech%20Tour%20TW_Lunar%20Lake%20AI%20Hardware%20Accelerators.pdf)다.

### 2.2 TPU의 칩·호스트·Pod는 다른 단위다

TPU v5e 문서에서 칩은 연산 장치, 호스트는 장치를 제어하는 CPU 시스템, Pod는 여러 칩의 연결 구성이다. v5e의 전체 호스트 구성은 8칩, Pod는 256칩이며, 1·4·8칩 VM 구성도 설명한다. 따라서 ‘TPU VM 하나’를 ‘TPU 칩 하나’와 같은 말로 사용하면 안 된다. (참고: v5e 칩마다 TensorCore 하나가 포함됩니다. TensorCore마다 행렬 곱셈 단위(MXU) 4개, 벡터 단위, 스칼라 단위가 있음.) 실제로 보이는 장치 수와 작업이 사용한 장치 수를 각각 기록한다. [TPU v5e — System architecture / Configurations](https://docs.cloud.google.com/tpu/docs/v5e)
<img width="1242" height="826" alt="image" src="https://github.com/user-attachments/assets/62b0fb09-f5ac-44e3-b87f-7b3b847353bc" />

```text
단일 칩 비교                         시스템 비교
GPU 1개 또는 TPU 1칩                 CPU 호스트 + 여러 가속기
  └─ 칩의 메모리·연산 능력             ├─ 장치별 메모리
                                     ├─ 장치 간 통신
                                     └─ 요청 배분·호스트·네트워크
```

8개 장치의 최대 처리율을 더한 값은 시스템의 이상적 연산 상한일 뿐이다. 작은 작업을 나누면 통신과 동기화가 이득보다 클 수 있다. 각 장치의 메모리 용량 합도 한 칩에서 같은 속도로 접근하는 단일 메모리 용량을 뜻하지 않는다.

### 2.3 사용 목적도 비교 조건이다

| 목적 | 함께 맞춰야 할 조건 |
|---|---|
| 학습 | 순전파·역전파·옵티마이저, 수렴 기준, 분산 통신 |
| 추론 | 입력·가중치·출력, 품질 기준, 배치·요청 패턴 |
| 클라우드 서비스 | 네트워크·큐·동시 요청 수, 사용자 지연시간 |
| 개인 기기 실행 | 전원 모드·발열·공유 메모리 경쟁, 기기 안에서의 전체 시간 |

이번 공통 과제는 추론이다. 학습 처리량을 가져와 추론 지연시간과 비교하지 않는다. 서버와 개인 기기를 서비스 관점에서 비교할 수는 있지만, 그때에는 칩 성능 외의 환경 차이도 결과의 일부다.

---

## 3. 하드웨어 구조 — 계산과 데이터 이동의 담당자

### 3.1 H100 SXM — 여러 SM과 계층적인 메모리

```text
호스트 CPU·RAM
      ↕ PCIe: 입력 전달·실행 요청·결과 회수
H100 SXM 1개
  HBM ── L2 캐시 ── 여러 SM
                     ├─ shared memory / L1, 레지스터
                     ├─ Tensor Core: 행렬곱
                     ├─ CUDA 연산 유닛: 원소별·일반 연산
                     └─ warp scheduler: 준비된 워프의 명령 발행
      ↕ NVLink: 다른 GPU와의 통신
```

프로그래머나 라이브러리는 블록·타일·데이터 배치를 정하고, 컴파일러는 이를 명령으로 내린다. 하드웨어는 워프 실행과 캐시를 관리한다. 지난주의 shared-memory tiling은 입력 조각을 명시적으로 재사용하는 예다. 다만 `ReLU`가 항상 별도 커널이 되는 것은 아니다. GEMM의 후처리에 결합될 수도 있다. [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html), [Triton 행렬곱 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

### 3.2 TPU v5e — TensorCore 안의 MXU·벡터·스칼라 장치

```text
호스트 CPU·RAM
      ↕ 호스트 연결: 데이터 전달·프로그램 실행
TPU v5e 칩 1개
  HBM ↔ 온칩 VMEM·레지스터 ↔ TensorCore 1개
                              ├─ MXU 4개: 행렬곱
                              ├─ Vector unit: 벡터·원소별 연산
                              └─ Scalar unit: 스칼라 계산·제어
      ↕ ICI: 다른 TPU 칩, 이번 단일 칩 과제에서는 제외
```
<img width="1238" height="774" alt="image" src="https://github.com/user-attachments/assets/802557d3-580d-44da-b595-eac18f89d64b" />


v5e의 TensorCore 구성은 [Google 제품 문서](https://docs.cloud.google.com/tpu/docs/v5e)에서 확인한다. 온칩 메모리와 계산의 배치·재사용·파이프라이닝은 컴파일러와 커널 작성 방식의 영향을 받는다. Pallas의 TPU 설명은 HBM에서 VMEM으로 블록을 가져오는 관점으로 이를 보여준다. [Pallas — Writing TPU kernels](https://docs.jax.dev/en/latest/pallas/tpu/details.html)

pallas 사용 예시 (출력 배열을 상수로 채우는 커널)<br>
임시 VMEM 버퍼를 할당하고, 거기에 상수를 쓰고, 출력 HBM Ref로 복사합니다.
```
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

@pl.kernel(
    out_type=jax.ShapeDtypeStruct((128,), jnp.float32),
    mesh=pltpu.TensorCoreMesh(axis_name='core'),
    scratch_types=dict(o_vmem=pltpu.VMEM((128,), jnp.float32)),
)
def fill_42(o_ref, o_vmem):
  # Compute in VMEM
  o_vmem[...] = jnp.full_like(o_vmem, 42.0)

  # VMEM → HBM (blocks until the transfer completes)
  pltpu.sync_copy(o_vmem, o_ref)

result = fill_42()  # [42.0, 42.0, ...]
```

**NVIDIA의 Tensor Core는 행렬 연산 유닛이고, TPU의 TensorCore는 MXU·벡터·스칼라 장치를 포함하는 더 큰 구성 단위다.** 이름이 비슷하다고 개수를 직접 비교하지 않는다. 또한 Scaling Book의 칩 비교 예에 등장하는 TPU v5p의 구성 개수를 v5e에 옮겨 적지 않는다. [Scaling Book — GPUs vs. TPUs at the chip level](https://jax-ml.github.io/scaling-book/gpus/#gpus-vs-tpus-at-the-chip-level)

### 3.3 288V의 NPU 4 — 행렬 엔진과 DSP, DMA

```text
같은 기기의 CPU: OpenVINO Runtime·컴파일러·드라이버
      │ 실행 요청
시스템 LPDDR 메모리 ↔ SoC 내부 연결 ↔ NPU 4
                                      ├─ Global control / MMU
                                      ├─ DMA ↔ scratchpad RAM
                                      └─ Neural Compute Engine들
                                          ├─ MAC array: 행렬곱
                                          ├─ activation·변환 경로
                                          └─ SHAVE DSP: 벡터 처리
```

Intel의 Lunar Lake NPU 4 자료는 6개의 Neural Compute Engine과 12개의 SHAVE DSP, DMA·scratchpad·제어부를 설명한다. 그래프 컴파일 결과에 따라 연산과 데이터 이동을 배치한다. 이 구성은 Lunar Lake 사례이며 다른 Core Ultra 세대에 그대로 적용하지 않는다. [Intel AI Hardware Accelerators, NPU 4 Architecture 부분](https://cdrdv2-public.intel.com/824436/2024_Intel_Tech%20Tour%20TW_Lunar%20Lake%20AI%20Hardware%20Accelerators.pdf)

공유 시스템 메모리를 사용해도 복사·레이아웃 변환·동기화 비용이 모두 0이 되는 것은 아니다. 위 그림은 역할을 연결한 개념도이며 버스의 실제 폭·실효 대역폭·버퍼 용량을 나타내지는 않는다.

### 3.4 지난주의 Tiling은 어디로 이어질까?

| 질문 | H100 SXM | TPU v5e | 288V NPU |
|---|---|---|---|
| 입력 조각을 가까이 두는 곳 | shared memory·레지스터·캐시 | VMEM·레지스터 | scratchpad 등 온칩 저장 공간 |
| 재사용을 설계하는 경로 | 라이브러리 또는 CUDA·Triton | XLA 또는 Pallas | NPU 그래프 컴파일러 |
| 큰 타일의 비용 | 저장 공간·상주 작업 제약 | VMEM·파이프라인·배치 제약 | 컴파일러의 메모리·연산 배치 제약 |
| 실제로 확인할 것 | 실행 커널·프로파일 | 컴파일 결과·프로파일 | 지원 여부·컴파일 결과·실행 장치 |

세 경우 모두 ‘입력을 재사용하면 이동량을 줄일 수 있다’는 원리는 같다. 반면 GPU의 occupancy 정의를 TPU나 NPU에 그대로 적용할 수는 없다.

---

## 4. 성능표 읽기 — 정밀도와 희소성과 집계 단위

### 4.1 숫자 옆에 붙여야 할 여섯 조건

```text
성능 수치 = 값 + 연산 종류 + 입력 정밀도 + 누산 정밀도
                + dense/sparse 조건 + 칩/시스템 집계 범위
```
- 값: 초당얼마나 많은 연산을 처리하는지 (예시의 100 TFLOPs/s는 초당 부동소수점 연산 100조 회라는 뜻, 사양표의 수치는 보통 특정 조건에서 이론적 최대 처리율을 의미함)
- 연산 종류: 무엇을 하는 속도인지 (곱셈, 덧셈, 행렬곱, 나눗셈 등은 모두 계산이지만 하드웨어가 처리하는 속도는 같지 않다.)
- 입력 정밀도: 곱할 숫자를 어떤 형식으로 표현하는지 (예시로 BF16)
- 누산 정밀도: 곱한 결과를 계속 더할 때 어떤 형식을 사용하는지 (많은 값을 더할 때에는 중간 합의 정밀도가 결과 오차에 영향을 줌)
    - 곱셈에 사용하는 입력 숫자는 BF16으로 표현한다.
    - 그 곱들을 더해 가는 중간 합은 FP32로 유지한다.
- dense/sparse 조건: 0인 값을 이용해서 계산을 생략한 성능인가?
    - Dense 성능: 희소성을 이용한 계산 생략을 전제로 하지 않는 성능입니다.
    - Sparse 성능: 지원되는 0의 배치 등을 이용해 일부 계산을 생략할 수 있을 때의 성능입니다.
- 칩/시스템 집계범위: 어느 장치를 몇 개 합친 수치인가?
    - 예시: 100 TFLOPs/s가 다음 중 무엇인지 구분해야 합니다.
        - 가속기 한 개의 성능
        - 가속기 여러 개의 성능 합계
        - CPU·GPU·NPU 등 여러 종류의 연산 장치를 합한 성능
> “가속기 한 개에서, BF16 숫자를 입력으로 받아 FP32로 합을 누적하는 행렬 연산을, 희소성에 의한 계산 생략 없이 수행할 때의 최대 처리율이 100 TFLOPs/s다.”
### TOPS와 FLOPs/s를 풀어서 읽기

| 표기 | 풀어 쓴 뜻 | 읽는 방법 |
|---|---|---|
| operation | 기본 연산 한 번 | 곱셈·덧셈 등 무엇을 1회로 세는지 확인 |
| OPS | Operations Per Second | 초당 기본 연산 횟수 |
| TOPS | Tera Operations Per Second | 초당 연산 횟수를 1조 단위로 표시 |
| FLOPs | Floating-point Operations | 부동소수점 연산의 총횟수; 이 문서에서는 작업량 표기 |
| FLOPs/s | Floating-point Operations Per Second | 초당 부동소수점 연산 횟수 |
| TFLOPs/s | Tera Floating-point Operations Per Second | 초당 부동소수점 연산 1조 회를 1단위로 표시 |
| INT8 | 8비트 정수 표현 | 정수 연산의 자료형 |
| FP16 / BF16 | 각각 16비트 부동소수점 표현 | 같은 16비트라도 지수·가수 구성이 다름 |
| FP32 | 32비트 부동소수점 표현 | 입력·계산·중간 합 중 어디에 쓰는지 구분 |

$$
1\;\mathrm{TOPS}=10^{12}\;\mathrm{operations/s},\qquad
1\;\mathrm{TFLOPs/s}=10^{12}\;\mathrm{FLOPs/s}
$$

예를 들어 **INT8 48 TOPS**는 해당 사양의 조건에서 8비트 정수 연산을 초당 최대 48조 회 처리한다는 뜻이다. TOPS 자체가 INT8을 뜻하는 것은 아니므로 자료형을 함께 적어야 한다. **INT8 TOPS와 BF16 TFLOPs/s는 숫자 크기만으로 우열을 정할 수 없다.**

연산을 세는 방식도 확인한다. `a × b + c`처럼 곱한 뒤 더하는 계산을 MAC(Multiply-Accumulate, 곱셈·누산)이라고 부른다. 곱셈 1회와 덧셈 1회를 따로 세면 **1 MAC = 2 operations**다. 이 기준의 48 TOPS는 초당 24조 MAC에 해당한다. 부동소수점 FMA(Fused Multiply-Add, 곱셈과 덧셈을 결합한 연산)도 이 문서에서는 2 FLOPs로 센다. 명령 하나와 연산 한 번이 항상 같은 뜻은 아니다.

### 4.2 공개 사양을 조건과 함께 옮기기

| 항목 | H100 SXM 1개 | TPU v5e 1칩 | 288V 내부 NPU 1개 |
|---|---|---|---|
| 부동소수점 행렬 성능 | BF16·FP16 Tensor Core **1,979 TFLOPs/s, sparse** [H] | BF16 **197 TFLOPs/s** [T] | FP16 성능 수치는 이번 조사에서 미확인 |
| 정수 행렬 성능 | INT8 **3,958 TOPS, sparse** [H] | INT8 **393 TOPS** [T] | NPU INT8 **48 TOPS** [I] |
| 희소성 조건 | 별표의 `With sparsity` 적용 [H] | 이 표에 희소성 배율 별도 표시 없음 [T] | sparsity 지원은 Yes이나 48의 적용 조건은 해당 사양표에서 미확인 [I] |
| 표의 누산 정밀도 | 제품표만으로 해당 커널의 누산은 확정 불가 | BF16 MXU의 FP32 누산은 별도 구조 문서 참고 [A] | 이번 조사에서 미확인 |
| 큰 메모리 용량 | GPU 메모리 80 GB [H] | 칩당 HBM 16 GB [T] | 시스템 최대 32 GB; NPU 전용 용량 아님 [I] |
| 큰 메모리 대역폭 | 3.35 TB/s [H] | 800 GiB/s [T] | NPU에 보장되는 실효 대역폭은 미확인 |
| 장치 간 연결 | NVLink 900 GB/s 표기 [H] | ICI 칩당 양방향 합계 400 GB/s [T] | 대응하는 다중 NPU 연결 수치는 미확인 |
| 호스트 연결 | PCIe Gen5 128 GB/s 표기 [H] | 본 조사에서 수치 미확인 | SoC 내부 경로; PCIe GPU의 값으로 대체 불가 |
| 전력 관련 사양 | 최대 TDP 700 W, 설정 가능 [H] | 칩 TDP는 이번 조사에서 미확인 | 프로세서 기본 전력 30 W·최대 터보 37 W; NPU 단독 값 아님 [I] |

출처: [H: NVIDIA H100 SXM 열과 각주](https://www.nvidia.com/en-us/data-center/h100/), [T: Google TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e), [I: Intel 288V 사양](https://www.intel.com/content/www/us/en/products/sku/240961/intel-core-ultra-9-processor-288v-12m-cache-up-to-5-10-ghz/specifications.html), [A: Google TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm).

H100의 dense BF16 성능은 약 **989 TFLOPs/s**로 구분해 사용한다. NVIDIA의 자체 벤치마크 설명에서도 이 dense 값을 사용한다. 무작위 dense 행렬에 sparse 피크를 대입하지 않는다. [NVIDIA H100 dense BF16 기준 설명](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/training/benchmark/E2E_BENCHMARK.md)

288V의 **전체 INT8 120 TOPS**는 NPU 단독 48 TOPS와 다른 범위다. 전체 수치를 NPU 성능으로 쓰거나, 여러 엔진의 피크 합이 단일 모델에서 동시에 실현된다고 가정하지 않는다. [Intel 288V — Overall / NPU Peak TOPS](https://www.intel.com/content/www/us/en/products/sku/240961/intel-core-ultra-9-processor-288v-12m-cache-up-to-5-10-ghz/specifications.html)

### 4.3 메모리 대역폭과 연결 대역폭을 구분하기

$$
800\;\mathrm{GiB/s}=800\times2^{30}\;\mathrm{Byte/s}
\approx859.0\;\mathrm{GB/s}
$$

GB/s와 GiB/s의 차이는 약 7.4%다. 그리고 HBM→연산 장치, CPU↔가속기, 가속기↔가속기는 서로 다른 경계다. 세 값을 더해 하나의 BW로 사용하지 않는다.

여기서 GB는 $10^9$ Byte, GiB는 $2^{30}$ Byte, TB는 $10^{12}$ Byte다. `/s`는 ‘초당’을 뜻한다. 저장 용량인 GB와 데이터 이동 속도인 GB/s도 구분한다. Byte는 8 bit이며, 이후 시간 표의 1 ms는 $10^{-3}$초, 1 μs는 $10^{-6}$초다.

양방향 합계도 단방향 전송 속도와 다르다. 예를 들어 H100의 NVLink 900 GB/s는 양방향 집계로 읽어야 하며, 한 방향의 900 GB/s 전송을 뜻하지 않는다. 연결 개수·방향·토폴로지까지 맞춰야 한다. [Scaling Book — Networking](https://jax-ml.github.io/scaling-book/gpus/#networking)

### 4.4 실제 제품의 Roofline에 대입할 때

분기점 $P/BW$에서 P는 최대 연산 처리율(FLOPs/s), BW는 메모리 대역폭(Byte/s)이다. 나누면 단위는 FLOPs/Byte가 된다. 이는 연산 시간과 메모리 시간이 같아지는 산술 강도이며, 작업의 산술 강도 $I=F/Q$와 비교한다.

H100의 dense BF16 약 989 TFLOPs/s와 3.35 TB/s를 사용하면 분기점은 약 295 FLOPs/Byte다. v5e의 BF16 197 TFLOPs/s와 800 GiB/s에서는 약 229 FLOPs/Byte다. 이는 각각 해당 연산·메모리 경계의 **이론적 비율**이다.

뒤에서 계산하는 배치 128의 I≈102.32를 대입하면 두 장치 모두 이 단순 모델에서는 메모리 측 상한 아래에 놓인다. 하지만 작은 행렬이 캐시에 남거나 실행 준비 비용이 지배하면 실제 병목은 달라진다. 288V NPU는 같은 부동소수점 P와 해당 메모리 BW를 확보하지 못했으므로 같은 방식의 제품별 시간을 채우지 않는다.

---

## 5. 프로그래밍 방식 — 모델에서 장치 실행까지

### 5.1 프레임워크 호출과 직접 커널 작성은 다르다

| 경로 | 모델 수준에서 하는 일 | 더 낮은 수준에서 조절하는 일 |
|---|---|---|
| NVIDIA GPU | 프레임워크가 연산을 CUDA 라이브러리·컴파일된 커널로 전달 | CUDA·Triton으로 타일, 데이터 배치, 로드·계산·저장 구성 |
| Google TPU | JAX 함수를 `jax.jit`으로 XLA 컴파일 | Pallas로 블록별 접근·VMEM 재사용·커널 구성 |
| Intel NPU | OpenVINO 모델을 NPU 플러그인으로 컴파일·추론 | 컴파일러가 그래프를 하드웨어에 배치; 공개 모델 API를 임의 NPU 커널 작성 API와 동일시하지 않음 |

CUDA는 호스트와 장치, 커널과 스레드 계층을 명시한다. Triton의 행렬곱 튜토리얼은 블록별 포인터·누산·출력 저장을 작성하는 예다. JAX JIT는 함수를 추적하고 XLA로 컴파일하며, Pallas는 별도의 커널 작성 경로다. [CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html), [Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), [JAX JIT](https://docs.jax.dev/en/latest/jit-compilation.html#jit-compiling-a-function), [Pallas TPU](https://docs.jax.dev/en/latest/pallas/tpu/details.html)

### 5.2 OpenVINO 2026에서 특히 확인할 것

```python
import openvino as ov

core = ov.Core()
model = core.read_model("model.xml")  # 같은 XW+b→ReLU 그래프를 미리 준비
compiled = core.compile_model(model, "NPU")
```

이 코드는 실행 경로의 예시이며 이 저장소에 `model.xml`이 있다는 뜻은 아니다. 문서에는 내부 primitive의 F32·F16·U8 지원과 하드웨어 FP16 계산이 구분되어 있다. 따라서 FP32 모델 입력이 FP32 하드웨어 계산을 보장하지 않는다. 누산 정밀도도 따로 확인한다.

동적 shape는 preview다. Lunar Lake의 NPU40XX 이상, 상한이 있는 차원, Compiler-In-Plugin 경로, Windows 드라이버 32.0.100.4621 또는 Linux 1.30 이상이라는 조건이 있다. 검증 범위도 제한되어 있으므로 이 공통 그래프의 동적 배치 지원을 보장하지 않는다. 이번에는 배치 1과 128의 **정적 그래프를 각각 컴파일**한다.

AUTO는 NPU 단독 실행 증거가 아니며 HETERO 지원도 제한적이다. 버전 간 컴파일 blob 호환성도 보장되지 않는다. [OpenVINO 2026 — NPU Device, Data Types / Dynamic Shapes / Limitations](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)

### 5.3 지원하지 않는 연산을 만났다면

| 단계 | 확인할 자료·행동 | 기록할 내용 |
|---|---|---|
| 프레임워크 변환 | 모델 변환 로그, 연산 지원 문서 | 그래프가 원래 계산을 보존하는가? |
| 장치 컴파일 | 선택한 dtype·shape의 컴파일 결과 | 실패 연산·오류, 컴파일러 버전 |
| 실제 실행 | 실행 장치 정보·프로파일 | CPU fallback, 분할 실행, 추가 복사 여부 |
| 결과 검증 | 같은 기준 출력과 오차 비교 | 허용오차 통과 여부 |

OpenVINO에서는 `core.query_model(model, "NPU")`로 지원 매핑을 조사한 뒤 실제 컴파일·추론까지 확인하는 절차를 사용할 수 있다. 지원 조회만으로 성능이나 전체 실행 성공을 확정하지 않는다. [OpenVINO Core.query_model](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.Core.html#openvino.Core.query_model)

CPU로 일부 연산이 넘어갔다면 ‘NPU 커널 단독’ 결과와 분리한다. 반면 전체 시스템 추론 비교에서는 그 전송·CPU 시간을 포함한 결과가 유효할 수 있다. 어느 질문에 답하는 측정인지 먼저 정하는 것이 중요하다.

---

## 6. 공통 예제의 수학과 비교 조건

### 6.1 기호와 shape

$$
Y=\operatorname{ReLU}(XW+b),\qquad
Y_{ij}=\max\left(0,\sum_{k=0}^{1023}X_{ik}W_{kj}+b_j\right)
$$

| 텐서 | shape | 역할 |
|---|---|---|
| X | `[M, 1024]` | M개 샘플의 입력 |
| W | `[1024, 1024]` | 고정 가중치 |
| b | `[1024]` | 행마다 broadcast하는 bias |
| Y | `[M, 1024]` | 최종 출력 |
| M | 1 또는 128 | 한 호출에 들어가는 배치 크기 |

일반적인 행렬곱 표기에서는 X의 shape를 `[M, K]`, W를 `[K, N]`, Y를 `[M, N]`으로 쓴다. **K는 입력 특징 수이자 곱해서 더할 축의 길이**, **N은 출력 특징 수**다. 이번에는 K=N=1024이며 M만 1과 128로 바꾼다. shape는 각 축의 원소 수를 순서대로 적은 것이다.

식의 i는 샘플 행 번호(0부터 M−1), j는 출력 특징 번호(0부터 N−1), k는 입력 특징 번호(0부터 K−1)다. $\sum$는 k를 바꾸며 곱한 값을 모두 더하라는 뜻이다. $b_j$는 출력 특징 j에 더하는 bias이며 모든 행에 같은 값을 적용하는 것을 broadcast라고 한다. ReLU(Rectified Linear Unit)는 음수를 0으로 바꾸고 나머지는 그대로 두는 함수로, $\max(0,z)$는 0과 z 중 큰 값을 고른다.

학습·역전파는 없다. 같은 seed로 한 번 만든 입력 파일을 공유하고, 배치 1은 배치 128 입력의 첫 행을 사용한다. W·b는 두 배치에서 같다. 프레임워크별 난수 생성기를 각각 호출하면 seed가 같아도 값이 다를 수 있으므로 입력 파일의 해시까지 기록하는 편이 좋다.

### 6.2 두 종류의 공정한 비교


| 비교 | 알고 싶은 것 |
|---|---|
| 동일 연산 조건의 커널 비교 | 같은 수치 계산을 시켰을 때 어느 구현이 더 빠른가? |
| 동일 품질 목표의 모델·시스템 비교 | 내가 요구하는 품질의 결과를 어느 구성이 더 빨리 제공하는가? |

| 구분 | 반드시 맞출 것 | 달라도 되는 것 | 내릴 수 있는 결론 |
|---|---|---|---|
| 동일 연산 조건의 커널 비교 | 입력값, shape, 저장·계산·누산 dtype, dense 조건, 후처리·출력 반올림, 측정 구간 | 내부 타일·스케줄 | 이 조건의 커널 구현 중 어느 쪽이 빠른가 |
| 동일 품질 목표의 모델·시스템 비교 | 원본 입력·작업, 품질 기준, 요청 패턴, 측정 경계 | 검증된 정밀도 변환·양자화·최적화 | 같은 품질을 만족하는 서비스 중 어느 쪽이 빠른가 |

① 동일 연산 조건의 커널 비교: 계산 조건을 고정한다.  
문서의 Y = ReLU(XW + b)를 두 장치에서 실행한다고 해보자.

- X·W·b의 값과 행렬 크기
- 입력은 BF16, 중간 합은 FP32 같은 수치 형식
- 희소성에 따른 계산 생략 여부
- bias와 ReLU의 계산 정밀도, 최종 출력 형식과 반올림 위치
- 실행시간에 포함하는 구간

이렇게 해야 시간이 다른 이유를 정해진 계산 조건에서의 하드웨어와 구현 차이로 해석할 수 있다.

② 동일 품질 목표의 모델·시스템 비교: 결과의 합격 기준을 고정한다

이번에는 사용자의 요구가 다음과 같다고 해보자.  
> “내 입력을 처리해서, 정해 둔 오차 이내의 결과를 최대한 빨리 돌려줘.”

이 경우 내부 계산이 BF16인지 FP16인지 INT8인지는 달라도 된다. 대신 같은 원본 입력과 공통 품질 기준으로 검사해서 합격해야 한다.


**이번 세 제품을 곧바로 하나의 엄격한 커널 순위로 묶지는 않는다.** H100·v5e는 BF16 입력과 FP32 누산 경로를 비교 후보로 삼을 수 있다. Intel NPU 경로는 FP16 계산을 사용하므로 같은 조건이라고 적지 않는다. 세 제품을 함께 비교할 때에는 우선 같은 원본 입력·오차 기준을 사용하고, 실제 내부 정밀도 차이를 표에 남긴다.

### 6.3 기준 출력과 허용오차를 맞추는 방법

다음은 이 과제에서 제안하는 검증 규칙이며 제조사가 보장하는 오차 범위가 아니다.

아래에서 `Y`는 장치가 계산한 출력, `Y_ref`는 비교 기준 출력, `abs`는 절댓값이다. 오차 부등식의 `1e-2`는 0.01이며, 첫 0.01은 절대오차 허용량, 두 번째 0.01은 기준값 크기에 비례하는 상대오차 허용 비율이다. RMSE(Root Mean Square Error)는 각 원소의 오차를 제곱해 평균한 뒤 제곱근을 취한 값으로, 전체 출력의 오차 크기를 요약한다.

1. CPU에서 원본 FP32 입력을 만들고 동일한 파일을 세 환경에 전달한다. 예를 들어 X는 `[-1, 1]`, W는 `[-1/32, 1/32]`, b는 `[-0.1, 0.1]` 범위로 고정한다.
2. 기준 출력은 이 FP32 값들을 FP64로 올려 계산한 `ReLU(XW+b)`로 만든다. 이는 비교용 수치 기준이고 기준 CPU의 실행시간은 가속기 성능으로 사용하지 않는다.
3. 세 장치 품질 비교에서는 원본 기준에 대해 `abs(Y - Y_ref) <= 1e-2 + 1e-2 * abs(Y_ref)`를 모든 원소에 적용한다. 최대 절대오차·RMSE·위반 원소 비율도 함께 보고한다.
4. 동일 BF16 커널 비교에서는 **먼저 BF16으로 반올림한 동일 입력값**을 기준으로 사용하고, bias·출력 dtype과 반올림 위치도 맞춘다. 원본 FP32 대비 오차와 이 커널 기준 대비 오차를 분리한다.
5. 실패한 결과를 보고 장치마다 다른 허용오차를 적용하지 않는다. 공통 기준을 변경한다면 이유를 적고 모든 장치를 다시 검증한다.

0에 가까운 ReLU 출력은 상대오차만으로 평가하기 어렵기 때문에 절대오차 항을 둔다. BF16과 FP16은 모두 2 Byte지만 지수·가수 구성이 달라 같은 수치 조건이 아니다. INT8을 선택한다면 calibration 데이터·scale·zero point·양자화 방식도 별도로 맞춰야 한다.

이 작은 그래프의 오차 통과는 실제 모델의 정확도 통과와 다르다. 모델·시스템 비교로 확장할 때에는 고정 평가 데이터와 작업별 품질 지표를 추가한다.

---

## 7. 배치 1과 128 — 연산량과 이동량 계산

### 7.1 계산에 사용할 가정

이 절의 F는 GEMM(General Matrix Multiplication, 일반 행렬곱)의 연산 횟수, Q는 입력 읽기와 출력 쓰기를 합한 이동량, I=F/Q는 산술 강도다. M은 배치 크기, K는 입력 특징 수, N은 출력 특징 수이며 K=N=1024다. P는 선택한 행렬 연산의 최대 처리율이다.

- X·W·b·Y의 저장 크기를 모두 원소당 2 Byte로 둔다. 이는 이동량 계산용 가정이며 실제 제품 경로의 dtype을 확정하지 않는다.
- 입력은 이미 장치의 큰 메모리에 있다. 호스트 전송은 이 Q에서 제외한다.
- 한 호출 동안 X·W·b를 각각 한 번 읽고 Y를 한 번 쓴다고 가정한다.
- 행렬곱 중간 합과 bias·ReLU의 중간 결과는 온칩에서 처리하고, 최종 Y만 저장한다.
- 호출을 시작할 때 캐시에서 입력을 공급받지 않는다고 가정한다. 호출 간 캐시 재사용은 별도 조건이다.
- 연산량은 GEMM의 관례적 값 $F=2MKN$으로 센다. FMA는 2 FLOPs다. bias 덧셈 MN개와 ReLU 비교 MN개는 별도 후처리로 표시하며 행렬 피크 P에 합쳐 나누지 않는다.

정확히 곱셈과 덧셈 개수를 세면 GEMM은 $MN(2K-1)$, bias까지 더하면 $2MKN$이다. 여기서는 지난주와 같이 GEMM 자체의 관례적 2MKN 표기를 유지한다. 실제 명령 수와 유효 작업량도 구분한다.

### 7.2 FLOPs와 Byte 계산

$$
F=2M\times1024\times1024
$$

$$
Q=2\underbrace{(M\times1024)}_{X}
+2\underbrace{(1024\times1024)}_{W}
+2\underbrace{1024}_{b}
+2\underbrace{(M\times1024)}_{Y}
$$

F의 앞에 붙은 2는 곱셈과 덧셈을 각각 1 FLOP로 세기 때문이다. 반면 Q의 각 항 앞에 붙은 2는 **원소 하나의 저장 크기가 2 Byte**이기 때문이다. 같은 숫자 2지만 역할과 단위가 다르다.

| 계산 항목 | 배치 1 | 배치 128 |
|---|---:|---:|
| GEMM의 관례적 F | 2,097,152 FLOPs | 268,435,456 FLOPs |
| bias 덧셈 / ReLU 비교 | 각각 1,024회 | 각각 131,072회 |
| X 읽기 | 2,048 Byte | 262,144 Byte |
| W 읽기 | 2,097,152 Byte | 2,097,152 Byte |
| b 읽기 | 2,048 Byte | 2,048 Byte |
| Y 쓰기 | 2,048 Byte | 262,144 Byte |
| 총 Q | **2,103,296 Byte** | **2,623,488 Byte** |
| I=F/Q | **약 0.9971 FLOPs/Byte** | **약 102.3201 FLOPs/Byte** |

배치를 128배로 늘리면 GEMM 작업량도 128배가 된다. 하지만 W를 한 번 가져와 128개 행에 재사용할 수 있다는 가정 때문에 Q는 약 1.247배만 증가한다. 이것이 지난주의 데이터 재사용이 배치 크기와 연결되는 지점이다.

### 7.3 이 Byte가 실제 HBM 이동량일까?

위 Q는 단일 호출의 이상적 데이터 이동 모델이다. 실제로는 타일 사이 재읽기, 패딩, dtype 변환, 별도 후처리 커널 때문에 이동량이 늘 수 있다. 반대로 W가 이전 호출에서 캐시에 남아 있으면 HBM 읽기량은 더 작아질 수 있다.

특히 W의 저장 크기는 2 MiB다. 반복 실행에서는 어떤 메모리에서 W를 공급했는지가 중요한 조건이 된다. ‘가중치가 장치에 있다’와 ‘매번 HBM에서 읽는다’는 같은 말이 아니다.

| 실행 가정 | Q를 바꿔야 하는 이유 |
|---|---|
| bias·ReLU를 하나의 후처리로 분리 | GEMM 중간 출력을 한 번 쓰고 다시 읽음 |
| 중간 출력도 2 Byte인 경우 | 위 경우 추가 이동량은 $4MN$ Byte |
| 중간 출력이 FP32인 경우 | 추가 이동량은 $8MN$ Byte |
| 호스트 입출력 포함 | X·Y 전송량을 별도의 연결 구간에서 계산 |
| 다중 장치 분할 | 장치 간 통신량과 동기화 추가 |

이상적 Q가 실제 트래픽이라고 단정하지 않고, 프로파일로 확인할 항목으로 남긴다.

위 표에서 MN은 중간 출력의 원소 수다. 2 Byte 중간값은 쓰기 2 Byte와 읽기 2 Byte를 합쳐 원소당 4 Byte가 추가된다. FP32 중간값은 각각 4 Byte이므로 원소당 8 Byte가 추가된다.

---

## 8. 가상 장치 계산 — 연산 성능이 커도 느릴 수 있다

### 8.1 실물 제품과 무관한 두 장치

아래는 원리를 설명하기 위해 정한 **가상 조건**이다. A·B는 H100·TPU·Intel NPU 특정지은 것이 아니며, 제공되지 않은 상세 참고자료의 수치를 재현한 것도 아니다. 두 장치는 같은 dense 연산·정밀도를 실행한다고 가정한다.

| 항목 | 가상 A | 가상 B |
|---|---:|---:|
| P | 100 TFLOPs/s | 20 TFLOPs/s |
| BW | 100 GB/s | 400 GB/s |
| 분기점 P/BW | 1,000 FLOPs/Byte | 50 FLOPs/Byte |

$$
t_{\mathrm{compute}}=F/P,\quad
t_{\mathrm{memory}}=Q/BW,\quad
t_{\mathrm{ideal}}=\max(t_{\mathrm{compute}},t_{\mathrm{memory}})
$$

여기서 $t_{\mathrm{compute}}$는 연산 시간, $t_{\mathrm{memory}}$는 데이터 이동 시간, $t_{\mathrm{ideal}}$은 둘 중 큰 값인 시간 하한이다. F·Q는 7절의 작업량과 이동량이고, P·BW는 바로 위 가상 장치 표의 처리율과 대역폭이다. 식의 시간 단위는 초(s)이며 아래 표에서는 $10^6$을 곱해 마이크로초(μs)로 표시한다. A·B는 장치 이름이다.

### 8.2 직접 계산한 시간 하한

| 배치 | 장치 | 연산 시간 | 메모리 시간 | 이상적 시간 | 이 모델의 제한 요인 |
|---|---|---:|---:|---:|---|
| 1 | A | 0.02097 μs | 21.03296 μs | **21.03296 μs** | 메모리 |
| 1 | B | 0.10486 μs | 5.25824 μs | **5.25824 μs** | 메모리 |
| 128 | A | 2.68435 μs | 26.23488 μs | **26.23488 μs** | 메모리 |
| 128 | B | 13.42177 μs | 6.55872 μs | **13.42177 μs** | 연산 |

A의 P는 B의 5배지만 배치 1에서는 B의 시간 하한이 4배 유리하다. 배치 128에서도 B의 시간 하한이 약 1.95배 유리하다. 데이터를 공급하는 속도를 함께 봐야 하기 때문이다.

배치 128의 B에서는 I≈102.32가 분기점 50을 넘는다. 이 상태에서 Q만 더 줄이면 이미 더 큰 $F/P$는 줄어들지 않는다. 지난주의 Roofline 수평 구간과 같은 해석이다.

### 8.3 계산을 재현하는 코드

```python
# 표준 Python만 필요하다. 실제 장치 벤치마크가 아니다.
devices = {"A": (100e12, 100e9), "B": (20e12, 400e9)}
for batch in (1, 128):
    flops = 2 * batch * 1024 * 1024
    traffic = 2 * (batch * 1024 + 1024 * 1024 + 1024 + batch * 1024)
    print(f"M={batch}, F={flops}, Q={traffic}, I={flops / traffic:.4f}")
    for name, (peak, bandwidth) in devices.items():
        compute = flops / peak
        memory = traffic / bandwidth
        print(name, "compute_us=", compute * 1e6,
              "memory_us=", memory * 1e6,
              "ideal_us=", max(compute, memory) * 1e6)
```

표의 계산은 이 식으로 검산했다. 컴파일·명령 발행·후처리·동기화·전송 비용은 넣지 않았으므로 이 시간을 실제 사용자 지연시간으로 제시하지 않는다.

---

## 9. 같은 작업의 제품별 비교 조건표

### 9.1 문헌 조사로 채운 실행 계획

아래 표는 **문헌으로 확인한 경로와 제안한 비교 조건**을 함께 정리한 것이다. 각 칸의 설정이 실제 장치에서 검증되었다는 뜻은 아니다. 기준 출력과 품질 조건은 6절, 배치별 기록 단위는 10절을 따른다.

| 항목 | H100 SXM 1개 | TPU v5e 1칩 | 288V NPU 1개 |
|---|---|---|---|
| 실행 경로 | 프레임워크→CUDA 라이브러리, 또는 Triton | JAX→JIT→XLA, 필요시 Pallas | 모델 변환→OpenVINO 2026 NPU 컴파일→추론 |
| X·W 저장/입력 후보 | BF16 | BF16 | FP16 모델 입력 후보 |
| b·Y 저장 후보 | BF16로 맞추되 반올림 위치 명시 | 왼쪽과 동일하게 명시 | FP16; 실제 입출력과 변환 확인 |
| 곱셈 계산 후보 | BF16 Tensor Core | BF16 MXU | FP16 하드웨어 경로 |
| 누산 조건 | FP32로 명시하는 구현 선택, 실제 결과 확인 | FP32 MXU 누산 경로, lowering 확인 | 이번 조사에서 미확인; FP32라고 가정하지 않음 |
| 정수 양자화·희소성 | 사용하지 않는 dense 조건 | 동일 | 양자화·희소 최적화 설정과 적용 여부 확인 |
| shape | M=1·128 별도 정적 입력 | M=1·128 별도 컴파일 조건 | M=1·128 별도 정적 모델 컴파일 |
| 장치에 유지할 데이터 | W·b·X, 출력 Y 버퍼 | W·b·X, 출력 Y 버퍼 | W·b의 컴파일/런타임 배치와 X·Y 버퍼 확인 |
| 호스트 작업 | 입력 준비·실행 요청·결과 회수 | 동일 | 동일, 변환·버퍼 복사 여부 추가 확인 |
| 다른 장치의 계산 | 없는 것을 목표로 확인 | sharding 없이 1칩 사용 확인 | CPU fallback·HETERO 여부 확인 |
| 엄격한 커널 비교 | BF16/FP32 조건의 후보 | BF16/FP32 조건의 후보 | 위 두 열과 동일 조건이라는 근거 부족 |
| 동일 품질 비교 | 공통 원본 기준 오차 검사 | 동일 | 동일, FP16 경로임을 병기 |
| 출처 | [CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)·[Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | [TPU 구조](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)·[JAX 정밀도 설정](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html) | [OpenVINO NPU](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) |

H100·v5e도 BF16이라는 이름만 같다고 비교가 끝나지 않는다. 누산 결과를 언제 BF16으로 반올림하는지, bias·ReLU를 어느 정밀도로 실행하는지, 출력이 같은 의미인지 확인해야 한다. 맞추지 못한 경우에는 품질 목표 비교로 분류한다.

### 9.2 소프트웨어 버전 기록표

문서 버전과 실행 버전은 다르다. 지금 확인하지 않은 설치 버전에 임의의 숫자를 넣지 않는다.

| 대상 | 이번 문헌 기준 | 실행할 때 반드시 채울 항목 | 현재 상태 |
|---|---|---|---|
| H100 | 확인일의 CUDA Programming Guide·Triton 문서 | OS, GPU driver, CUDA runtime/toolkit, 프레임워크·cuBLAS·Triton 버전, 설정 | 실행 버전 미확인 |
| v5e | 확인일의 Cloud TPU·JAX 문서 | VM 이미지, JAX, jaxlib, libtpu, 컴파일 옵션, 장치 수 | 실행 버전 미확인 |
| 288V NPU | OpenVINO **2026** 문서 | 정확한 OpenVINO 빌드, NPU driver, compiler 종류·버전, OS, 전원 모드 | 실행 버전 미확인 |

### 9.3 공개 사실과 실행 확인 항목을 분리하기

| 문헌·계산으로 확인한 것 | 실행해야 확인할 것 |
|---|---|
| 제품·세대·사양의 집계 범위 | 실제 배정 장치, 분할·공유 여부, 클록·전력 설정 |
| 지원 dtype·컴파일 경로의 설명 | 이 그래프의 전체 컴파일·추론 성공 |
| 공통 shape의 논리적 FLOPs | 패딩·변환·분할로 늘어난 실제 작업 |
| 이상적인 최소 이동 모델 | 실제 HBM·캐시·시스템 메모리 트래픽 |
| 비교할 오차 규칙 | 기준 출력 대비 수치 오차·품질 통과 |
| 기록할 시간의 정의 | 컴파일·p50/p95·처리량·에너지 |

현재 내릴 수 있는 결론은 **사양표의 P만으로 이 작업의 실측 순위를 정할 수 없다**는 것이다. 세 장치의 실제 속도 순위, 정확도 통과, 전력 효율은 아직 결론 내리지 않는다.

---

## 10. 실제로 빠르다는 뜻 — 시간과 처리량과 에너지

### 10.1 시계의 시작과 끝부터 정하기

```text
모델 로드 → 변환·컴파일 → 준비 실행(warm-up)
                              ↓
요청 도착 → 대기 → 입력 준비·전송 → 장치 계산 → 출력 전송·후처리 → 응답
                                  └─ 장치 연산 시간 ─┘
            └──────────── 사용자 요청의 전체 시간 ─────────────┘
```

위 시간들은 측정 범위를 설명하는 구간이다. 비동기 전송과 계산이 겹치면 각 시간의 합이 전체 벽시계 시간과 같지 않을 수 있다.

| 측정 이름 | 포함할 것 | 별도로 기록할 것 |
|---|---|---|
| Cold start | 로드·변환·컴파일·첫 추론 등 정의한 초기화 구간 | 컴파일 캐시의 유무 |
| 장치 연산 시간 | 정한 장치 커널/그래프의 시작부터 완료 | 호스트 전송 제외 여부, 후처리 포함 여부 |
| Warm 호출 지연시간 | 준비된 입력으로 호출해 완료를 기다리는 시간 | 호스트 dispatch·대기 포함, 장치 시간과 구분 |
| End-to-end | 입력 요청부터 사용할 수 있는 출력까지 | 입력 준비·전송·후처리·네트워크 범위 |

JAX는 비동기 실행을 하므로 반환 시점만 재면 완료시간보다 짧게 나올 수 있다. `block_until_ready()`로 완료를 기다리고, 첫 JIT 호출은 측정 반복과 분리한다. 동기화를 넣은 호스트 타이머도 순수 장치 커널 타이머와는 다르다. [JAX JIT — timing and warm-up](https://docs.jax.dev/en/latest/jit-compilation.html#jit-compiling-a-function)

CUDA에서는 장치 이벤트·완료 동기화 등 측정 도구의 범위를 확인한다. OpenVINO에서도 동기 infer 완료와 비동기 요청 제출 시점을 구분한다. 세 경로에 같은 Python 타이머를 쓴다는 것만으로 측정 경계가 같아지지 않는다.

### 10.2 배치별 지연시간·처리량 기록 단위

이 과제에서는 **한 호출이 배치 M개 샘플을 처리하는 요청 하나**라고 정한다. R회 호출을 전체 T초 동안 완료했다면 다음과 같다.

$$
\mathrm{requests/s}=\frac{R}{T},\qquad
\mathrm{samples/s}=\frac{RM}{T}
$$

| 기호·표기 | 뜻 | 단위 |
|---|---|---|
| R | 측정 구간에서 완료한 호출, 즉 요청의 총개수 | 요청 수 |
| M | 호출 하나가 처리하는 샘플 수, 즉 배치 크기 | 샘플/요청 |
| T | 그 요청들을 완료하는 데 걸린 전체 벽시계 시간 | s |
| RM | 완료한 샘플의 총개수, R×M | 샘플 수 |
| requests/s | 초당 완료 요청 수 | 요청/s |
| samples/s | 초당 완료 샘플 수 | 샘플/s |

예를 들어 배치 128인 요청 100개를 2초 동안 완료했다면 요청 처리량은 100/2=50 요청/s이고, 샘플 처리량은 100×128/2=6,400 샘플/s다. 여기서 R은 **요청 개수**이며, 지난주에 처리율 기호로 사용한 R과 다른 정의다.

| 항목 | 배치 1 | 배치 128 |
|---|---|---|
| 지연시간 | μs/호출 또는 ms/호출, p50·p95 | 동일; 128개 결과가 완료되는 시간 |
| 처리량 | 요청/s와 샘플/s가 같음 | 샘플/s는 요청/s의 128배 |
| 평균 샘플당 비용 | 호출 시간/1 | 호출 시간/128 |
| 사용자 관점 주의 | 요청 대기·전송 포함 여부 | 시간/128은 한 사용자의 응답 대기시간이 아님 |

동시 요청이 여러 개라면 평균 지연시간의 역수를 그대로 전체 처리량으로 쓰지 않는다. 실제 완료 샘플 수와 벽시계 시간으로 계산한다. 실시간 입력 128개를 모아 배치를 만든다면 **배치가 찰 때까지의 대기시간**도 사용자 지연시간에 포함한다.

p50·p95는 측정한 지연시간의 백분위수다. p50은 가운데 값인 중앙값, p95는 요청의 약 95%가 그 시간 이내에 완료된다는 뜻이다. 평균값 하나로 드러나지 않는 느린 요청을 함께 살펴보기 위해 기록한다.

실습의 제안 조건은 배치별 별도 컴파일, warm-up 10회, 순차 호출 100회다. 처리량용 비동기 실행은 별도 실험으로 두고 동시 요청 수를 적는다. 이는 스터디용 프로토콜이며 MLPerf 규칙을 대체하지 않는다.

### 10.3 MLPerf 결과표는 무엇을 함께 기록할까?

MLPerf의 Closed는 기준 모델을 유지하는 비교이고, Open은 모델 변경·재학습 등의 여지가 있다. 결과에는 소프트웨어, 시스템, 가속기 종류·수, 상세 설정과 코드가 연결된다. 서로 다른 모델·품질 목표·division·버전의 행을 바로 비교하지 않는다. [MLPerf Datacenter — Divisions / Submission Information](https://mlcommons.org/benchmarks/inference-datacenter/)

| 시나리오 | 요청 방식과 읽을 지표 |
|---|---|
| Offline | 준비된 작업 묶음을 처리하는 처리량 중심 |
| Server | 도착하는 요청을 지연 제약 안에서 처리하는 처리량 중심 |

Server의 요청 생성은 Poisson 도착을 사용하며 모델별 지연 조건을 확인해야 한다. 본 과제의 ‘동기 호출 100회’나 ‘배치 128’은 MLPerf Server/Offline 실행과 동의어가 아니다. 정확한 요구사항은 선택한 벤치마크 릴리스의 규칙으로 고정한다. [MLPerf Inference Rules — Scenarios](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)

### 10.4 LLM으로 확장한다면

TTFT는 요청부터 첫 출력 토큰까지의 시간이다. 전체 출력 토큰 처리량은 여러 요청의 출력 토큰 합계를 측정 구간으로 나눈 값이다. 동시 요청을 늘려 전체 tokens/s가 높아져도 한 사용자의 TTFT나 완료시간은 나빠질 수 있다. 입력·출력 길이와 동시성도 맞춘다. [NVIDIA NIM Benchmarking — Metrics](https://docs.nvidia.com/nim/benchmarking/llm/metrics)

### 10.5 TDP와 작업당 에너지는 다르다

$$
E_{\mathrm{workload}}=\int_{t_0}^{t_1}p_{\mathrm{power}}(t)\,dt,\qquad
E_{\mathrm{sample}}=\frac{E_{\mathrm{workload}}}{\text{완료 샘플 수}}
$$

| 기호 | 뜻 | 단위 |
|---|---|---|
| $E_{\mathrm{workload}}$ | 정한 측정 구간에서 작업을 실행하며 소비한 총에너지 | J, 줄 |
| $E_{\mathrm{sample}}$ | 완료 샘플 하나당 소비한 평균 에너지 | J/샘플 |
| $p_{\mathrm{power}}(t)$ | 시간 t에서 측정한 소비전력; 앞의 연산 처리율 P와 다른 값 | W, 와트 = J/s |
| $t_0$, $t_1$ | 측정 시작 시각과 종료 시각 | s |
| $dt$ | 에너지를 합산할 때 사용하는 아주 작은 시간 간격 | s |
| $\int_{t_0}^{t_1}$ | 시작부터 끝까지 ‘전력×짧은 시간’을 모두 더한다는 적분 기호 | 결과는 J |

즉, 시간에 따라 달라지는 소비전력을 누적하면 에너지가 된다. 해당 구간의 **평균 전력**을 알고 있다면 ‘평균 전력 × 측정 시간’으로 같은 에너지를 구할 수 있다. TDP(Thermal Design Power)는 열 설계를 위한 전력 사양이다.

TDP는 이 적분에 넣을 실제 소비전력 측정값이 아니다. 칩 전력과 전체 시스템 벽전력을 비교할 때도 측정 범위를 맞춘다. MLPerf의 전력 측정은 해당 벤치마크 실행 중 전체 시스템의 AC 전력을 대상으로 하며 사양표 TDP를 검증한 결과가 아니다. [MLPerf — Power Measurements](https://mlcommons.org/benchmarks/inference-datacenter/)

이번 문서의 전력 효율은 **아직 미측정**이다. 필요하면 평균 W, 측정 시간 s, 총 J, J/샘플을 기록하고 idle 전력을 뺐는지도 명시한다.


---

## 11. 자주 생기는 오해와 문제 찾기

### 11.1 자주 생기는 오해

| 오해 | 더 정확한 설명 |
|---|---|
| GPU는 범용이고 NPU는 모두 같은 구조다 | H100·v5e·Lunar Lake NPU의 구체적인 구조와 소프트웨어를 구분한다. |
| Tensor Core 개수로 GPU와 TPU를 비교할 수 있다 | 두 이름은 가리키는 구성 단위가 다르다. |
| INT8 48 TOPS는 BF16 48 TFLOPs/s와 같다 | 연산·수치 표현·집계 조건이 다르다. |
| H100의 BF16 1,979는 dense 행렬에도 적용된다 | 해당 제품표의 별표는 sparsity 조건이다. |
| 288V NPU는 120 TOPS다 | NPU 단독 값과 전체 프로세서의 표기를 구분한다. |
| 메모리 32 GB를 모두 NPU가 전용으로 쓴다 | 시스템 메모리와 NPU 전용 저장 공간은 다르다. |
| FP32 입력이면 내부 곱셈도 FP32다 | 컴파일된 계산 정밀도와 누산은 별도로 확인한다. |
| NPU는 동적 shape를 전혀 지원하지 않는다 | OpenVINO 2026의 제한적인 preview 지원 조건을 확인한다. |
| API 호출이 성공하면 모든 연산이 NPU에서 실행됐다 | 실행 장치와 분할·fallback을 확인한다. |
| 배치 128의 시간을 128로 나누면 응답 지연시간이다 | 평균 샘플당 비용과 실제 대기시간은 다르다. |
| FLOPs/s가 높으면 전력 효율도 좋다 | 같은 품질·범위의 실제 에너지가 필요하다. |
| 공개되지 않은 값은 적당한 추정치로 채우면 된다 | 미확인·미측정과 계산 가정을 분리한다. |

### 11.2 계산과 측정이 다르면 어디부터 볼까?

1. 실제 제품과 장치 수, 분할 여부가 사양표와 같은지 확인한다.
2. dense/sparse·dtype·누산·출력 반올림이 맞는지 확인한다.
3. 기준 출력과 허용오차를 통과했는지 확인한다.
4. 타이머가 완료까지 기다렸는지, 컴파일·전송·후처리를 어디까지 포함했는지 확인한다.
5. 작은 작업의 실행 비용, 캐시 재사용, 패딩·변환·fallback을 조사한다.

실측이 가상 Roofline 시간보다 짧다면 먼저 Q의 메모리 경계와 캐시 가정을 점검한다. 길다면 실행 비용·실효 대역폭·작업 병렬성 등 모델에서 빠진 비용을 살펴본다.

---

## 12. 연습문제 3개

### 문제 1. 성능표에서 비교 조건 찾기

어떤 발표자가 ‘H100은 BF16 1,979 TFLOPs/s이고 288V는 120 TOPS이므로 NPU보다 약 16.5배 빠르다’고 말했다.

1. 잘못 맞춘 조건을 세 가지 이상 찾아보자.
2. 두 장치의 이 공통 작업 속도를 비교하려면 무엇을 추가로 확인해야 할까?

### 문제 2. 배치와 데이터 재사용

7절의 2 Byte 저장·입력 1회 읽기·출력 1회 쓰기 조건을 사용한다.

1. 배치 1·128의 F, Q, I를 구하자.
2. 배치를 128배 늘려도 Q가 128배가 되지 않는 이유는 무엇일까?
3. 배치 128의 bias·ReLU를 GEMM에서 분리하고 2 Byte 중간 출력을 한 번 저장·재읽기하면 Q와 I는 어떻게 바뀔까?

### 문제 3. 사용자 지연시간과 처리량

다음은 실제 제품과 무관한 가상 측정이다. 두 구성의 품질 기준은 같고, 순차 실행하며 별도 대기·전송 비용은 없다고 가정한다.

| 구성 | 배치 | 한 호출 완료시간 |
|---|---:|---:|
| C | 1 | 0.10 ms |
| D | 128 | 2.00 ms |

1. 각 구성의 요청/s와 샘플/s는 얼마인가?
2. D에서 한 샘플의 결과를 받는 시간은 2/128 ms인가?
3. D의 해당 작업 구간 평균 시스템 전력이 20 W라면 배치당·샘플당 에너지는 얼마인가? 20 W가 TDP 표기뿐이라면 같은 계산을 실측 에너지로 보고할 수 있을까?

---

## 13. 해답과 해설

### 13.1 문제 1 해설

1,979는 sparse BF16 Tensor Core 수치다. 120은 288V의 전체 INT8 TOPS이며 NPU 단독 값이 아니다. 연산 정밀도, 희소성, 집계 범위가 모두 어긋난다. 이를 나눈 값은 공통 작업의 속도비가 아니다.

먼저 제품·장치 수·정밀도·품질·배치를 정하고, 같은 그래프를 어떤 소프트웨어 버전으로 실행하는지 확인한다. 이후 누산·fallback·전송·측정 구간을 확인하고 실제 완료시간을 측정해야 한다. NPU 단독 48을 대입해 다시 나누는 것만으로도 문제가 해결되지 않는다.

### 13.2 문제 2 해설

배치 1은 F=2,097,152 FLOPs, Q=2,103,296 Byte, I≈0.9971이다. 배치 128은 F=268,435,456 FLOPs, Q=2,623,488 Byte, I≈102.3201이다.

W의 크기는 배치와 무관하며 한 호출에서 여러 입력 행에 재사용된다. 배치에 비례하는 것은 X·Y의 크기다.

별도 후처리가 생기면 중간 출력 쓰기·읽기로 다음이 추가된다.

여기서 $\Delta Q$는 추가 이동량, $Q'=Q+\Delta Q$는 변경 후 총이동량, $I'=F/Q'$는 변경 후 산술 강도다. Δ는 ‘변화량’을, 작은 따옴표 모양의 prime(′)은 ‘변경 후 값’을 표시한다. 앞의 4는 원소당 쓰기 2 Byte와 읽기 2 Byte의 합이다.

$$
\Delta Q=4\times128\times1024=524,288\;\mathrm{Byte}
$$

$$
Q'=3,147,776\;\mathrm{Byte},\qquad
I'=\frac{268,435,456}{3,147,776}\approx85.2778\;\mathrm{FLOPs/Byte}
$$

논리적 GEMM FLOPs는 같지만 이동량이 증가해 산술 강도가 낮아진다. 실제 중간값이 캐시에 남으면 HBM에서 이 이동이 전부 발생하지 않을 수 있다는 점도 기록한다.

### 13.3 문제 3 해설

| 구성 | 요청/s | 샘플/s | 한 호출에 들어간 샘플의 완료시간 |
|---|---:|---:|---:|
| C | 10,000 | 10,000 | 0.10 ms |
| D | 500 | 64,000 | 2.00 ms |

D의 평균 샘플당 처리 비용은 0.015625 ms이지만, 그 호출의 결과를 받는 시간은 2 ms다. 실제 서비스에서는 배치 형성 대기가 더해질 수도 있다.

아래에서 $E_{\mathrm{batch}}$는 배치 하나의 에너지, $E_{\mathrm{sample}}$은 샘플 하나의 평균 에너지다. 20은 평균 전력(W), 0.002는 호출 완료시간(s), 128은 배치의 샘플 수다.

$$
E_{\mathrm{batch}}=20\times0.002=0.04\;\mathrm{J},\qquad
E_{\mathrm{sample}}=0.04/128=0.0003125\;\mathrm{J}
$$

따라서 샘플당 0.3125 mJ다. 이는 해당 측정 구간의 평균 시스템 전력이 실제로 20 W라는 가정에서만 성립한다. TDP 20 W만 알면 실제 에너지는 아직 미측정이다.

## 14. 공식 자료와 추가 읽기

### 14.1 공통으로 읽을 자료

| 자료 | 읽을 범위 | 확인할 질문 |
|---|---|---|
| [Scaling Book 12장 — How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/#gpus-vs-tpus-at-the-chip-level) | GPUs vs. TPUs at the chip level, 필요시 What Is a GPU?·Memory | 실행 제어와 온칩 재사용이 어떻게 다른가? |
| [OpenVINO 2026 — NPU Device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) | 컴파일 예, Supported Inference Data Types, Dynamic Shapes, Limitations | 모델·shape·컴파일러·드라이버가 실행 가능성을 어떻게 결정하는가? |
| [MLPerf Inference — Datacenter](https://mlcommons.org/benchmarks/inference-datacenter/) | Scenarios & Metrics, Divisions, Submission Information | 결과와 함께 모델·품질·장치 수·소프트웨어를 적는 이유는 무엇인가? |

### 14.2 구조와 사양 보충 자료

| 자료 | 읽을 범위·주의점 |
|---|---|
| [Scaling Book 2장 — What Is a TPU?](https://jax-ml.github.io/scaling-book/tpus/#what-is-a-tpu) | MXU·VPU·메모리·제어, 설명 대상 세대 구분 |
| [Google TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e) | 칩·호스트·Pod, BF16 성능, HBM·ICI의 단위 |
| [Google TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) | TensorCore와 MXU, BF16 곱셈·FP32 누산 |
| [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) | SXM 열과 희소성 각주, 메모리와 연결 구분 |
| [Intel Core Ultra 9 288V](https://www.intel.com/content/www/us/en/products/sku/240961/intel-core-ultra-9-processor-288v-12m-cache-up-to-5-10-ghz/specifications.html) | NPU와 Overall TOPS, 시스템 메모리, 프로세서 전력 |
| [Intel Lunar Lake AI Hardware Accelerators](https://cdrdv2-public.intel.com/824436/2024_Intel_Tech%20Tour%20TW_Lunar%20Lake%20AI%20Hardware%20Accelerators.pdf) | NPU 4 아키텍처: MAC·DSP·DMA·scratchpad |

### 14.3 프로그래밍과 측정 보충 자료

| 자료 | 읽을 범위·확인할 것 |
|---|---|
| [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) | 호스트·장치·커널·스레드 계층 |
| [Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | 입력 타일·누산·출력 후처리 |
| [JAX JIT compilation](https://docs.jax.dev/en/latest/jit-compilation.html#jit-compiling-a-function) | 추적·컴파일·재사용·완료 대기 |
| [JAX matmul precision](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html) | 입력 dtype과 계산 알고리즘의 구분 |
| [Pallas TPU kernels](https://docs.jax.dev/en/latest/pallas/tpu/details.html) | HBM·VMEM 블록 배치 |
| [MLPerf Inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) | 시나리오·품질·지연 조건; 실제 비교에서는 릴리스 고정 |
| [NVIDIA LLM metrics](https://docs.nvidia.com/nim/benchmarking/llm/metrics) | TTFT·전체 출력 토큰 처리량·요청 지연시간 |
| [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) | 무료 자원과 할당·사용 제한 |
