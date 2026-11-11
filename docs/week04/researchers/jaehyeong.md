<WeekNavBar :week="4" topic="Roofline과 Tiling" :researchers="[{'id': 'mbsong', 'name': 'mbsong', 'label': 'mbsong (송무복)', 'badge': 'Roofline & 타일 크기별 분석'}, {'id': 'singon', 'name': 'singon', 'label': 'singon (김신곤)', 'badge': '4×4 & 4096×4096 완전 해설'}, {'id': 'junpark', 'name': 'junpark', 'label': 'junpark (박준)', 'badge': '2D Block Tiling 직관'}, {'id': 'injokim', 'name': 'injokim', 'label': 'injokim (김인조)', 'badge': '이론 유도 & 연습문제 풀이'}, {'id': 'jaehyeong', 'name': 'jaehyeong', 'label': 'jaehyeong (이재형)', 'badge': 'Roofline & Tiling 핵심 노트'}]" />

> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **jaehyeong (이재형)** (`Roofline & Tiling 핵심 노트`)
>
> 💡 **핵심 기여 요약**: Compute-bound와 Bandwidth-bound의 경계 조건 및 타일 행렬곱에서 입력 조각이 재사용되는 횟수를 직관적으로 요약했습니다.

---

# 4주차: Roofline과 Tiling

[참고사항](https://app.notion.com/p/3f15e414b6c280c7ad27d0b239c5e050?pvs=21)

- https://youtu.be/9Qf_QLt7IWY
- https://dan.naver.com/data/deview/session/attach/LLM%E1%84%80%E1%85%AAGPU%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A5%E1%86%A8%E1%84%92%E1%85%AA%E1%84%8B%E1%85%B4%E1%84%86%E1%85%A9%E1%84%83%E1%85%B3%E1%86%AB%E1%84%80%E1%85%A5%E1%86%BA.pdf(LLM과 GPU 최적화의 모든 것, 2024, NAVER)

---

### &lt; 이번 주차 핵심 &gt;

1. **Arithmetic Intensity(AI)** = FLOPs / Byte
2. **Performance(Pref)** = min(Peak_FLOPs, BW x AI)
3. **criticla_AI** = Peak_FLOPs / BW
4. **AI < Critical_AI** → Memory bound
5. **AI > Critical_AI** → Compute bound
6. **Tiling** = 데이터를 HW accelerator memory에 올려 재사용해서 HBM traffic 감소
7. Tiling 전후 연산량(FLOPs)은 같고, 메모리량(Byte)가 줄어 AI 증가
8. Tile 증가 → reuse 증가, 하지만 register/ shared memory 사용 증가 → occupancy 감소 가능

---

## 1. Arithmetic Intensity(산술 강도)

- **목적**: 연산/커널이 Compute-bound 인지, Communication-bound 인지 판단하기 위해 AI라는 지표를 도입한다.
- **정의**: 알고리즘의 `Arithmetic Intensity(AI)`는 알고리즘 처리에 소요된 total FLOPs를 알고리즘이 HBM과 통신(R/W)해야하는 Byte수로 나눈 값이다.
    
    > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
    
    다시말해, 알고리즘 연산에 필요한 Byte당 실행되는 FLOPs 연산 수이다.
    
- **값의 의미:**
    
    `AI 높다` → 계산 횟수에 비해 메모리 접근이 적음. 데이터 한번 가져오면 많은 연산 수행 가능.
    
    `AI 낮다` → 계산 횟수에 비해 메모리 접근이 많음. 데이터 가져올 때 마다 적은 연산 수행, **메모리에 여러번 접근해야.**
    

### 예시) Vector Add와 MatMul 데이터 재사용 차이

- **Vector Add**
    
    Vector Add는 N개의 원소에 대해,
    
    > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
    
    로 계산된다. 
    
    FP32 기준, A,B 를 각각 4N Byte읽어오고, C를 4N byte 쓰므로, 총 메모리 이동량은 **12N Byte**이다.
    
    덧셈은 원소당 1번 이기 때문에 **N FLOPs**이다.
    
    따라서 Vector Add의 Arithmetic Intensity는
    
    **Arithmetic Intensity** = N(FLOPs)/12N(Bytes) = **0.083** FLOPs/Byte
    
- **MatMul**
    
    FP32기준, **D(N,N) = A(N,N) X B(N,N)** 행렬 연산을 생각해보자.
    
    > 🖼️ *[도식 참고: figure 1]*
    
    figure 1
    
    총 메모리량의 경우, A와 B matrix `2N^2 * 4Byte` 만큼을 읽어오고, 계산 결과인 C matrix `N^2 * 4Byte`를 쓴다. 따라서
    
    **메모리량 = 3N^2 * 4Byte = 12N^2 Byte**
    
    이다.
    
    총 연산량의 경우, C의 각 element마다 
    
    > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
    
    이런 N번의 곱셈과 N번의 덧셈으로 총 2N번의 연산이 발생한다. C의 크기가 N^2 이므로,
    
    **연산량 = 2N^3 FLOPs**
    
    따라서, 
    
    > 🖼️ *[도식 참고: 원본 노션 첨부 도식]*
    
    이 된다.
    
- **결론**
    
    **AI_vecadd = 0.083 (FLOPs/Byte)
    AI_matmul = N/6 (FLOPs/Byte) = 42.67 (if N=256)**
    

## 2. **Roofline — 어느 자원이 성능을 제한할까?**

- **목적**
    
    특정 연산/커널이 현재 HW에서 Memory BW에 제한되는지, Compute 성능에 제한되는지 판단하기 위해 roofline model을 사용한다.
    

### Roofline Plot 보는 법

> 🖼️ *[도식 참고: figure 2]*

figure 2

**&lt; Plot 설명 &gt;**

**상황**: 동일한 Peak FLOPs를 가지는 가속기를 memory BW만 다르게 한 상황에서, 서로 다른 산술 강도(Arithmetic Intensity)를 가진 Algo1과 Algo2의 성능(FLOPs/s)을 비교

**plot 구성 요소 설명**

- **x축**: Arithmetic Intensity, `FLOPs/Byte` , 1B당 얼마나 많은 연산을 수행하는지 
$\boxed{I=\frac{\text{총 FLOPs} }{\text{HBM에서 읽고 쓴 총 Byte} }}$
- **y축**: Performance, `FLOPs/s` , 해당 AI에서 HW가 낼 수 있는 연산 처리율. GPU의 최대 성능 P와, BW x I 중 더 작은 값으로 제한된다.
 $\boxed{Performance=\min(P,\ BW\times I)}$
- **Critical AI(Ridge Point)**: $\boxed{I_{critical}=\frac{P}{BW} }$ 알고리즘 총 FLOPs와 필요한 data량이 HW 한계에 딱 맞아 떨어지는 경우.

**figure 2 설명**

- **`red area`**: **Memory-bound**. Algorithm의 Arithmetic Intensity가 너무 낮아서 HW1/2의 Peak Performance에 도달하지 못한다. 이 구간에서 AI를 높이면 FLOPs/s가 선형적으로 올라간다.
- **`yellow area`**: HW의 BW에 따라 알고리즘 처리 성능이 갈리는 구간. 낮은 memory BW1을 갖는 HW에서 Algorithm은 여전히 memory-bound에 부딪혀 HW 최고 성능에 도달하지 못한다. 하지만, 높은 BW2를 사용하는 HW는 Algorithm을 Peak 성능으로 처리할 수 있는 구간.
- **`green area`**: **Compute bound**. Algorithm의 Arithmetic Intensity나 HW의 BW를 아무리 높이더라도 이미 HW의 Peak FLOPs를 사용중이기 때문에 이득이 없는 구간.

**&lt; Algo1, Algo2 설명 &gt;**

- Algo1 처럼 알고리즘의 메모리 재사용성(AI)이 안좋아 AI가 낮으면, 데이터 공급 시간이 병목이 되어 HW의 연산기는 대기 시간이 생기고, 성능(FLOPs/s)이 제대로 안나오게 된다.
- Algo2 처럼 알고리즘의 메모리 재사용성(AI)이 좋아 AI가 임계점 이상으로 높으면, 메모리 공급보단 GPU의 최대 연산 성능이 병목이 된다.

**&lt; 왜 Memory Bandwidth 한계는 기울어진 선이고, Compute 한계는 수평선일까? &gt;**

Memory BW 한계에 걸리는 경우, 알고리즘의 Arithmetic Intensity(메모리 재사용성)을 높이면 남아돌고있던 HW의 연산장치들을 더욱 활용 가능해져, Peak FLOPs/s 까지 AI에 비례해 선형적으로 증가하는, 기울어진 선이 나오게 된다.

반면, Compute 한계에 부딪힌 경우 아무리 Byte당 FLOPs 처리량을 늘린다 한들 더 많은 연산을 처리할 수 없기 때문에 saturation되는 수평선이 나오게 된다.

### 실제 Kernel을 plot 해보면

> 🖼️ *[도식 참고: roofline-analysis.png]*

- **왼쪽 아래 파란 점인 Achieved Value**는, 실제 profiling 된 kernel의 측정값(AI/Perf)을 의미한다.
- **`takeaway1`**: 이론적인 roofline 위에 실제 kerenl의 측정값을 찍어서 kernel이 현재 어느 bottleneck 영역(mem, compute)에 있는지 알 수 있다.
- **`takeaway2`**: achieved value와 해당 roofline boundary 사이의 거리(흰색 점선)는 성능 개선 가능성을 나타낸다. achieved value가 roofline boundary에 가까울수록 성능은 더 최적화된 상태다. 커널의 성능을 높이려면 메모리 재사용성(AI)을 높여야 한다.

## 3. Tiling — 작은 조각으로 나누면 왜 다시 쓸 수 있을까?

https://www.youtube.com/watch?v=Q3GgbfGTnVc

*https://developer.nvidia.com/blog/how-to-write-high-performance-matrix-multiply-in-nvidia-cuda-tile/*

#### **&lt; 핵심 &gt;**

MatMul의 연산 특징상, 같은 입력 데이터를 계산에 **재사용이 가능하다**
→ 큰 행렬을 작은 조각(tile)으로 나누어 공유 메모리에 놓고 계산 
→ HBM Read 감소 
→ 산술 강도(AI) 증가 
→ 커널 실행 성능(FLOPs/s) 증가

### Tiling

Q. Local tiling이랑 shared memory tiling은 각각 어떤 동작의 차이가 있고, 각 장단점? 

**용어 정리**

- `Thread`: GPU에서 output 타일의 특정 원소나 일부를 연산하는 가장 작은 실행 단위. 각 thread들은 자기만의 독립된 register을 가진다.
- `warp`: 32개 thread 묶음. NVIDIA GPU에서 보통 같은 instruction을 32개 thread가 동시에 실행한다.
- `thread block`: 여러 thread/warp 그룹. 같은 block의 thread들은 shared memory(on-chip 자원)를 공유한다.

**HW 구조 개념 복기**

> 🖼️ *[도식 참고: 원본 노션 첨부 도식]*

- `register`: thread별 전용 작은 메모리
- `shared memory`: block 별 공용 메모리
- `HBM`/`Global memory`: 가장 큰 범주의 전체 공용 메모리

아래 행렬곱을 연산할 때, C의 한 element인 c_ij를 연산하기 위해선

$$
C = A @ B
$$

**`(a_i1 * b_i1) + (a_i2 * b_i2) + … + (a_ij * b_ij)`** 연산이 필요하다.

이를 모든 C의 원소에 대해서 반복한다면, 연산을 위해 접근하는 A, B elements에 많은 중복이 발생하게된다.

A, B는 용량은 크지만 접근 속도가 느린 Global Memory에 저장되어있기 때문에, 이 부분에 memory 병목이 발생하게된다.

> 🖼️ *[도식 참고: 원본 노션 첨부 도식]*

따라서, 속도가 빠르지만 용량은 작은 shared memory를 활용하여 A, B의 element loading 재활용을 수행해줄 수 있다.

> 🖼️ *[도식 참고: 원본 노션 첨부 도식]*

다음 그림의 주황색 block인, shared memory에 A와 B의 elemnt 일부를 load한 뒤,

> 🖼️ *[도식 참고: 원본 노션 첨부 도식]*

각 A,B shared memory끼리의 partial dot product를 수행한 결과를 C output tile 해당 영역에 더해준다.

> 🖼️ *[도식 참고: figure 5: *Illustration of matrix multiply (A * B = C) broken into tiles*]*

figure 5: *Illustration of matrix multiply (A * B = C) broken into tiles*

이 방식을 K 번 반복하면서 partial dot product 결과를 output tile의 accumulator에 계속 더하고, K방향 처리가 끝나면 최종 결과를 global memory(HBM)에 저장한다.

CNN의 stride처럼, Tile의 크기만큼 연산 보폭이 늘어나기 때문에, HBM에서 A,B element를 읽어오는 총 횟수는 T배 줄어들게 된다.

**Tile size ↑ → HBM traffic ↓ (대략 1/T) → AI ↑**

이므로, MatMul layer의 Performance를 올릴 수 있다.
