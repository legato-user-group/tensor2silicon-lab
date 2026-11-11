<script setup lang="ts">
import { ref, computed } from 'vue'
import { useMasteryStore } from '../composables/useMasteryStore'

const props = defineProps<{
  week?: number
  checkpointId?: string
  anchorId?: string
}>()

const { isCompleted, markCompleted } = useMasteryStore()
const questFeedback = ref<{ ok: boolean; msg: string } | null>(null)
const questSolved = computed(() =>
  props.week && props.checkpointId ? isCompleted(props.week, props.checkpointId) : false
)

interface StageInfo {
  id: string
  step: string
  name: string
  abstraction: string
  keyChange: string
  codeSnippet: string
  a100Note: string
  rtx5090Note: string
}

const stages: StageInfo[] = [
  {
    id: 'python',
    step: 'Stage 1',
    name: 'Python / Triton DSL',
    abstraction: '블록(Tile) 수준 텐서 연산',
    keyChange: '프로그래머가 2D 포인터 블록(`a_ptrs`, `b_ptrs`)과 K축 루프(`tl.load`, `tl.dot`, `tl.store`)만 선언합니다. 워프 분할이나 Shared Memory 주소는 아직 드러나지 않습니다.',
    codeSnippet: `# [Stage 1] Triton Python 커널 (프로그래머 작성 코드)
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk`,
    a100Note: 'A100에서도 동일한 Python 코드를 그대로 사용합니다 (@triton.autotune으로 메타파라미터만 선택).',
    rtx5090Note: 'RTX 5090(sm_120)에서도 단 한 줄의 커널 코드 수정 없이 동일하게 실행됩니다.'
  },
  {
    id: 'ttir',
    step: 'Stage 2',
    name: 'TTIR (Triton IR)',
    abstraction: '하드웨어 독립적 MLIR `tt` Dialect',
    keyChange: 'Python AST가 SSA(Single Static Assignment) 형태의 MLIR로 변환됩니다. `tt.splat`, `tt.make_range`, `tt.broadcast`, `tt.load`, `tt.dot` 등 타일 텐서 연산과 `scf.for` 루프가 명시됩니다.',
    codeSnippet: `// [Stage 2] TTIR (레이아웃 인코딩 없음, 순수 블록 텐서 타입)
%acc_init = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
%acc_loop:3 = scf.for %k = %c0 to %K step %c32 iter_args(%acc = %acc_init, %ap = %a_ptrs, %bp = %b_ptrs) -> (...) {
  %a_tile = tt.load %ap, %a_mask, %zero : tensor<128x32x!tt.ptr<f16>> -> tensor<128x32xf16>
  %b_tile = tt.load %bp, %b_mask, %zero : tensor<32x128x!tt.ptr<f16>> -> tensor<32x128xf16>
  %acc_next = tt.dot %a_tile, %b_tile, %acc : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
  scf.yield %acc_next, %ap_next, %bp_next
}`,
    a100Note: '타겟 아키텍처와 무관하게 텐서 형태(`tensor<128x32xf16>`)만 표현합니다.',
    rtx5090Note: 'A100과 동일한 TTIR 구조를 가집니다 (하드웨어 종속 최적화 이전 단계).'
  },
  {
    id: 'ttgir',
    step: 'Stage 3',
    name: 'TTGIR (Triton GPU IR)',
    abstraction: '워프·메모리 레이아웃·비동기 파이프라인 명시 (`ttg` Dialect)',
    keyChange: '텐서 타입 뒤에 `#blocked`, `#shared`, `#mma`, `#dot_op` 레이아웃 인코딩이 붙습니다! 또한 K 루프에 소프트웨어 파이프라이닝(`ttg.local_alloc`, `ttg.async_copy_global_to_local`, `ttg.async_wait`)이 삽입됩니다.',
    codeSnippet: `// [Stage 3] TTGIR (레이아웃 + Shared Memory 다중 버퍼 + 비동기 복사 삽입)
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>

%smem_a = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #shared_a, #ttg.shared_memory, mutable>
// 프롤로그: 루프 진입 전 k=0..2 타일 비동기 복사 예약
%tok = ttg.async_copy_global_to_local %a_ptrs, %smem_a_slice mask %mask : tensor<128x32x!tt.ptr<f16>, #blocked> -> <128x32xf16, #shared_a>
// 본 루프: 대기(wait) -> Shared에서 레지스터(#dot_op)로 적재 -> tt.dot(..., -> tensor<128x128xf32, #mma>)`,
    a100Note: 'A100(sm_80)은 `nvidia_mma` versionMajor=2 (`m16n8k16`)와 `num_stages-1`칸의 순환 Shared Memory 버퍼를 할당합니다 (mbsong 실측 확인).',
    rtx5090Note: 'RTX 5090(sm_120)에서도 `versionMajor = 2, instrShape = [16, 8]` 기반 MMA 레이아웃과 3칸 순환 버퍼(`num_stages=4`일 때)가 생성됩니다 (junpark 실측 확인).'
  },
  {
    id: 'llir',
    step: 'Stage 4',
    name: 'LLVM IR (NVVM)',
    abstraction: '스레드(Thread) 수준 포인터·레지스터 연산 및 타겟 Intrinsic',
    keyChange: '블록 텐서 개념이 사라지고 개별 GPU 스레드가 담당하는 레지스터 조각과 NVVM intrinsic(`llvm.nvvm.cp.async.*`, `ldmatrix`, `mma.sync`) 호출로 완전히 풀립니다(Unroll).',
    codeSnippet: `; [Stage 4] LLVM IR (스레드 단위 인라인 PTX / NVVM Intrinsic 호출)
call void asm sideeffect "cp.async.cg.shared.global [$0], [$1], 16, $2;", "r,l,r"(i32 %smem_ptr, ptr addrspace(1) %gmem_ptr, i32 %src_size)
call void @llvm.nvvm.cp.async.commit.group()
call void @llvm.nvvm.cp.async.wait.group(i32 2)
call void @llvm.nvvm.barrier0()
; Shared Memory -> 레지스터 적재 후 Tensor Core MMA 호출
%res = call { float, float, float, float } asm "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ...", ...`,
    a100Note: '16B 단위 글로벌→Shared 비동기 복사(`cp.async.cg.shared.global`)와 `ldmatrix`가 인라인 어셈블리로 결합됩니다.',
    rtx5090Note: '워프당 담당하는 [16, 8] 출력 조각 수만큼 `mma.sync` 인라인 어셈블리가 반복 호출됩니다.'
  },
  {
    id: 'ptx',
    step: 'Stage 5',
    name: 'NVIDIA PTX',
    abstraction: '가상 GPU ISA (어셈블리)',
    keyChange: '실제 GPU 명령어 수준에서 비동기 복사(`cp.async`), 행렬 조각 로드(`ldmatrix.sync.aligned.m8n8.x4.shared.b16`), Tensor Core 연산(`mma.sync.aligned.m16n8k16`)의 개수를 직접 셀 수 있습니다.',
    codeSnippet: `// [Stage 5] NVIDIA PTX
cp.async.cg.shared.global [ %r_smem + 0 ], [ %rd_gmem + 0 ], 16, %r_valid;
cp.async.commit_group;
cp.async.wait_group 2;
bar.sync 0;

ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%r_smem_read];
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f1, %f2, %f3, %f4}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%f1, %f2, %f3, %f4};`,
    a100Note: 'BM=128, BN=256, BK=64, 8 warps 설정에서 K 1회 반복당 m16n8k16 명령이 정확히 (128×256×64)/(16×8×16)/8 = 128회 생성됨을 확인했습니다.',
    rtx5090Note: 'BM=128, BN=128, BK=32, 4 warps 설정에서 K 1회 반복당 워프별로 m16n8k16 명령이 정확히 64회(전체 256회) 생성됨을 확인했습니다.'
  },
  {
    id: 'sass',
    step: 'Stage 6',
    name: 'CUBIN / SASS',
    abstraction: '실제 실리콘 실행 바이너리 (SM 하드웨어 기계어)',
    keyChange: '`ptxas` 어셈블러가 가상 레지스터를 실제 하드웨어 레지스터(`R0`~`R255`)에 할당하고, PTX의 `mma.sync`를 물리적 Tensor Core 명령어(`HMMA.16816.F32`)로 변환합니다.',
    codeSnippet: `// [Stage 6] SASS (cuobjdump -sass 결과)
LDGSTS.E.BYPASS.LTC128B.128 [R_SMEM], [R_GMEM] ;  // cp.async -> 글로벌에서 Shared로 직행
DEPBAR.LE SB0, 0x2 ;                              // wait_group 2
BAR.SYNC 0x0 ;                                    // CTA 동기화
LDSM.16.M88.4 R4, [R_SMEM_READ] ;                 // ldmatrix -> Shared에서 레지스터로
HMMA.16816.F32 R8, R4, R6, R8 ;                   // Tensor Core FP16xFP16 -> FP32 누적`,
    a100Note: 'A100 SASS에서 `LDGSTS`(비동기 복사), `LDSM`(행렬 로드), `HMMA.16816.F32`(Tensor Core) 3박자 명령어 시퀀스로 최종 실행됩니다.',
    rtx5090Note: 'Blackwell 소비자용 SM(sm_120) 기계어로 어셈블되어 GPU 스트리밍 멀티프로세서에서 실행됩니다.'
  }
]

const activeIdx = ref(0)

function selectStage(idx: number) {
  activeIdx.value = idx
  questFeedback.value = null
}

function verifyStageQuest() {
  if (activeIdx.value === 2) {
    questFeedback.value = {
      ok: true,
      msg: '정답입니다! Stage 2(TTIR)까지는 하드웨어 독립적인 블록 텐서였지만, Stage 3(TTGIR)로 내려오면서 워프 분할(#mma), 스위즐된 Shared Memory(#shared_a), 그리고 다중 버퍼 비동기 복사(ttg.async_copy_global_to_local)가 처음으로 구체화됩니다!'
    }
    if (props.week && props.checkpointId) {
      markCompleted(props.week, props.checkpointId)
    }
  } else if (activeIdx.value < 2) {
    questFeedback.value = {
      ok: false,
      msg: `선택하신 Stage ${activeIdx.value + 1}(${stages[activeIdx.value].name})은 아직 하드웨어 독립적인 블록 추상화 단계입니다! 코드 조각을 보면 아직 #mma / #shared 레이아웃이나 비동기 복사 명령이 없습니다. 한 단계 더 내려가 보세요!`
    }
  } else {
    questFeedback.value = {
      ok: false,
      msg: `선택하신 Stage ${activeIdx.value + 1}(${stages[activeIdx.value].name})은 이미 블록 텐서가 개별 스레드 명령어(cp.async, mma.sync, HMMA)로 모두 풀린 단계입니다! MLIR 텐서 타입 뒤에 #mma / #shared 레이아웃이 '처음으로' 부착되는 단계를 찾아보세요.`
    }
  }
}
</script>

<template>
  <div :id="anchorId || checkpointId" class="t2s-ir-viewer" :class="{ 'quest-solved': questSolved }">
    <div v-if="checkpointId" class="t2s-ir-quest-banner" :class="{ done: questSolved }">
      <div class="ir-quest-top">
        <span class="ir-quest-tag">🎯 CHECKPOINT 4 · 컴파일 파이프라인 탐정 미션</span>
        <span class="ir-quest-state">{{ questSolved ? '✅ 탐정 미션 통과!' : '⏳ Stage 1~6 탭을 눌러 정답 단계를 찾아보세요' }}</span>
      </div>
      <p class="ir-quest-desc">
        <strong>미션:</strong> 아래 <code>1 Python</code> ~ <code>6 SASS</code> 탭을 하나씩 클릭해 코드 변화를 비교해 보세요!
        하드웨어 독립적이던 블록 텐서 타입 뒤에 <strong><code>#mma</code> / <code>#shared</code> 레이아웃 인코딩이 처음으로 붙고</strong>,
        다중 버퍼 비동기 복사(<code>ttg.async_copy_global_to_local</code>)가 삽입되는 단계를 고른 뒤 하단 검증 버튼을 누르세요.
      </p>
    </div>

    <div class="t2s-ir-header">
      <div>
        <span class="t2s-ir-badge">🔍 COMPILER PIPELINE EXPLORER</span>
        <h4 class="t2s-ir-title">Triton 다단계 컴파일 파이프라인 단계별 해부 (Python → SASS)</h4>
      </div>
    </div>

    <!-- Stepper Buttons -->
    <div class="t2s-ir-steps">
      <button
        v-for="(s, idx) in stages"
        :key="s.id"
        class="ir-step-btn"
        :class="{ active: activeIdx === idx }"
        @click="selectStage(idx)"
      >
        <span class="step-num">{{ idx + 1 }}</span>
        <span class="step-label">{{ s.name }}</span>
      </button>
    </div>

    <!-- Active Stage Content -->
    <div class="t2s-ir-body">
      <div class="ir-meta-row">
        <div class="ir-meta-pill">
          <strong>추상화 수준:</strong> {{ stages[activeIdx].abstraction }}
        </div>
      </div>

      <p class="ir-desc">{{ stages[activeIdx].keyChange }}</p>

      <pre class="ir-code"><code>{{ stages[activeIdx].codeSnippet }}</code></pre>

      <div class="ir-hw-compare">
        <div class="hw-box a100">
          <div class="hw-title">⚡ NVIDIA A100 (`sm_80`) 실측 포인트 (by `mbsong`)</div>
          <p>{{ stages[activeIdx].a100Note }}</p>
        </div>
        <div class="hw-box rtx5090">
          <div class="hw-title">🔥 NVIDIA RTX 5090 (`sm_120`) 실측 포인트 (by `junpark`)</div>
          <p>{{ stages[activeIdx].rtx5090Note }}</p>
        </div>
      </div>
    </div>

    <div v-if="checkpointId" class="t2s-ir-quest-footer">
      <button type="button" class="ir-verify-btn" @click="verifyStageQuest">
        🎯 현재 선택한 단계 ({{ stages[activeIdx].step }}: {{ stages[activeIdx].name }}) 검증하기
      </button>
      <div v-if="questFeedback" class="ir-quest-feedback" :class="questFeedback.ok ? 'ok' : 'retry'">
        <strong>{{ questFeedback.ok ? '🎉 탐정 미션 클리어!' : '🔎 다른 단계를 살펴보세요' }}</strong>
        <p>{{ questFeedback.msg }}</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.t2s-ir-viewer {
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: var(--vp-c-bg-soft);
  padding: 18px 20px;
  margin: 24px 0;
}

.t2s-ir-viewer.quest-solved {
  border-color: rgba(16, 185, 129, 0.45);
}

.t2s-ir-quest-banner {
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.35);
  border-radius: 10px;
  padding: 12px 14px;
  margin-bottom: 16px;
}

.t2s-ir-quest-banner.done {
  background: rgba(16, 185, 129, 0.1);
  border-color: rgba(16, 185, 129, 0.4);
}

.ir-quest-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}

.ir-quest-tag {
  font-size: 0.72rem;
  font-weight: 800;
  color: #d97706;
}

.dark .ir-quest-tag {
  color: #fbbf24;
}

.t2s-ir-quest-banner.done .ir-quest-tag,
.t2s-ir-quest-banner.done .ir-quest-state {
  color: #10b981;
}

.ir-quest-state {
  font-size: 0.75rem;
  font-weight: 700;
}

.ir-quest-desc {
  margin: 0 !important;
  font-size: 0.84rem;
  line-height: 1.55;
}

.t2s-ir-quest-footer {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px dashed var(--vp-c-divider);
}

.ir-verify-btn {
  padding: 9px 18px;
  border-radius: 9px;
  border: none;
  background: linear-gradient(90deg, #2563eb, #0ea5e9);
  color: #fff;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 3px 10px rgba(37, 99, 235, 0.25);
}

.ir-verify-btn:hover {
  opacity: 0.94;
}

.ir-quest-feedback {
  margin-top: 10px;
  padding: 10px 14px;
  border-radius: 9px;
  font-size: 0.84rem;
  line-height: 1.5;
}

.ir-quest-feedback.ok {
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.35);
}

.ir-quest-feedback.retry {
  background: rgba(245, 158, 11, 0.12);
  border: 1px solid rgba(245, 158, 11, 0.35);
}

.ir-quest-feedback p {
  margin: 4px 0 0 0 !important;
}

.t2s-ir-header {
  margin-bottom: 14px;
}

.t2s-ir-badge {
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  padding: 2px 8px;
  border-radius: 5px;
}

.t2s-ir-title {
  margin: 4px 0 0 0 !important;
  font-size: 1.05rem;
}

.t2s-ir-steps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
  gap: 8px;
  margin-bottom: 16px;
}

.ir-step-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  text-align: left;
  transition: all 0.15s;
}

.ir-step-btn:hover {
  border-color: var(--vp-c-brand-1);
}

.ir-step-btn.active {
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  color: #fff;
  font-weight: 700;
}

.step-num {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.72rem;
  font-weight: 700;
  flex-shrink: 0;
}

.ir-step-btn.active .step-num {
  background: rgba(255, 255, 255, 0.25);
  color: #fff;
}

.step-label {
  font-size: 0.8rem;
  line-height: 1.2;
}

.ir-meta-row {
  margin-bottom: 8px;
}

.ir-meta-pill {
  display: inline-block;
  font-size: 0.8rem;
  padding: 4px 10px;
  border-radius: 6px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
}

.ir-desc {
  font-size: 0.88rem;
  margin: 8px 0 12px 0;
  line-height: 1.55;
}

.ir-code {
  background: #0f172a;
  color: #e2e8f0;
  padding: 14px 16px;
  border-radius: 10px;
  font-size: 0.8rem;
  line-height: 1.5;
  overflow-x: auto;
  margin-bottom: 14px;
}

.ir-hw-compare {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 10px;
}

.hw-box {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 10px 14px;
}

.hw-box.a100 {
  border-left: 4px solid #10b981;
}

.hw-box.rtx5090 {
  border-left: 4px solid #f59e0b;
}

.hw-title {
  font-size: 0.8rem;
  font-weight: 700;
  margin-bottom: 4px;
}

.hw-box p {
  margin: 0;
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
  line-height: 1.45;
}
</style>
