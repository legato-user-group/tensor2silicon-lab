<script setup lang="ts">
import { ref, computed } from 'vue'

interface HwPreset {
  id: string
  name: string
  peakTflops: number
  bwTbs: number
  sramLimitKb: number
  defaultBytes: number
}

const presets: HwPreset[] = [
  {
    id: 'edu_fp32',
    name: '🎓 4주차 가상 교육용 GPU (FP32: 20 TFLOPs, 1.0 TB/s)',
    peakTflops: 20,
    bwTbs: 1.0,
    sramLimitKb: 164,
    defaultBytes: 4
  },
  {
    id: 'edu_fp16',
    name: '🎓 7주차 가상 교육용 GPU (FP16: 120 TFLOPs, 1.5 TB/s)',
    peakTflops: 120,
    bwTbs: 1.5,
    sramLimitKb: 96,
    defaultBytes: 2
  },
  {
    id: 'a100',
    name: '⚡ NVIDIA A100 SXM (FP16 TC: 312 TFLOPs, 2.0 TB/s)',
    peakTflops: 312,
    bwTbs: 2.0,
    sramLimitKb: 164,
    defaultBytes: 2
  },
  {
    id: 'rtx5090',
    name: '🔥 NVIDIA RTX 5090 (FP16 TC: 419 TFLOPs, 1.79 TB/s)',
    peakTflops: 419,
    bwTbs: 1.79,
    sramLimitKb: 128,
    defaultBytes: 2
  },
  {
    id: 'h100',
    name: '🚀 NVIDIA H100 SXM (FP16 TC: 989 TFLOPs, 3.35 TB/s)',
    peakTflops: 989,
    bwTbs: 3.35,
    sramLimitKb: 228,
    defaultBytes: 2
  },
  {
    id: 'tpuv5e',
    name: '🔷 Google TPU v5e (BF16 MXU: 197 TFLOPs, 0.82 TB/s)',
    peakTflops: 197,
    bwTbs: 0.82,
    sramLimitKb: 16384,
    defaultBytes: 2
  }
]

const selectedPresetId = ref('edu_fp32')
const matrixN = ref(4096)
const elemBytes = ref(4) // 4 = FP32, 2 = FP16/BF16
const tileBM = ref(64)
const tileBN = ref(64)
const tileBK = ref(32)
const numStages = ref(3)

function onPresetChange() {
  const found = presets.find((p) => p.id === selectedPresetId.value)
  if (found) {
    elemBytes.value = found.defaultBytes
    if (found.id === 'edu_fp32') {
      tileBM.value = 64
      tileBN.value = 64
      tileBK.value = 64
      numStages.value = 2
    } else if (found.id === 'edu_fp16') {
      tileBM.value = 128
      tileBN.value = 128
      tileBK.value = 32
      numStages.value = 4
    }
  }
}

const currentHw = computed(() => presets.find((p) => p.id === selectedPresetId.value) || presets[0])

const totalFlops = computed(() => 2 * Math.pow(matrixN.value, 3))

// Naive (no reuse: every C[i,j] reads row i of A and col j of B from HBM)
const qNaiveBytes = computed(() => {
  const N = matrixN.value
  return (2 * Math.pow(N, 3) + Math.pow(N, 2)) * elemBytes.value
})

// Tiled HBM traffic (each BMxBN output tile reads BMxN of A and NxBN of B, writes BMxBN of C once)
const qTiledBytes = computed(() => {
  const N = matrixN.value
  const bm = tileBM.value
  const bn = tileBN.value
  const numTilesM = N / bm
  const numTilesN = N / bn
  const numTiles = numTilesM * numTilesN
  const readsPerTile = (bm * N + N * bn) * elemBytes.value
  const writesTotal = N * N * elemBytes.value
  return numTiles * readsPerTile + writesTotal
})

// Ideal Compulsory Traffic (read A once, read B once, write C once)
const qIdealBytes = computed(() => 3 * Math.pow(matrixN.value, 2) * elemBytes.value)

const trafficReductionRatio = computed(() => qNaiveBytes.value / qTiledBytes.value)

const arithmeticIntensity = computed(() => totalFlops.value / qTiledBytes.value)
const naiveIntensity = computed(() => totalFlops.value / qNaiveBytes.value)

const ridgePoint = computed(() => currentHw.value.peakTflops / currentHw.value.bwTbs)

const rooflinePerf = computed(() =>
  Math.min(currentHw.value.peakTflops, currentHw.value.bwTbs * arithmeticIntensity.value)
)

const isComputeBound = computed(() => arithmeticIntensity.value >= ridgePoint.value)

// Shared Memory calculation
const singleStageKb = computed(() => {
  const elems = tileBM.value * tileBK.value + tileBK.value * tileBN.value
  return (elems * elemBytes.value) / 1024
})

const pipelineSmemKb = computed(() => {
  const bufferMultiplier = Math.max(1, numStages.value - 1)
  return singleStageKb.value * bufferMultiplier
})

const fullStagesSmemKb = computed(() => singleStageKb.value * numStages.value)

const accumulatorKb = computed(() => (tileBM.value * tileBN.value * 4) / 1024) // FP32 accumulator

// SVG Roofline Plot coordinates (log10 scale from I = 0.25 to 512)
const minLogX = Math.log10(0.25)
const maxLogX = Math.log10(512)
const svgW = 560
const svgH = 220
const padL = 52
const padR = 24
const padT = 22
const padB = 34

function xToPx(iVal: number) {
  const clamped = Math.max(0.25, Math.min(512, iVal))
  const ratio = (Math.log10(clamped) - minLogX) / (maxLogX - minLogX)
  return padL + ratio * (svgW - padL - padR)
}

function yToPx(tflops: number) {
  const maxY = currentHw.value.peakTflops * 1.35
  const minY = Math.max(0.5, currentHw.value.peakTflops * 0.01)
  const clamped = Math.max(minY, Math.min(maxY, tflops))
  const ratio = (Math.log10(clamped) - Math.log10(minY)) / (Math.log10(maxY) - Math.log10(minY))
  return svgH - padB - ratio * (svgH - padT - padB)
}

const ridgePx = computed(() => ({
  x: xToPx(ridgePoint.value),
  y: yToPx(currentHw.value.peakTflops)
}))

const startPx = computed(() => ({
  x: xToPx(0.25),
  y: yToPx(currentHw.value.bwTbs * 0.25)
}))

const endPx = computed(() => ({
  x: xToPx(512),
  y: yToPx(currentHw.value.peakTflops)
}))

const currentPointPx = computed(() => ({
  x: xToPx(arithmeticIntensity.value),
  y: yToPx(rooflinePerf.value)
}))

function formatBytes(bytes: number) {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB (' + (bytes / Math.pow(1024, 3)).toFixed(2) + ' GiB)'
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB (' + (bytes / Math.pow(1024, 2)).toFixed(1) + ' MiB)'
  return (bytes / 1024).toFixed(1) + ' KB'
}
</script>

<template>
  <div class="t2s-calc-box">
    <div class="t2s-calc-header">
      <div>
        <span class="t2s-calc-badge">🎛️ INTERACTIVE SIMULATOR</span>
        <h4 class="t2s-calc-title">Roofline & 2D Tiled MatMul 실시간 계산기</h4>
      </div>
      <span class="t2s-calc-status" :class="isComputeBound ? 'compute' : 'memory'">
        {{ isComputeBound ? '⚡ 연산 병목 구간 (Compute-Bound)' : '🚧 메모리 대역폭 병목 구간 (Memory-Bound)' }}
      </span>
    </div>

    <div class="t2s-calc-controls">
      <div class="control-row full">
        <label>하드웨어 프리셋 선택</label>
        <select v-model="selectedPresetId" @change="onPresetChange" class="calc-select">
          <option v-for="p in presets" :key="p.id" :value="p.id">{{ p.name }}</option>
        </select>
      </div>

      <div class="control-grid">
        <div class="control-item">
          <label>행렬 크기 (M = N = K): <strong>{{ matrixN }}</strong></label>
          <div class="btn-group">
            <button v-for="n in [1024, 2048, 4096, 8192]" :key="n" :class="{ active: matrixN === n }" @click="matrixN = n">
              {{ n }}
            </button>
          </div>
        </div>

        <div class="control-item">
          <label>입력/출력 자료형: <strong>{{ elemBytes === 4 ? 'FP32 (4B)' : 'FP16/BF16 (2B)' }}</strong></label>
          <div class="btn-group">
            <button :class="{ active: elemBytes === 4 }" @click="elemBytes = 4">FP32 (4 Byte)</button>
            <button :class="{ active: elemBytes === 2 }" @click="elemBytes = 2">FP16 / BF16 (2 Byte)</button>
          </div>
        </div>

        <div class="control-item">
          <label>출력 타일 (BM × BN): <strong>{{ tileBM }} × {{ tileBN }}</strong></label>
          <div class="btn-group">
            <button
              v-for="t in [16, 32, 64, 128, 256]"
              :key="t"
              :class="{ active: tileBM === t && tileBN === t }"
              @click="tileBM = t; tileBN = t"
            >
              {{ t }}×{{ t }}
            </button>
          </div>
        </div>

        <div class="control-item">
          <label>K축 타일 폭 (BK) & 파이프라인 (`num_stages`)</label>
          <div class="btn-group">
            <button v-for="bk in [16, 32, 64, 128]" :key="'bk'+bk" :class="{ active: tileBK === bk }" @click="tileBK = bk">
              BK={{ bk }}
            </button>
            <span class="sep">|</span>
            <button v-for="s in [2, 3, 4]" :key="'s'+s" :class="{ active: numStages === s }" @click="numStages = s">
              {{ s }}stg
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- SVG Roofline Chart -->
    <div class="t2s-roofline-chart-wrap">
      <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="t2s-roofline-svg">
        <!-- Grid lines -->
        <line
          v-for="gx in [0.5, 1, 4, 16, 64, 256]"
          :key="'gx'+gx"
          :x1="xToPx(gx)"
          :y1="padT"
          :x2="xToPx(gx)"
          :y2="svgH - padB"
          stroke="currentColor"
          stroke-opacity="0.1"
          stroke-dasharray="3,3"
        />
        <text
          v-for="gx in [0.5, 1, 4, 16, 64, 256]"
          :key="'gxt'+gx"
          :x="xToPx(gx)"
          :y="svgH - 10"
          text-anchor="middle"
          font-size="10"
          fill="currentColor"
          opacity="0.65"
        >
          I={{ gx }}
        </text>

        <!-- Axes -->
        <line :x1="padL" :y1="svgH - padB" :x2="svgW - padR" :y2="svgH - padB" stroke="currentColor" stroke-opacity="0.3" />
        <line :x1="padL" :y1="padT" :x2="padL" :y2="svgH - padB" stroke="currentColor" stroke-opacity="0.3" />

        <!-- Ridge vertical guide -->
        <line
          :x1="ridgePx.x"
          :y1="ridgePx.y"
          :x2="ridgePx.x"
          :y2="svgH - padB"
          stroke="#f59e0b"
          stroke-width="1.5"
          stroke-dasharray="4,4"
        />
        <text :x="ridgePx.x" :y="padT + 10" text-anchor="middle" font-size="10" fill="#f59e0b" font-weight="bold">
          분기점 I* = {{ ridgePoint.toFixed(1) }}
        </text>

        <!-- Roofline Curve -->
        <polyline
          :points="`${startPx.x},${startPx.y} ${ridgePx.x},${ridgePx.y} ${endPx.x},${endPx.y}`"
          fill="none"
          stroke="#3b82f6"
          stroke-width="3"
        />

        <!-- Current Tiled Configuration Point -->
        <circle
          :cx="currentPointPx.x"
          :cy="currentPointPx.y"
          r="7"
          :fill="isComputeBound ? '#10b981' : '#ef4444'"
          stroke="#fff"
          stroke-width="2"
        />
        <text
          :x="Math.min(svgW - 95, Math.max(padL + 60, currentPointPx.x))"
          :y="Math.max(padT + 22, currentPointPx.y - 12)"
          text-anchor="middle"
          font-size="11"
          font-weight="bold"
          :fill="isComputeBound ? '#10b981' : '#ef4444'"
        >
          현재 타일: I={{ arithmeticIntensity.toFixed(1) }} ({{ rooflinePerf.toFixed(1) }} TFLOPs/s)
        </text>
      </svg>
    </div>

    <!-- Metric Cards -->
    <div class="t2s-metrics-grid">
      <div class="metric-card">
        <span class="m-label">총 연산량 (F = 2N³)</span>
        <strong class="m-val">{{ (totalFlops / 1e12).toFixed(3) }} TFLOPs</strong>
        <span class="m-sub">타일 크기와 무관하게 일정</span>
      </div>

      <div class="metric-card">
        <span class="m-label">HBM 메모리 이동량 (Q)</span>
        <strong class="m-val">{{ formatBytes(qTiledBytes) }}</strong>
        <span class="m-sub">
          재사용 없을 때 대비 <strong>{{ trafficReductionRatio.toFixed(1) }}배 감소</strong>
          (최소 필수 이동량: {{ formatBytes(qIdealBytes) }})
        </span>
      </div>

      <div class="metric-card">
        <span class="m-label">산술 강도 (I = F / Q) vs 분기점 (I*)</span>
        <strong class="m-val">{{ arithmeticIntensity.toFixed(2) }} FLOPs/B</strong>
        <span class="m-sub">
          하드웨어 분기점 I* = {{ ridgePoint.toFixed(1) }} FLOPs/B
          (재사용 없을 때 I = {{ naiveIntensity.toFixed(2) }})
        </span>
      </div>

      <div class="metric-card">
        <span class="m-label">Roofline 성능 상한 R = min(P, BW×I)</span>
        <strong class="m-val highlight">{{ rooflinePerf.toFixed(1) }} TFLOPs/s</strong>
        <span class="m-sub">
          피크 연산 성능({{ currentHw.value?.peakTflops || currentHw.peakTflops }} TFLOPs/s)의
          <strong>{{ ((rooflinePerf / currentHw.peakTflops) * 100).toFixed(0) }}%</strong> 도달
        </span>
      </div>

      <div class="metric-card wide">
        <span class="m-label">온칩 메모리 요구량 (CTA 1개 기준 Shared Memory & 누산기)</span>
        <div class="sram-bars">
          <div>
            <strong>1단계(A+B 타일):</strong> {{ singleStageKb.toFixed(1) }} KB
            &nbsp;|&nbsp;
            <strong>Triton 파이프라인 버퍼(num_stages-1={{ Math.max(1, numStages - 1) }}칸):</strong>
            <span :class="{ warn: pipelineSmemKb > currentHw.sramLimitKb }">
              {{ pipelineSmemKb.toFixed(1) }} KB
            </span>
            &nbsp;|&nbsp;
            <strong>전체 {{ numStages }}칸 기준:</strong>
            <span :class="{ warn: fullStagesSmemKb > currentHw.sramLimitKb }">
              {{ fullStagesSmemKb.toFixed(1) }} KB
            </span>
            (SM 한도: {{ currentHw.sramLimitKb }} KB)
          </div>
          <div class="m-sub">
            FP32 누산기(`accumulator` [{{ tileBM }}×{{ tileBN }}]): <strong>{{ accumulatorKb.toFixed(1) }} KB</strong> (레지스터 파일에 상주)
            <span v-if="fullStagesSmemKb > currentHw.sramLimitKb" class="sram-alert">
              ⚠️ 주의: Shared Memory 한도({{ currentHw.sramLimitKb }} KB) 초과 위험! 타일 크기나 num_stages를 줄여야 컴파일됩니다.
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.t2s-calc-box {
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: var(--vp-c-bg-soft);
  padding: 18px 20px;
  margin: 24px 0;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
}

.t2s-calc-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--vp-c-divider);
}

.t2s-calc-badge {
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  padding: 2px 8px;
  border-radius: 5px;
}

.t2s-calc-title {
  margin: 4px 0 0 0 !important;
  font-size: 1.05rem;
}

.t2s-calc-status {
  font-size: 0.8rem;
  font-weight: 700;
  padding: 5px 12px;
  border-radius: 20px;
}

.t2s-calc-status.compute {
  background: rgba(16, 185, 129, 0.15);
  color: #10b981;
  border: 1px solid rgba(16, 185, 129, 0.35);
}

.t2s-calc-status.memory {
  background: rgba(239, 68, 68, 0.14);
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.35);
}

.control-row.full {
  margin-bottom: 12px;
}

.control-row.full label,
.control-item label {
  display: block;
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
  margin-bottom: 5px;
}

.calc-select {
  width: 100%;
  padding: 7px 10px;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.86rem;
  font-weight: 600;
}

.control-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.btn-group {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
}

.btn-group button {
  padding: 4px 9px;
  font-size: 0.78rem;
  font-weight: 600;
  border-radius: 6px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.15s;
}

.btn-group button.active {
  background: var(--vp-c-brand-1);
  color: #fff;
  border-color: var(--vp-c-brand-1);
}

.sep {
  color: var(--vp-c-divider);
  margin: 0 2px;
}

.t2s-roofline-chart-wrap {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 8px 12px;
  margin-bottom: 14px;
}

.t2s-roofline-svg {
  width: 100%;
  max-height: 220px;
  display: block;
}

.t2s-metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 10px;
}

.metric-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 10px 13px;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.metric-card.wide {
  grid-column: 1 / -1;
}

.m-label {
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
  font-weight: 600;
}

.m-val {
  font-size: 1.12rem;
  color: var(--vp-c-text-1);
}

.m-val.highlight {
  color: var(--vp-c-brand-1);
}

.m-sub {
  font-size: 0.76rem;
  color: var(--vp-c-text-2);
  line-height: 1.4;
}

.sram-bars {
  font-size: 0.84rem;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.warn {
  color: #ef4444;
  font-weight: 700;
}

.sram-alert {
  display: block;
  margin-top: 4px;
  color: #ef4444;
  font-weight: 600;
}
</style>
