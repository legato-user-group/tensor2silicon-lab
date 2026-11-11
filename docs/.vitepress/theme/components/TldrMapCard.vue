<script setup lang="ts">
import { computed } from 'vue'
import { useMasteryStore } from '../composables/useMasteryStore'

export interface CheckpointMeta {
  id: string
  shortLabel: string
  anchorId: string
}

const props = defineProps<{
  week: number
  intuition: string
  steps: Array<{ tag: string; title: string; desc: string }>
  keyQuestions: string[]
  checkpoints: CheckpointMeta[]
}>()

const { isCompleted, getCompletedCount, resetWeek } = useMasteryStore()

const doneCount = computed(() =>
  getCompletedCount(
    props.week,
    props.checkpoints.map((c) => c.id)
  )
)

const totalCount = computed(() => props.checkpoints.length)
const percent = computed(() =>
  totalCount.value > 0 ? Math.round((doneCount.value / totalCount.value) * 100) : 0
)

function scrollToCheckpoint(anchorId: string) {
  if (typeof document === 'undefined') return
  const el = document.getElementById(anchorId)
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}

function handleReset() {
  resetWeek(props.week)
}
</script>

<template>
  <div class="t2s-tldr-wrap">
    <!-- Top Mastery Progress Bar -->
    <div class="t2s-progress-banner">
      <div class="progress-top">
        <div class="progress-title-group">
          <span class="progress-badge" :class="{ complete: doneCount === totalCount }">
            {{ doneCount === totalCount ? '🏅 완독 인증 완료' : '🎯 능동 학습 트래커' }}
          </span>
          <strong class="progress-label">
            이번 장 이해도 점검: {{ doneCount }} / {{ totalCount }} 완료 ({{ percent }}%)
          </strong>
        </div>
        <button v-if="doneCount > 0" type="button" class="reset-btn" @click="handleReset">
          초기화
        </button>
      </div>

      <div class="progress-track">
        <div class="progress-fill" :style="{ width: percent + '%' }"></div>
      </div>

      <div class="checkpoint-pills">
        <button
          v-for="(cp, idx) in checkpoints"
          :key="cp.id"
          type="button"
          class="cp-pill"
          :class="{ done: isCompleted(week, cp.id) }"
          @click="scrollToCheckpoint(cp.anchorId)"
          title="클릭하면 해당 체크포인트 위치로 이동합니다"
        >
          <span class="cp-icon">{{ isCompleted(week, cp.id) ? '✅' : `${idx + 1}` }}</span>
          <span>{{ cp.shortLabel }}</span>
        </button>
      </div>
    </div>

    <!-- 3-Minute TL;DR Mental Map -->
    <div class="t2s-tldr-body">
      <div class="tldr-header">
        <span class="tldr-eyebrow">🧭 읽기 전 3분 핵심 요약 지도 (TL;DR Mental Map)</span>
        <p class="tldr-intuition">{{ intuition }}</p>
      </div>

      <div class="tldr-steps-grid">
        <div v-for="(s, idx) in steps" :key="idx" class="tldr-step-card">
          <span class="step-tag">{{ s.tag }}</span>
          <strong>{{ s.title }}</strong>
          <p>{{ s.desc }}</p>
        </div>
      </div>

      <div class="tldr-questions">
        <div class="q-title">📌 이 장을 다 읽고 나면 스스로 답할 수 있어야 하는 3가지 질문</div>
        <ul>
          <li v-for="(q, i) in keyQuestions" :key="i">{{ q }}</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<style scoped>
.t2s-tldr-wrap {
  border: 1px solid var(--vp-c-brand-1);
  border-radius: 14px;
  overflow: hidden;
  margin: 20px 0 28px 0;
  background: var(--vp-c-bg-soft);
  box-shadow: 0 6px 20px rgba(37, 99, 235, 0.07);
}

.t2s-progress-banner {
  background: var(--vp-c-bg);
  padding: 14px 18px;
  border-bottom: 1px solid var(--vp-c-divider);
}

.progress-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.progress-title-group {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.progress-badge {
  font-size: 0.72rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

.progress-badge.complete {
  background: rgba(16, 185, 129, 0.16);
  color: #10b981;
}

.progress-label {
  font-size: 0.88rem;
  color: var(--vp-c-text-1);
}

.reset-btn {
  font-size: 0.74rem;
  color: var(--vp-c-text-3);
  border: 1px solid var(--vp-c-divider);
  background: transparent;
  padding: 2px 8px;
  border-radius: 6px;
  cursor: pointer;
}

.reset-btn:hover {
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-text-2);
}

.progress-track {
  height: 8px;
  border-radius: 999px;
  background: var(--vp-c-bg-soft);
  overflow: hidden;
  margin-bottom: 10px;
  border: 1px solid var(--vp-c-divider);
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #10b981);
  transition: width 0.35s ease;
}

.checkpoint-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.cp-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 600;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.15s;
}

.cp-pill:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.cp-pill.done {
  background: rgba(16, 185, 129, 0.12);
  border-color: rgba(16, 185, 129, 0.45);
  color: #10b981;
}

.cp-icon {
  font-size: 0.72rem;
  font-weight: 700;
}

.t2s-tldr-body {
  padding: 16px 18px;
}

.tldr-eyebrow {
  display: inline-block;
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  margin-bottom: 6px;
}

.tldr-intuition {
  margin: 0 0 14px 0 !important;
  font-size: 0.94rem;
  font-weight: 600;
  line-height: 1.55;
  color: var(--vp-c-text-1);
  padding: 10px 14px;
  border-left: 4px solid var(--vp-c-brand-1);
  background: var(--vp-c-bg);
  border-radius: 0 8px 8px 0;
}

.tldr-steps-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}

.tldr-step-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.step-tag {
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--vp-c-brand-1);
}

.tldr-step-card strong {
  font-size: 0.86rem;
  color: var(--vp-c-text-1);
}

.tldr-step-card p {
  margin: 0 !important;
  font-size: 0.78rem;
  line-height: 1.45;
  color: var(--vp-c-text-2);
}

.tldr-questions {
  background: var(--vp-c-bg);
  border: 1px dashed var(--vp-c-divider);
  border-radius: 10px;
  padding: 10px 14px;
}

.q-title {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin-bottom: 6px;
}

.tldr-questions ul {
  margin: 0 !important;
  padding-left: 18px !important;
}

.tldr-questions li {
  font-size: 0.82rem;
  line-height: 1.5;
  color: var(--vp-c-text-2);
  margin: 3px 0 !important;
}
</style>
