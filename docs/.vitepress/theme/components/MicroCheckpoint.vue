<script setup lang="ts">
import { ref, computed } from 'vue'
import { useMasteryStore } from '../composables/useMasteryStore'

export interface ChoiceOption {
  label: string
  correct?: boolean
  feedback: string
}

const props = withDefaults(
  defineProps<{
    week: number
    checkpointId: string
    anchorId?: string
    mode?: 'choice' | 'number'
    badge: string
    title: string
    question: string
    options?: ChoiceOption[]
    expectedAnswer?: number
    tolerance?: number
    unit?: string
    placeholder?: string
    hint?: string
    solutionExplanation?: string
  }>(),
  {
    mode: 'choice',
    tolerance: 0.05,
    unit: '',
    placeholder: '숫자 입력'
  }
)

const { isCompleted, markCompleted } = useMasteryStore()

const selectedIdx = ref<number | null>(null)
const numInput = ref<string>('')
const numFeedback = ref<{ ok: boolean; msg: string } | null>(null)
const showHint = ref(false)

const alreadyDone = computed(() => isCompleted(props.week, props.checkpointId))

function selectOption(idx: number) {
  selectedIdx.value = idx
  const opt = props.options?.[idx]
  if (opt?.correct) {
    markCompleted(props.week, props.checkpointId)
  }
}

function checkNumber() {
  const val = parseFloat(numInput.value.replace(/,/g, '').trim())
  if (Number.isNaN(val)) {
    numFeedback.value = {
      ok: false,
      msg: '숫자를 입력해 주세요. (예: 2)'
    }
    return
  }
  const target = props.expectedAnswer ?? 0
  if (Math.abs(val - target) <= props.tolerance) {
    numFeedback.value = {
      ok: true,
      msg: props.solutionExplanation || `정답입니다! (${target} ${props.unit})`
    }
    markCompleted(props.week, props.checkpointId)
  } else {
    const ratioHint =
      val > target
        ? '입력하신 값이 정답보다 큽니다. 타일 한 변(t)이 분모에 있다는 점을 다시 확인해 보세요!'
        : '입력하신 값이 정답보다 작습니다. 힌트를 눌러 식을 점검해 보세요!'
    numFeedback.value = {
      ok: false,
      msg: `아쉽습니다 (${val} ${props.unit}). ${ratioHint}`
    }
  }
}
</script>

<template>
  <div :id="anchorId || checkpointId" class="t2s-cp-card" :class="{ solved: alreadyDone }">
    <div class="cp-header">
      <span class="cp-badge">{{ badge }}</span>
      <span class="cp-status" :class="{ done: alreadyDone }">
        {{ alreadyDone ? '✅ 이해도 검증 완료' : '⏳ 직접 풀어보세요' }}
      </span>
    </div>

    <h4 class="cp-title">{{ title }}</h4>
    <p class="cp-question">{{ question }}</p>

    <!-- Mode 1: Choice (Predict / Misconception O-X / Bug Spot) -->
    <div v-if="mode === 'choice' && options" class="cp-choices">
      <button
        v-for="(opt, idx) in options"
        :key="idx"
        type="button"
        class="cp-choice-btn"
        :class="{
          selected: selectedIdx === idx,
          correct: (selectedIdx === idx && opt.correct) || (alreadyDone && opt.correct && selectedIdx === null),
          wrong: selectedIdx === idx && !opt.correct
        }"
        @click="selectOption(idx)"
      >
        <span class="choice-idx">{{ String.fromCharCode(65 + idx) }}</span>
        <span class="choice-label">{{ opt.label }}</span>
      </button>

      <div
        v-if="selectedIdx !== null"
        class="cp-feedback"
        :class="options[selectedIdx].correct ? 'ok' : 'retry'"
      >
        <strong>
          {{ options[selectedIdx].correct ? '🎉 정확합니다!' : '💡 다시 생각해 볼까요?' }}
        </strong>
        <p>{{ options[selectedIdx].feedback }}</p>
      </div>
      <div v-else-if="alreadyDone" class="cp-feedback ok">
        <strong>🎉 이미 통과한 체크포인트입니다!</strong>
        <p>{{ options.find((o) => o.correct)?.feedback }}</p>
      </div>
    </div>

    <!-- Mode 2: Number Fill-in-the-blank (Micro-Calculation) -->
    <div v-else-if="mode === 'number'" class="cp-number-box">
      <div class="num-input-row">
        <input
          v-model="numInput"
          type="text"
          inputmode="decimal"
          class="cp-num-input"
          :placeholder="alreadyDone ? String(expectedAnswer) : placeholder"
          @keyup.enter="checkNumber"
        />
        <span v-if="unit" class="cp-unit">{{ unit }}</span>
        <button type="button" class="cp-submit-btn" @click="checkNumber">
          정답 확인
        </button>
        <button v-if="hint" type="button" class="cp-hint-btn" @click="showHint = !showHint">
          {{ showHint ? '힌트 닫기' : '💡 힌트' }}
        </button>
      </div>

      <div v-if="showHint && hint" class="cp-hint-box">
        <strong>💡 손계산 힌트:</strong> {{ hint }}
      </div>

      <div v-if="numFeedback" class="cp-feedback" :class="numFeedback.ok ? 'ok' : 'retry'">
        <strong>{{ numFeedback.ok ? '🎉 손계산 검증 통과!' : '🔎 다시 계산해 볼까요?' }}</strong>
        <p>{{ numFeedback.msg }}</p>
      </div>
      <div v-else-if="alreadyDone && solutionExplanation" class="cp-feedback ok">
        <strong>🎉 이미 통과한 손계산 체크포인트입니다! (정답: {{ expectedAnswer }} {{ unit }})</strong>
        <p>{{ solutionExplanation }}</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.t2s-cp-card {
  border: 1.5px solid var(--vp-c-divider);
  border-left: 5px solid #f59e0b;
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  padding: 16px 18px;
  margin: 22px 0;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.t2s-cp-card.solved {
  border-left-color: #10b981;
  border-color: rgba(16, 185, 129, 0.4);
}

.cp-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.cp-badge {
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.03em;
  padding: 2px 8px;
  border-radius: 5px;
  background: rgba(245, 158, 11, 0.16);
  color: #d97706;
}

.dark .cp-badge {
  color: #fbbf24;
}

.t2s-cp-card.solved .cp-badge {
  background: rgba(16, 185, 129, 0.16);
  color: #10b981;
}

.cp-status {
  font-size: 0.76rem;
  font-weight: 700;
  color: var(--vp-c-text-2);
}

.cp-status.done {
  color: #10b981;
}

.cp-title {
  margin: 4px 0 8px 0 !important;
  font-size: 1rem;
  color: var(--vp-c-text-1);
}

.cp-question {
  margin: 0 0 14px 0 !important;
  font-size: 0.9rem;
  line-height: 1.6;
  color: var(--vp-c-text-1);
}

.cp-choices {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.cp-choice-btn {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  text-align: left;
  padding: 10px 13px;
  border-radius: 9px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  cursor: pointer;
  font-size: 0.86rem;
  line-height: 1.45;
  transition: all 0.15s;
}

.cp-choice-btn:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateX(2px);
}

.cp-choice-btn.correct {
  border-color: #10b981;
  background: rgba(16, 185, 129, 0.1);
  font-weight: 600;
}

.cp-choice-btn.wrong {
  border-color: #ef4444;
  background: rgba(239, 68, 68, 0.08);
}

.choice-idx {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.76rem;
  font-weight: 700;
  flex-shrink: 0;
}

.cp-choice-btn.correct .choice-idx {
  background: #10b981;
  border-color: #10b981;
  color: #fff;
}

.cp-choice-btn.wrong .choice-idx {
  background: #ef4444;
  border-color: #ef4444;
  color: #fff;
}

.num-input-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.cp-num-input {
  width: 130px;
  padding: 7px 12px;
  border-radius: 8px;
  border: 1.5px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.92rem;
  font-weight: 700;
}

.cp-num-input:focus {
  outline: none;
  border-color: var(--vp-c-brand-1);
}

.cp-unit {
  font-size: 0.88rem;
  font-weight: 700;
  color: var(--vp-c-text-2);
}

.cp-submit-btn {
  padding: 7px 14px;
  border-radius: 8px;
  border: none;
  background: var(--vp-c-brand-1);
  color: #fff;
  font-size: 0.84rem;
  font-weight: 700;
  cursor: pointer;
}

.cp-submit-btn:hover {
  opacity: 0.92;
}

.cp-hint-btn {
  padding: 7px 11px;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
}

.cp-hint-box {
  margin-top: 10px;
  padding: 9px 12px;
  border-radius: 8px;
  background: var(--vp-c-bg);
  border: 1px dashed #f59e0b;
  font-size: 0.82rem;
  color: var(--vp-c-text-2);
}

.cp-feedback {
  margin-top: 10px;
  padding: 11px 14px;
  border-radius: 9px;
  font-size: 0.85rem;
  line-height: 1.55;
}

.cp-feedback.ok {
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.35);
}

.cp-feedback.retry {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.cp-feedback p {
  margin: 4px 0 0 0 !important;
  color: var(--vp-c-text-1);
}
</style>
