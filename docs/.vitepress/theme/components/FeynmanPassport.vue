<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useMasteryStore } from '../composables/useMasteryStore'

const props = defineProps<{
  week: number
  weekTitle: string
  checkpointId: string
  anchorId?: string
  allCheckpointIds: string[]
  promptQuestion: string
  modelAnswer: string
  selfCheckItems: string[]
}>()

const {
  state,
  isCompleted,
  markCompleted,
  getCompletedCount,
  saveFeynmanNote,
  getFeynmanNote,
  setReaderName
} = useMasteryStore()

const userAnswer = ref('')
const unlockedModel = ref(false)
const checkedRubrics = ref<boolean[]>([])
const nameInput = ref('')
const copied = ref(false)
const previewPassport = ref(false)

onMounted(() => {
  const saved = getFeynmanNote(props.week)
  if (saved) {
    userAnswer.value = saved
    unlockedModel.value = true
  }
  nameInput.value = state.value.readerName || ''
  checkedRubrics.value = props.selfCheckItems.map(() => isCompleted(props.week, props.checkpointId))
})

const feynmanDone = computed(() => isCompleted(props.week, props.checkpointId))
const doneCount = computed(() => getCompletedCount(props.week, props.allCheckpointIds))
const totalCount = computed(() => props.allCheckpointIds.length)
const allDone = computed(() => doneCount.value === totalCount.value)

function unlockComparison() {
  if (userAnswer.value.trim().length < 10) return
  saveFeynmanNote(props.week, userAnswer.value.trim())
  unlockedModel.value = true
}

function toggleRubric(idx: number) {
  checkedRubrics.value[idx] = !checkedRubrics.value[idx]
  if (checkedRubrics.value.every(Boolean)) {
    markCompleted(props.week, props.checkpointId)
  }
}

function updateName() {
  setReaderName(nameInput.value.trim())
}

const verificationCode = computed(() => {
  const baseStr = `${nameInput.value.trim() || 'T2S-READER'}-W0${props.week}-${userAnswer.value.length}`
  let hash = 2166136261
  for (let i = 0; i < baseStr.length; i++) {
    hash ^= baseStr.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  const hex = (hash >>> 0).toString(16).toUpperCase().padStart(8, '0')
  return `T2S-W0${props.week}-${hex.slice(0, 6)}`
})

const shareMarkdown = computed(() => {
  const reader = nameInput.value.trim() || '익명 연구원'
  const note = userAnswer.value.trim() || '(직접 작성한 한 줄 요약)'
  return (
    `### 🏅 [Week 0${props.week} 완독 인증] ${props.weekTitle}\n` +
    `- **독자**: ${reader}\n` +
    `- **검증 코드**: \`${verificationCode.value}\` (${doneCount.value}/${totalCount.value} 체크포인트 통과)\n` +
    `- **나의 핵심 질문 한 줄 설명**:\n  > ${note}\n`
  )
})

async function copyCertification() {
  if (typeof navigator !== 'undefined' && navigator.clipboard) {
    await navigator.clipboard.writeText(shareMarkdown.value)
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 2500)
  }
}
</script>

<template>
  <div :id="anchorId || checkpointId" class="t2s-feynman-wrap">
    <!-- Part 1: Feynman Write-to-Unlock -->
    <div class="feynman-box" :class="{ solved: feynmanDone }">
      <div class="feynman-head">
        <span class="f-badge">✍️ FINAL CHECKPOINT · 파인만 한 줄 설명</span>
        <span class="f-status" :class="{ done: feynmanDone }">
          {{ feynmanDone ? '✅ 자가 설명 & 점검 완료' : '⏳ 내 언어로 정리하기' }}
        </span>
      </div>

      <h4 class="f-title">읽은 내용을 내 말로 설명할 수 있어야 진짜 내 지식입니다</h4>
      <p class="f-prompt">{{ promptQuestion }}</p>

      <textarea
        v-model="userAnswer"
        class="f-textarea"
        rows="3"
        placeholder="동료에게 설명하듯 1~2문장으로 편하게 적어보세요 (10자 이상 입력 시 모범 답안이 열립니다)..."
      ></textarea>

      <div class="f-actions">
        <button
          type="button"
          class="f-unlock-btn"
          :disabled="userAnswer.trim().length < 10"
          @click="unlockComparison"
        >
          🔓 내 설명 저장 & 모범 답안과 비교하기
        </button>
        <span v-if="userAnswer.trim().length < 10" class="char-hint">
          현재 {{ userAnswer.trim().length }}자 / 최소 10자 이상 입력해 주세요
        </span>
      </div>

      <!-- Unlocked Model Answer + Self-Check Rubric -->
      <div v-if="unlockedModel || feynmanDone" class="f-model-panel">
        <div class="model-answer-card">
          <strong>📘 교안 모범 답안</strong>
          <p>{{ modelAnswer }}</p>
        </div>

        <div class="rubric-card">
          <strong>✅ 자가 점검: 내가 쓴 설명에 아래 핵심 원리가 포함되었나요? (모두 체크 시 완료)</strong>
          <label
            v-for="(item, idx) in selfCheckItems"
            :key="idx"
            class="rubric-item"
          >
            <input
              type="checkbox"
              :checked="checkedRubrics[idx]"
              @change="toggleRubric(idx)"
            />
            <span>{{ item }}</span>
          </label>
        </div>
      </div>
    </div>

    <!-- Part 2: Mastery Passport & Community Verification -->
    <div class="passport-section">
      <div class="passport-header">
        <div>
          <span class="p-eyebrow">🏅 COMPLETION PASSPORT & COMMUNITY PROOF</span>
          <h4 class="p-title">Week 0{{ week }} 완독 인증 패스포트 & 한 줄 방명록</h4>
        </div>
        <button
          v-if="!allDone"
          type="button"
          class="preview-toggle-btn"
          @click="previewPassport = !previewPassport"
        >
          {{ previewPassport ? '미리보기 닫기' : '👀 인증서 양식 미리보기' }}
        </button>
      </div>

      <div v-if="!allDone && !previewPassport" class="passport-locked">
        <p>
          현재 <strong>{{ doneCount }} / {{ totalCount }}</strong>개의 이해도 관문을 완료했습니다.
          본문 중간의 체크포인트와 실습 미션을 모두 통과하면 <strong>Week 0{{ week }} 완독 인증 패스포트</strong>가 발급됩니다!
        </p>
      </div>

      <div v-else class="passport-card-container">
        <div v-if="!allDone" class="preview-notice">
          ※ 현재 미리보기 상태입니다 ({{ doneCount }}/{{ totalCount }} 완료). 5개 관문을 모두 통과하면 정식 인증 코드가 활성화됩니다.
        </div>

        <div class="passport-input-row">
          <label>인증서에 표시할 이름 또는 GitHub ID:</label>
          <input
            v-model="nameInput"
            type="text"
            class="p-name-input"
            placeholder="예: @yblee110 또는 홍길동"
            @input="updateName"
          />
        </div>

        <!-- Visual Digital Certificate Card -->
        <div class="digital-certificate" :class="{ official: allDone }">
          <div class="cert-top">
            <span class="cert-org">TENSOR2SILICON LAB · OPEN RESEARCH WEBBOOK</span>
            <span class="cert-code">{{ verificationCode }}</span>
          </div>
          <div class="cert-main">
            <div class="cert-week">WEEK 0{{ week }} MASTERY CERTIFICATE</div>
            <div class="cert-topic">{{ weekTitle }}</div>
            <div class="cert-reader">
              완독 연구원: <strong>{{ nameInput.trim() || '이름을 입력해 주세요' }}</strong>
            </div>
          </div>
          <div class="cert-quote" v-if="userAnswer.trim()">
            “{{ userAnswer.trim() }}”
          </div>
          <div class="cert-footer">
            <span>검증 항목: 개념 예측 · 오개념 진단 · 손계산 · 시뮬레이터 미션 · 파인만 설명 ({{ doneCount }}/{{ totalCount }})</span>
            <span class="cert-stamp">{{ allDone ? '✅ VERIFIED' : '⏳ IN PROGRESS' }}</span>
          </div>
        </div>

        <!-- Share / Community Verification Actions -->
        <div class="community-share-row">
          <button type="button" class="share-btn primary" @click="copyCertification">
            {{ copied ? '✅ 클립보드에 복사되었습니다!' : '📋 완독 인증 마크다운 복사하기' }}
          </button>
          <a
            :href="`https://github.com/legato-user-group/tensor2silicon-lab/issues/new?title=${encodeURIComponent(`[Week 0${week} 완독 인증] ${weekTitle}`)}`"
            target="_blank"
            rel="noopener"
            class="share-btn secondary"
          >
            💬 GitHub 레포에 완독 인증 & 인사이트 남기기 ↗
          </a>
        </div>
        <p class="share-help">
          💡 위 <strong>복사하기</strong> 버튼을 누른 뒤 GitHub 인증 스레드나 스터디 채널에 붙여넣으면, 누가 어떤 인사이트를 얻으며 완독했는지 팀 전체가 한눈에 확인할 수 있습니다.
        </p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.t2s-feynman-wrap {
  margin: 28px 0;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.feynman-box {
  border: 1.5px solid var(--vp-c-divider);
  border-left: 5px solid #8b5cf6;
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  padding: 18px 20px;
}

.feynman-box.solved {
  border-left-color: #10b981;
  border-color: rgba(16, 185, 129, 0.4);
}

.feynman-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}

.f-badge {
  font-size: 0.72rem;
  font-weight: 800;
  padding: 2px 8px;
  border-radius: 5px;
  background: rgba(139, 92, 246, 0.15);
  color: #7c3aed;
}

.dark .f-badge {
  color: #a78bfa;
}

.f-status {
  font-size: 0.76rem;
  font-weight: 700;
  color: var(--vp-c-text-2);
}

.f-status.done {
  color: #10b981;
}

.f-title {
  margin: 4px 0 8px 0 !important;
  font-size: 1.02rem;
}

.f-prompt {
  margin: 0 0 12px 0 !important;
  font-size: 0.9rem;
  line-height: 1.55;
  color: var(--vp-c-text-1);
}

.f-textarea {
  width: 100%;
  box-sizing: border-box;
  border-radius: 9px;
  border: 1.5px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  padding: 10px 12px;
  font-size: 0.88rem;
  line-height: 1.5;
  resize: vertical;
}

.f-textarea:focus {
  outline: none;
  border-color: var(--vp-c-brand-1);
}

.f-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 10px;
}

.f-unlock-btn {
  padding: 8px 15px;
  border-radius: 8px;
  border: none;
  background: #7c3aed;
  color: #fff;
  font-size: 0.84rem;
  font-weight: 700;
  cursor: pointer;
}

.f-unlock-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.char-hint {
  font-size: 0.78rem;
  color: var(--vp-c-text-3);
}

.f-model-panel {
  margin-top: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.model-answer-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 9px;
  padding: 12px 14px;
  font-size: 0.86rem;
}

.model-answer-card p {
  margin: 6px 0 0 0 !important;
  line-height: 1.55;
}

.rubric-card {
  background: rgba(16, 185, 129, 0.08);
  border: 1px solid rgba(16, 185, 129, 0.3);
  border-radius: 9px;
  padding: 12px 14px;
  font-size: 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.rubric-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  cursor: pointer;
  line-height: 1.45;
}

.passport-section {
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: var(--vp-c-bg-soft);
  padding: 18px 20px;
}

.passport-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 12px;
}

.p-eyebrow {
  font-size: 0.7rem;
  font-weight: 800;
  color: var(--vp-c-brand-1);
}

.p-title {
  margin: 2px 0 0 0 !important;
  font-size: 1.05rem;
}

.preview-toggle-btn {
  font-size: 0.78rem;
  font-weight: 600;
  padding: 5px 10px;
  border-radius: 7px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
}

.passport-locked {
  background: var(--vp-c-bg);
  border: 1px dashed var(--vp-c-divider);
  border-radius: 10px;
  padding: 14px 16px;
  font-size: 0.88rem;
  color: var(--vp-c-text-2);
}

.passport-locked p {
  margin: 0 !important;
}

.preview-notice {
  font-size: 0.78rem;
  color: #d97706;
  margin-bottom: 10px;
  font-weight: 600;
}

.passport-input-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}

.passport-input-row label {
  font-size: 0.84rem;
  font-weight: 600;
}

.p-name-input {
  padding: 6px 12px;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.86rem;
  min-width: 220px;
}

.digital-certificate {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  color: #f8fafc;
  border-radius: 14px;
  padding: 20px 22px;
  border: 1.5px solid #334155;
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.22);
  margin-bottom: 14px;
}

.digital-certificate.official {
  border-color: #10b981;
}

.cert-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.72rem;
  color: #94a3b8;
  letter-spacing: 0.05em;
  margin-bottom: 12px;
}

.cert-code {
  font-family: monospace;
  background: rgba(255, 255, 255, 0.1);
  padding: 2px 8px;
  border-radius: 4px;
  color: #38bdf8;
}

.cert-week {
  font-size: 0.78rem;
  font-weight: 800;
  color: #60a5fa;
  letter-spacing: 0.06em;
}

.cert-topic {
  font-size: 1.25rem;
  font-weight: 800;
  margin: 4px 0 8px 0;
  color: #ffffff;
}

.cert-reader {
  font-size: 0.9rem;
  color: #cbd5e1;
}

.cert-quote {
  margin-top: 12px;
  padding: 10px 14px;
  border-left: 3px solid #38bdf8;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 0 8px 8px 0;
  font-size: 0.84rem;
  font-style: italic;
  color: #e2e8f0;
}

.cert-footer {
  margin-top: 14px;
  padding-top: 10px;
  border-top: 1px solid rgba(255, 255, 255, 0.12);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 0.74rem;
  color: #94a3b8;
}

.cert-stamp {
  font-weight: 800;
  color: #34d399;
}

.community-share-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.share-btn {
  padding: 8px 15px;
  border-radius: 8px;
  font-size: 0.84rem;
  font-weight: 700;
  cursor: pointer;
  text-decoration: none !important;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.share-btn.primary {
  background: var(--vp-c-brand-1);
  color: #fff;
  border: none;
}

.share-btn.secondary {
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1) !important;
  border: 1px solid var(--vp-c-divider);
}

.share-btn.secondary:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1) !important;
}

.share-help {
  margin: 10px 0 0 0 !important;
  font-size: 0.79rem;
  color: var(--vp-c-text-2);
}
</style>
