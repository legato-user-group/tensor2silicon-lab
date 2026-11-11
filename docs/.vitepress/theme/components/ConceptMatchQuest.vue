<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useMasteryStore } from '../composables/useMasteryStore'

export interface MatchPair {
  id: string
  left: string
  right: string
}

const props = defineProps<{
  week: number
  checkpointId: string
  anchorId?: string
  badge: string
  title: string
  instruction: string
  pairs: MatchPair[]
}>()

const { isCompleted, markCompleted } = useMasteryStore()

const selectedLeftId = ref<string | null>(null)
const matchedIds = ref<Record<string, boolean>>({})
const wrongFlashRightId = ref<string | null>(null)
const feedbackMsg = ref<string>('')

// Deterministic shuffle so SSR and client match cleanly
const shuffledRight = computed(() => {
  const arr = [...props.pairs]
  // Simple deterministic permutation (reverse + swap middle)
  if (arr.length >= 4) {
    return [arr[2], arr[0], arr[3], arr[1], ...arr.slice(4)]
  }
  return arr.reverse()
})

const alreadyDone = computed(() => isCompleted(props.week, props.checkpointId))

onMounted(() => {
  if (alreadyDone.value) {
    const all: Record<string, boolean> = {}
    props.pairs.forEach((p) => {
      all[p.id] = true
    })
    matchedIds.value = all
  }
})

const matchedCount = computed(() => Object.keys(matchedIds.value).length)

function pickLeft(id: string) {
  if (matchedIds.value[id]) return
  selectedLeftId.value = selectedLeftId.value === id ? null : id
  feedbackMsg.value = ''
}

function pickRight(rightItem: MatchPair) {
  if (matchedIds.value[rightItem.id]) return
  if (!selectedLeftId.value) {
    feedbackMsg.value = '먼저 왼쪽 항목(개념/코드)을 하나 클릭한 뒤 오른쪽 설명을 선택해 주세요!'
    return
  }

  if (selectedLeftId.value === rightItem.id) {
    matchedIds.value = { ...matchedIds.value, [rightItem.id]: true }
    selectedLeftId.value = null
    feedbackMsg.value = ''
    if (Object.keys(matchedIds.value).length === props.pairs.length) {
      markCompleted(props.week, props.checkpointId)
    }
  } else {
    wrongFlashRightId.value = rightItem.id
    feedbackMsg.value = '💡 짝이 맞지 않습니다! 선택한 왼쪽 항목의 핵심 역할을 다시 확인해 보세요.'
    setTimeout(() => {
      wrongFlashRightId.value = null
    }, 650)
  }
}
</script>

<template>
  <div :id="anchorId || checkpointId" class="t2s-match-box" :class="{ solved: alreadyDone }">
    <div class="match-header">
      <span class="match-badge">{{ badge }}</span>
      <span class="match-progress" :class="{ done: alreadyDone }">
        {{ alreadyDone ? `✅ ${pairs.length}/${pairs.length} 매칭 완료!` : `🔗 매칭 현황: ${matchedCount} / ${pairs.length}` }}
      </span>
    </div>

    <h4 class="match-title">{{ title }}</h4>
    <p class="match-desc">{{ instruction }}</p>

    <div class="match-grid">
      <!-- Left Column -->
      <div class="match-col">
        <div class="col-head">① 개념 · 계층 · 코드 선택</div>
        <button
          v-for="item in pairs"
          :key="'L-' + item.id"
          type="button"
          class="match-card left-card"
          :class="{
            active: selectedLeftId === item.id,
            matched: matchedIds[item.id]
          }"
          @click="pickLeft(item.id)"
        >
          <span class="card-indicator">
            {{ matchedIds[item.id] ? '✅' : selectedLeftId === item.id ? '👉' : '○' }}
          </span>
          <span class="card-text">{{ item.left }}</span>
        </button>
      </div>

      <!-- Right Column -->
      <div class="match-col">
        <div class="col-head">② 알맞은 역할 · 실행 의미 연결</div>
        <button
          v-for="item in shuffledRight"
          :key="'R-' + item.id"
          type="button"
          class="match-card right-card"
          :class="{
            matched: matchedIds[item.id],
            wrong: wrongFlashRightId === item.id
          }"
          @click="pickRight(item)"
        >
          <span class="card-indicator">{{ matchedIds[item.id] ? '✅' : '🔗' }}</span>
          <span class="card-text">{{ item.right }}</span>
        </button>
      </div>
    </div>

    <div v-if="feedbackMsg" class="match-feedback retry">
      {{ feedbackMsg }}
    </div>
    <div v-else-if="alreadyDone" class="match-feedback ok">
      🎉 모든 개념과 실행 동작을 정확히 연결했습니다!
    </div>
  </div>
</template>

<style scoped>
.t2s-match-box {
  border: 1.5px solid var(--vp-c-divider);
  border-left: 5px solid #3b82f6;
  border-radius: 14px;
  background: var(--vp-c-bg-soft);
  padding: 18px 20px;
  margin: 24px 0;
}

.t2s-match-box.solved {
  border-left-color: #10b981;
  border-color: rgba(16, 185, 129, 0.45);
}

.match-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}

.match-badge {
  font-size: 0.72rem;
  font-weight: 800;
  padding: 2px 8px;
  border-radius: 5px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

.t2s-match-box.solved .match-badge {
  background: rgba(16, 185, 129, 0.16);
  color: #10b981;
}

.match-progress {
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--vp-c-text-2);
}

.match-progress.done {
  color: #10b981;
}

.match-title {
  margin: 4px 0 6px 0 !important;
  font-size: 1.02rem;
}

.match-desc {
  margin: 0 0 14px 0 !important;
  font-size: 0.86rem;
  color: var(--vp-c-text-2);
}

.match-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
}

.match-col {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.col-head {
  font-size: 0.76rem;
  font-weight: 700;
  color: var(--vp-c-text-2);
  margin-bottom: 2px;
}

.match-card {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  text-align: left;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1.5px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  cursor: pointer;
  font-size: 0.83rem;
  line-height: 1.45;
  transition: all 0.15s ease;
}

.match-card:hover:not(.matched) {
  border-color: var(--vp-c-brand-1);
}

.match-card.active {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  font-weight: 600;
}

.match-card.matched {
  border-color: rgba(16, 185, 129, 0.5);
  background: rgba(16, 185, 129, 0.1);
  cursor: default;
}

.match-card.wrong {
  border-color: #ef4444;
  background: rgba(239, 68, 68, 0.12);
}

.card-indicator {
  flex-shrink: 0;
  font-size: 0.82rem;
}

.left-card .card-text {
  font-family: var(--vp-font-family-mono);
  font-size: 0.8rem;
  font-weight: 600;
}

.match-feedback {
  margin-top: 12px;
  padding: 10px 14px;
  border-radius: 9px;
  font-size: 0.84rem;
  font-weight: 600;
}

.match-feedback.ok {
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.35);
  color: #10b981;
}

.match-feedback.retry {
  background: rgba(245, 158, 11, 0.12);
  border: 1px solid rgba(245, 158, 11, 0.35);
  color: #d97706;
}
</style>
