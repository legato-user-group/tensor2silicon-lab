<script setup lang="ts">
import { ref, computed } from 'vue'
import { withBase } from 'vitepress'

const props = withDefaults(
  defineProps<{
    src: string
    title: string
    author?: string
    description?: string
    height?: number
  }>(),
  {
    height: 620
  }
)

const isExpanded = ref(false)
const iframeRef = ref<HTMLIFrameElement | null>(null)
const fullUrl = computed(() => withBase(props.src))

function openFullscreen() {
  if (iframeRef.value && iframeRef.value.requestFullscreen) {
    iframeRef.value.requestFullscreen()
  } else {
    window.open(fullUrl.value, '_blank')
  }
}
</script>

<template>
  <div class="t2s-deck-card">
    <div class="t2s-deck-toolbar">
      <div class="t2s-deck-info">
        <span class="t2s-deck-tag">🖥️ INTERACTIVE HTML</span>
        <strong class="t2s-deck-title">{{ title }}</strong>
        <span v-if="author" class="t2s-deck-author">by {{ author }}</span>
      </div>
      <div class="t2s-deck-actions">
        <button type="button" class="deck-btn" @click="isExpanded = !isExpanded">
          {{ isExpanded ? '🔽 기본 크기' : '🔼 크게 보기' }}
        </button>
        <button type="button" class="deck-btn" @click="openFullscreen">
          ⛶ 전체화면
        </button>
        <a :href="fullUrl" target="_blank" rel="noopener" class="deck-btn primary">
          ↗ 새 탭에서 열기
        </a>
      </div>
    </div>
    <p v-if="description" class="t2s-deck-desc">{{ description }}</p>
    <div class="t2s-deck-frame-wrap" :style="{ height: (isExpanded ? 860 : height) + 'px' }">
      <iframe
        ref="iframeRef"
        :src="fullUrl"
        :title="title"
        class="t2s-deck-iframe"
        loading="lazy"
        allowfullscreen
      ></iframe>
    </div>
  </div>
</template>

<style scoped>
.t2s-deck-card {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  overflow: hidden;
  background: var(--vp-c-bg-soft);
  margin: 20px 0;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
}

.t2s-deck-toolbar {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: var(--vp-c-bg);
  border-bottom: 1px solid var(--vp-c-divider);
}

.t2s-deck-info {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.t2s-deck-tag {
  font-size: 0.7rem;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 5px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

.t2s-deck-title {
  font-size: 0.92rem;
  color: var(--vp-c-text-1);
}

.t2s-deck-author {
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
}

.t2s-deck-actions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.deck-btn {
  font-size: 0.78rem;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  cursor: pointer;
  text-decoration: none !important;
  transition: all 0.15s;
}

.deck-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.deck-btn.primary {
  background: var(--vp-c-brand-1);
  color: #fff !important;
  border-color: var(--vp-c-brand-1);
}

.t2s-deck-desc {
  margin: 0;
  padding: 8px 14px;
  font-size: 0.83rem;
  color: var(--vp-c-text-2);
  border-bottom: 1px solid var(--vp-c-divider);
}

.t2s-deck-frame-wrap {
  width: 100%;
  transition: height 0.25s ease;
  background: #0f172a;
}

.t2s-deck-iframe {
  width: 100%;
  height: 100%;
  border: none;
  display: block;
}
</style>
