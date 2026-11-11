<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, withBase } from 'vitepress'

const props = defineProps<{
  week: number
  topic: string
  researchers: Array<{
    id: string
    name: string
    label: string
    badge?: string
  }>
}>()

const route = useRoute()

const weekPrefix = computed(() => `/week0${props.week}`)

function isActive(targetPath: string) {
  const fullTarget = withBase(targetPath).replace(/\.html$/, '').replace(/\/$/, '')
  const current = route.path.replace(/\.html$/, '').replace(/\/$/, '')
  return current === fullTarget
}
</script>

<template>
  <div class="t2s-week-nav">
    <div class="t2s-week-nav-header">
      <div class="t2s-week-nav-left">
        <span class="t2s-week-badge">WEEK 0{{ week }}</span>
        <span class="t2s-week-topic">{{ topic }}</span>
      </div>
      <a
        href="https://github.com/yblee110"
        target="_blank"
        rel="noopener"
        class="t2s-lead-author-chip"
        title="메인 교안 집필 및 커리큘럼 설계: 이영빈 (@yblee110)"
      >
        <img src="https://github.com/yblee110.png" alt="이영빈" class="author-chip-avatar" />
        <span>✍️ 메인 교안 집필: <strong>이영빈 (yblee)</strong></span>
      </a>
    </div>
    <div class="t2s-week-nav-tabs">
      <div class="t2s-tab-group main-tabs">
        <a
          :href="withBase(`${weekPrefix}/`)"
          class="t2s-tab"
          :class="{ active: isActive(`${weekPrefix}/`) || isActive(`${weekPrefix}/index`) }"
        >
          <span>📘 메인 교안</span>
          <span class="pill-sub">집필: 이영빈</span>
        </a>
        <a
          :href="withBase(`${weekPrefix}/guide`)"
          class="t2s-tab"
          :class="{ active: isActive(`${weekPrefix}/guide`) }"
        >
          <span>🧭 학습 안내 & 예제</span>
          <span class="pill-sub">yblee</span>
        </a>
      </div>
      <div class="t2s-tab-divider"></div>
      <div class="t2s-tab-group researcher-tabs">
        <span class="t2s-tab-group-label">🧑‍🔬 연구원 리서치 노트:</span>
        <a
          v-for="r in researchers"
          :key="r.id"
          :href="withBase(`${weekPrefix}/researchers/${r.id}`)"
          class="t2s-tab researcher-pill"
          :class="{ active: isActive(`${weekPrefix}/researchers/${r.id}`) }"
        >
          <span>{{ r.name }}</span>
          <span v-if="r.badge" class="pill-sub">{{ r.badge }}</span>
        </a>
      </div>
    </div>
  </div>
</template>

<style scoped>
.t2s-week-nav {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.t2s-week-nav-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 12px;
  padding-bottom: 10px;
  border-bottom: 1px dashed var(--vp-c-divider);
}

.t2s-week-nav-left {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.t2s-lead-author-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px 3px 5px;
  border-radius: 999px;
  background: var(--vp-c-bg);
  border: 1px solid rgba(37, 99, 235, 0.3);
  font-size: 0.76rem;
  color: var(--vp-c-text-2);
  text-decoration: none !important;
  transition: border-color 0.15s ease, color 0.15s ease;
}

.t2s-lead-author-chip strong {
  color: var(--vp-c-brand-1);
}

.t2s-lead-author-chip:hover {
  border-color: var(--vp-c-brand-1);
}

.author-chip-avatar {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  object-fit: cover;
}

.t2s-week-badge {
  background: var(--vp-c-brand-1);
  color: #fff;
  font-size: 0.75rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 6px;
  letter-spacing: 0.04em;
}

.t2s-week-topic {
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--vp-c-text-1);
}

.t2s-week-nav-tabs {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.t2s-tab-group {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.t2s-tab-divider {
  width: 1px;
  height: 22px;
  background: var(--vp-c-divider);
  margin: 0 4px;
}

.t2s-tab-group-label {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--vp-c-text-2);
  margin-right: 4px;
}

.t2s-tab {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 11px;
  border-radius: 8px;
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  text-decoration: none !important;
  transition: all 0.15s ease;
}

.t2s-tab:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  transform: translateY(-1px);
}

.t2s-tab.active {
  background: var(--vp-c-brand-soft);
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  font-weight: 700;
}

.pill-sub {
  font-size: 0.7rem;
  padding: 1px 5px;
  border-radius: 4px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-3);
}

.t2s-tab.active .pill-sub {
  background: var(--vp-c-brand-1);
  color: #fff;
}
</style>
