import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { h, onMounted, watch } from 'vue'
import { useRoute } from 'vitepress'
import WeekNavBar from './components/WeekNavBar.vue'
import HtmlDeckViewer from './components/HtmlDeckViewer.vue'
import RooflineCalculator from './components/RooflineCalculator.vue'
import IrPipelineViewer from './components/IrPipelineViewer.vue'
import TldrMapCard from './components/TldrMapCard.vue'
import MicroCheckpoint from './components/MicroCheckpoint.vue'
import FeynmanPassport from './components/FeynmanPassport.vue'
import ConceptMatchQuest from './components/ConceptMatchQuest.vue'
import { useMasteryStore } from './composables/useMasteryStore'
import './custom.css'

const WEEK_CP_MAP: Record<number, string[]> = {
  2: ['cp1-check', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman'],
  3: ['cp1-check', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman'],
  4: ['cp1-roofline', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman'],
  5: ['cp1-check', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman'],
  6: ['cp1-check', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman'],
  7: ['cp1-check', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman']
}

function updateSidebarBadges() {
  if (typeof document === 'undefined') return
  const { getCompletedCount } = useMasteryStore()

  const links = document.querySelectorAll<HTMLAnchorElement>('.VPSidebar a')
  links.forEach((link) => {
    const href = link.getAttribute('href') || ''
    const match = href.match(/\/week0([2-7])\/?(?:$|\?|#)/)
    if (match && !href.includes('/guide') && !href.includes('/researchers')) {
      const weekNum = Number(match[1])
      const cpIds = WEEK_CP_MAP[weekNum]
      if (!cpIds) return

      const done = getCompletedCount(weekNum, cpIds)
      const total = cpIds.length

      const target = link.querySelector<HTMLElement>('.text') || link
      let badge = target.querySelector<HTMLSpanElement>('.t2s-sidebar-mastery-pill')
      if (!badge) {
        badge = document.createElement('span')
        badge.className = 't2s-sidebar-mastery-pill'
        target.appendChild(badge)
      }
      if (done === total) {
        badge.textContent = `✅ ${done}/${total}`
        badge.classList.add('done')
      } else if (done > 0) {
        badge.textContent = `🔥 ${done}/${total}`
        badge.classList.remove('done')
      } else {
        badge.textContent = `🎯 0/${total}`
        badge.classList.remove('done')
      }
    }
  })
}

export default {
  extends: DefaultTheme,
  Layout() {
    const route = useRoute()
    onMounted(() => {
      setTimeout(updateSidebarBadges, 120)
      window.addEventListener('t2s-mastery-updated', updateSidebarBadges)
    })
    watch(
      () => route.path,
      () => {
        setTimeout(updateSidebarBadges, 120)
      }
    )
    return h(DefaultTheme.Layout)
  },
  enhanceApp({ app }) {
    app.component('WeekNavBar', WeekNavBar)
    app.component('HtmlDeckViewer', HtmlDeckViewer)
    app.component('RooflineCalculator', RooflineCalculator)
    app.component('IrPipelineViewer', IrPipelineViewer)
    app.component('TldrMapCard', TldrMapCard)
    app.component('MicroCheckpoint', MicroCheckpoint)
    app.component('FeynmanPassport', FeynmanPassport)
    app.component('ConceptMatchQuest', ConceptMatchQuest)
  }
} satisfies Theme
