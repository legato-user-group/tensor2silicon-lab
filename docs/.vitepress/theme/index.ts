import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import WeekNavBar from './components/WeekNavBar.vue'
import HtmlDeckViewer from './components/HtmlDeckViewer.vue'
import RooflineCalculator from './components/RooflineCalculator.vue'
import IrPipelineViewer from './components/IrPipelineViewer.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('WeekNavBar', WeekNavBar)
    app.component('HtmlDeckViewer', HtmlDeckViewer)
    app.component('RooflineCalculator', RooflineCalculator)
    app.component('IrPipelineViewer', IrPipelineViewer)
  }
} satisfies Theme
