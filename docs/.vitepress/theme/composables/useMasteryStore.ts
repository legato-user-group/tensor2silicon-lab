import { ref } from 'vue'

export interface MasteryState {
  completed: Record<string, Record<string, boolean>>
  feynmanNotes: Record<string, string>
  readerName: string
}

const STORAGE_KEY = 't2s_mastery_v1'

const state = ref<MasteryState>({
  completed: {},
  feynmanNotes: {},
  readerName: ''
})

let initialized = false

function initStore() {
  if (initialized || typeof window === 'undefined') return
  initialized = true
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      state.value = {
        completed: parsed.completed || {},
        feynmanNotes: parsed.feynmanNotes || {},
        readerName: parsed.readerName || ''
      }
    }
  } catch {
    // ignore storage errors
  }
}

function persist() {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state.value))
    window.dispatchEvent(new CustomEvent('t2s-mastery-updated'))
  } catch {
    // ignore storage errors
  }
}

export function useMasteryStore() {
  initStore()

  function weekKey(week: number) {
    return `week0${week}`
  }

  function isCompleted(week: number, checkpointId: string): boolean {
    const wk = weekKey(week)
    return Boolean(state.value.completed[wk]?.[checkpointId])
  }

  function markCompleted(week: number, checkpointId: string) {
    const wk = weekKey(week)
    if (!state.value.completed[wk]) {
      state.value.completed[wk] = {}
    }
    if (!state.value.completed[wk][checkpointId]) {
      state.value.completed[wk][checkpointId] = true
      persist()
    }
  }

  function getCompletedCount(week: number, checkpointIds: string[]): number {
    return checkpointIds.filter((id) => isCompleted(week, id)).length
  }

  function saveFeynmanNote(week: number, note: string) {
    state.value.feynmanNotes[weekKey(week)] = note
    persist()
  }

  function getFeynmanNote(week: number): string {
    return state.value.feynmanNotes[weekKey(week)] || ''
  }

  function setReaderName(name: string) {
    state.value.readerName = name
    persist()
  }

  function resetWeek(week: number) {
    const wk = weekKey(week)
    state.value.completed[wk] = {}
    delete state.value.feynmanNotes[wk]
    persist()
  }

  return {
    state,
    isCompleted,
    markCompleted,
    getCompletedCount,
    saveFeynmanNote,
    getFeynmanNote,
    setReaderName,
    resetWeek
  }
}
