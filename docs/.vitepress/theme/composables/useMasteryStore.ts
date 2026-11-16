import { ref } from 'vue'

export interface MasteryState {
  completed: Record<string, Record<string, boolean>>
  feynmanNotes: Record<string, string>
  readerName: string
}

const STORAGE_KEY = 't2s_mastery_book_v1'

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

  function chapterKey(chapter: number) {
    return `ch0${chapter}`
  }

  function isCompleted(chapter: number, checkpointId: string): boolean {
    const ck = chapterKey(chapter)
    return Boolean(state.value.completed[ck]?.[checkpointId])
  }

  function markCompleted(chapter: number, checkpointId: string) {
    const ck = chapterKey(chapter)
    if (!state.value.completed[ck]) {
      state.value.completed[ck] = {}
    }
    if (!state.value.completed[ck][checkpointId]) {
      state.value.completed[ck][checkpointId] = true
      persist()
    }
  }

  function getCompletedCount(chapter: number, checkpointIds: string[]): number {
    return checkpointIds.filter((id) => isCompleted(chapter, id)).length
  }

  function saveFeynmanNote(chapter: number, note: string) {
    state.value.feynmanNotes[chapterKey(chapter)] = note
    persist()
  }

  function getFeynmanNote(chapter: number): string {
    return state.value.feynmanNotes[chapterKey(chapter)] || ''
  }

  function setReaderName(name: string) {
    state.value.readerName = name
    persist()
  }

  function resetWeek(chapter: number) {
    const ck = chapterKey(chapter)
    state.value.completed[ck] = {}
    delete state.value.feynmanNotes[ck]
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
