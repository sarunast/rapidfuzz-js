import type { IndexedMatcherWorkload } from './shared.ts'

const LOWER = [...'abcdefghijklmnopqrstuvwxyz']

/** A cyclic order-2 de Bruijn sequence, linearized with its first character. */
export function lowercaseBigramQuery(): string {
  const sequence: number[] = []
  const work = [0]
  const edges = new Array<number>(LOWER.length).fill(0)
  while (work.length > 0) {
    const vertex = work[work.length - 1]
    const edge = edges[vertex]
    if (edge < LOWER.length) {
      edges[vertex]++
      work.push(edge)
    } else {
      const popped = work.pop()
      if (popped === undefined) throw new Error('de Bruijn traversal underflowed')
      sequence.push(popped)
    }
  }
  sequence.reverse()
  return sequence.map((ordinal) => LOWER[ordinal]).join('')
}

/** The temporary query and result both die on return. */
export function runTouchedSetSpike(matcher: IndexedMatcherWorkload): void {
  matcher.best(lowercaseBigramQuery())
}

export interface TouchedValidation {
  readonly touched: number
  readonly fraction: number
  readonly widestPosting: number
  readonly denseCutoff: number
}

export function validateTouchedCorpus(corpus: readonly string[]): TouchedValidation {
  const postings = new Map<string, number>()
  let touched = 0
  for (const choice of corpus) {
    const seen = new Set<string>()
    for (let index = 0; index + 1 < choice.length; index++) {
      seen.add(choice.slice(index, index + 2))
    }
    if (seen.size > 0) touched++
    for (const gram of seen) postings.set(gram, (postings.get(gram) ?? 0) + 1)
  }
  const widestPosting = Math.max(0, ...postings.values())
  const denseCutoff = (2 / 3) * corpus.length
  if (widestPosting >= denseCutoff) {
    throw new Error(`a selected posting (${widestPosting}) reaches the dense cutoff`)
  }
  return {
    touched,
    fraction: corpus.length === 0 ? 0 : touched / corpus.length,
    widestPosting,
    denseCutoff,
  }
}
