import type {
  CandidateIndex,
  CandidateIndexBuilder,
} from '#core/scoring/candidateIndex.js'
import type {
  ChoiceIndex,
  ChoiceIndexBuilder,
  SelectedChoices,
} from '#core/scoring/choiceIndex.js'
import type { PreparedKernel } from '#core/scoring/compilation.js'
import type { Sequence } from '#core/types.js'

import type { SoftQuery, SoftTverskyChoice } from '../soft.js'

export interface PreparedSoftQuery {
  readonly query: SoftQuery
  readonly scoreChoice: PreparedKernel
}

interface ScoredChoice {
  readonly id: number
  readonly score: number
}

function worse(left: ScoredChoice, right: ScoredChoice): boolean {
  return left.score < right.score || (left.score === right.score && left.id > right.id)
}

function pushWorstHeap(heap: ScoredChoice[], row: ScoredChoice): void {
  let child = heap.length
  heap.push(row)
  while (child > 0) {
    const parent = (child - 1) >>> 1
    if (!worse(row, heap[parent])) break
    heap[child] = heap[parent]
    child = parent
  }
  heap[child] = row
}

function replaceWorst(heap: ScoredChoice[], row: ScoredChoice): void {
  let parent = 0
  while (true) {
    const left = parent * 2 + 1
    if (left >= heap.length) break
    const right = left + 1
    let child = left
    if (right < heap.length && worse(heap[right], heap[left])) child = right
    if (!worse(heap[child], row)) break
    heap[parent] = heap[child]
    parent = child
  }
  heap[parent] = row
}

function retain(heap: ScoredChoice[], row: ScoredChoice, limit: number): void {
  if (heap.length < limit) pushWorstHeap(heap, row)
  else if (worse(heap[0], row)) replaceWorst(heap, row)
}

class SoftTverskyIndex implements ChoiceIndex {
  #ids = new Uint32Array(0)
  #scores = new Float64Array(0)
  readonly #marks: Uint8Array
  readonly #touched: number[] = []

  constructor(
    readonly choices: readonly SoftTverskyChoice[],
    readonly prepare: (query: Sequence) => PreparedSoftQuery,
    readonly postingOrdinals: ReadonlyMap<unknown, number>,
    readonly postingOffsets: Uint32Array,
    readonly postingIds: Uint32Array,
    readonly vocabularyPostingOrdinals: Uint32Array,
    readonly inner: CandidateIndex,
    readonly innerThreshold: number,
    readonly cappedChoices: Uint32Array,
  ) {
    this.#marks = new Uint8Array(choices.length)
  }

  select(
    query: Sequence,
    threshold: number | null,
    limit: number | null,
  ): SelectedChoices {
    const prepared = this.prepare(query)
    if (limit === null) return this.#selectUnlimited(prepared, threshold)
    // `search` refuses a zero limit long before the index sees one, but the
    // interface admits it and the other implementations answer it, so this one
    // answers it too rather than depending on its current caller.
    if (limit === 0) return { ids: this.#ids, scores: this.#scores, length: 0 }

    const candidateThreshold = threshold === null ? Number.MIN_VALUE : threshold
    const candidates = this.#candidateIds(candidateThreshold, prepared.query)
    const retained: ScoredChoice[] = []
    let settled = false
    for (const id of candidates) {
      const score = prepared.scoreChoice(this.choices[id], threshold)
      if (threshold === null || score >= threshold) {
        retain(retained, { id, score }, limit)
        if (retained.length === limit && retained[0].score === 1) {
          settled = true
          break
        }
      }
    }

    if (
      !settled &&
      threshold === null &&
      (retained.length < limit || retained[0].score === 0)
    ) {
      for (let id = 0; id < this.#marks.length; id++) {
        if (this.#marks[id] === 1) continue
        retain(retained, { id, score: 0 }, limit)
        if (
          retained.length === limit &&
          retained[0].score === 0 &&
          retained[0].id <= id
        ) {
          break
        }
      }
    }

    retained.sort((left, right) => right.score - left.score || left.id - right.id)
    const length = retained.length
    this.#reserve(length)
    for (let at = 0; at < length; at++) {
      this.#ids[at] = retained[at].id
      this.#scores[at] = retained[at].score
    }
    return { ids: this.#ids, scores: this.#scores, length }
  }

  #selectUnlimited(
    prepared: PreparedSoftQuery,
    threshold: number | null,
  ): SelectedChoices {
    const scored: ScoredChoice[] = []
    for (const id of this.#candidateIds(threshold, prepared.query)) {
      const score = prepared.scoreChoice(this.choices[id], threshold)
      if (threshold === null || score >= threshold) scored.push({ id, score })
    }
    scored.sort((left, right) => right.score - left.score || left.id - right.id)
    this.#reserve(scored.length)
    for (let at = 0; at < scored.length; at++) {
      this.#ids[at] = scored[at].id
      this.#scores[at] = scored[at].score
    }
    return { ids: this.#ids, scores: this.#scores, length: scored.length }
  }

  scan(query: Sequence, threshold: number | null): SelectedChoices {
    const prepared = this.prepare(query)
    this.#reserve(this.choices.length)
    let length = 0
    for (const id of this.#candidateIds(threshold, prepared.query)) {
      const score = prepared.scoreChoice(this.choices[id], threshold)
      if (threshold !== null && score < threshold) continue
      this.#ids[length] = id
      this.#scores[length++] = score
    }
    return { ids: this.#ids, scores: this.#scores, length }
  }

  #reserve(length: number): void {
    if (this.#ids.length >= length) return
    this.#ids = new Uint32Array(length)
    this.#scores = new Float64Array(length)
  }

  #candidateIds(threshold: number | null, query: SoftQuery): readonly number[] {
    this.#clearMarks()
    // Collected in a loop rather than by `filter` so the operand narrows to a
    // string here, instead of needing a second null test at the use site.
    const fuzzy: string[] = []
    for (const entry of query.table.entries) {
      if (entry.operand !== null) fuzzy.push(entry.operand)
    }
    if (
      threshold === null ||
      threshold <= 0 ||
      query.occurrenceCount === 0 ||
      (query.weighted !== null && query.table.entries.length === 0) ||
      fuzzy.length > 32
    ) {
      for (let id = 0; id < this.choices.length; id++) this.#touch(id)
      return this.#touched
    }
    for (const entry of query.table.entries) {
      const posting = this.postingOrdinals.get(entry.canonical)
      if (posting !== undefined) this.#touchPosting(posting)
    }
    for (const operand of fuzzy) {
      const found = this.inner.candidates(operand, this.innerThreshold)
      for (let at = 0; at < found.length; at++) {
        this.#touchPosting(this.vocabularyPostingOrdinals[found.ids[at]])
      }
    }
    if (fuzzy.length !== 0) for (const id of this.cappedChoices) this.#touch(id)
    this.#touched.sort((left, right) => left - right)
    return this.#touched
  }

  #touchPosting(posting: number): void {
    for (
      let at = this.postingOffsets[posting];
      at < this.postingOffsets[posting + 1];
      at++
    ) {
      this.#touch(this.postingIds[at])
    }
  }

  #touch(id: number): void {
    if (this.#marks[id] === 1) return
    this.#marks[id] = 1
    this.#touched.push(id)
  }

  // Clearing only what the last query marked keeps this O(candidates) and
  // leaves no counter to wrap, which a generation stamp would need at 2^32.
  #clearMarks(): void {
    for (const id of this.#touched) this.#marks[id] = 0
    this.#touched.length = 0
  }
}

export function createSoftTverskyIndexBuilder(
  prepareChoice: (choice: Sequence) => SoftTverskyChoice,
  prepareQuery: (query: Sequence) => PreparedSoftQuery,
  innerBuilder: CandidateIndexBuilder,
  innerThreshold: number,
): ChoiceIndexBuilder {
  const choices: SoftTverskyChoice[] = []
  const postingOrdinals = new Map<unknown, number>()
  const postingLists: number[][] = []
  const vocabularyIds = new Map<string, number>()
  const vocabularyPostingOrdinals: number[] = []
  const cappedChoices: number[] = []
  let sealed = false
  return {
    add(choice) {
      if (sealed) throw new TypeError('choice index builder is already sealed')
      const prepared = prepareChoice(choice)
      const id = choices.length
      choices.push(prepared)
      let fuzzyCount = 0
      for (const entry of prepared.counts.entries) {
        let posting = postingOrdinals.get(entry.canonical)
        if (posting === undefined) {
          posting = postingLists.length
          postingOrdinals.set(entry.canonical, posting)
          postingLists.push([])
        }
        postingLists[posting].push(id)
        if (entry.operand === null) continue
        fuzzyCount++
        if (!vocabularyIds.has(entry.operand)) {
          vocabularyIds.set(entry.operand, vocabularyIds.size)
          vocabularyPostingOrdinals.push(posting)
          innerBuilder.add(entry.operand)
        }
      }
      if (fuzzyCount >= 33) cappedChoices.push(id)
    },
    seal() {
      if (sealed) throw new TypeError('choice index builder is already sealed')
      sealed = true
      const postingOffsets = new Uint32Array(postingLists.length + 1)
      let postingCount = 0
      for (let posting = 0; posting < postingLists.length; posting++) {
        postingOffsets[posting] = postingCount
        postingCount += postingLists[posting].length
      }
      postingOffsets[postingLists.length] = postingCount
      const postingIds = new Uint32Array(postingCount)
      let at = 0
      for (const posting of postingLists) for (const id of posting) postingIds[at++] = id
      return new SoftTverskyIndex(
        choices,
        prepareQuery,
        postingOrdinals,
        postingOffsets,
        postingIds,
        Uint32Array.from(vocabularyPostingOrdinals),
        innerBuilder.seal(),
        innerThreshold,
        Uint32Array.from(cappedChoices),
      )
    },
  }
}
