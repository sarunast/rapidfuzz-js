import { scoreFromDistance } from '#core/scoring/builtIn/cutoff.js'
import type {
  CandidateChoices,
  CandidateIndex,
  CandidateIndexBuilder,
} from '#core/scoring/candidateIndex.js'
import type { PreparedKernel } from '#core/scoring/compilation.js'
import { scorerSequence, snapshotSequence } from '#core/sequence.js'
import type { Sequence } from '#core/types.js'

import { createDiceIndexBuilder } from '../ngram/inverted/dice.js'

export const INDEL_GRAM_SIZE = 2

export function maximumQualifyingDistance(maximum: number, threshold: number): number {
  if (!(scoreFromDistance('normalizedSimilarity', 0, maximum, null) >= threshold)) {
    return -1
  }
  let low = 0
  let high = maximum
  while (low < high) {
    const middle = Math.ceil((low + high) / 2)
    if (scoreFromDistance('normalizedSimilarity', middle, maximum, null) >= threshold) {
      low = middle
    } else high = middle - 1
  }
  return low
}

class IndelCandidateIndex implements CandidateIndex {
  readonly #marks: Uint8Array
  #ids = new Uint32Array(0)
  readonly #touched: number[] = []

  constructor(
    readonly prepared: readonly unknown[],
    readonly bucketOfId: Uint32Array,
    readonly bucketLengths: Uint32Array,
    readonly bucketOffsets: Uint32Array,
    readonly bucketIds: Uint32Array,
    readonly dice: import('#core/scoring/choiceIndex.js').ChoiceIndex,
    readonly prepareQuery: (query: Sequence) => PreparedKernel,
    readonly gramSize: number,
  ) {
    this.#marks = new Uint8Array(prepared.length)
  }

  candidates(query: Sequence, threshold: number): CandidateChoices {
    const size = this.prepared.length
    if (this.#ids.length < size) this.#ids = new Uint32Array(size)
    if (Number.isNaN(threshold) || threshold > 1 || size === 0) {
      return { ids: this.#ids, length: 0 }
    }
    if (threshold <= 0) {
      for (let id = 0; id < size; id++) this.#ids[id] = id
      return { ids: this.#ids, length: size }
    }

    this.#clearMarks()
    const queryLength = scorerSequence(query).length
    const queryGrams = Math.max(queryLength - this.gramSize + 1, 0)
    const minimumByBucket = new Float64Array(this.bucketLengths.length)
    let minimumDice = Number.POSITIVE_INFINITY
    for (let bucket = 0; bucket < this.bucketLengths.length; bucket++) {
      const choiceLength = this.bucketLengths[bucket]
      const maximum = queryLength + choiceLength
      let distance = maximumQualifyingDistance(maximum, threshold)
      if (distance < Math.abs(queryLength - choiceLength)) {
        minimumByBucket[bucket] = Number.POSITIVE_INFINITY
        continue
      }
      if ((distance & 1) !== (maximum & 1)) distance--
      const choiceGrams = Math.max(choiceLength - this.gramSize + 1, 0)
      const minimum = Math.max(
        0,
        Math.ceil((queryGrams + choiceGrams - (2 * this.gramSize - 1) * distance) / 2),
      )
      minimumByBucket[bucket] = minimum
      if (queryGrams + choiceGrams === 0 || minimum === 0) this.#touchBucket(bucket)
      else {
        minimumDice = Math.min(minimumDice, (2 * minimum) / (queryGrams + choiceGrams))
      }
    }

    if (Number.isFinite(minimumDice)) {
      const found = this.dice.scan(query, minimumDice)
      for (let at = 0; at < found.length; at++) {
        const id = found.ids[at]
        const bucket = this.bucketOfId[id]
        const minimum = minimumByBucket[bucket]
        if (!Number.isFinite(minimum)) continue
        const choiceLength = this.bucketLengths[bucket]
        const choiceGrams = Math.max(choiceLength - this.gramSize + 1, 0)
        // Both gram counts sum to less than 2^33 and shared overlap is below
        // 2^32 under the index limits. Dice's accumulated Float64 error is
        // therefore far below half an integer, so rounding recovers overlap.
        const shared = Math.round((found.scores[at] * (queryGrams + choiceGrams)) / 2)
        if (shared >= minimum) this.#touch(id)
      }
    }

    this.#touched.sort((left, right) => left - right)
    const kernel = this.prepareQuery(query)
    let length = 0
    for (const id of this.#touched) {
      if (kernel(this.prepared[id], null) >= threshold) this.#ids[length++] = id
    }
    return { ids: this.#ids, length }
  }

  #touchBucket(bucket: number): void {
    for (let at = this.bucketOffsets[bucket]; at < this.bucketOffsets[bucket + 1]; at++) {
      this.#touch(this.bucketIds[at])
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

export function createIndelCandidateIndexBuilder(
  prepareQuery: (query: Sequence) => PreparedKernel,
  gramSize: number = INDEL_GRAM_SIZE,
): CandidateIndexBuilder {
  const prepared: unknown[] = []
  const buckets = new Map<number, number[]>()
  const dice = createDiceIndexBuilder(gramSize)
  let sealed = false
  return {
    add(choice) {
      if (sealed) throw new TypeError('candidate index builder is already sealed')
      const owned = snapshotSequence(choice)
      const value = scorerSequence(owned)
      const id = prepared.length
      prepared.push(value)
      let bucket = buckets.get(value.length)
      if (bucket === undefined) buckets.set(value.length, (bucket = []))
      bucket.push(id)
      dice.add(owned)
    },
    seal() {
      if (sealed) throw new TypeError('candidate index builder is already sealed')
      sealed = true
      const ordered = [...buckets].sort((left, right) => left[0] - right[0])
      const bucketLengths = Uint32Array.from(ordered.map(([length]) => length))
      const bucketOffsets = new Uint32Array(ordered.length + 1)
      const bucketIds = new Uint32Array(prepared.length)
      const bucketOfId = new Uint32Array(prepared.length)
      let at = 0
      for (let bucket = 0; bucket < ordered.length; bucket++) {
        bucketOffsets[bucket] = at
        for (const id of ordered[bucket][1]) {
          bucketOfId[id] = bucket
          bucketIds[at++] = id
        }
      }
      bucketOffsets[ordered.length] = at
      return new IndelCandidateIndex(
        prepared,
        bucketOfId,
        bucketLengths,
        bucketOffsets,
        bucketIds,
        dice.seal(),
        prepareQuery,
        gramSize,
      )
    },
  }
}
