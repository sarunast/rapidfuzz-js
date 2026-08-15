import type {
  ChoiceIndex,
  ChoiceIndexBuilder,
  SelectedChoices,
} from '../../../../core/scoring/choiceIndex.js'
import type { Sequence } from '../../../../core/types.js'
import { convSequence } from '../../sequence.js'
import { NGramIndexBuilder, type SealedIndex } from './builder.js'
import { extractGrams } from './keys.js'
import {
  fillZeroes,
  gramlessResult,
  outranks,
  QueryState,
  rankSelected,
  reachesDenseList,
  roomFor,
  sortTouched,
  zeroesQualify,
} from './query.js'

export function assertCosineExact(queryGrams: number, maxChoiceGrams: number): void {
  if (queryGrams * maxChoiceGrams > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(
      'a cosine query of this many grams cannot be scored exactly against a choice this long',
    )
  }
}

export function assertCosineNormsExact(
  querySquaredNorm: number,
  maxSquaredNorm: number,
): void {
  if (
    querySquaredNorm > Number.MAX_SAFE_INTEGER ||
    maxSquaredNorm > Number.MAX_SAFE_INTEGER
  ) {
    throw new RangeError(
      'a cosine query with grams repeated this often cannot be scored exactly against this corpus',
    )
  }
}

function clamp(similarity: number): number {
  return similarity < 1 ? similarity : 1
}

class CosineIndex implements ChoiceIndex {
  private readonly state = new QueryState()
  private readonly accumulator: Float64Array

  constructor(private readonly sealed: SealedIndex<Float64Array>) {
    this.accumulator = new Float64Array(sealed.choiceCount)
  }

  private begin(query: Sequence): ArrayLike<unknown> {
    const elements = convSequence(query)
    assertCosineExact(
      elements.length - this.sealed.gramSize + 1,
      this.sealed.maxGramCount,
    )
    return elements
  }

  select(
    query: Sequence,
    threshold: number | null,
    limit: number | null,
  ): SelectedChoices {
    if (limit === null) return rankSelected(this.collect(query, threshold, false))
    const sealed = this.sealed
    const state = this.state
    const elements = this.begin(query)
    if (elements.length < sealed.gramSize) {
      return gramlessResult(sealed, state, elements, threshold, limit, false)
    }
    const querySquaredNorm = extractGrams(
      elements,
      sealed.gramSize,
      sealed.radix,
      false,
      state.keys,
      state.counts,
    )
    assertCosineNormsExact(querySquaredNorm, sealed.maxSquaredNorm)
    this.accumulate()
    const room = roomFor(limit, sealed.choiceCount)
    state.reserve(room)
    const length = fillZeroes(
      sealed,
      state,
      this.accumulator,
      this.top(querySquaredNorm, threshold, room),
      threshold,
      room,
    )
    this.reset()
    return { ids: state.ids, scores: state.scores, length }
  }

  scan(query: Sequence, threshold: number | null): SelectedChoices {
    return this.collect(query, threshold, true)
  }

  private collect(
    query: Sequence,
    threshold: number | null,
    ascending: boolean,
  ): SelectedChoices {
    const sealed = this.sealed
    const state = this.state
    const elements = this.begin(query)
    if (elements.length < sealed.gramSize) {
      return gramlessResult(sealed, state, elements, threshold, null, ascending)
    }
    const querySquaredNorm = extractGrams(
      elements,
      sealed.gramSize,
      sealed.radix,
      false,
      state.keys,
      state.counts,
    )
    assertCosineNormsExact(querySquaredNorm, sealed.maxSquaredNorm)
    this.accumulate()
    const everyChoice = state.scannedAll || zeroesQualify(threshold)
    const source = everyChoice ? null : ascending ? sortTouched(state) : state.touched
    const total = source === null ? sealed.choiceCount : source.length
    state.reserve(total)
    const ids = state.ids
    const scores = state.scores
    const accumulator = this.accumulator
    const squaredNorm = sealed.squaredNorm
    const base = state.base
    let length = 0
    for (let index = 0; index < total; index++) {
      const id = source === null ? index : source[index]
      const choiceSquaredNorm = squaredNorm[id]
      const score =
        choiceSquaredNorm === 0
          ? 0
          : clamp(
              (base + accumulator[id]) / Math.sqrt(querySquaredNorm * choiceSquaredNorm),
            )
      if (threshold !== null && score < threshold) continue
      ids[length] = id
      scores[length] = score
      length++
    }
    this.reset()
    return { ids, scores, length }
  }

  private accumulate(): void {
    const sealed = this.sealed
    const state = this.state
    const postings = sealed.postings
    const accumulator = this.accumulator
    const touched = state.touched
    const keys = state.keys
    const queryCounts = state.counts
    const ids = postings.ids
    const postingCounts = postings.counts
    const offsets = postings.offsets
    const dense = postings.dense
    state.base = 0
    if (dense !== null && reachesDenseList(postings, dense, keys)) state.scannedAll = true
    const tracking = !state.scannedAll
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      if (ordinal === undefined) continue
      const queryCount = queryCounts[index]
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      if (dense !== null && dense[ordinal] === 1) {
        state.base += queryCount
        if (postingCounts === null) {
          for (let at = from; at < upto; at++) accumulator[ids[at]] -= queryCount
          continue
        }
        for (let at = from; at < upto; at++) {
          accumulator[ids[at]] += queryCount * (postingCounts[at] - 1)
        }
        continue
      }
      if (!tracking) {
        if (postingCounts === null) {
          for (let at = from; at < upto; at++) accumulator[ids[at]] += queryCount
          continue
        }
        for (let at = from; at < upto; at++) {
          accumulator[ids[at]] += queryCount * postingCounts[at]
        }
        continue
      }
      if (postingCounts === null) {
        for (let at = from; at < upto; at++) {
          const id = ids[at]
          if (accumulator[id] === 0) touched.push(id)
          accumulator[id] += queryCount
        }
        continue
      }
      for (let at = from; at < upto; at++) {
        const id = ids[at]
        if (accumulator[id] === 0) touched.push(id)
        accumulator[id] += queryCount * postingCounts[at]
      }
    }
  }

  private top(querySquaredNorm: number, threshold: number | null, room: number): number {
    if (room === 0) return 0
    const sealed = this.sealed
    const state = this.state
    const touched = state.touched
    const accumulator = this.accumulator
    const squaredNorm = sealed.squaredNorm
    const base = state.base
    const everyChoice = state.scannedAll
    const total = everyChoice ? sealed.choiceCount : touched.length
    const ids = state.ids
    const scores = state.scores
    let length = 0
    for (let index = 0; index < total; index++) {
      const id = everyChoice ? index : touched[index]
      const choiceSquaredNorm = squaredNorm[id]
      const score =
        choiceSquaredNorm === 0
          ? 0
          : clamp(
              (base + accumulator[id]) / Math.sqrt(querySquaredNorm * choiceSquaredNorm),
            )
      if (threshold !== null && score < threshold) continue
      let at = length
      if (at === room) {
        if (!outranks(score, id, scores[room - 1], ids[room - 1])) continue
        at = room - 1
      } else {
        length++
      }
      while (at > 0 && outranks(score, id, scores[at - 1], ids[at - 1])) {
        ids[at] = ids[at - 1]
        scores[at] = scores[at - 1]
        at--
      }
      ids[at] = id
      scores[at] = score
    }
    return length
  }

  private reset(): void {
    const state = this.state
    const accumulator = this.accumulator
    const touched = state.touched
    if (state.scannedAll) accumulator.fill(0)
    else
      for (let index = 0; index < touched.length; index++) accumulator[touched[index]] = 0
    touched.length = 0
    state.scannedAll = false
    state.base = 0
  }
}

export function createCosineIndexBuilder(gramSize: number): ChoiceIndexBuilder {
  return new NGramIndexBuilder(
    gramSize,
    (values) => Float64Array.from(values),
    (sealed) => new CosineIndex(sealed),
  )
}
