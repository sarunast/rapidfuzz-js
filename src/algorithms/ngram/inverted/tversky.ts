import type {
  ChoiceIndex,
  ChoiceIndexBuilder,
  SelectedChoices,
} from '#core/scoring/choiceIndex.js'
import { convSequence } from '#core/sequence.js'
import type { Sequence } from '#core/types.js'

import { NGramIndexBuilder, type SealedIndex } from './builder.js'
import { accumulateSharedFrequency, assertSharedAccumulatorExact } from './overlap.js'
import {
  fillZeroes,
  gramlessResult,
  outranks,
  prepareQueryGrams,
  QueryState,
  rankSelected,
  resetQuery,
  roomFor,
  sortTouched,
  zeroesQualify,
} from './query.js'

class TverskyIndex implements ChoiceIndex {
  private readonly state = new QueryState()
  private readonly accumulator: Int32Array
  private readonly scale: number
  private readonly scaledAlpha: number
  private readonly scaledBeta: number

  constructor(
    private readonly sealed: SealedIndex<null>,
    alpha: number,
    beta: number,
  ) {
    this.accumulator = new Int32Array(sealed.choiceCount)
    this.scale = Math.max(1, alpha, beta)
    this.scaledAlpha = alpha / this.scale
    this.scaledBeta = beta / this.scale
  }

  private begin(query: Sequence): ArrayLike<unknown> {
    const elements = convSequence(query)
    assertSharedAccumulatorExact(elements.length - this.sealed.gramSize + 1)
    return elements
  }

  // `shared / scale` stays a division so the quotient is bit-identical to
  // `tverskyScore`'s, and the two penalty terms are summed before the
  // numerator joins for the same commutation guarantee.
  private score(shared: number, queryGrams: number, choiceGrams: number): number {
    const numerator = shared / this.scale
    const unmatched =
      this.scaledAlpha * (queryGrams - shared) + this.scaledBeta * (choiceGrams - shared)
    return numerator / (numerator + unmatched)
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
    const queryGrams = elements.length - sealed.gramSize + 1
    prepareQueryGrams(sealed, elements, state)
    accumulateSharedFrequency(sealed, state, this.accumulator)
    const room = roomFor(limit, sealed.choiceCount)
    state.reserve(room)
    const length = fillZeroes(
      sealed,
      state,
      this.accumulator,
      this.top(queryGrams, threshold, room),
      threshold,
      room,
    )
    resetQuery(state, this.accumulator)
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
    const queryGrams = elements.length - sealed.gramSize + 1
    prepareQueryGrams(sealed, elements, state)
    accumulateSharedFrequency(sealed, state, this.accumulator)
    const everyChoice = state.scannedAll || zeroesQualify(threshold)
    const source = everyChoice ? null : ascending ? sortTouched(state) : state.touched
    const total = source === null ? sealed.choiceCount : source.length
    state.reserve(total)
    const ids = state.ids
    const scores = state.scores
    const accumulator = this.accumulator
    const choiceGramCounts = sealed.gramCount
    const base = state.base
    let length = 0
    for (let index = 0; index < total; index++) {
      const id = source === null ? index : source[index]
      const choiceGrams = choiceGramCounts[id]
      const score =
        choiceGrams === 0
          ? 0
          : this.score(base + accumulator[id], queryGrams, choiceGrams)
      if (threshold !== null && score < threshold) continue
      ids[length] = id
      scores[length] = score
      length++
    }
    resetQuery(state, this.accumulator)
    return { ids, scores, length }
  }

  private top(queryGrams: number, threshold: number | null, room: number): number {
    if (room === 0) return 0
    const sealed = this.sealed
    const state = this.state
    const touched = state.touched
    const accumulator = this.accumulator
    const choiceGramCounts = sealed.gramCount
    const base = state.base
    const everyChoice = state.scannedAll
    const total = everyChoice ? sealed.choiceCount : touched.length
    const ids = state.ids
    const scores = state.scores
    let length = 0
    for (let index = 0; index < total; index++) {
      const id = everyChoice ? index : touched[index]
      const choiceGrams = choiceGramCounts[id]
      const score =
        choiceGrams === 0
          ? 0
          : this.score(base + accumulator[id], queryGrams, choiceGrams)
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
}

export function createTverskyIndexBuilder(
  gramSize: number,
  alpha: number,
  beta: number,
): ChoiceIndexBuilder {
  return new NGramIndexBuilder(
    gramSize,
    () => null,
    (sealed) => new TverskyIndex(sealed, alpha, beta),
  )
}
