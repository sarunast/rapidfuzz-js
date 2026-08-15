import type {
  ChoiceIndex,
  ChoiceIndexBuilder,
  SelectedChoices,
} from '../../../../core/scoring/choiceIndex.js'
import { convSequence } from '../../../../core/sequence.js'
import type { Sequence } from '../../../../core/types.js'
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

export function assertDiceAccumulatorExact(gramCount: number): void {
  if (gramCount > 0x7fff_ffff) {
    throw new RangeError('a query of more than 2147483647 grams cannot be indexed')
  }
}

class DiceIndex implements ChoiceIndex {
  private readonly state = new QueryState()
  private readonly accumulator: Int32Array

  constructor(private readonly sealed: SealedIndex<null>) {
    this.accumulator = new Int32Array(sealed.choiceCount)
  }

  private begin(query: Sequence): ArrayLike<unknown> {
    const elements = convSequence(query)
    assertDiceAccumulatorExact(elements.length - this.sealed.gramSize + 1)
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
    const queryGrams = elements.length - sealed.gramSize + 1
    extractGrams(elements, sealed.gramSize, sealed.radix, false, state.keys, state.counts)
    this.accumulate()
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
    const queryGrams = elements.length - sealed.gramSize + 1
    extractGrams(elements, sealed.gramSize, sealed.radix, false, state.keys, state.counts)
    this.accumulate()
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
          : (2 * (base + accumulator[id])) / (queryGrams + choiceGrams)
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
        state.base += 1
        if (postingCounts === null) {
          for (let at = from; at < upto; at++) accumulator[ids[at]] -= 1
          continue
        }
        for (let at = from; at < upto; at++) {
          const count = postingCounts[at]
          accumulator[ids[at]] += (queryCount < count ? queryCount : count) - 1
        }
        continue
      }
      if (!tracking) {
        if (postingCounts === null) {
          for (let at = from; at < upto; at++) accumulator[ids[at]] += 1
          continue
        }
        for (let at = from; at < upto; at++) {
          const count = postingCounts[at]
          accumulator[ids[at]] += queryCount < count ? queryCount : count
        }
        continue
      }
      if (postingCounts === null) {
        for (let at = from; at < upto; at++) {
          const id = ids[at]
          if (accumulator[id] === 0) touched.push(id)
          accumulator[id] += 1
        }
        continue
      }
      for (let at = from; at < upto; at++) {
        const id = ids[at]
        if (accumulator[id] === 0) touched.push(id)
        const count = postingCounts[at]
        accumulator[id] += queryCount < count ? queryCount : count
      }
    }
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
          : (2 * (base + accumulator[id])) / (queryGrams + choiceGrams)
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

export function createDiceIndexBuilder(gramSize: number): ChoiceIndexBuilder {
  return new NGramIndexBuilder(
    gramSize,
    () => null,
    (sealed) => new DiceIndex(sealed),
  )
}
