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

/**
 * Dice accumulates `Σ min(a, b)`, which cannot exceed the query's own gram
 * count, so this is the whole of its `Int32Array` accumulator's exactness
 * condition. Unreachable for any real text — it is 2.1 billion grams — and
 * checked rather than assumed because its failure mode is a wrong score rather
 * than a thrown error.
 */
export function assertDiceAccumulatorExact(gramCount: number): void {
  if (gramCount > 0x7fff_ffff) {
    throw new RangeError('a query of more than 2147483647 grams cannot be indexed')
  }
}

/**
 * A Sørensen-Dice index: `2 · Σ min(a, b) / (gramsA + gramsB)`, with the shared
 * count coming straight out of accumulation. No kernel call is left — the index
 * *is* the scorer.
 *
 * Its accumulator is an `Int32Array` where Cosine's is a `Float64Array`, which
 * is why the two are separate classes rather than one carrying a mode: Dice's
 * overlap is a sum of `min(queryCount, choiceCount)` terms and so bounded by the
 * query's own gram count, and narrowing it measured 1.05–1.68x. The
 * read-modify-write did not shrink so much as vanish — an integer add on four
 * bytes where a `Float64Array` cost a conversion, a double add and eight.
 */
class DiceIndex implements ChoiceIndex {
  private readonly state = new QueryState()
  private readonly accumulator: Int32Array

  constructor(private readonly sealed: SealedIndex<null>) {
    this.accumulator = new Int32Array(sealed.choiceCount)
  }

  /** Converts the query and checks the overlap accumulator's exactness bound. */
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
    // Collect and sort rather than insert into place: with no limit there is no
    // room bound to make the insertion walk in `top` cheap.
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

  /**
   * Every qualifying choice, in the cheapest order the caller can use: `scan`
   * needs ascending ids and pays for them, while a ranked call sorts by score
   * afterwards and would throw that order away. Ordering the touched set is not
   * a rounding error — it measured 84% of a `threshold: 0.5` query over 10,000
   * choices, where accumulation itself was 5%.
   */
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

  /**
   * Two literal loops per posting-list shape rather than one with a branch in
   * it: this is the innermost frame of the whole representation, and the shape
   * is known before it starts.
   *
   * No membership marks. Where the touched set is read no dense list was
   * reached, so every contribution below is strictly positive and an untouched
   * accumulator entry is still exactly zero; where a dense list was reached the
   * set is never read at all. A generation-mark array cost 26% of this loop for
   * a set that is either unread or already implied.
   */
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
        // Every choice holds this gram once unless the slice says otherwise, so
        // the whole corpus takes `min(queryCount, 1)` in one addition and the
        // loop walks only the exceptions.
        // `min(queryCount, 1)` is `1`: `extractGrams` starts every frequency at
        // one and only increments, so a query gram is never seen zero times.
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

  /**
   * Top-k with Dice's arithmetic in the loop rather than behind a callback.
   *
   * The callback was a good trade while selection ran once per *touched* choice.
   * A dense list makes it run once per choice in the corpus, and inlining the
   * arithmetic measured 1.41–1.92x end to end where closing the callback over
   * locals recovered only 1.04–1.18x. The rest is the call boundary itself,
   * which only duplication removes.
   */
  private top(queryGrams: number, threshold: number | null, room: number): number {
    // A caller may ask for nothing, and then there is no result array to insert
    // into: `room - 1` would read off the front of one.
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
      // Scored rather than skipped: a dense list puts every choice into this
      // walk, gramless ones included, and they are then not zero-filled either.
      // A gramless choice against a query that has grams shares nothing.
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

  /**
   * Clears only what the query touched. Walking the whole accumulator would put
   * a cost proportional to the corpus back into every query, which is the one
   * thing this representation exists to avoid — except where a dense list has
   * already made the walk the whole corpus, and then `fill` beats it.
   */
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

/** A builder for a Sørensen-Dice index over grams of `gramSize` elements. */
export function createDiceIndexBuilder(gramSize: number): ChoiceIndexBuilder {
  return new NGramIndexBuilder(
    gramSize,
    () => null,
    (sealed) => new DiceIndex(sealed),
  )
}
