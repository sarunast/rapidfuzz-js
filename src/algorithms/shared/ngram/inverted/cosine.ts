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
 * Cosine's dot product is `Σ qᵢ·cᵢ`, which is bounded by
 * `gramCount(query) · gramCount(choice)` — so while that product is a safe
 * integer, every term and every partial sum is exact whatever order they are
 * added in, and the index matches the exhaustive scorer to the bit.
 *
 * Above it they can disagree, because a dense list decomposes a repeated gram's
 * contribution as `q·(c-1) + q` where a sparse one computes `q·c`: at
 * `q = 116,982,125` and `c = 105,643,526` those are 12358404163972748 and
 * 12358404163972750. Checked rather than assumed, for the reason the Dice bound
 * in `assertDiceAccumulatorExact` is: the failure mode is a wrong score rather than a
 * thrown error. It takes ~100-million-gram sequences on both sides to reach.
 */
export function assertCosineExact(queryGrams: number, maxChoiceGrams: number): void {
  if (queryGrams * maxChoiceGrams > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(
      'a cosine query of this many grams cannot be scored exactly against a choice this long',
    )
  }
}

/**
 * The other half of Cosine's denominator, and a second boundary since prepared
 * profiles began packing their grams: `Σ count²` is summed here as `2c + 1` per
 * occurrence, and by a packed profile as `c²` per distinct gram, because
 * counting a run of sorted keys is where its counts come from. Both are exact
 * while every squared norm is a safe integer, so both sides answer alike.
 *
 * Above it neither is exact and they need not agree — one gram repeated
 * 268,435,459 times puts them 16 apart, one ulp at that magnitude — while a
 * merely large norm usually survives: the same pair agrees at 200,000,001.
 * Agreement up there is luck rather than a property, so the bound is the one
 * that can be proved.
 *
 * A norm rather than a length, deliberately. `Σ count² ≤ gramCount²` would make
 * `gramCount ≤ 94,906,265` a sufficient test, and it would refuse a
 * 100-million-gram query of distinct grams whose norm is nowhere near the
 * boundary. What decides this is repetition, so repetition is what it reads.
 */
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

/**
 * One square root of the product and then a clamp, which is the arithmetic the
 * exhaustive kernel uses and for its reason: `Math.sqrt(3) * Math.sqrt(3)` is
 * `3.0000000000000004`, which would leave a profile scored against itself just
 * short of `1`.
 */
function clamp(similarity: number): number {
  return similarity < 1 ? similarity : 1
}

/**
 * A Cosine index: `Σ a·b / √(‖a‖² ‖b‖²)`, clamped.
 *
 * Its accumulator stays `Float64Array`. The dot product is bounded by
 * `queryGrams × choiceGrams` rather than by the query alone, so a long query
 * against a long choice can carry it past what an `Int32Array` holds, and the
 * failure mode would be a wrong score rather than a thrown error.
 *
 * Two exactness conditions rather than one, and the denominator owns the
 * second: a packed profile sums the same squared norm in a different order, so
 * a query checks `assertCosineExact` for the numerator before extraction and
 * `assertCosineNormsExact` for the norms after it, once the query's own is
 * known.
 */
class CosineIndex implements ChoiceIndex {
  private readonly state = new QueryState()
  private readonly accumulator: Float64Array

  constructor(private readonly sealed: SealedIndex<Float64Array>) {
    this.accumulator = new Float64Array(sealed.choiceCount)
  }

  /**
   * `Σ qᵢ·cᵢ ≤ gramCount(query) · gramCount(choice)`, and the longest choice in
   * the index is the one that can carry it past a double's exact integers —
   * where a dense list and a sparse one stop agreeing to the bit.
   */
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
    // Collect and sort rather than insert into place: with no limit there is no
    // room bound to make the insertion walk in `top` cheap.
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

  /**
   * Every qualifying choice, in the cheapest order the caller can use: `scan`
   * needs ascending ids and pays for them, while a ranked call sorts by score
   * afterwards and would throw that order away.
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
        // The dot product's default term is `queryCount × 1`, and an exception
        // replaces it: an absent choice gives back the whole term, a repeated
        // gram adds the extra `count − 1` copies.
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
    // A caller may ask for nothing, and then there is no result array to insert
    // into: `room - 1` would read off the front of one.
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
      // Scored rather than skipped: a dense list puts every choice into this
      // walk, gramless ones included. The zero-norm guard is load-bearing — the
      // dense `base` applies to them too, so without it a gramless choice would
      // divide a positive numerator by zero and the clamp would turn the
      // infinity into a perfect score, which is what it used to get.
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

/** A builder for a Cosine index over grams of `gramSize` elements. */
export function createCosineIndexBuilder(gramSize: number): ChoiceIndexBuilder {
  return new NGramIndexBuilder(
    gramSize,
    (values) => Float64Array.from(values),
    (sealed) => new CosineIndex(sealed),
  )
}
