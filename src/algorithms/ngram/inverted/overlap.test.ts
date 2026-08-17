import { describe, expect, it } from 'vitest'

import type { ChoiceIndex } from '#core/scoring/choiceIndex.js'
import { convSequence } from '#core/sequence.js'

import { NGramIndexBuilder, type SealedIndex } from './builder.js'
import { extractGrams } from './keys.js'
import { accumulateSharedFrequency } from './overlap.js'
import { QueryState, resetQuery } from './query.js'

// A builder hands its sealed structure to the callback rather than returning
// it, so the walk is reached by capturing what a real seal produced.
const stubIndex: ChoiceIndex = {
  select() {
    throw new Error('the captured index answers no queries')
  },
  scan() {
    throw new Error('the captured index answers no queries')
  },
}

function sealedOf(gramSize: number, choices: readonly string[]): SealedIndex<null> {
  let captured: SealedIndex<null> | undefined
  const builder = new NGramIndexBuilder<null>(
    gramSize,
    () => null,
    (sealed) => {
      captured = sealed
      return stubIndex
    },
  )
  for (const choice of choices) builder.add(choice)
  builder.seal()
  if (captured === undefined) throw new Error('the builder did not seal')
  return captured
}

/** The shared count per choice id, which is `state.base + accumulator[id]`. */
function accumulate(
  sealed: SealedIndex<null>,
  query: string,
  state: QueryState,
  accumulator: Int32Array,
): number[] {
  const elements = convSequence(query)
  extractGrams(elements, sealed.gramSize, sealed.radix, null, state.keys, state.counts)
  accumulateSharedFrequency(sealed, state, accumulator)
  return Array.from(accumulator, (value) => state.base + value)
}

function scratchFor(sealed: SealedIndex<null>): {
  state: QueryState
  accumulator: Int32Array
} {
  return { state: new QueryState(), accumulator: new Int32Array(sealed.choiceCount) }
}

describe('the shared-frequency posting walk', () => {
  it('accumulates the lists a sparse query touches, and nothing else', () => {
    const sealed = sealedOf(2, ['abc', 'xyd', 'abzz', 'qqq'])
    const { state, accumulator } = scratchFor(sealed)

    expect(sealed.postings.dense).toBeNull()

    const shared = accumulate(sealed, 'abc', state, accumulator)

    expect(state.scannedAll).toBe(false)
    expect(state.base).toBe(0)
    expect(shared).toEqual([2, 0, 1, 0])
    // Insertion order is traversal order, which is not a contract: what the
    // walk owes is every touched candidate exactly once and no untouched one.
    expect(new Set(state.touched).size).toBe(state.touched.length)
    expect([...state.touched].sort((left, right) => left - right)).toEqual([0, 2])
  })

  it('shares the smaller frequency, whichever side holds it', () => {
    // One gram, `a`, held three times by the query against choices holding it
    // twice and four times.
    const sealed = sealedOf(1, ['aa', 'aaaa'])
    const { state, accumulator } = scratchFor(sealed)

    expect(sealed.postings.dense).toBeNull()
    expect(sealed.postings.counts).not.toBeNull()

    expect(accumulate(sealed, 'aaa', state, accumulator)).toEqual([2, 3])
  })

  it('reads a dense list that stores absences alone', () => {
    // `ab` is in six of seven choices and nothing repeats anywhere, so the
    // exceptions are the absent choice and the posting carries no counts.
    const sealed = sealedOf(2, ['abc', 'abd', 'abe', 'abf', 'abg', 'abh', 'xyz'])
    const { state, accumulator } = scratchFor(sealed)

    expect(sealed.postings.dense).not.toBeNull()
    expect(sealed.postings.counts).toBeNull()

    // `bc` is a sparse list holding choice 0 alone, so that choice reads the
    // dense contribution and a sparse one.
    const shared = accumulate(sealed, 'abc', state, accumulator)

    expect(state.scannedAll).toBe(true)
    expect(state.touched).toEqual([])
    expect(state.base).toBe(1)
    expect(shared).toEqual([2, 1, 1, 1, 1, 1, 0])
  })

  it('reads a dense list whose exceptions are frequencies', () => {
    // `aa` is dense, two choices hold it twice, and the query holds it twice —
    // so `min(queryCount, count) - 1` runs with a query count above one.
    const sealed = sealedOf(2, ['aab', 'aaab', 'aaac', 'aad', 'aae', 'aaf', 'zz'])
    const { state, accumulator } = scratchFor(sealed)

    expect(sealed.postings.dense).not.toBeNull()
    expect(sealed.postings.counts).not.toBeNull()

    const shared = accumulate(sealed, 'aaa', state, accumulator)

    expect(state.scannedAll).toBe(true)
    expect(state.base).toBe(1)
    expect(shared).toEqual([1, 2, 2, 1, 1, 1, 0])
  })

  it('returns the scratch to zero between a widened query and a tracked one', () => {
    // `ab` is dense and `zq` is not, so the three queries run both of
    // `resetQuery`'s strategies: the full fill after a widened scan, and the
    // touched walk after a tracked one.
    const choices = ['abc', 'abd', 'abe', 'abf', 'abg', 'abzq', 'zq']
    const sealed = sealedOf(2, choices)
    const { state, accumulator } = scratchFor(sealed)
    const zeroes = new Array<number>(choices.length).fill(0)

    function reset(): void {
      resetQuery(state, accumulator)
      expect([...accumulator]).toEqual(zeroes)
      expect(state.keys).toEqual([])
      expect(state.counts).toEqual([])
      expect(state.touched).toEqual([])
      expect(state.base).toBe(0)
      expect(state.scannedAll).toBe(false)
    }

    const widened = accumulate(sealed, 'abc', state, accumulator)
    expect(state.scannedAll).toBe(true)
    expect(widened).toEqual([2, 1, 1, 1, 1, 1, 0])
    reset()

    const tracked = accumulate(sealed, 'zq', state, accumulator)
    expect(state.scannedAll).toBe(false)
    expect(tracked).toEqual([0, 0, 0, 0, 0, 1, 1])
    reset()

    expect(accumulate(sealed, 'abc', state, accumulator)).toEqual(widened)
    reset()
  })
})
