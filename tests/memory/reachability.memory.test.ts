import { queryObjects } from 'node:v8'

import { describe, expect, it } from 'vitest'

import { similarity as cosineSimilarity } from '../../src/algorithms/cosine/index.js'
import { similarity as diceSimilarity } from '../../src/algorithms/dice/index.js'
import { QueryState } from '../../src/algorithms/ngram/inverted/query.js'
import { createScorer } from '../../src/core/scoring/scorer.js'
import { createIndexedMatcher } from '../../src/search/matcher/createIndexedMatcher.js'

const choices = ['ab', 'bc', 'cd', 'de'] as const
const dice = createScorer(diceSimilarity, { gramSize: 2 })
const cosine = createScorer(cosineSimilarity, { gramSize: 2 })

function count(constructor: Function): number {
  return queryObjects(constructor, { format: 'count' })
}

function constructUseAndDrop(kind: 'dice' | 'cosine'): void {
  const matcher =
    kind === 'dice'
      ? createIndexedMatcher([...choices], { scorer: dice })
      : createIndexedMatcher([...choices], { scorer: cosine })
  matcher.best('bc')
  matcher.search('cd', { limit: 2 })
}

class CorpusItem {
  constructor(readonly text: string) {}
}

function withLiveCorpusMatcher(check: () => void): void {
  const matcher = createIndexedMatcher([new CorpusItem('ab')], {
    scorer: dice,
    getText: (item) => item.text,
  })
  matcher.best('ab')
  check()
  if (matcher.size !== 1) throw new Error('matcher unexpectedly changed')
}

class NumericQuery implements ArrayLike<number> {
  readonly length = 4;
  [index: number]: number

  constructor() {
    this[0] = 97
    this[1] = 98
    this[2] = 99
    this[3] = 100
  }
}

class IteratorQuery implements ArrayLike<number> {
  readonly length = 2;
  [index: number]: number

  constructor() {
    this[0] = 97
    this[1] = 98
  }
}

function submitNumericQuery(
  matcher: import('../../bench/memory/workloads/shared.ts').IndexedMatcherWorkload,
): void {
  matcher.best(new NumericQuery())
}

function startAndDropSearchIterator(matcher: {
  searchIter(query: IteratorQuery): IterableIterator<unknown>
}): void {
  const iterator = matcher.searchIter(new IteratorQuery())
  expect(iterator.next().done).toBe(false)
}

class InvalidSentinel {
  toString(): string {
    return 'invalid-sentinel'
  }
}

class LateInvalidWrapper implements ArrayLike<unknown> {
  readonly length = 10_001;
  [index: number]: unknown

  constructor(sentinel: InvalidSentinel) {
    for (let index = 0; index < this.length - 1; index++) this[index] = index
    this[this.length - 1] = sentinel
  }
}

function submitLateInvalid(
  matcher: import('../../bench/memory/workloads/shared.ts').IndexedMatcherWorkload,
): void {
  expect(() => matcher.best(new LateInvalidWrapper(new InvalidSentinel()))).toThrow(
    /integer elements only/,
  )
}

describe.sequential('indexed matcher reachability', () => {
  it('returns Dice and Cosine query states to baseline after destruction', () => {
    const baseline = count(QueryState)
    for (const kind of ['dice', 'cosine'] as const) {
      for (let repeat = 0; repeat < 5; repeat++) constructUseAndDrop(kind)
      expect(count(QueryState)).toBe(baseline)
    }
  })

  it('holds exactly one query state for one live matcher across queries', () => {
    const baseline = count(QueryState)
    ;(() => {
      const matcher = createIndexedMatcher(choices, { scorer: dice })
      for (let repeat = 0; repeat < 20; repeat++) matcher.best(`a${repeat}`)
      const observed = count(QueryState)
      expect(matcher.size).toBe(choices.length)
      expect(observed).toBe(baseline + 1)
    })()
    expect(count(QueryState)).toBe(baseline)
  })

  it('holds corpus items only for the matcher lifetime', () => {
    const baseline = count(CorpusItem)
    withLiveCorpusMatcher(() => expect(count(CorpusItem)).toBe(baseline + 1))
    expect(count(CorpusItem)).toBe(baseline)
  })

  it('does not retain a numeric array-like query', () => {
    const matcher = createIndexedMatcher(choices, { scorer: dice })
    const baseline = count(NumericQuery)
    submitNumericQuery(matcher)
    const observed = count(NumericQuery)
    expect(matcher.size).toBe(choices.length)
    expect(observed).toBe(baseline)
  })

  it('releases a partially consumed searchIter query when its iterator is dropped', () => {
    const matcher = createIndexedMatcher(choices, { scorer: dice })
    const baseline = count(IteratorQuery)
    startAndDropSearchIterator(matcher)
    const observed = count(IteratorQuery)
    expect(matcher.size).toBe(choices.length)
    expect(observed).toBe(baseline)
  })

  it('collects late-invalid wrappers and sentinels after a successful query', () => {
    const matcher = createIndexedMatcher(choices, { scorer: dice })
    const wrapperBaseline = count(LateInvalidWrapper)
    const sentinelBaseline = count(InvalidSentinel)
    submitLateInvalid(matcher)
    matcher.best('ab')
    const wrapperObserved = count(LateInvalidWrapper)
    const sentinelObserved = count(InvalidSentinel)
    expect(matcher.size).toBe(choices.length)
    expect(wrapperObserved).toBe(wrapperBaseline)
    expect(sentinelObserved).toBe(sentinelBaseline)
  })
})
