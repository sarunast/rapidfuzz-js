import { queryObjects } from 'node:v8'

import { describe, expect, it } from 'vitest'

import { similarity as cosineSimilarity } from '../../src/algorithms/cosine/index.js'
import { similarity as diceSimilarity } from '../../src/algorithms/dice/index.js'
import { QueryState } from '../../src/algorithms/ngram/inverted/query.js'
import { similarity as tverskySimilarity } from '../../src/algorithms/tversky/index.js'
import { createScorer } from '../../src/core/scoring/scorer.js'
import { createIndexedMatcher } from '../../src/search/matcher/createIndexedMatcher.js'

const choices = ['ab', 'bc', 'cd', 'de'] as const
const dice = createScorer(diceSimilarity, { gramSize: 2 })
const cosine = createScorer(cosineSimilarity, { gramSize: 2 })
// Non-default weights, so this exercises the Tversky index rather than the
// Dice index the default configuration routes to.
const tversky = createScorer(tverskySimilarity, { gramSize: 2, alpha: 1, beta: 0.1 })

function count(constructor: Function): number {
  return queryObjects(constructor, { format: 'count' })
}

function constructUseAndDrop(kind: 'dice' | 'cosine' | 'tversky'): void {
  const matcher =
    kind === 'dice'
      ? createIndexedMatcher([...choices], { scorer: dice })
      : kind === 'cosine'
        ? createIndexedMatcher([...choices], { scorer: cosine })
        : createIndexedMatcher([...choices], { scorer: tversky })
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

class ArbitrarySentinel {
  toString(): string {
    return 'arbitrary-sentinel'
  }
}

class ArbitraryWrapper implements ArrayLike<unknown> {
  readonly length = 10_001;
  [index: number]: unknown

  constructor(sentinel: ArbitrarySentinel) {
    for (let index = 0; index < this.length - 1; index++) this[index] = index
    this[this.length - 1] = sentinel
  }
}

function submitArbitraryQuery(
  matcher: import('../../bench/memory/workloads/shared.ts').IndexedMatcherWorkload,
): void {
  matcher.best(new ArbitraryWrapper(new ArbitrarySentinel()))
}

class TokenItem {
  constructor(readonly name: string) {}
}

function withLiveTokenMatcher(check: () => void): void {
  const matcher = createIndexedMatcher(
    [
      [new TokenItem('a'), new TokenItem('b')],
      [new TokenItem('c'), new TokenItem('d')],
    ],
    { scorer: dice },
  )
  matcher.best(['unrelated', 'tokens'])
  check()
  if (matcher.size !== 2) throw new Error('matcher unexpectedly changed')
}

// The tokens are made inside `getText` and never handed back, so the index is
// the only thing that could still be holding one when the matcher is live.
function withLiveUnreachableTokenMatcher(check: () => void): void {
  const matcher = createIndexedMatcher(['dead', 'react typescript'], {
    scorer: dice,
    getText: (item) => (item === 'dead' ? [new TokenItem('dead'), NaN] : item.split(' ')),
  })
  matcher.best(['react', 'typescript'])
  check()
  if (matcher.size !== 2) throw new Error('matcher unexpectedly changed')
}

class TokenChoice implements ArrayLike<unknown> {
  readonly length = 2;
  [index: number]: unknown

  constructor() {
    this[0] = 'react'
    this[1] = 'typescript'
  }
}

// The check runs while the matcher is live, because a matcher already dead
// retains nothing whatever the index did with a choice. The sequences are made
// inside `getText` rather than being the items themselves — a matcher holds its
// items to return them, which would answer the question before the index got
// to.
function withLiveTokenChoiceMatcher(check: () => void): void {
  const matcher = createIndexedMatcher(['first', 'second'], {
    scorer: dice,
    getText: () => new TokenChoice(),
  })
  check()
  if (matcher.size !== 2) throw new Error('matcher unexpectedly changed')
}

describe.sequential('indexed matcher reachability', () => {
  it('returns Dice, Cosine and Tversky query states to baseline after destruction', () => {
    const baseline = count(QueryState)
    for (const kind of ['dice', 'cosine', 'tversky'] as const) {
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

  it('collects an arbitrary query and its elements when the call returns', () => {
    // The rare path ordinalizes the whole query into a table of its own, which
    // holds every element it saw — so this is the one query shape that could
    // outlive its call. Query state is retained and reused, so the release has
    // to happen on return rather than at whatever the next query is.
    const matcher = createIndexedMatcher(choices, { scorer: dice })
    const wrapperBaseline = count(ArbitraryWrapper)
    const sentinelBaseline = count(ArbitrarySentinel)
    submitArbitraryQuery(matcher)
    const wrapperObserved = count(ArbitraryWrapper)
    const sentinelObserved = count(ArbitrarySentinel)
    expect(wrapperObserved).toBe(wrapperBaseline)
    expect(sentinelObserved).toBe(sentinelBaseline)
    expect(matcher.best('ab')?.item).toBe('ab')
    expect(matcher.size).toBe(choices.length)
  })

  it('holds token elements for the matcher lifetime, because identity is the key', () => {
    const baseline = count(TokenItem)
    withLiveTokenMatcher(() => expect(count(TokenItem)).toBe(baseline + 4))
    expect(count(TokenItem)).toBe(baseline)
  })

  it('holds no element that sits in a window an unmatchable one poisons', () => {
    // Every window `dead` could appear in also holds the `NaN` beside it, so it
    // can never name a posting. An ordinal for it would be held for the life of
    // the index and push every later element further up the radix ladder.
    const baseline = count(TokenItem)
    withLiveUnreachableTokenMatcher(() => expect(count(TokenItem)).toBe(baseline))
  })

  it('retains no choice sequence once its elements have ordinals', () => {
    // What a gram-bearing choice leaves behind is its postings and its
    // elements' ordinals, never the sequence they were read from — so nothing
    // may hold the caller's object, before conversion copies it or after.
    const baseline = count(TokenChoice)
    withLiveTokenChoiceMatcher(() => expect(count(TokenChoice)).toBe(baseline))
  })
})
