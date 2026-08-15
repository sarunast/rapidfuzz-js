// Not ported from RapidFuzz — upstream has no index, and this proves an
// optimisation upstream does not make.
//
// `tokenSetRatio` answers 100 exactly when one non-empty token set
// contains the other, so `tokenContainmentProof` can name every perfect match
// without scoring a pair. The risk it carries is a plausible wrong answer: an
// id that is not the earliest, or a match reported where the scorer would have
// found none. Only a comparison against independently computed results can
// catch that, so every assertion below is against a brute force that scores
// every choice through the public scorer and applies the documented ordering —
// descending score, then ascending id.
import { describe, expect, it } from 'vitest'

import { createScorer } from '#core/scoring/scorer.js'
import { createMatcher } from '#search/matcher/createMatcher.js'

import { tokenSetRatio } from './tokenSetRatio.js'

const scorer = createScorer(tokenSetRatio)

interface Expected {
  readonly key: number
  readonly score: number
}

/** What scoring every choice and ordering the results has to produce. */
function bruteForce(choices: readonly string[], query: string): readonly Expected[] {
  const scored = choices.map((choice, key) => ({
    key,
    score: scorer.score(query, choice),
  }))
  return scored.slice().sort((a, b) => b.score - a.score || a.key - b.key)
}

function expectAgrees(choices: readonly string[], queries: readonly string[]): void {
  const matcher = createMatcher(choices, { scorer })
  for (const query of queries) {
    const oracle = bruteForce(choices, query)

    const best = matcher.best(query)
    expect(best?.key, `best(${JSON.stringify(query)})`).toBe(oracle[0]?.key)
    expect(best?.score, `best(${JSON.stringify(query)})`).toBe(oracle[0]?.score)

    for (const limit of [1, 2, 3, 5]) {
      const found = matcher.search(query, { limit })
      const wanted = oracle.slice(0, limit)
      expect(
        found.map((match) => match.key),
        `search(${JSON.stringify(query)}, { limit: ${limit} })`,
      ).toEqual(wanted.map((entry) => entry.key))
      expect(found.map((match) => match.score)).toEqual(
        wanted.map((entry) => entry.score),
      )
    }
  }
}

describe('token containment proof', () => {
  it('agrees on containment both ways, and on ties', () => {
    // `data engineer` is contained by two later entries and contains an earlier
    // one, so both channels fire and the earliest id has to win.
    const choices = [
      'engineer',
      'data engineer',
      'senior data engineer',
      'data engineer cloud',
      'unrelated words here',
      'data engineer',
    ]
    expectAgrees(choices, [
      'data engineer',
      'engineer data',
      'senior data engineer',
      'engineer',
      'data',
      'cloud engineer data',
      'nothing shared at all',
    ])
  })

  it('agrees when nothing is contained', () => {
    const choices = ['alpha beta', 'gamma delta', 'epsilon zeta']
    expectAgrees(choices, ['theta iota', 'alpha gamma', 'beta delta'])
  })

  it('agrees on empty and whitespace-only entries and queries', () => {
    // Two inputs with no tokens between them score 0, not 100 — RapidFuzz issue
    // 110 — so nothing here may be proven a perfect match.
    const choices = ['', '   ', 'alpha', '\t\n', 'alpha beta']
    expectAgrees(choices, ['', '   ', 'alpha', 'alpha beta', 'beta'])
  })

  it('agrees on astral text, which stays in the packed key space', () => {
    // `tokenKey` packs a non-BMP code point two UTF-16 units at a time, so
    // astral tokens are ordinary identities and must be proven, not declined.
    const choices = ['\u{1f600} alpha', 'alpha', '\u{1f600}', 'alpha \u{1f600} beta']
    expectAgrees(choices, ['\u{1f600} alpha', '\u{1f600}', 'alpha', '\u{1f600} beta'])
  })

  it('agrees on a query longer than the subset cap', () => {
    // Thirteen tokens: past `SUBSET_CAP`, where the proof declines rather than
    // truncating its enumeration, and the scan has to answer instead.
    const long = 'a b c d e f g h i j k l m'
    const choices = ['a b c', long, 'a b c d e f g h i j k l m n', 'z']
    expectAgrees(choices, [long, `${long} n`])
  })

  it('agrees when a token needs elementwise equality', () => {
    // An object element hashes into a collision bucket rather than an identity,
    // so key comparison alone cannot decide containment. Choices holding one
    // still serve the posting side; a query holding one declines outright.
    const shared = { toString: () => 'shared' }
    // Single-character elements convert to code points, so `[...'alpha']` is a
    // packed token; a multi-character element like `'alpha'` is not, and an
    // object never is. The third entry therefore holds one of each, which is
    // the asymmetric case: it can still be proven to *contain* a packed-only
    // query, but can never be contained by one, so it serves the posting side
    // while staying out of the set-key map.
    const choices: unknown[][] = [
      [...'alpha'],
      ['alpha', ' ', shared],
      [...'alpha', ' ', shared],
      [shared],
      [...'alpha beta'],
    ]
    const matcher = createMatcher(choices, { scorer })
    const queries: unknown[][] = [
      [...'alpha'],
      [shared],
      ['alpha', ' ', shared],
      [...'alpha beta'],
    ]
    for (const query of queries) {
      const oracle = choices
        .map((choice, key) => ({ key, score: scorer.score(query, choice) }))
        .sort((a, b) => b.score - a.score || a.key - b.key)
      const best = matcher.best(query)
      expect(best?.key).toBe(oracle[0]?.key)
      expect(best?.score).toBe(oracle[0]?.score)
    }
  })

  it('agrees under thresholds at and around the optimum', () => {
    const choices = ['alpha beta', 'alpha', 'alpha beta gamma', 'delta']
    const matcher = createMatcher(choices, { scorer })
    for (const threshold of [0, 50, 100]) {
      const oracle = bruteForce(choices, 'alpha beta').filter(
        (entry) => entry.score >= threshold,
      )
      const best = matcher.best('alpha beta', { threshold })
      expect(best?.key).toBe(oracle[0]?.key)
      expect(best?.score).toBe(oracle[0]?.score)
      expect(
        matcher.search('alpha beta', { threshold, limit: 3 }).map((m) => m.key),
      ).toEqual(oracle.slice(0, 3).map((entry) => entry.key))
    }
  })

  it('refuses a non-finite threshold the way the scan does', () => {
    const matcher = createMatcher(['alpha beta'], { scorer })
    expect(() => matcher.best('alpha beta', { threshold: Number.NaN })).toThrow(
      RangeError,
    )
    expect(() => matcher.search('alpha beta', { threshold: Number.NaN })).toThrow(
      RangeError,
    )
  })

  it('leaves an unlimited search to the scan, which reports every score', () => {
    // `limit: null` must still return the choices below the optimum, which the
    // proof knows nothing about.
    const choices = ['alpha beta', 'alpha', 'zeta']
    const matcher = createMatcher(choices, { scorer })
    const found = matcher.search('alpha beta', { limit: null })
    expect(found.map((match) => match.key)).toEqual(
      bruteForce(choices, 'alpha beta').map((entry) => entry.key),
    )
  })

  it('agrees over a random corpus, where nobody chose the shapes', () => {
    let seed = 0x51f3d2b
    const rand = (): number => {
      seed ^= seed << 13
      seed ^= seed >>> 17
      seed ^= seed << 5
      seed >>>= 0
      return seed / 0x100000000
    }
    const words = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta']
    const phrase = (): string => {
      const count = Math.floor(rand() * 4)
      const parts: string[] = []
      for (let i = 0; i < count; i++) parts.push(words[Math.floor(rand() * words.length)])
      return parts.join(' ')
    }
    const choices: string[] = []
    for (let i = 0; i < 120; i++) choices.push(phrase())
    const queries: string[] = []
    for (let i = 0; i < 60; i++) queries.push(phrase())

    // The corpus has to actually contain what the proof looks for, or this
    // agrees about nothing: a generator that stopped producing containment
    // would leave every query falling through to the scan and the test would
    // still pass, green and vacuous.
    const settled = queries.filter(
      (query) => bruteForce(choices, query)[0]?.score === 100,
    )
    expect(settled.length).toBeGreaterThan(queries.length / 4)

    expectAgrees(choices, queries)
  })
})
