// Not ported from RapidFuzz — upstream has no equivalent, because in Python
// there is only ever one representation of a sequence to get wrong.
//
// `process` scores a query against many choices by preparing the query once,
// and `scorerSequence` in `src/_common.ts` keeps a BMP-only string as a string
// rather than expanding it into code points. So a prepared query and a
// converted choice can meet as `'a'` and `97` — the same character, spelled two
// ways — and every comparison written as `===` sees a mismatch. The bit-
// parallel kernels read either form and never notice; a common prefix, a
// Hamming count and Jaro's transposition pass do.
//
// These tests therefore run each pair through all four entry points — the
// scorer itself, `scoreMatrix`, `scorePairs` and `extractOne` — and require one answer
// from all of them. The expected values were produced by upstream's Python
// package (rapidfuzz 3.14.5).
import { describe, expect, it } from 'vitest'

import type { NormalizedScorer, Scorer, Sequence } from '../src/_common.js'
import { configure } from '../src/configure.js'
import { osaOneWord } from '../src/distance/_bitVector/index.js'
import { hammingDistance } from '../src/distance/hamming.js'
import {
  indelDistance,
  indelNormalizedDistance,
  indelNormalizedSimilarity,
  indelSimilarity,
} from '../src/distance/indel.js'
import { jaroSimilarity } from '../src/distance/jaro.js'
import { jaroWinklerSimilarity } from '../src/distance/jaroWinkler.js'
import {
  lcsSeqDistance,
  lcsSeqNormalizedDistance,
  lcsSeqNormalizedSimilarity,
  lcsSeqSimilarity,
} from '../src/distance/lcsSeq.js'
import {
  type LevenshteinOptions,
  levenshteinDistance,
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
  levenshteinSimilarity,
} from '../src/distance/levenshtein.js'
import { postfixDistance } from '../src/distance/postfix.js'
import { prefixDistance } from '../src/distance/prefix.js'
import {
  partialRatio,
  partialTokenRatio,
  partialTokenSetRatio,
  partialTokenSortRatio,
  qRatio,
  ratio,
  tokenRatio,
  tokenSetRatio,
  tokenSortRatio,
  wRatio,
} from '../src/fuzz.js'
import { extract, extractOne, type ScoreOptions } from '../src/search.js'
import { matrixScores, pairScores } from './matrix.js'

interface Expectations {
  readonly ratio: number
  readonly partialRatio: number
  readonly wRatio: number
  readonly qRatio: number
  readonly tokenSortRatio: number
  readonly tokenSetRatio: number
  readonly tokenRatio: number
  readonly partialTokenRatio: number
  readonly hammingDistance: number
  readonly prefixDistance: number
  readonly postfixDistance: number
  readonly jaroSimilarity: number
  readonly jaroWinklerSimilarity: number
}

interface Case extends Expectations {
  readonly s1: string
  readonly s2: string
}

/** Every scorer above, paired with the field holding its expected value. */
const scorers: ReadonlyArray<
  readonly [string, Scorer | NormalizedScorer, (c: Expectations) => number]
> = [
  ['ratio', ratio, (c) => c.ratio],
  ['partialRatio', partialRatio, (c) => c.partialRatio],
  ['wRatio', wRatio, (c) => c.wRatio],
  ['qRatio', qRatio, (c) => c.qRatio],
  ['tokenSortRatio', tokenSortRatio, (c) => c.tokenSortRatio],
  ['tokenSetRatio', tokenSetRatio, (c) => c.tokenSetRatio],
  ['tokenRatio', tokenRatio, (c) => c.tokenRatio],
  ['partialTokenRatio', partialTokenRatio, (c) => c.partialTokenRatio],
  ['hammingDistance', hammingDistance, (c) => c.hammingDistance],
  ['prefixDistance', prefixDistance, (c) => c.prefixDistance],
  ['postfixDistance', postfixDistance, (c) => c.postfixDistance],
  ['jaroSimilarity', jaroSimilarity, (c) => c.jaroSimilarity],
  ['jaroWinklerSimilarity', jaroWinklerSimilarity, (c) => c.jaroWinklerSimilarity],
]

const cases: readonly Case[] = [
  {
    s1: 'a',
    s2: 'a\u{1F600}',
    ratio: 66.66666666666667,
    partialRatio: 100,
    wRatio: 90,
    qRatio: 66.66666666666667,
    tokenSortRatio: 66.66666666666667,
    tokenSetRatio: 66.66666666666666,
    tokenRatio: 66.66666666666667,
    partialTokenRatio: 100,
    hammingDistance: 1,
    prefixDistance: 1,
    postfixDistance: 2,
    jaroSimilarity: 0.8333333333333334,
    jaroWinklerSimilarity: 0.8500000000000001,
  },
  {
    // `wRatio` picked the partial-scoring branch here when it measured the
    // length ratio in UTF-16 code units: 3 / 2 rather than 2 / 2.
    s1: 'aa',
    s2: 'a\u{1F600}',
    ratio: 50,
    partialRatio: 66.66666666666667,
    wRatio: 50,
    qRatio: 50,
    tokenSortRatio: 50,
    tokenSetRatio: 50,
    tokenRatio: 50,
    partialTokenRatio: 66.66666666666667,
    hammingDistance: 1,
    prefixDistance: 1,
    postfixDistance: 2,
    jaroSimilarity: 0.6666666666666666,
    jaroWinklerSimilarity: 0.6666666666666666,
  },
  {
    s1: 'a',
    s2: 'a\u{1F600}b',
    ratio: 50,
    partialRatio: 100,
    wRatio: 90,
    qRatio: 50,
    tokenSortRatio: 50,
    tokenSetRatio: 50,
    tokenRatio: 50,
    partialTokenRatio: 100,
    hammingDistance: 2,
    prefixDistance: 2,
    postfixDistance: 3,
    jaroSimilarity: 0.7777777777777777,
    jaroWinklerSimilarity: 0.7999999999999999,
  },
  {
    // The astral character on the query side rather than the choice side: the
    // prepared query is expanded and the choice is the one left as a string.
    s1: 'a\u{1F600}b',
    s2: 'a',
    ratio: 50,
    partialRatio: 100,
    wRatio: 90,
    qRatio: 50,
    tokenSortRatio: 50,
    tokenSetRatio: 50,
    tokenRatio: 50,
    partialTokenRatio: 100,
    hammingDistance: 2,
    prefixDistance: 2,
    postfixDistance: 3,
    jaroSimilarity: 0.7777777777777777,
    jaroWinklerSimilarity: 0.7999999999999999,
  },
  {
    s1: 'abc',
    s2: 'a\u{1F600}bc',
    ratio: 85.71428571428572,
    partialRatio: 80,
    wRatio: 85.71428571428572,
    qRatio: 85.71428571428572,
    tokenSortRatio: 85.71428571428572,
    tokenSetRatio: 85.71428571428571,
    tokenRatio: 85.71428571428572,
    partialTokenRatio: 80,
    hammingDistance: 3,
    prefixDistance: 3,
    postfixDistance: 2,
    jaroSimilarity: 0.9166666666666666,
    jaroWinklerSimilarity: 0.9249999999999999,
  },
  {
    s1: 'a\u{1F600}bcdefgh',
    s2: 'abcdefgh',
    ratio: 94.11764705882352,
    partialRatio: 93.33333333333333,
    wRatio: 94.11764705882352,
    qRatio: 94.11764705882352,
    tokenSortRatio: 94.11764705882352,
    tokenSetRatio: 94.11764705882354,
    tokenRatio: 94.11764705882354,
    partialTokenRatio: 93.33333333333333,
    hammingDistance: 8,
    prefixDistance: 8,
    postfixDistance: 2,
    jaroSimilarity: 0.9629629629629629,
    jaroWinklerSimilarity: 0.9666666666666667,
  },
  {
    s1: 'x y',
    s2: 'x\u{1F600} y',
    ratio: 85.71428571428572,
    partialRatio: 80,
    wRatio: 85.71428571428572,
    qRatio: 85.71428571428572,
    tokenSortRatio: 85.71428571428572,
    tokenSetRatio: 85.71428571428571,
    tokenRatio: 85.71428571428572,
    partialTokenRatio: 100,
    hammingDistance: 3,
    prefixDistance: 3,
    postfixDistance: 2,
    jaroSimilarity: 0.9166666666666666,
    jaroWinklerSimilarity: 0.9249999999999999,
  },
  {
    // One token, but padded: token sorting still changes the query, so
    // `wRatio` must not take its single-token shortcut past the token scorers.
    s1: ' a',
    s2: 'aa',
    ratio: 50,
    partialRatio: 66.66666666666667,
    wRatio: 63.333333333333336,
    qRatio: 50,
    tokenSortRatio: 66.66666666666667,
    tokenSetRatio: 66.66666666666666,
    tokenRatio: 66.66666666666667,
    partialTokenRatio: 100,
    hammingDistance: 1,
    prefixDistance: 2,
    postfixDistance: 1,
    jaroSimilarity: 0.6666666666666666,
    jaroWinklerSimilarity: 0.6666666666666666,
  },
  {
    s1: ' a ',
    s2: 'a',
    ratio: 50,
    partialRatio: 100,
    wRatio: 90,
    qRatio: 50,
    tokenSortRatio: 100,
    tokenSetRatio: 100,
    tokenRatio: 100,
    partialTokenRatio: 100,
    hammingDistance: 3,
    prefixDistance: 3,
    postfixDistance: 3,
    jaroSimilarity: 0,
    jaroWinklerSimilarity: 0,
  },
  {
    s1: 'a b',
    s2: 'ab',
    ratio: 80,
    partialRatio: 66.66666666666667,
    wRatio: 80,
    qRatio: 80,
    tokenSortRatio: 80,
    tokenSetRatio: 80,
    tokenRatio: 80,
    partialTokenRatio: 66.66666666666667,
    hammingDistance: 2,
    prefixDistance: 2,
    postfixDistance: 2,
    jaroSimilarity: 0.611111111111111,
    jaroWinklerSimilarity: 0.611111111111111,
  },
]

describe('every entry point agrees with upstream', () => {
  for (const testCase of cases) {
    const { s1, s2 } = testCase

    describe(`${JSON.stringify(s1)} vs ${JSON.stringify(s2)}`, () => {
      for (const [name, scorer, expected] of scorers) {
        it(name, () => {
          const want = expected(testCase)

          expect(scorer(s1, s2, {})).toBeCloseTo(want, 10)
          expect(matrixScores([s1], [s2], { scorer })[0][0]).toBeCloseTo(want, 10)
          expect(pairScores([s1], [s2], { scorer })[0]).toBeCloseTo(want, 10)
          expect(extractOne(s1, [s2], { scorer })?.score).toBeCloseTo(want, 10)
        })
      }
    })
  }
})

// A deterministic PRNG, so a failure names one seed rather than "sometimes".
function makeRng(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 0x1_0000_0000
  }
}

const alphabet = ['a', 'b', 'c', ' ', '\u{1F600}', '\u{1F601}', 'é', 'ab', '  ']

function randomText(rng: () => number, maxLength: number): string {
  let out = ''
  const length = Math.floor(rng() * (maxLength + 1))
  for (let i = 0; i < length; i++) out += alphabet[Math.floor(rng() * alphabet.length)]
  return out
}

// A prepared scorer holds a mask indexed from the whole query, so it cannot
// re-base itself on a shared prefix the way the direct path does — it marks
// those positions as taken and scans past them instead. Random short text
// almost never shares a prefix, so the pairs above exercise none of that.
// Lengths here straddle the 32-element word boundary on both sides of the
// prefix, which is where the claimed bits change word.
describe('a shared prefix scores the same prepared as direct', () => {
  const shared = [
    '',
    'a',
    'senior frontend ',
    'x'.repeat(31),
    'y'.repeat(32),
    'z'.repeat(40),
  ]
  const tails = ['', 'q', 'engineer', 'developer', 'engineering manager', 'w'.repeat(35)]

  for (const [name, scorer] of scorers) {
    it(name, () => {
      const disagreements: string[] = []

      for (const prefix of shared) {
        for (const tail1 of tails) {
          for (const tail2 of tails) {
            const s1 = prefix + tail1
            const s2 = prefix + tail2
            const direct = scorer(s1, s2, {})

            for (const [label, prepared] of [
              ['scoreMatrix', matrixScores([s1], [s2], { scorer })[0][0]],
              ['scorePairs', pairScores([s1], [s2], { scorer })[0]],
            ] satisfies ReadonlyArray<readonly [string, number]>) {
              if (Math.abs(direct - prepared) > 1e-9) {
                disagreements.push(
                  `${label} ${JSON.stringify(s1)} vs ${JSON.stringify(s2)}: ` +
                    `${direct} != ${prepared}`,
                )
              }
            }
          }
        }
      }

      expect(disagreements.slice(0, 5)).toEqual([])
    })
  }
})

describe('the prepared path never disagrees with the direct one', () => {
  for (const [name, scorer] of scorers) {
    it(name, () => {
      const rng = makeRng(0x5eed)
      const disagreements: string[] = []

      for (let trial = 0; trial < 500; trial++) {
        const s1 = randomText(rng, 12)
        const s2 = randomText(rng, 12)
        const direct = scorer(s1, s2, {})

        for (const [label, prepared] of [
          ['scoreMatrix', matrixScores([s1], [s2], { scorer })[0][0]],
          ['scorePairs', pairScores([s1], [s2], { scorer })[0]],
        ] satisfies ReadonlyArray<readonly [string, number]>) {
          if (Math.abs(direct - prepared) > 1e-9) {
            disagreements.push(
              `${label} ${JSON.stringify(s1)} vs ${JSON.stringify(s2)}: ` +
                `${direct} != ${prepared}`,
            )
          }
        }
      }

      expect(disagreements.slice(0, 5)).toEqual([])
    })
  }
})

// Rejecting a weight is part of the answer a scorer gives, so both paths owe
// the same one. They did not: `process` read its weights through the validator
// and the scorer read `options.weights` straight, so a shape only one of them
// refused came back from the direct call as a number — `NaN` for a pair that is
// two characters apart. Upstream raises on all of these.
describe('every path refuses the same weights', () => {
  // The malformed shapes are what a JavaScript caller reaches and TypeScript
  // does not, so they are set rather than written as a literal — the point of
  // the test is the check at run time, which is the only one those callers get.
  const withWeights = (weights: unknown): LevenshteinOptions => {
    const options: LevenshteinOptions = {}
    Reflect.set(options, 'weights', weights)
    return options
  }

  const levenshteinScorers: ReadonlyArray<readonly [string, Scorer<LevenshteinOptions>]> =
    [
      ['levenshteinDistance', levenshteinDistance],
      ['levenshteinSimilarity', levenshteinSimilarity],
      ['levenshteinNormalizedDistance', levenshteinNormalizedDistance],
      ['levenshteinNormalizedSimilarity', levenshteinNormalizedSimilarity],
    ]

  const refused: ReadonlyArray<readonly [string, unknown]> = [
    ['too few costs', [1, 1]],
    ['too many costs', [1, 1, 1, 1]],
    ['a value that is no sequence of costs', 'xyz'],
    ['a cost that is not a number', [1, '1', 1]],
    ['a negative cost', [-1, 1, 1]],
    ['a NaN cost', [Number.NaN, 1, 1]],
    ['an infinite cost', [Infinity, 1, 1]],
    ['a negatively infinite cost', [-Infinity, 1, 1]],
    // The named spelling reaches the same validator, so it has to refuse the
    // same things. A partial object is a compile error for a TypeScript caller;
    // these are what a JavaScript one can still reach.
    ['a cost object missing a key', { insertion: 1, deletion: 1 }],
    ['a cost object with a spare key', { insert: 1, delete: 1, replace: 1 }],
    [
      'a named cost that is not a number',
      { insertion: 1, deletion: 1, substitution: '1' },
    ],
    ['a negative named cost', { insertion: -1, deletion: 1, substitution: 1 }],
    ['a NaN named cost', { insertion: Number.NaN, deletion: 1, substitution: 1 }],
  ]

  for (const [what, weights] of refused) {
    it(`rejects ${what}`, () => {
      for (const [name, scorer] of levenshteinScorers) {
        expect(() => scorer('abc', 'adc', withWeights(weights)), name).toThrow(TypeError)
        expect(
          () =>
            matrixScores(['abc'], ['adc'], {
              scorer: configure(scorer, withWeights(weights)),
            }),
          name,
        ).toThrow(TypeError)
      }
    })
  }

  // The costs either side of the bound, which are answers rather than errors.
  it('takes a zero cost and a fractional one', () => {
    expect(levenshteinDistance('abc', 'adc', { weights: [1, 1, 0] })).toBe(0)
    expect(levenshteinDistance('abc', 'adc', { weights: [1, 1, 0.5] })).toBe(0.5)
    expect(levenshteinDistance('abc', 'adc', { weights: [0, 0, 1] })).toBe(0)
  })
})

// A prepared Levenshtein holds the query's match masks and scores every choice
// against them, which the direct path cannot do because it trims a common affix
// first and so works over a range that differs per choice. The two therefore run
// different kernels over different ranges and must still agree exactly.
//
// `preparedDistanceWorthwhile` decides which one runs, and every input below is
// chosen to land on one side of it or the other: lengths straddle the 32-element
// word boundary, a cutoff sends the scorer to the banded kernels the held masks
// cannot serve, and non-uniform weights leave the uniform kernel entirely.
describe('the held Levenshtein pattern agrees with the trimming kernel', () => {
  const kinds = [
    ['levenshteinDistance', levenshteinDistance, 3],
    ['levenshteinSimilarity', levenshteinSimilarity, 3],
    ['levenshteinNormalizedDistance', levenshteinNormalizedDistance, 0.5],
    ['levenshteinNormalizedSimilarity', levenshteinNormalizedSimilarity, 0.5],
  ] satisfies ReadonlyArray<readonly [string, Scorer | NormalizedScorer, number]>

  // Either side of every word boundary the masks can change width at.
  const lengths = [0, 1, 2, 31, 32, 33, 63, 64, 65, 96, 97]

  for (const [name, scorer, cutoff] of kinds) {
    it(name, () => {
      const rng = makeRng(0xc0ffee)
      const disagreements: string[] = []

      const agree = (s1: string, s2: string | readonly string[], label: string): void => {
        for (const options of [{}, { scoreCutoff: cutoff }, { scoreHint: 2 }]) {
          const direct = scorer(s1, s2, options)
          const prepared = matrixScores([s1], [s2], { scorer, ...options })[0][0]
          if (Math.abs(direct - prepared) > 1e-9) {
            disagreements.push(
              `${label} ${JSON.stringify(options)} ${JSON.stringify(s1)}: ` +
                `${direct} != ${prepared}`,
            )
          }
        }
      }

      // The banded kernel is chosen by budget, and it reports `budget + 1` for
      // what the exact kernel returns outright, so every budget either side of
      // the band's ceiling has to produce the trimming kernel's answer. Reading
      // the ceiling off the untrimmed query is what once returned 3 for a
      // one-edit pair: the band was twice a word wide and silently truncated.
      const agreeAcrossBudgets = (s1: string, s2: string): void => {
        const longest = Math.max(s1.length, s2.length) || 1
        for (let budget = 0; budget <= 34; budget++) {
          const scoreCutoff =
            name === 'levenshteinDistance'
              ? budget
              : name === 'levenshteinSimilarity'
                ? Math.max(0, longest - budget)
                : name === 'levenshteinNormalizedDistance'
                  ? // The sweep runs past the longest input, and a normalised
                    // cutoff above 1 is one both this and upstream refuse.
                    Math.min(1, budget / longest)
                  : Math.max(0, 1 - budget / longest)
          const direct = scorer(s1, s2, { scoreCutoff })
          const prepared = matrixScores([s1], [s2], { scorer, scoreCutoff })[0][0]
          if (Math.abs(direct - prepared) > 1e-9) {
            disagreements.push(
              `budget ${budget} ${JSON.stringify(s1)} vs ${JSON.stringify(s2)}: ` +
                `${direct} != ${prepared}`,
            )
          }
        }
      }

      for (const queryLength of lengths) {
        for (const choiceLength of lengths) {
          const s1 = randomText(rng, queryLength)
          const s2 = randomText(rng, choiceLength)
          agree(s1, s2, 'string')
          // A choice spelled as an array meets a query spelled as a string, so
          // the held masks have to read both representations.
          agree(s1, Array.from(s2), 'array')
          agreeAcrossBudgets(s1, s2)
          // A near-copy puts the true distance among the budgets swept above,
          // which is where reporting `budget + 1` instead of the answer shows.
          agreeAcrossBudgets(s1, s1.slice(0, Math.floor(s1.length / 2)) + s2)
        }
      }

      // Non-uniform weights never reach the held pattern; they must still match.
      const weighted = randomText(rng, 40)
      const other = randomText(rng, 40)
      const weightSets: ReadonlyArray<readonly [number, number, number]> = [
        [1, 1, 2],
        [2, 1, 1],
      ]
      for (const weights of weightSets) {
        const direct = scorer(weighted, other, { weights })
        const prepared = matrixScores([weighted], [other], {
          scorer: configure(scorer, { weights }),
        })[0][0]
        if (Math.abs(direct - prepared) > 1e-9) {
          disagreements.push(
            `weights ${JSON.stringify(weights)}: ${direct} != ${prepared}`,
          )
        }
      }

      expect(disagreements.slice(0, 5)).toEqual([])
    })
  }

  it('reuses one held pattern across many choices without going stale', () => {
    const rng = makeRng(0x1234)
    const query = randomText(rng, 45)
    const choices = Array.from({ length: 300 }, () => randomText(rng, 60))
    const row = matrixScores([query], choices, { scorer: levenshteinDistance })[0]

    expect(row).toEqual(choices.map((c) => levenshteinDistance(query, c, {})))
  })
})

// The same argument as the Levenshtein block above, for the other family that
// holds a pattern: LCS and Indel share one set of prepared kernels, and the
// scorers choose between them and the trimming kernel by a gate of their own
// (`preparedLengthWorthwhile`, `preparedDistanceWorthwhile`). Two things can
// therefore disagree here that cannot disagree in Python — which kernel ran,
// and whether it was allowed to stop early — and a cutoff is what selects both.
//
// The `scorers` table above does not carry these eight, because its expected
// values come from upstream and are written per case. This needs no fixtures:
// the direct path is already parity-checked in `tests/distance/distance.test.ts`,
// so agreeing with it is the whole obligation.
describe('the held LCS pattern agrees with the trimming kernel', () => {
  const kinds = [
    ['lcsSeqDistance', lcsSeqDistance, 3],
    ['lcsSeqSimilarity', lcsSeqSimilarity, 3],
    ['lcsSeqNormalizedDistance', lcsSeqNormalizedDistance, 0.5],
    ['lcsSeqNormalizedSimilarity', lcsSeqNormalizedSimilarity, 0.5],
    ['indelDistance', indelDistance, 6],
    ['indelSimilarity', indelSimilarity, 6],
    ['indelNormalizedDistance', indelNormalizedDistance, 0.5],
    ['indelNormalizedSimilarity', indelNormalizedSimilarity, 0.5],
  ] satisfies ReadonlyArray<readonly [string, Scorer | NormalizedScorer, number]>

  // Either side of every word boundary the masks can change width at, and past
  // the three the multi-word kernel writes out before it loops.
  const lengths = [0, 1, 2, 31, 32, 33, 63, 64, 65, 96, 97, 130]

  /** Whether this kind reads its cutoff as a distance, a count or a fraction. */
  function isNormalized(name: string): boolean {
    return name.startsWith('lcsSeqNormalized') || name.startsWith('indelNormalized')
  }

  for (const [name, scorer, cutoff] of kinds) {
    it(name, () => {
      const rng = makeRng(0xf0cacc1a)
      const disagreements: string[] = []

      const agree = (s1: string, s2: string | readonly string[], label: string): void => {
        for (const options of [{}, { scoreCutoff: cutoff }]) {
          const direct = scorer(s1, s2, options)
          const prepared = matrixScores([s1], [s2], { scorer, ...options })[0][0]
          if (Math.abs(direct - prepared) > 1e-9) {
            disagreements.push(
              `${label} ${JSON.stringify(options)} ${JSON.stringify(s1)}: ` +
                `${direct} != ${prepared}`,
            )
          }
        }
      }

      // The gate picks a kernel by cutoff, and the bounded one is free to give
      // up as soon as its target is out of reach — so every cutoff either side
      // of the true score has to come back with the trimming kernel's answer.
      // A cutoff sweep is the only thing that reaches both sides of the gate
      // and both sides of the early exit for one pair of inputs.
      const agreeAcrossCutoffs = (s1: string, s2: string): void => {
        const longest = Math.max(s1.length, s2.length) || 1
        for (let step = 0; step <= 34; step++) {
          const scoreCutoff = isNormalized(name)
            ? Math.min(1, step / longest)
            : name === 'lcsSeqSimilarity'
              ? Math.max(0, longest - step)
              : name === 'indelSimilarity'
                ? Math.max(0, s1.length + s2.length - step)
                : step
          const direct = scorer(s1, s2, { scoreCutoff })
          const prepared = matrixScores([s1], [s2], { scorer, scoreCutoff })[0][0]
          if (Math.abs(direct - prepared) > 1e-9) {
            disagreements.push(
              `cutoff ${scoreCutoff} ${JSON.stringify(s1)} vs ${JSON.stringify(s2)}: ` +
                `${direct} != ${prepared}`,
            )
          }
        }
      }

      for (const queryLength of lengths) {
        for (const choiceLength of lengths) {
          const s1 = randomText(rng, queryLength)
          const s2 = randomText(rng, choiceLength)
          agree(s1, s2, 'string')
          // A choice spelled as an array meets a query spelled as a string, so
          // the held masks have to read both representations.
          agree(s1, Array.from(s2), 'array')
          agreeAcrossCutoffs(s1, s2)
          // A near-copy puts the true score among the cutoffs swept above,
          // which is where an early exit that fires too soon shows.
          agreeAcrossCutoffs(s1, s1.slice(0, Math.floor(s1.length / 2)) + s2)
        }
      }

      expect(disagreements.slice(0, 5)).toEqual([])
    })
  }

  // The gate refuses the held pattern when the query is the longer side, so a
  // row of choices shorter than the query is the one shape that never reaches
  // the prepared kernel at all. It still owes the same answers.
  it('answers the same whichever side is longer', () => {
    const rng = makeRng(0x5a1ad)
    const query = randomText(rng, 130)
    const choices = Array.from({ length: 60 }, (_unused, i) =>
      randomText(rng, i % 2 === 0 ? 40 : 200),
    )

    for (const scoreCutoff of [0.5, 0.8, 0.95]) {
      const row = matrixScores([query], choices, {
        scorer: lcsSeqNormalizedSimilarity,
        scoreCutoff,
      })[0]

      expect(row).toEqual(
        choices.map((c) => lcsSeqNormalizedSimilarity(query, c, { scoreCutoff })),
      )
    }
  })
})

// Not ported — upstream cannot express the failure, because in Python there is
// only one path through a scorer. Here a raw cutoff is read twice: once to bound
// the kernel, and once to decide what to report. Truncating it in only one of
// those places makes a fractional cutoff mean two different numbers, and the
// prepared path is where the two meet — `extract` and `scoreMatrix` pass the
// cutoff straight to the kernel. Expected values are rapidfuzz 3.14.5's.
describe('a fractional raw cutoff is truncated on every path', () => {
  const cases: ReadonlyArray<readonly [string, Scorer, string, string, number, number]> =
    [
      ['levenshteinDistance', levenshteinDistance, 'abc', 'xyz', 1.9, 2],
      ['levenshteinSimilarity', levenshteinSimilarity, 'ab', 'ax', 1.9, 1],
      ['levenshteinSimilarity', levenshteinSimilarity, 'abcd', 'abce', 3.7, 3],
      ['prefixDistance', prefixDistance, 'abc', 'xyz', 1.9, 2],
      ['postfixDistance', postfixDistance, 'abc', 'xyz', 2.5, 3],
      ['hammingDistance', hammingDistance, 'abc', 'xyz', 1.9, 2],
    ]

  for (const [name, scorer, s1, s2, scoreCutoff, want] of cases) {
    it(`${name}(${s1}, ${s2}) at ${scoreCutoff}`, () => {
      expect(scorer(s1, s2, { scoreCutoff })).toBe(want)
      expect(matrixScores([s1], [s2], { scorer, scoreCutoff })[0][0]).toBe(want)
      expect(pairScores([s1], [s2], { scorer, scoreCutoff })[0]).toBe(want)
    })
  }
})

describe('choices a caller can legally pass', () => {
  // Upstream enumerates a `str` of choices character by character, because
  // that is what iterating a `str` does in Python. Here it is `['abc']`
  // written wrong, so it is refused rather than scored.
  it('rejects a bare string, which upstream would read as its characters', () => {
    expect(() => extract('a', 'abc', { scorer: ratio, limit: null })).toThrow(TypeError)
    expect(() => extractOne('a', 'abc', { scorer: ratio })).toThrow(TypeError)
    expect(extract('a', ['abc'], { scorer: ratio, limit: null })).toEqual([
      { choice: 'abc', score: 50, key: 0 },
    ])
  })

  it('rejects a limit the heap cannot honour', () => {
    expect(() => extract('a', ['a'], { limit: 2.5 })).toThrow(RangeError)
    expect(() => extract('a', ['a'], { limit: NaN })).toThrow(RangeError)
    expect(() => extract('a', ['a'], { limit: Infinity })).toThrow(RangeError)
  })

  it('still takes every match for a null limit and none for zero', () => {
    expect(extract('a', ['a', 'ab'], { limit: null })).toHaveLength(2)
    expect(extract('a', ['a', 'ab'], { limit: 0 })).toHaveLength(0)
  })
})

describe('token sorting is a total order', () => {
  it('canonicalises tokens that are the same objects in a different order', () => {
    const x = {}
    const y = {}

    expect(tokenSortRatio([x, 32, y], [y, 32, x])).toBe(100)
  })

  it('canonicalises numeric tokens', () => {
    expect(tokenSortRatio([3, 32, 1, 32, 2], [2, 32, 3, 32, 1])).toBe(100)
  })

  it('sorts NaN to a fixed place instead of leaving the comparison undefined', () => {
    // NaN is never equal to itself, so it cannot be *matched*; what this pins
    // down is that sorting it does not depend on which side it arrived on.
    expect(tokenSortRatio([NaN, 32, 1], [1, 32, NaN])).toBe(
      tokenSortRatio([1, 32, NaN], [NaN, 32, 1]),
    )
  })

  it('separates symbols that share a description', () => {
    const x = Symbol('x')
    const y = Symbol('x')

    expect(String(x)).toBe(String(y))
    expect(tokenSortRatio([x, 32, y], [y, 32, x])).toBe(100)
    expect(partialTokenSortRatio([x, 32, y], [y, 32, x])).toBe(100)
  })
})

describe('scoreMatrix and scorePairs', () => {
  // rapidfuzz 3.14.5 rounds an integral dtype half away from zero: with
  // `ratio('ab', 'ac') == 50`, a multiplier of 0.01 gives 0.5 -> 1 and -0.01
  // gives -0.5 -> -1. `Math.round` alone agrees on the first and not the
  // second, because it breaks ties towards positive infinity.
  //
  // `'i32'` rather than an unsigned kind: the negative results are the half of
  // this that `Math.round` gets wrong, and an unsigned store would wrap them
  // before the assertion could see them.
  it('rounds an integral kind half away from zero', () => {
    const opts: ScoreOptions<'i32'> = { scorer: ratio, into: 'i32' }

    expect(matrixScores(['ab'], ['ac'], { ...opts, scoreMultiplier: 0.01 })).toEqual([
      [1],
    ])
    expect(matrixScores(['ab'], ['ac'], { ...opts, scoreMultiplier: -0.01 })).toEqual([
      [-1],
    ])
    expect(matrixScores(['ab'], ['ac'], { ...opts, scoreMultiplier: 0.05 })).toEqual([
      [3],
    ])
    expect(matrixScores(['ab'], ['ac'], { ...opts, scoreMultiplier: -0.05 })).toEqual([
      [-3],
    ])
    expect(pairScores(['ab'], ['ac'], { ...opts, scoreMultiplier: -0.01 })).toEqual([-1])
  })

  it('scores both triangles when weights make the scorer asymmetric', () => {
    // `cdist` computes half the matrix and mirrors it when the scorer is
    // symmetric, so a weighting that makes it asymmetric has to be spotted.
    //
    // Only the tuple form is asserted here. `parseWeights` accepts any
    // sequence, so a typed array is a working weighting at runtime and was the
    // case that got mirrored — but `LevenshteinWeights` is a tuple, so no
    // TypeScript caller can pass one and no test can construct the call
    // without an assertion this project does not allow. The detection was
    // widened to match what the scorer accepts regardless.
    const queries = ['ab', 'abc']
    const expected = [
      [0, levenshteinDistance('ab', 'abc', { weights: [1, 2, 1] })],
      [levenshteinDistance('abc', 'ab', { weights: [1, 2, 1] }), 0],
    ]
    expect(expected[0][1]).not.toBe(expected[1][0])

    expect(
      matrixScores(queries, queries, {
        scorer: configure(levenshteinDistance, { weights: [1, 2, 1] }),
      }),
    ).toEqual(expected)
  })
})

describe('single-word OSA', () => {
  it('rejects a pattern wider than the word its shifts assume', () => {
    const pattern: number[] = Array(33).fill(1)
    const text = [...pattern]
    text[32] = 2

    expect(() => osaOneWord(pattern, text)).toThrow(RangeError)
    expect(osaOneWord(pattern.slice(0, 32), text.slice(0, 32))).toBe(0)
  })
})

// `partialRatio`'s window scan prunes with a set of the needle's elements,
// rejecting any window whose last element the needle does not hold. A prepared
// query builds that set once and reuses it across every candidate, the way
// upstream's `CachedPartialRatio` holds `s1_char_set` beside its cached ratio.
//
// Unlike the LCS masks, the set is *not* representation-blind: the scan compares
// with `===`, so `'a' !== 97`, and `alignRepresentation` decides which spelling
// the scan sees from what the *candidate* turned out to be. A set built for the
// wrong spelling prunes windows that should have been scored, which shows up as
// a silently lower score rather than as an error — so these check the value, in
// every mix of representations, both orders, and with one query reused across
// candidates of differing shape.
describe('the prepared partialRatio char set survives every representation', () => {
  const texts = [
    'hello world',
    'hello',
    'a wonderful hello world of text',
    '',
    'ababababab',
    '\u{1f600} smile \u{1f600}',
    'smile',
  ]

  /** The same text in each representation the scorers accept. */
  function shapes(text: string): ReadonlyArray<readonly [string, Sequence]> {
    const chars = [...text]
    const codes = chars.map((c) => c.codePointAt(0) ?? 0)
    return [
      ['string', text],
      ['array-of-chars', chars],
      ['array-of-codes', codes],
      ['uint32', Uint32Array.from(codes)],
    ]
  }

  it.each(texts)('agrees with the direct scorer for query %j', (query) => {
    for (const [, preparedQuery] of shapes(query)) {
      for (const choice of texts) {
        for (const [, preparedChoice] of shapes(choice)) {
          for (const scoreCutoff of [0, 50, 95]) {
            const direct = partialRatio(preparedQuery, preparedChoice, { scoreCutoff })
            const found = extract(preparedQuery, [preparedChoice], {
              scorer: partialRatio,
              scoreCutoff,
            })

            expect(found.length > 0 ? found[0].score : 0).toBeCloseTo(direct, 10)
          }
        }
      }
    }
  })

  // Both memoised spellings get asked for inside a single prepared query only
  // when the candidates disagree with each other about representation.
  it('holds both spellings at once when candidates are mixed', () => {
    const mixed = texts.flatMap((t) => shapes(t).map(([, value]) => value))

    for (const query of texts) {
      for (const [, preparedQuery] of shapes(query)) {
        const found = extract(preparedQuery, mixed, {
          scorer: partialRatio,
          limit: mixed.length,
        })

        for (const row of found) {
          expect(row.score).toBeCloseTo(partialRatio(preparedQuery, row.choice), 10)
        }
      }
    }
  })
})

// `prepared.ts` does not call `wRatio_impl` or `qRatio_impl`. It reproduces
// their strategy over held state, because by the time a prepared branch runs the
// validation and conversion those wrappers do is already done. So there are two
// copies of the ladder — the length-ratio tests, the 0.95/0.9/0.6 scalings, the
// empty-input answers — and nothing but these assertions keeps them in step.
//
// The random pairs above are short and uncutoff, so they never reach the seams:
// the branch boundaries at length ratio 1.5 and 8, and the cutoffs that the
// scalings push through 100 (a base ratio over 95 divided by 0.95, over 90
// divided by 0.9, over 60 divided by 0.6). These aim at exactly those.
describe('prepared wRatio and qRatio do not drift from their direct copies', () => {
  const composite: ReadonlyArray<readonly [string, NormalizedScorer]> = [
    ['wRatio', wRatio],
    ['qRatio', qRatio],
  ]

  /** Straddles both branch boundaries, in both orders, including equal lengths. */
  const lengthRatios = [1, 1.2, 1.49, 1.5, 1.51, 2, 7.9, 8, 8.1, 12]

  /** Around every cutoff the scalings can push past 100, and past it. */
  const cutoffs = [
    undefined,
    0,
    50,
    59,
    60,
    61,
    89,
    90,
    91,
    94,
    95,
    96,
    99,
    100,
    100.5,
    101,
    150,
  ]

  it.each(composite)(
    '%s agrees on inputs straddling the branch boundaries',
    (_n, scorer) => {
      const rng = makeRng(0xc0ffee)
      const disagreements: string[] = []

      for (const ratioTarget of lengthRatios) {
        for (let trial = 0; trial < 12; trial++) {
          const base = randomText(rng, 14)
          // Grown by repetition so the length ratio lands where intended while the
          // content still shares material — a pair with nothing in common would
          // score 0 on every branch and prove nothing.
          const long =
            base.repeat(Math.max(1, Math.round(ratioTarget))) + randomText(rng, 4)

          for (const [s1, s2] of [
            [base, long],
            [long, base],
          ]) {
            for (const scoreCutoff of cutoffs) {
              const options = scoreCutoff === undefined ? {} : { scoreCutoff }
              const direct = scorer(s1, s2, options)
              const viaMatrix = matrixScores([s1], [s2], { scorer, ...options })[0][0]
              const viaPairs = pairScores([s1], [s2], { scorer, ...options })[0]

              if (
                Math.abs(direct - viaMatrix) > 1e-9 ||
                Math.abs(direct - viaPairs) > 1e-9
              ) {
                disagreements.push(
                  `${JSON.stringify(s1)} vs ${JSON.stringify(s2)} cutoff=${scoreCutoff}: ` +
                    `direct=${direct} matrix=${viaMatrix} pairs=${viaPairs}`,
                )
              }
            }
          }
        }
      }

      expect(disagreements.slice(0, 5)).toEqual([])
    },
  )

  // The empty-input answers are stated separately in each copy, and `qRatio`
  // differs from `ratio` only here.
  it.each(composite)('%s agrees on empty and whitespace-only inputs', (_n, scorer) => {
    const edges = ['', ' ', '   ', 'a', 'a b', ' a ', '\u{1F600}', '\u{1F600} \u{1F601}']
    const disagreements: string[] = []

    for (const s1 of edges) {
      for (const s2 of edges) {
        for (const scoreCutoff of [0, 50, 100, 101]) {
          const direct = scorer(s1, s2, { scoreCutoff })
          const viaMatrix = matrixScores([s1], [s2], { scorer, scoreCutoff })[0][0]
          const viaPairs = pairScores([s1], [s2], { scorer, scoreCutoff })[0]

          if (Math.abs(direct - viaMatrix) > 1e-9 || Math.abs(direct - viaPairs) > 1e-9) {
            disagreements.push(
              `${JSON.stringify(s1)} vs ${JSON.stringify(s2)} cutoff=${scoreCutoff}: ` +
                `direct=${direct} matrix=${viaMatrix} pairs=${viaPairs}`,
            )
          }
        }
      }
    }

    expect(disagreements.slice(0, 5)).toEqual([])
  })

  // Whitespace decides `wRatio`'s shortcut past the token scorers, and lazy
  // preparation means the shortcut is now the path that tokenises nothing — so
  // the two copies have to agree about which side holds a separator.
  it('wRatio agrees whether or not either side holds whitespace', () => {
    const withSpace = ['a b', 'hello world', 'x y z']
    const without = ['ab', 'helloworld', 'xyz']
    const disagreements: string[] = []

    for (const s1 of [...withSpace, ...without]) {
      for (const s2 of [...withSpace, ...without]) {
        for (const scoreCutoff of [0, 90, 95, 100]) {
          const direct = wRatio(s1, s2, { scoreCutoff })
          const viaMatrix = matrixScores([s1], [s2], {
            scorer: wRatio,
            scoreCutoff,
          })[0][0]

          if (Math.abs(direct - viaMatrix) > 1e-9) {
            disagreements.push(
              `${JSON.stringify(s1)} vs ${JSON.stringify(s2)} cutoff=${scoreCutoff}: ` +
                `direct=${direct} matrix=${viaMatrix}`,
            )
          }
        }
      }
    }

    expect(disagreements.slice(0, 5)).toEqual([])
  })
})

// The `scorers` table above drives the expectation-based tests, so it only holds
// scorers with a column in `cases`. `partialTokenSortRatio` and
// `partialTokenSetRatio` have none, which left them outside the prepared-vs-direct
// sweep entirely — and `partialTokenSortRatio` is the one that now scores through
// the query's held masks and pruning set rather than rebuilding them per
// candidate. A wrong cache there prunes windows that should have been scored,
// which lowers a score silently instead of raising an error.
describe('the partial token scorers agree prepared and direct', () => {
  const partials: ReadonlyArray<readonly [string, NormalizedScorer]> = [
    ['partialTokenSortRatio', partialTokenSortRatio],
    ['partialTokenSetRatio', partialTokenSetRatio],
    ['partialTokenRatio', partialTokenRatio],
  ]

  it.each(partials)('%s over random pairs', (_name, scorer) => {
    const rng = makeRng(0xbeef)
    const disagreements: string[] = []

    for (let trial = 0; trial < 400; trial++) {
      const s1 = randomText(rng, 16)
      const s2 = randomText(rng, 16)

      for (const scoreCutoff of [0, 60, 90, 100]) {
        const options = { scoreCutoff }
        const direct = scorer(s1, s2, options)
        const viaMatrix = matrixScores([s1], [s2], { scorer, ...options })[0][0]
        const viaPairs = pairScores([s1], [s2], { scorer, ...options })[0]
        const viaExtract = extract(s1, [s2], { scorer, ...options })

        const fromExtract = viaExtract.length > 0 ? viaExtract[0].score : 0
        if (
          Math.abs(direct - viaMatrix) > 1e-9 ||
          Math.abs(direct - viaPairs) > 1e-9 ||
          Math.abs(direct - fromExtract) > 1e-9
        ) {
          disagreements.push(
            `${JSON.stringify(s1)} vs ${JSON.stringify(s2)} cutoff=${scoreCutoff}: ` +
              `direct=${direct} matrix=${viaMatrix} pairs=${viaPairs} extract=${fromExtract}`,
          )
        }
      }
    }

    expect(disagreements.slice(0, 5)).toEqual([])
  })

  // The held sorted-query state is only allowed to serve the scan where the
  // sorted query is the shorter side; `partialAlignmentConverted` drops it
  // otherwise. These pairs put the query on both sides of that gate, and the
  // token sets both overlapping and disjoint — disjoint is the case that reaches
  // the sorted-whole comparison at all, since any shared token answers 100 first.
  it.each(partials)(
    '%s with the query long, short, overlapping and disjoint',
    (_n, scorer) => {
      const queries = ['alpha bravo charlie', 'alphax bravox', 'zz']
      const choices = [
        'alpha bravo charlie',
        'charlie bravo alpha',
        'zulu yankee charlie bravo alpha whiskey victor',
        'alphay bravoy charliey zuluy yankeey',
        'q',
        '',
      ]
      const disagreements: string[] = []

      for (const query of queries) {
        for (const choice of choices) {
          for (const [s1, s2] of [
            [query, choice],
            [choice, query],
          ]) {
            for (const scoreCutoff of [0, 50, 85, 100]) {
              const direct = scorer(s1, s2, { scoreCutoff })
              const viaMatrix = matrixScores([s1], [s2], { scorer, scoreCutoff })[0][0]

              if (Math.abs(direct - viaMatrix) > 1e-9) {
                disagreements.push(
                  `${JSON.stringify(s1)} vs ${JSON.stringify(s2)} cutoff=${scoreCutoff}: ` +
                    `direct=${direct} matrix=${viaMatrix}`,
                )
              }
            }
          }
        }
      }

      expect(disagreements.slice(0, 5)).toEqual([])
    },
  )

  // One prepared query against many candidates is what makes the held state
  // worth having, and also what would expose it leaking between candidates.
  it.each(partials)(
    '%s holds query state across a mixed candidate list',
    (_n, scorer) => {
      const query = 'alphax bravox charliex'
      const choices = [
        'alphay bravoy charliey zuluy',
        'alphax bravox charliex',
        '',
        'q',
        'zulu yankee xray whiskey victor uniform tango sierra',
        'charliex',
        'alphax',
      ]

      const prepared = extract(query, choices, { scorer, limit: choices.length })
      for (const row of prepared) {
        expect(row.score).toBeCloseTo(scorer(query, row.choice), 9)
      }
    },
  )
})

// The fuzz scorers accept `scoreHint` and ignore it, which is a decision rather
// than an oversight — see `FuzzOptions.scoreHint`. Wiring it into prepared
// `ratio` was measured at 1.49-1.71x the kernel iterations, because the only
// lever available is an early exit and one sized by an estimate has to be
// redone when the estimate was optimistic.
//
// Ignoring a hint is trivially safe; *honouring* one is where the risk lives.
// These tests pin the observable contract, so that an attempt to act on the
// hint has to keep every answer identical to prove itself.
describe('scoreHint never changes a fuzz answer', () => {
  const hints = [0, 10, 50, 70, 90, 95, 99, 100, 120]
  const cutoffs = [undefined, 0, 50, 70, 90, 100]

  // Long enough to reach the bounded kernel: `ratioPrepared` only budgets its
  // scan once the two inputs together pass 128 elements, so a corpus of short
  // strings would exercise none of the code a hint could possibly reach.
  const texts = [
    '',
    'a',
    'senior frontend engineer',
    'frontend engineer senior',
    'senior frontend engineer react typescript zurich remote hybrid full time role',
    'senior frontend engineer react typescript zurich remote hybrid full time post',
    'a posting describing a backend platform sre data warehouse position elsewhere',
    'x'.repeat(200),
    `${'x'.repeat(190)}${'y'.repeat(10)}`,
  ]

  it.each(scorers.filter(([name]) => name.endsWith('Ratio') || name === 'ratio'))(
    '%s answers identically for every hint',
    (name, scorer) => {
      const disagreements: string[] = []

      for (const s1 of texts) {
        for (const s2 of texts) {
          for (const scoreCutoff of cutoffs) {
            const options: ScoreOptions = scoreCutoff === undefined ? {} : { scoreCutoff }
            const base = scorer(s1, s2, options)
            const viaExtract = extractOne(s1, [s2], { scorer, ...options })?.score ?? null

            for (const scoreHint of hints) {
              const hinted = scorer(s1, s2, { ...options, scoreHint })
              const hintedPrepared =
                extractOne(s1, [s2], { scorer, ...options, scoreHint })?.score ?? null

              if (Math.abs(base - hinted) > 1e-9) {
                disagreements.push(
                  `${name} ${JSON.stringify(s1.slice(0, 20))} vs ` +
                    `${JSON.stringify(s2.slice(0, 20))} cutoff=${scoreCutoff} ` +
                    `hint=${scoreHint}: direct ${base} != ${hinted}`,
                )
              }
              if (viaExtract !== hintedPrepared) {
                disagreements.push(
                  `${name} ${JSON.stringify(s1.slice(0, 20))} vs ` +
                    `${JSON.stringify(s2.slice(0, 20))} cutoff=${scoreCutoff} ` +
                    `hint=${scoreHint}: prepared ${viaExtract} != ${hintedPrepared}`,
                )
              }
            }
          }
        }
      }

      expect(disagreements.slice(0, 5)).toEqual([])
    },
  )

  // `extract` tightens its own cutoff as it goes, so a hint meets a moving
  // target rather than a fixed one — including the ranking, not just the scores.
  it('leaves extract rankings untouched', () => {
    for (const [name, scorer] of scorers.filter(
      ([n]) => n === 'ratio' || n === 'wRatio',
    )) {
      const base = JSON.stringify(extract(texts[4], texts, { scorer, limit: 5 }))
      for (const scoreHint of hints) {
        const hinted = JSON.stringify(
          extract(texts[4], texts, { scorer, limit: 5, scoreHint }),
        )
        expect(`${name} hint=${scoreHint}: ${hinted}`).toBe(
          `${name} hint=${scoreHint}: ${base}`,
        )
      }
    }
  })
})
