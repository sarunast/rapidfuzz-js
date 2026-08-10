// Not ported from RapidFuzz — upstream's `CachedScorer` classes are the same
// code as its free functions with the query bound, so there is nothing there to
// disagree with. Here a prepared scorer is a second implementation: it holds
// masks, or a token view, or nothing at all, and it re-derives each of the four
// cutoff conventions from the bound it was handed.
//
// So the contract worth pinning is that the two never differ. Every metric is
// run through all four conventions, at a sweep of cutoffs, both ways round —
// once as a direct call and once through the prepared factory `process` uses.
// The metric-specific expected values live in the ported suites; what is
// checked here is agreement, plus the handful of answers only the prepared path
// can give: a missing choice, a choice that is not a sequence, and a query that
// is not one either.
import { describe, expect, it } from 'vitest'

import {
  prepareScorerOf,
  type MaybeSequence,
  type PrepareScorer,
  type Sequence,
} from '../../src/_common.js'
import {
  damerauLevenshteinDistance,
  damerauLevenshteinNormalizedDistance,
  damerauLevenshteinNormalizedSimilarity,
  damerauLevenshteinSimilarity,
} from '../../src/distance/damerauLevenshtein.js'
import {
  hammingDistance,
  hammingNormalizedDistance,
  hammingNormalizedSimilarity,
  hammingSimilarity,
} from '../../src/distance/hamming.js'
import {
  indelDistance,
  indelNormalizedDistance,
  indelNormalizedSimilarity,
  indelSimilarity,
} from '../../src/distance/indel.js'
import {
  jaroDistance,
  jaroNormalizedDistance,
  jaroNormalizedSimilarity,
  jaroSimilarity,
} from '../../src/distance/jaro.js'
import {
  jaroWinklerDistance,
  jaroWinklerNormalizedDistance,
  jaroWinklerNormalizedSimilarity,
  jaroWinklerSimilarity,
} from '../../src/distance/jaroWinkler.js'
import {
  lcsSeqDistance,
  lcsSeqNormalizedDistance,
  lcsSeqNormalizedSimilarity,
  lcsSeqSimilarity,
} from '../../src/distance/lcsSeq.js'
import {
  levenshteinDistance,
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
  levenshteinSimilarity,
} from '../../src/distance/levenshtein.js'
import {
  osaDistance,
  osaNormalizedDistance,
  osaNormalizedSimilarity,
  osaSimilarity,
} from '../../src/distance/osa.js'
import {
  postfixDistance,
  postfixNormalizedDistance,
  postfixNormalizedSimilarity,
  postfixSimilarity,
} from '../../src/distance/postfix.js'
import {
  prefixDistance,
  prefixNormalizedDistance,
  prefixNormalizedSimilarity,
  prefixSimilarity,
} from '../../src/distance/prefix.js'
import { callUntyped } from '../common.js'

interface Scorers {
  readonly distance: (
    s1: Sequence,
    s2: Sequence,
    options?: { scoreCutoff?: number },
  ) => number
  readonly similarity: (
    s1: Sequence,
    s2: Sequence,
    options?: { scoreCutoff?: number },
  ) => number
  readonly normalizedDistance: (
    s1: MaybeSequence,
    s2: MaybeSequence,
    options?: { scoreCutoff?: number },
  ) => number
  readonly normalizedSimilarity: (
    s1: MaybeSequence,
    s2: MaybeSequence,
    options?: { scoreCutoff?: number },
  ) => number
}

const FAMILIES: ReadonlyMap<string, Scorers> = new Map([
  [
    'DamerauLevenshtein',
    {
      distance: damerauLevenshteinDistance,
      similarity: damerauLevenshteinSimilarity,
      normalizedDistance: damerauLevenshteinNormalizedDistance,
      normalizedSimilarity: damerauLevenshteinNormalizedSimilarity,
    },
  ],
  [
    'Hamming',
    {
      distance: hammingDistance,
      similarity: hammingSimilarity,
      normalizedDistance: hammingNormalizedDistance,
      normalizedSimilarity: hammingNormalizedSimilarity,
    },
  ],
  [
    'Indel',
    {
      distance: indelDistance,
      similarity: indelSimilarity,
      normalizedDistance: indelNormalizedDistance,
      normalizedSimilarity: indelNormalizedSimilarity,
    },
  ],
  [
    'Jaro',
    {
      distance: jaroDistance,
      similarity: jaroSimilarity,
      normalizedDistance: jaroNormalizedDistance,
      normalizedSimilarity: jaroNormalizedSimilarity,
    },
  ],
  [
    'JaroWinkler',
    {
      distance: jaroWinklerDistance,
      similarity: jaroWinklerSimilarity,
      normalizedDistance: jaroWinklerNormalizedDistance,
      normalizedSimilarity: jaroWinklerNormalizedSimilarity,
    },
  ],
  [
    'LCSseq',
    {
      distance: lcsSeqDistance,
      similarity: lcsSeqSimilarity,
      normalizedDistance: lcsSeqNormalizedDistance,
      normalizedSimilarity: lcsSeqNormalizedSimilarity,
    },
  ],
  [
    'Levenshtein',
    {
      distance: levenshteinDistance,
      similarity: levenshteinSimilarity,
      normalizedDistance: levenshteinNormalizedDistance,
      normalizedSimilarity: levenshteinNormalizedSimilarity,
    },
  ],
  [
    'OSA',
    {
      distance: osaDistance,
      similarity: osaSimilarity,
      normalizedDistance: osaNormalizedDistance,
      normalizedSimilarity: osaNormalizedSimilarity,
    },
  ],
  [
    'Prefix',
    {
      distance: prefixDistance,
      similarity: prefixSimilarity,
      normalizedDistance: prefixNormalizedDistance,
      normalizedSimilarity: prefixNormalizedSimilarity,
    },
  ],
  [
    'Postfix',
    {
      distance: postfixDistance,
      similarity: postfixSimilarity,
      normalizedDistance: postfixNormalizedDistance,
      normalizedSimilarity: postfixNormalizedSimilarity,
    },
  ],
])

/** Conventions in the order `PreparedMetricKind` names them. */
const CONVENTIONS = [
  'distance',
  'similarity',
  'normalizedDistance',
  'normalizedSimilarity',
] as const

/** Raw conventions count elements; normalised ones live in `[0, 1]`. */
const RAW_CUTOFFS = [0, 1, 2, 3, 5, 17, 64, 200]
const NORMALIZED_CUTOFFS = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]

const LONG_A = 'the quick brown fox jumps over the lazy dog, twice over and again'
const LONG_B = 'the quick brown fox leaps over the lazy dog, twice over and again!'

/**
 * Pairs chosen for the dispatchers rather than for any one metric: empty on
 * either side, equal lengths, very unequal ones, and a pair long enough that
 * every kernel family leaves its one-word path.
 */
const PAIRS: ReadonlyArray<readonly [Sequence, Sequence]> = [
  ['', ''],
  ['', 'abcd'],
  ['abcd', ''],
  ['abcd', 'abcd'],
  ['abcd', 'abdc'],
  ['abcd', 'wxyz'],
  ['abcd', 'abcdefghij'],
  ['abcdefghij', 'abcd'],
  [LONG_A, LONG_B],
  [LONG_B, LONG_A],
  [LONG_A, 'fox'],
  [LONG_A.repeat(3), LONG_B.repeat(3)],
  // A string against the same text as elements, which is the pair a prepared
  // query and a converted choice actually meet as.
  [LONG_A, [...LONG_A].map((c) => c.codePointAt(0))],
  [[...LONG_A], LONG_B],
]

function preparedOf(scorer: object): PrepareScorer {
  const factory = prepareScorerOf(scorer)
  if (factory === null) throw new Error('scorer has no prepared factory')
  return factory
}

describe('a prepared scorer agrees with a direct call', () => {
  for (const [name, scorers] of FAMILIES) {
    it(`for ${name}, at every cutoff of every convention`, () => {
      for (const convention of CONVENTIONS) {
        const scorer = scorers[convention]
        const prepare = preparedOf(scorer)
        // Jaro and Jaro-Winkler are normalised in all four conventions —
        // their raw "distance" is `1 - similarity` — so every cutoff they take
        // is checked against `[0, 1]`.
        const cutoffs =
          convention.startsWith('normalized') || name.startsWith('Jaro')
            ? NORMALIZED_CUTOFFS
            : RAW_CUTOFFS

        for (const [s1, s2] of PAIRS) {
          const what = `${name}.${convention} on ${String(s1).slice(0, 12)}`
          const score = prepare(s1, {})
          expect(score(s2, null, null), what).toBeCloseTo(scorer(s1, s2), 9)

          for (const scoreCutoff of cutoffs) {
            expect(score(s2, scoreCutoff, null), `${what} at ${scoreCutoff}`).toBeCloseTo(
              scorer(s1, s2, { scoreCutoff }),
              9,
            )
          }
        }
      }
    })
  }
})

describe('what only a prepared scorer is asked', () => {
  for (const [name, scorers] of FAMILIES) {
    it(`reports maximum dissimilarity for a missing choice — ${name}`, () => {
      expect(preparedOf(scorers.normalizedDistance)('abcd', {})(null, null, null)).toBe(1)
      expect(preparedOf(scorers.normalizedSimilarity)('abcd', {})(null, null, null)).toBe(
        0,
      )
      expect(
        preparedOf(scorers.normalizedDistance)('abcd', {})(Number.NaN, null, null),
      ).toBe(1)
    })

    // Only the normalised conventions have an answer for a missing input; the
    // raw ones fall through to the refusal any non-sequence gets.
    it(`refuses a missing choice for a raw convention — ${name}`, () => {
      expect(() => preparedOf(scorers.distance)('abcd', {})(null, null, null)).toThrow(
        TypeError,
      )
      expect(() => preparedOf(scorers.similarity)('abcd', {})(null, null, null)).toThrow(
        TypeError,
      )
    })

    it(`refuses a choice that is not a sequence — ${name}`, () => {
      for (const convention of CONVENTIONS) {
        const score = preparedOf(scorers[convention])('abcd', {})
        expect(() => score(7, null, null)).toThrow(TypeError)
      }
    })

    it(`refuses a query that is not a sequence — ${name}`, () => {
      for (const convention of CONVENTIONS) {
        const prepare = preparedOf(scorers[convention])
        expect(() => callUntyped(prepare, 7, {})).toThrow(TypeError)
      }
    })
  }
})

// Hamming's `pad` is parsed once, when the query is prepared, rather than per
// candidate — so the prepared path has its own copy of the refusal.
describe('a prepared Hamming scorer carries its pad option', () => {
  it('refuses a length mismatch when padding is off', () => {
    const score = preparedOf(hammingDistance)('abcd', { pad: false })
    expect(score('abcd', null, null)).toBe(0)
    expect(() => score('abcde', null, null)).toThrow('Sequences are not the same length.')
  })

  it('pads by default', () => {
    expect(preparedOf(hammingDistance)('abcd', {})('abcde', null, null)).toBe(1)
  })
})

// The bounded loop is a second implementation of the count, and it splits again
// on whether both sides are strings.
describe('bounded Hamming over elements rather than characters', () => {
  it('agrees with the exact count inside the bound', () => {
    const a = [...'abcdefghij'].map((c) => c.codePointAt(0))
    const b = [...'abcdefghXY'].map((c) => c.codePointAt(0))
    expect(hammingDistance(a, b)).toBe(2)
    expect(hammingDistance(a, b, { scoreCutoff: 2 })).toBe(2)
    expect(hammingDistance(a, b, { scoreCutoff: 1 })).toBe(2)
    expect(hammingDistance(a, b, { scoreCutoff: 0 })).toBe(1)
  })
})

// Damerau-Levenshtein keeps its rows in `Int16Array`s while the largest value
// fits one, and switches to `Int32Array`s past that. The switch is on the
// longer side alone, so a long input against a short one reaches it without the
// quadratic cost that a long pair would carry.
describe('the Damerau-Levenshtein row store grows past 16 bits', () => {
  it('scores a sequence longer than a signed short can count', () => {
    const long = 'a'.repeat(40_000)
    expect(damerauLevenshteinDistance(long, 'b')).toBe(40_000)
    expect(damerauLevenshteinDistance('b', long)).toBe(40_000)

    // Longer again, so the rows have to grow rather than be reused, and then
    // shorter, so the grown ones are reused rather than replaced.
    const longer = 'a'.repeat(90_000)
    expect(damerauLevenshteinDistance(longer, 'b')).toBe(90_000)
    expect(damerauLevenshteinDistance(long, 'b')).toBe(40_000)
  })
})
