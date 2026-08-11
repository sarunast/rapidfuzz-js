// Ported from RapidFuzz tests/distance/test_distance.py
import { describe, expect, it } from 'vitest'

import {
  damerauLevenshteinNormalizedDistance,
  damerauLevenshteinNormalizedSimilarity,
} from '../../src/algorithms/damerauLevenshtein/implementation.js'
import {
  hammingNormalizedDistance,
  hammingNormalizedSimilarity,
} from '../../src/algorithms/hamming/implementation.js'
import {
  indelNormalizedDistance,
  indelNormalizedSimilarity,
  indelSimilarity,
} from '../../src/algorithms/indel/implementation.js'
import {
  jaroNormalizedDistance,
  jaroNormalizedSimilarity,
} from '../../src/algorithms/jaro/implementation.js'
import {
  jaroWinklerNormalizedDistance,
  jaroWinklerNormalizedSimilarity,
} from '../../src/algorithms/jaroWinkler/implementation.js'
import {
  lcsSeqNormalizedDistance,
  lcsSeqNormalizedSimilarity,
} from '../../src/algorithms/lcs/implementation.js'
import {
  levenshteinDistance,
  levenshteinSimilarity,
} from '../../src/algorithms/levenshtein/implementation.js'
import {
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
} from '../../src/algorithms/levenshtein/implementation.js'
import {
  osaNormalizedDistance,
  osaNormalizedSimilarity,
} from '../../src/algorithms/osa/implementation.js'
import {
  postfixNormalizedDistance,
  postfixNormalizedSimilarity,
} from '../../src/algorithms/postfix/implementation.js'
import {
  prefixDistance,
  prefixNormalizedDistance,
  prefixNormalizedSimilarity,
} from '../../src/algorithms/prefix/implementation.js'
import type {
  MaybeSequence,
  ScorerOptions,
} from '../../src/algorithms/shared/scorerSupport.js'
import { callUntyped } from '../common.js'

interface NormalizedScorer {
  normalizedDistance: (s1: MaybeSequence, s2: MaybeSequence) => number
  normalizedSimilarity: (s1: MaybeSequence, s2: MaybeSequence) => number
}

/** Mirrors `all_scorer_modules` from `tests/distance/common.py`. */
const SCORERS: ReadonlyArray<readonly [string, NormalizedScorer]> = [
  [
    'DamerauLevenshtein',
    {
      normalizedDistance: damerauLevenshteinNormalizedDistance,
      normalizedSimilarity: damerauLevenshteinNormalizedSimilarity,
    },
  ],
  [
    'Hamming',
    {
      normalizedDistance: hammingNormalizedDistance,
      normalizedSimilarity: hammingNormalizedSimilarity,
    },
  ],
  [
    'Indel',
    {
      normalizedDistance: indelNormalizedDistance,
      normalizedSimilarity: indelNormalizedSimilarity,
    },
  ],
  [
    'Jaro',
    {
      normalizedDistance: jaroNormalizedDistance,
      normalizedSimilarity: jaroNormalizedSimilarity,
    },
  ],
  [
    'JaroWinkler',
    {
      normalizedDistance: jaroWinklerNormalizedDistance,
      normalizedSimilarity: jaroWinklerNormalizedSimilarity,
    },
  ],
  [
    'LCSseq',
    {
      normalizedDistance: lcsSeqNormalizedDistance,
      normalizedSimilarity: lcsSeqNormalizedSimilarity,
    },
  ],
  [
    'Levenshtein',
    {
      normalizedDistance: levenshteinNormalizedDistance,
      normalizedSimilarity: levenshteinNormalizedSimilarity,
    },
  ],
  [
    'OSA',
    {
      normalizedDistance: osaNormalizedDistance,
      normalizedSimilarity: osaNormalizedSimilarity,
    },
  ],
  [
    'Postfix',
    {
      normalizedDistance: postfixNormalizedDistance,
      normalizedSimilarity: postfixNormalizedSimilarity,
    },
  ],
  [
    'Prefix',
    {
      normalizedDistance: prefixNormalizedDistance,
      normalizedSimilarity: prefixNormalizedSimilarity,
    },
  ],
]

describe('normalized scorers handle a missing value', () => {
  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      for (const missing of [null, undefined, Number.NaN]) {
        expect(callUntyped(scorer.normalizedDistance, missing, 'test')).toBe(1)
        expect(callUntyped(scorer.normalizedSimilarity, missing, 'test')).toBe(0)
        expect(callUntyped(scorer.normalizedDistance, 'test', missing)).toBe(1)
        expect(callUntyped(scorer.normalizedSimilarity, 'test', missing)).toBe(0)
      }
    })
  }
})

describe('normalized scorers treat two empty strings as identical', () => {
  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer.normalizedDistance('', '')).toBe(0)
      expect(scorer.normalizedSimilarity('', '')).toBe(1)
    })
  }
})

describe('arrays are treated like strings', () => {
  const text = 'the wonderful new york mets'

  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer.normalizedSimilarity(Array.from(text), Array.from(text))).toBe(1)
      expect(scorer.normalizedSimilarity(text, Array.from(text))).toBe(1)
      expect(scorer.normalizedSimilarity(Array.from(text), text)).toBe(1)
    })
  }
})

describe('byte arrays are treated like strings', () => {
  const text = 'the wonderful new york mets'
  const bytes = new TextEncoder().encode(text)

  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer.normalizedSimilarity(bytes, bytes)).toBe(1)
      expect(scorer.normalizedSimilarity(text, bytes)).toBe(1)
      expect(scorer.normalizedSimilarity(bytes, text)).toBe(1)
    })
  }
})

describe('string elements behave like their code points', () => {
  const mixed = [0x61, 0x61, 'a', 'a']

  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer.normalizedSimilarity(mixed, mixed)).toBe(1)
      expect(scorer.normalizedSimilarity('aaaa', mixed)).toBe(1)
      expect(scorer.normalizedSimilarity(mixed, 'aaaa')).toBe(1)
    })
  }
})

describe('-1 and -2 are distinct elements', () => {
  // Upstream guards this case because hash(-1) == hash(-2) in CPython.
  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer.normalizedSimilarity([0, -1], [0, -2])).not.toBe(1)
    })
  }
})

describe('non-character elements compare by value', () => {
  // Upstream's CustomHashable checks that elements are compared via `hash`.
  // JavaScript has no hash protocol, so the equivalent is `===`, which is
  // value equality for multi-character strings.
  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer.normalizedSimilarity(['aa', 'aa'], ['aa', 'aa'])).toBe(1)
      expect(scorer.normalizedSimilarity(['aa', 'aa'], ['aa', 'bb'])).not.toBe(1)
    })
  }
})

// Not ported — upstream raises `TypeError` when a processor returns something
// that is not a sequence, but its tests do not cover the case, and here the
// same return produced a *score* rather than an error.
describe('a processor has to return a sequence', () => {
  // `convSequence` reads a `length` off whatever it is handed, and
  // `new Array(undefined)` is an array of one `undefined` — so two unrelated
  // inputs both became `[undefined]` and scored as identical. The processor is
  // set rather than written as a literal because these are returns TypeScript
  // already refuses and a JavaScript caller does not.
  const returning = (value: unknown): ScorerOptions => {
    const options: ScorerOptions = {}
    Reflect.set(options, 'processor', () => value)
    return options
  }

  for (const value of [123, null, undefined, true, Symbol('s'), { a: 1 }]) {
    it(`rejects ${String(value)}`, () => {
      expect(() => levenshteinDistance('abc', 'zzzz', returning(value))).toThrow(
        TypeError,
      )
      expect(() => indelNormalizedSimilarity('abc', 'zzzz', returning(value))).toThrow(
        TypeError,
      )
    })
  }

  it('still takes every sequence form a scorer accepts', () => {
    expect(levenshteinDistance('abc', 'abd', { processor: (s) => s })).toBe(1)
    expect(levenshteinDistance('abc', 'abd', returning('abd'))).toBe(0)
    expect(levenshteinDistance('abc', 'abd', returning([97, 98, 100]))).toBe(0)
    expect(levenshteinDistance('abc', 'abd', returning(Uint8Array.of(97, 98, 100)))).toBe(
      0,
    )
  })
})

// Not ported — upstream has no test for these, but it does refuse them, and
// every expected value below was taken from rapidfuzz 3.14.5 directly.
//
// Upstream reaches the refusal through its bindings rather than by checking:
// a raw cutoff is converted to `uint64_t`, so a NaN or an infinity fails the
// conversion to an integer, and a value that does not fit an unsigned 64-bit
// integer raises `OverflowError`. A normalised cutoff is range-checked
// outright. Both are refused here, as `RangeError` — which is what JavaScript
// reserves for an argument outside its range, where Python's choice of class is
// an artefact of how it converts.
//
// The conversion truncates before it range-checks, which is why `-0.5` is
// *accepted* below and behaves as `0`; it is the value after truncation that
// has to be a `uint64_t`.
//
// One half of that conversion is deliberately *not* reproduced: upstream also
// refuses a cutoff at or above `2 ** 64`, because that is where its C type runs
// out. There is no `uint64_t` here, a cutoff is a count of elements held in a
// `number`, and the limit that means something is finiteness — so `2 ** 64` is
// taken. No distance over a JavaScript string can reach either bound, so the
// two answer alike wherever the answer is a score.
//
// The `fuzz` scorers score out of 100 and validate nothing, upstream included,
// so they keep taking any cutoff at all.
describe('a scoreCutoff outside its range is refused', () => {
  const raw = [levenshteinDistance, levenshteinSimilarity]
  const normalized = [levenshteinNormalizedDistance, levenshteinNormalizedSimilarity]

  it('refuses a raw cutoff that is no count of elements', () => {
    for (const scorer of raw) {
      for (const scoreCutoff of [-1, Number.NaN, Infinity, -Infinity]) {
        expect(() => scorer('abc', 'abd', { scoreCutoff })).toThrow(RangeError)
      }
    }
  })

  // A cutoff no distance can reach is the exact score, whatever its size. The
  // first of these is upstream's largest — `2 ** 64 - 1` is not a `number`, so
  // it is stated the way the doubles fall — and the rest are past the ceiling
  // upstream converts against and are taken here anyway.
  it('takes any finite cutoff, including ones past upstream ceiling', () => {
    for (const scoreCutoff of [2 ** 64 - 2048, 2 ** 64, 2 ** 70, Number.MAX_VALUE]) {
      expect(levenshteinDistance('abc', 'abd', { scoreCutoff })).toBe(1)
      expect(levenshteinSimilarity('abc', 'abd', { scoreCutoff })).toBe(0)
    }
  })

  it('refuses a normalised cutoff outside zero to one', () => {
    for (const scorer of normalized) {
      for (const scoreCutoff of [2, -1, 1.5, Infinity, -Infinity]) {
        expect(() => scorer('abc', 'abd', { scoreCutoff })).toThrow(RangeError)
      }
    }
  })

  // Upstream's range check is a pair of comparisons, and NaN fails both — so it
  // survives to be compared against the score, fails that too, and yields the
  // worst score. Refusing it would be stricter than the thing being ported.
  it('lets a NaN cutoff through to the worst score, as upstream does', () => {
    expect(levenshteinNormalizedDistance('abc', 'abd', { scoreCutoff: NaN })).toBe(1)
    expect(levenshteinNormalizedSimilarity('abc', 'abd', { scoreCutoff: NaN })).toBe(0)
  })

  it('takes the cutoffs upstream takes, including a fractional raw one', () => {
    expect(levenshteinDistance('abc', 'abd', { scoreCutoff: 1.5 })).toBe(1)
    expect(levenshteinSimilarity('abc', 'abd', { scoreCutoff: 1.5 })).toBe(2)
    expect(levenshteinDistance('abc', 'abd', { scoreCutoff: 0 })).toBe(1)
    expect(levenshteinNormalizedDistance('abc', 'abd', { scoreCutoff: 0 })).toBe(1)
    expect(levenshteinNormalizedDistance('abc', 'abd', { scoreCutoff: 1 })).toBeCloseTo(
      1 / 3,
      12,
    )
  })

  // The cutoff a raw scorer is held to is the truncated one, everywhere it is
  // read: `1.9` means `1`, so a distance of 2 is rejected as `1 + 1` rather
  // than reported as `1.9 + 1`, and a similarity of 1 *clears* the cutoff
  // instead of being rejected. The similarity direction is the one that needs a
  // scorer to truncate before it bounds its kernel — bounding at `max - 1.9`
  // stops a search that `max - 1` would have let finish.
  it('truncates a fractional raw cutoff, as the uint64_t conversion does', () => {
    expect(levenshteinDistance('abc', 'xyz', { scoreCutoff: 1.9 })).toBe(2)
    expect(prefixDistance('abc', 'xyz', { scoreCutoff: 1.9 })).toBe(2)
    expect(levenshteinSimilarity('ab', 'ax', { scoreCutoff: 1.9 })).toBe(1)
    expect(levenshteinSimilarity('abcd', 'abce', { scoreCutoff: 3.7 })).toBe(3)
    expect(indelSimilarity('ab', 'ax', { scoreCutoff: 2.5 })).toBe(2)
  })

  // Truncation is what makes a cutoff in `(-1, 0)` legal: it becomes `0`, and
  // only then is it checked for fitting an unsigned integer.
  it('takes a raw cutoff that truncates to zero', () => {
    expect(levenshteinDistance('abc', 'abd', { scoreCutoff: -0.5 })).toBe(1)
    expect(levenshteinSimilarity('abc', 'abd', { scoreCutoff: -0.5 })).toBe(2)
  })

  // A missing input answers before any cutoff is looked at. Upstream does the
  // same: `normalized_distance(None, 'abd', score_cutoff=2)` returns 1.0.
  it('answers a missing input before it reaches the cutoff', () => {
    expect(levenshteinNormalizedDistance(null, 'abd', { scoreCutoff: 2 })).toBe(1)
    expect(levenshteinNormalizedSimilarity(null, 'abd', { scoreCutoff: 2 })).toBe(0)
  })
})
