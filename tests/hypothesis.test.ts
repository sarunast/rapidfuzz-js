// Ported from RapidFuzz tests/test_hypothesis.py
//
// Hypothesis's `@given(st.text())` maps to fast-check's `fc.property` with
// `fc.string`. The reference implementations below are ports of the slow,
// obviously-correct ones upstream keeps for exactly this purpose.
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import { indelDistance, indelNormalizedSimilarity } from '../src/distance/indel.js'
import { indelEditops, indelOpcodes } from '../src/distance/indel.js'
import { jaroWinklerSimilarity } from '../src/distance/jaroWinkler.js'
import {
  levenshteinDistance,
  levenshteinEditops,
  levenshteinNormalizedSimilarity,
  levenshteinOpcodes,
} from '../src/distance/levenshtein.js'
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
} from '../src/_fuzz/legacy.js'
import { extract } from '../src/search.js'
import { defaultProcess } from '../src/utils.js'
import { matrixScores } from './matrix.js'

const RUNS = 100
const BLOCK_RUNS = 50

/** Upstream's `HYPOTHESIS_ALPHABET`: letters, digits and punctuation. */
const ALPHABET =
  'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

const text = (
  options: { minLength?: number; maxLength?: number } = {},
): fc.Arbitrary<string> =>
  fc
    .array(fc.constantFrom(...ALPHABET.split('')), {
      minLength: options.minLength ?? 0,
      maxLength: options.maxLength ?? 32,
    })
    .map((chars) => chars.join(''))

/** Text of at least 65 characters, exercising the multi-word bit-parallel path. */
const longText = (): fc.Arbitrary<string> => text({ minLength: 65, maxLength: 96 })

/**
 * The same alphabet with astral characters mixed in.
 *
 * Upstream has no counterpart: in Python every one of these is a single
 * element already. Here they are two UTF-16 units each, so a scorer or an
 * `apply` that indexes in the wrong units cuts one in half — which shows up as
 * a lone surrogate in the output rather than as an error.
 */
const astralText = (): fc.Arbitrary<string> =>
  fc
    .array(
      fc.constantFrom(...ALPHABET.split(''), '\u{1F600}', '\u{1F44D}', '\u{1F1E8}'),
      {
        maxLength: 32,
      },
    )
    .map((chars) => chars.join(''))

function isclose(a: number, b: number, relTol = 1e-9): boolean {
  return Math.abs(a - b) <= relTol * Math.max(Math.abs(a), Math.abs(b))
}

/**
 * Straightforward O(nm) Levenshtein — the "much less error prone" reference
 * upstream keeps to check the bit-parallel implementation against.
 */
function levenshtein(
  s1: string,
  s2: string,
  weights: [number, number, number] = [1, 1, 1],
): number {
  const a = Array.from(s1)
  const b = Array.from(s2)
  const [insert, delete_, substitute] = weights

  const dist: number[][] = Array.from({ length: a.length + 1 }, () =>
    new Array<number>(b.length + 1).fill(0),
  )

  for (let row = 1; row <= a.length; row++) dist[row][0] = row * delete_
  for (let col = 1; col <= b.length; col++) dist[0][col] = col * insert

  for (let col = 1; col <= b.length; col++) {
    for (let row = 1; row <= a.length; row++) {
      const cost = a[row - 1] === b[col - 1] ? 0 : substitute

      dist[row][col] = Math.min(
        dist[row - 1][col] + delete_,
        dist[row][col - 1] + insert,
        dist[row - 1][col - 1] + cost,
      )
    }
  }

  return dist[a.length][b.length]
}

function normalizeDistance(
  dist: number,
  s1: string,
  s2: string,
  weights: [number, number, number] = [1, 1, 1],
): number {
  const [insert, delete_, substitute] = weights
  const len1 = Array.from(s1).length
  const len2 = Array.from(s2).length

  const maxDist =
    len1 > len2
      ? Math.min(
          len1 * delete_ + len2 * insert,
          len2 * substitute + (len1 - len2) * delete_,
        )
      : Math.min(
          len1 * delete_ + len2 * insert,
          len1 * substitute + (len2 - len1) * insert,
        )

  return maxDist ? 1 - dist / maxDist : 1
}

/** Reference Jaro, ported from upstream's `jaro_similarity`. */
function jaroSimilarityRef(pattern: string, textValue: string): number {
  const p = Array.from(pattern)
  const t = Array.from(textValue)

  if (p.length === 0 && t.length === 0) return 1
  if (p.length === 0 || t.length === 0) return 0

  const bound = Math.max(Math.floor(Math.max(p.length, t.length) / 2) - 1, 0)

  const flagsP = new Array<boolean>(p.length).fill(false)
  const flagsT = new Array<boolean>(t.length).fill(false)
  let common = 0

  for (let i = 0; i < t.length; i++) {
    const lo = Math.max(0, i - bound)
    const hi = Math.min(p.length, i + bound + 1)

    for (let j = lo; j < hi; j++) {
      if (!flagsP[j] && p[j] === t[i]) {
        flagsP[j] = true
        flagsT[i] = true
        common++
        break
      }
    }
  }

  if (common === 0) return 0

  let transpositions = 0
  let j = 0

  for (let i = 0; i < t.length; i++) {
    if (!flagsT[i]) continue
    while (!flagsP[j]) j++
    if (t[i] !== p[j]) transpositions++
    j++
  }

  transpositions = Math.floor(transpositions / 2)

  return (common / p.length + common / t.length + (common - transpositions) / common) / 3
}

function jaroWinklerSimilarityRef(
  pattern: string,
  textValue: string,
  prefixWeight = 0.1,
): number {
  const p = Array.from(pattern)
  const t = Array.from(textValue)
  const maxPrefix = Math.min(Math.min(p.length, t.length), 4)

  let prefix = 0
  while (prefix < maxPrefix && t[prefix] === p[prefix]) prefix++

  let sim = jaroSimilarityRef(pattern, textValue)
  if (sim > 0.7) sim += prefix * prefixWeight * (1 - sim)

  return sim
}

/** Reference partial_ratio: try every alignment and take the best. */
function partialRatioShortNeedleImpl(s1: string, s2: string): number {
  const a = Array.from(s1)
  const b = Array.from(s2)

  if (a.length === 0 && b.length === 0) return 100
  if (a.length === 0 || b.length === 0) return 0
  if (a.length > b.length) return partialRatioShortNeedleImpl(s2, s1)

  let res = 0
  for (let i = -a.length; i < b.length; i++) {
    const part = b.slice(Math.max(0, i), Math.min(b.length, i + a.length)).join('')
    res = Math.max(res, ratio(s1, part))
  }

  return res
}

function partialRatioShortNeedle(s1: string, s2: string): number {
  if (Array.from(s1).length !== Array.from(s2).length) {
    return partialRatioShortNeedleImpl(s1, s2)
  }

  return Math.max(
    partialRatioShortNeedleImpl(s1, s2),
    partialRatioShortNeedleImpl(s2, s1),
  )
}

it('converts matching blocks consistently', () => {
  fc.assert(
    fc.property(text(), text(), (s1, s2) => {
      const ops = levenshteinEditops(s1, s2)
      // Records, so structural equality compares them directly.
      expect(ops.toMatchingBlocks()).toEqual(ops.toOpcodes().toMatchingBlocks())
    }),
    { numRuns: RUNS },
  )
})

describe('editops reproduce the destination when applied', () => {
  const CASES = [
    ['Levenshtein editops', levenshteinEditops, text, RUNS],
    ['Levenshtein editops (long)', levenshteinEditops, longText, BLOCK_RUNS],
    ['Indel editops', indelEditops, text, RUNS],
    ['Indel editops (long)', indelEditops, longText, BLOCK_RUNS],
    ['Levenshtein editops (astral)', levenshteinEditops, astralText, RUNS],
    ['Indel editops (astral)', indelEditops, astralText, RUNS],
  ] as const

  for (const [name, editops, arbitrary, runs] of CASES) {
    it(name, () => {
      fc.assert(
        fc.property(arbitrary(), arbitrary(), (s1, s2) => {
          expect(editops(s1, s2).apply(s1, s2)).toBe(s2)
        }),
        { numRuns: runs },
      )
    })
  }
})

describe('opcodes reproduce the destination when applied', () => {
  const CASES = [
    ['Levenshtein opcodes', levenshteinOpcodes, text, RUNS],
    ['Levenshtein opcodes (long)', levenshteinOpcodes, longText, BLOCK_RUNS],
    ['Indel opcodes', indelOpcodes, text, RUNS],
    ['Indel opcodes (long)', indelOpcodes, longText, BLOCK_RUNS],
    ['Levenshtein opcodes (astral)', levenshteinOpcodes, astralText, RUNS],
    ['Indel opcodes (astral)', indelOpcodes, astralText, RUNS],
  ] as const

  for (const [name, opcodes, arbitrary, runs] of CASES) {
    it(name, () => {
      fc.assert(
        fc.property(arbitrary(), arbitrary(), (s1, s2) => {
          expect(opcodes(s1, s2).apply(s1, s2)).toBe(s2)
        }),
        { numRuns: runs },
      )
    })
  }
})

it('matches the reference partial_ratio for short needles', () => {
  fc.assert(
    fc.property(text({ maxLength: 64 }), text(), (s1, s2) => {
      expect(isclose(partialRatio(s1, s2), partialRatioShortNeedle(s1, s2))).toBe(true)
    }),
    { numRuns: BLOCK_RUNS },
  )
})

it('token_ratio is the max of token_sort_ratio and token_set_ratio', () => {
  fc.assert(
    fc.property(text(), text(), (s1, s2) => {
      expect(tokenRatio(s1, s2)).toBe(
        Math.max(tokenSortRatio(s1, s2), tokenSetRatio(s1, s2)),
      )
    }),
    { numRuns: BLOCK_RUNS },
  )
})

it('partial_token_ratio is the max of its two components', () => {
  fc.assert(
    fc.property(text(), text(), (s1, s2) => {
      expect(partialTokenRatio(s1, s2)).toBe(
        Math.max(partialTokenSortRatio(s1, s2), partialTokenSetRatio(s1, s2)),
      )
    }),
    { numRuns: BLOCK_RUNS },
  )
})

describe('Levenshtein matches the reference implementation', () => {
  const CASES = [
    ['short', text({ maxLength: 64 })],
    ['long', longText()],
    ['random', text()],
  ] as const

  for (const [name, arbitrary] of CASES) {
    it(name, () => {
      fc.assert(
        fc.property(arbitrary, arbitrary, (s1, s2) => {
          const referenceDist = levenshtein(s1, s2)
          expect(levenshteinDistance(s1, s2)).toBe(referenceDist)
          expect(
            isclose(
              levenshteinNormalizedSimilarity(s1, s2),
              normalizeDistance(referenceDist, s1, s2),
            ),
          ).toBe(true)

          const indelRef = levenshtein(s1, s2, [1, 1, 2])
          expect(indelDistance(s1, s2)).toBe(indelRef)
          expect(
            isclose(
              indelNormalizedSimilarity(s1, s2),
              normalizeDistance(indelRef, s1, s2, [1, 1, 2]),
            ),
          ).toBe(true)
        }),
        { numRuns: BLOCK_RUNS },
      )
    })
  }
})

it('the default processor is idempotent', () => {
  fc.assert(
    fc.property(text(), (sentence) => {
      expect(defaultProcess(sentence)).toBe(defaultProcess(defaultProcess(sentence)))
    }),
    { numRuns: BLOCK_RUNS },
  )
})

describe('only identical strings reach a score of 100', () => {
  const SCORERS = { ratio, wRatio, qRatio }
  const PROCESSORS = {
    identity: (s: string | ArrayLike<unknown>) => s,
    defaultProcess,
  }

  for (const [scorerName, scorer] of Object.entries(SCORERS)) {
    for (const [processorName, processor] of Object.entries(PROCESSORS)) {
      it(`${scorerName} + ${processorName}`, () => {
        fc.assert(
          fc.property(
            fc.array(text(), { minLength: 1, maxLength: 8 }),
            fc.nat(),
            (choices, pick) => {
              const query = choices[pick % choices.length]
              // Upstream's `assume(processor(query))` — skip empty queries.
              fc.pre(String(processor(query)).length > 0)

              const matches = extract(query, choices, {
                scorer,
                processor,
                scoreCutoff: 100,
                limit: null,
              })

              expect(matches).not.toEqual([])
              for (const match of matches) {
                expect(processor(query)).toEqual(processor(String(match.choice)))
              }
            },
          ),
          { numRuns: BLOCK_RUNS },
        )
      })
    }
  }
})

it('cdist agrees with a naive double loop', () => {
  fc.assert(
    fc.property(
      fc.array(text(), { minLength: 1, maxLength: 6 }),
      fc.array(text(), { minLength: 1, maxLength: 6 }),
      (queries, choices) => {
        const reference = queries.map((query) =>
          choices.map((choice) => levenshteinDistance(query, choice)),
        )

        expect(matrixScores(queries, choices, { scorer: levenshteinDistance })).toEqual(
          reference,
        )
      },
    ),
    { numRuns: BLOCK_RUNS },
  )
})

describe('Jaro-Winkler matches the reference implementation', () => {
  const CASES = [
    ['short', text({ maxLength: 64 })],
    ['long', longText()],
    ['random', text()],
  ] as const

  for (const [name, arbitrary] of CASES) {
    it(name, () => {
      fc.assert(
        fc.property(arbitrary, arbitrary, (s1, s2) => {
          expect(
            isclose(jaroWinklerSimilarityRef(s1, s2), jaroWinklerSimilarity(s1, s2)),
          ).toBe(true)
        }),
        { numRuns: BLOCK_RUNS },
      )
    })
  }
})
