import { describe, expect, test } from 'vitest'

import * as damerauLevenshtein from '../../src/algorithms/damerauLevenshtein/index.js'
import * as hamming from '../../src/algorithms/hamming/index.js'
import * as indel from '../../src/algorithms/indel/index.js'
import * as jaro from '../../src/algorithms/jaro/index.js'
import * as jaroWinkler from '../../src/algorithms/jaroWinkler/index.js'
import * as lcs from '../../src/algorithms/lcs/index.js'
import * as levenshtein from '../../src/algorithms/levenshtein/index.js'
import * as osa from '../../src/algorithms/osa/index.js'
import * as postfix from '../../src/algorithms/postfix/index.js'
import * as prefix from '../../src/algorithms/prefix/index.js'
import type { Editops, Opcodes } from '../../src/algorithms/shared/editops/index.js'
import { scoreMatrix, scorePairs } from '../../src/batch/index.js'
import type { ScoreArrayKind } from '../../src/batch/storage.js'
import { normalizeText } from '../../src/core/normalize.js'
import type { Metric } from '../../src/core/scoring/metric.js'
import { createScorer, type Scorer } from '../../src/core/scoring/scorer.js'
import type { Direction, Sequence } from '../../src/core/types.js'
import * as fuzz from '../../src/fuzz/index.js'
import { bestMatch, createMatcher, search, searchIter } from '../../src/search/index.js'
import fixture from '../fixtures/rapidfuzz-3.14.5.json' with { type: 'json' }

interface MetricSuite {
  readonly distance: Metric<'distance', never>
  readonly similarity: Metric<'similarity', never>
  readonly normalizedDistance: Metric<'distance', never>
  readonly normalizedSimilarity: Metric<'similarity', never>
}

interface Scores {
  readonly distance: number
  readonly similarity: number
  readonly normalizedDistance: number
  readonly normalizedSimilarity: number
}

function metricSuite(family: string): MetricSuite {
  switch (family) {
    case 'damerauLevenshtein':
      return damerauLevenshtein
    case 'hamming':
      return hamming
    case 'indel':
      return indel
    case 'jaro':
      return jaro
    case 'jaroWinkler':
      return jaroWinkler
    case 'lcs':
      return lcs
    case 'levenshtein':
      return levenshtein
    case 'osa':
      return osa
    case 'postfix':
      return postfix
    case 'prefix':
      return prefix
    default:
      throw new TypeError(`unknown parity family ${family}`)
  }
}

function directScores(suite: MetricSuite, left: Sequence, right: Sequence): Scores {
  return {
    distance: suite.distance(left, right),
    similarity: suite.similarity(left, right),
    normalizedDistance: suite.normalizedDistance(left, right),
    normalizedSimilarity: suite.normalizedSimilarity(left, right),
  }
}

function configuredScores(
  family: string,
  configuration: object,
  left: Sequence,
  right: Sequence,
): Scores {
  if (family === 'levenshtein') {
    const weights = Reflect.get(configuration, 'weights')
    if (
      !Array.isArray(weights) ||
      weights.length !== 3 ||
      typeof weights[0] !== 'number' ||
      typeof weights[1] !== 'number' ||
      typeof weights[2] !== 'number'
    ) {
      throw new TypeError('invalid oracle weights')
    }
    const options = {
      weights: {
        insertion: weights[0],
        deletion: weights[1],
        substitution: weights[2],
      },
    }
    return {
      distance: createScorer(levenshtein.distance, options).score(left, right),
      similarity: createScorer(levenshtein.similarity, options).score(left, right),
      normalizedDistance: createScorer(levenshtein.normalizedDistance, options).score(
        left,
        right,
      ),
      normalizedSimilarity: createScorer(levenshtein.normalizedSimilarity, options).score(
        left,
        right,
      ),
    }
  }
  if (family === 'hamming') {
    const pad = Reflect.get(configuration, 'pad')
    if (typeof pad !== 'boolean') throw new TypeError('invalid oracle pad')
    const options = { pad }
    return {
      distance: createScorer(hamming.distance, options).score(left, right),
      similarity: createScorer(hamming.similarity, options).score(left, right),
      normalizedDistance: createScorer(hamming.normalizedDistance, options).score(
        left,
        right,
      ),
      normalizedSimilarity: createScorer(hamming.normalizedSimilarity, options).score(
        left,
        right,
      ),
    }
  }
  if (family === 'jaroWinkler') {
    const prefixWeight = Reflect.get(configuration, 'prefixWeight')
    if (typeof prefixWeight !== 'number') {
      throw new TypeError('invalid oracle prefixWeight')
    }
    const options = { prefixWeight }
    return {
      distance: createScorer(jaroWinkler.distance, options).score(left, right),
      similarity: createScorer(jaroWinkler.similarity, options).score(left, right),
      normalizedDistance: createScorer(jaroWinkler.normalizedDistance, options).score(
        left,
        right,
      ),
      normalizedSimilarity: createScorer(jaroWinkler.normalizedSimilarity, options).score(
        left,
        right,
      ),
    }
  }
  throw new TypeError(`unexpected configured parity family ${family}`)
}

interface EditopsSuite {
  readonly editops: (a: Sequence, b: Sequence) => Editops
  readonly opcodes: (a: Sequence, b: Sequence) => Opcodes
}

/** The four families whose alignments upstream exposes, and ours with them. */
function editopsSuite(family: string): EditopsSuite {
  switch (family) {
    case 'hamming':
      return hamming
    case 'indel':
      return indel
    case 'lcs':
      return lcs
    case 'levenshtein':
      return levenshtein
    default:
      throw new TypeError(`unknown edit operation family ${family}`)
  }
}

const SCORE_ARRAY_KINDS = [
  'f64',
  'f32',
  'i32',
  'i16',
  'i8',
  'u32',
  'u16',
  'u8',
  'u8c',
] as const

function isScoreArrayKind(value: string): value is ScoreArrayKind {
  return SCORE_ARRAY_KINDS.some((kind) => kind === value)
}

function batchScorer(metric: string): Scorer<Direction> {
  switch (metric) {
    case 'fuzz.similarity':
      return createScorer(fuzz.similarity)
    case 'levenshtein.distance':
      return createScorer(levenshtein.distance)
    case 'levenshtein.normalizedSimilarity':
      return createScorer(levenshtein.normalizedSimilarity)
    default:
      throw new TypeError(`unknown batch oracle metric ${metric}`)
  }
}

function fuzzScore(name: string, left: string, right: string): number {
  switch (name) {
    case 'similarity':
      return fuzz.similarity(left, right)
    case 'partialSimilarity':
      return fuzz.partialSimilarity(left, right)
    case 'tokenSortSimilarity':
      return fuzz.tokenSortSimilarity(left, right)
    case 'tokenSetSimilarity':
      return fuzz.tokenSetSimilarity(left, right)
    case 'tokenSimilarity':
      return fuzz.tokenSimilarity(left, right)
    case 'partialTokenSortSimilarity':
      return fuzz.partialTokenSortSimilarity(left, right)
    case 'partialTokenSetSimilarity':
      return fuzz.partialTokenSetSimilarity(left, right)
    case 'partialTokenSimilarity':
      return fuzz.partialTokenSimilarity(left, right)
    case 'weightedSimilarity':
      return fuzz.weightedSimilarity(left, right)
    default:
      throw new TypeError(`unknown fuzz oracle ${name}`)
  }
}

describe(`RapidFuzz ${fixture.rapidfuzzVersion} parity`, () => {
  test('matches all four public metric operations', () => {
    for (const entry of fixture.metricCases) {
      const actual =
        Object.keys(entry.configuration).length === 0
          ? directScores(metricSuite(entry.family), entry.left, entry.right)
          : configuredScores(entry.family, entry.configuration, entry.left, entry.right)
      for (const operation of [
        'distance',
        'similarity',
        'normalizedDistance',
        'normalizedSimilarity',
      ] as const) {
        expect(actual[operation], `${entry.family}.${operation}`).toBeCloseTo(
          entry.scores[operation],
          12,
        )
      }
    }
  })

  test('matches the retained fuzz scorers', () => {
    for (const entry of fixture.fuzzCases) {
      for (const [name, expected] of Object.entries(entry.scores)) {
        expect(fuzzScore(name, entry.left, entry.right), name).toBeCloseTo(expected, 12)
      }
    }
  })

  test('matches default text normalization', () => {
    for (const entry of fixture.normalization) {
      expect(normalizeText(entry.input)).toBe(entry.output)
    }
  })

  test('matches edit operations and everything derived from them', () => {
    for (const entry of fixture.editopsCases) {
      const suite = editopsSuite(entry.family)
      const label = `${entry.family} ${JSON.stringify([entry.left, entry.right])}`
      const edits = suite.editops(entry.left, entry.right)
      const blocks = suite.opcodes(entry.left, entry.right)

      expect(edits.operations, `${label} editops`).toEqual(entry.editops)
      expect(blocks.operations, `${label} opcodes`).toEqual(entry.opcodes)
      expect([edits.srcLen, edits.destLen], `${label} lengths`).toEqual([
        entry.srcLen,
        entry.destLen,
      ])
      expect(edits.toMatchingBlocks(), `${label} matching blocks`).toEqual(
        entry.matchingBlocks,
      )
      expect(blocks.toMatchingBlocks(), `${label} opcode matching blocks`).toEqual(
        entry.opcodeMatchingBlocks,
      )
      expect(edits.inverse().operations, `${label} inverse`).toEqual(entry.inverse)
      expect(blocks.inverse().operations, `${label} opcode inverse`).toEqual(
        entry.opcodeInverse,
      )
      expect(edits.apply(entry.left, entry.right), `${label} apply`).toBe(entry.applied)
      expect(blocks.apply(entry.left, entry.right), `${label} opcode apply`).toBe(
        entry.opcodeApplied,
      )
      expect(edits.toOpcodes().operations, `${label} editops to opcodes`).toEqual(
        entry.opcodes,
      )
      expect(blocks.toEditops().operations, `${label} opcodes to editops`).toEqual(
        entry.editops,
      )
    }
  })

  test('matches hamming edit operations with padding disabled', () => {
    for (const entry of fixture.hammingEditopsCases) {
      const options = { pad: entry.configuration.pad }
      const label = `hamming pad:${options.pad} ${JSON.stringify([entry.left, entry.right])}`
      const edits = hamming.editops(entry.left, entry.right, options)
      const blocks = hamming.opcodes(entry.left, entry.right, options)

      expect(edits.operations, `${label} editops`).toEqual(entry.editops)
      expect(blocks.operations, `${label} opcodes`).toEqual(entry.opcodes)
      expect([edits.srcLen, edits.destLen], `${label} lengths`).toEqual([
        entry.srcLen,
        entry.destLen,
      ])
      expect(edits.toMatchingBlocks(), `${label} matching blocks`).toEqual(
        entry.matchingBlocks,
      )
      expect(edits.apply(entry.left, entry.right), `${label} apply`).toBe(entry.applied)
    }

    for (const entry of fixture.hammingEditopsErrors) {
      expect(() =>
        hamming.editops(entry.left, entry.right, { pad: entry.configuration.pad }),
      ).toThrow(entry.error)
    }
  })

  test('matches removing one edit script from another', () => {
    for (const entry of fixture.removeSubsequenceCases) {
      const suite = editopsSuite(entry.family)
      const label = `${entry.source} -> ${entry.target} less ${entry.subset}`
      const full = suite.editops(entry.source, entry.target)
      const part = suite.editops(entry.source, entry.subset)

      expect(full.operations, `${label} full script`).toEqual(entry.full)
      expect(part.operations, `${label} subsequence`).toEqual(entry.subsequence)

      const remainder = full.removeSubsequence(part)
      expect(remainder.operations, `${label} remainder`).toEqual(entry.operations)
      expect([remainder.srcLen, remainder.destLen], `${label} lengths`).toEqual([
        entry.srcLen,
        entry.destLen,
      ])
    }
  })

  test('matches partial similarity alignment', () => {
    for (const entry of fixture.alignmentCases) {
      const label = `alignment ${JSON.stringify([entry.left, entry.right])}`
      const alignment = fuzz.partialSimilarityAlignment(entry.left, entry.right)
      if (alignment === null) throw new TypeError(`${label} produced no alignment`)

      expect(alignment.score, `${label} score`).toBeCloseTo(entry.alignment.score, 12)
      expect(
        [alignment.srcStart, alignment.srcEnd, alignment.destStart, alignment.destEnd],
        `${label} bounds`,
      ).toEqual([
        entry.alignment.srcStart,
        entry.alignment.srcEnd,
        entry.alignment.destStart,
        entry.alignment.destEnd,
      ])
    }
  })

  test('matches batch output for every score array kind', () => {
    const { queries, choices, kinds } = fixture.batchDtypes
    for (const entry of kinds) {
      if (!isScoreArrayKind(entry.into)) {
        throw new TypeError(`unknown score array kind ${entry.into}`)
      }
      const scores = scorePairs(queries, choices, {
        scorer: createScorer(levenshtein.normalizedSimilarity),
        into: entry.into,
        scoreMultiplier: entry.scoreMultiplier,
      })
      expect(Array.from(scores), `into: ${entry.into}`).toEqual(entry.pairs)
    }
  })

  test('matches the score a rejected pair is stored as', () => {
    const { queries, choices, cases } = fixture.batchRejections
    for (const entry of cases) {
      const scores = scorePairs(queries, choices, {
        scorer: batchScorer(entry.metric),
        threshold: entry.threshold,
      })
      expect(Array.from(scores), `${entry.metric} at ${entry.threshold}`).toEqual(
        entry.pairs,
      )
    }
  })

  test('matches ranked results over a record of choices', () => {
    const { choices, top } = fixture.search.record
    // `limit` belongs to `search` alone: each entry point takes the keys it
    // defines, and one carrying a key another ignores is refused.
    const options = {
      scorer: createScorer(fuzz.similarity),
      threshold: fixture.search.threshold,
    }
    expect(search(fixture.search.query, choices, { ...options, limit: null })).toEqual(
      top,
    )
    expect(bestMatch(fixture.search.query, choices, options)).toEqual(top[0])

    // A Map keys by its own keys, so the same pairs give the same answer.
    const mapped = new Map(Object.entries(choices))
    expect(search(fixture.search.query, mapped, { ...options, limit: null })).toEqual(top)
  })

  test('matches a search that normalizes text before scoring', () => {
    const entry = fixture.search.normalized
    const options = {
      scorer: createScorer(fuzz.similarity),
      normalize: normalizeText,
      threshold: entry.threshold,
    }
    expect(search(entry.query, entry.choices, { ...options, limit: null })).toEqual(
      entry.top,
    )
    expect(bestMatch(entry.query, entry.choices, options)).toEqual(entry.best)
  })

  test('matches ranked and streaming process results', () => {
    const scorer = createScorer(fuzz.similarity)
    const options = { scorer, threshold: fixture.search.threshold }
    expect(bestMatch(fixture.search.query, fixture.search.choices, options)).toEqual(
      fixture.search.best,
    )
    expect(
      search(fixture.search.query, fixture.search.choices, { ...options, limit: 4 }),
    ).toEqual(fixture.search.top)
    expect(
      Array.from(searchIter(fixture.search.query, fixture.search.choices, options)),
    ).toEqual(fixture.search.iter)
    const matcher = createMatcher(fixture.search.choices, { scorer })
    // A Matcher method takes the threshold alone: the scorer is the one the
    // Matcher was built with, and naming another here would change nothing.
    expect(
      Array.from(
        matcher.searchIter(fixture.search.query, { threshold: fixture.search.threshold }),
      ),
    ).toEqual(fixture.search.iter)
  })

  test('matches cutoff, scaling, and integral batch output', () => {
    const scorer = createScorer(levenshtein.normalizedSimilarity)
    const options = {
      scorer,
      threshold: fixture.batch.threshold,
      scoreMultiplier: fixture.batch.scoreMultiplier,
      into: 'u8' as const,
    }
    expect(
      Array.from(scorePairs(fixture.batch.queries, fixture.batch.choices, options)),
    ).toEqual(fixture.batch.pairs)
    expect(
      scoreMatrix(fixture.batch.queries, fixture.batch.matrixChoices, options).toArray(),
    ).toEqual(fixture.batch.matrix)
  })
})
