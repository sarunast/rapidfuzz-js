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
import { scoreMatrix, scorePairs } from '../../src/batch/index.js'
import type { Metric } from '../../src/core/metric.js'
import { normalizeText } from '../../src/core/normalize.js'
import { createScorer } from '../../src/core/scorer.js'
import type { Sequence } from '../../src/core/types.js'
import * as fuzz from '../../src/fuzz/index.js'
import { bestMatch, createMatcher, search, searchIter } from '../../src/search/index.js'
import fixture from '../fixtures/rapidfuzz-3.14.5.json' with { type: 'json' }

interface MetricSuite {
  readonly distance: Metric<'distance', object>
  readonly similarity: Metric<'similarity', object>
  readonly normalizedDistance: Metric<'distance', object>
  readonly normalizedSimilarity: Metric<'similarity', object>
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
    case 'fuzzySimilarity':
      return fuzz.fuzzySimilarity(left, right)
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

  test('matches edit operations', () => {
    for (const entry of fixture.editopsCases) {
      const suite = metricSuite(entry.family)
      if (!('editops' in suite) || !('opcodes' in suite)) {
        throw new TypeError(`missing edit operations for ${entry.family}`)
      }
      const editops = Reflect.get(suite, 'editops')
      const opcodes = Reflect.get(suite, 'opcodes')
      if (typeof editops !== 'function' || typeof opcodes !== 'function') {
        throw new TypeError(`invalid edit operations for ${entry.family}`)
      }
      const edits = Reflect.apply(editops, undefined, [entry.left, entry.right])
      const blocks = Reflect.apply(opcodes, undefined, [entry.left, entry.right])
      if (
        typeof edits !== 'object' ||
        edits === null ||
        typeof blocks !== 'object' ||
        blocks === null
      ) {
        throw new TypeError('edit operation oracle returned an invalid value')
      }
      expect(Reflect.get(edits, 'operations')).toEqual(entry.editops)
      expect(Reflect.get(blocks, 'operations')).toEqual(entry.opcodes)
    }
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
    expect(Array.from(matcher.searchIter(fixture.search.query, options))).toEqual(
      fixture.search.iter,
    )
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
