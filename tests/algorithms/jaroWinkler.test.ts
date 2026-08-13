// Ported from RapidFuzz tests/distance/test_JaroWinkler.py
import { expect, it } from 'vitest'

import { jaroWinklerSimilarity } from '../../src/algorithms/jaroWinkler/implementation.js'
import { similarity as jaroWinklerMetric } from '../../src/algorithms/jaroWinkler/index.js'
import { scoreMatrix } from '../../src/batch/scoreMatrix.js'
import { normalizeText as defaultProcess } from '../../src/core/normalize.js'
import { createScorer } from '../../src/core/scoring/scorer.js'
import { createMatcher } from '../../src/search/index.js'
import { prepareScorerOf } from '../support/preparation.js'
import { JaroWinkler } from '../support/scorers.js'

it('handles sequences of numbers', () => {
  expect(JaroWinkler.similarity([0, -1], [0, -2])).toBeCloseTo(0.666666, 5)
})

it('clamps to 1.0 with a large prefix weight', () => {
  expect(
    JaroWinkler.similarity('milyarder', 'milyarderlik', { prefixWeight: 0.5 }),
  ).toBeCloseTo(1, 6)
  expect(
    JaroWinkler.similarity('milyarder', 'milyarderlik', { prefixWeight: 1 }),
  ).toBeCloseTo(1, 6)
})

it('reads thresholds on its natural normalized scale', () => {
  expect(JaroWinkler.similarity('abcd', 'abce')).toBeCloseTo(0.883333, 5)
  expect(JaroWinkler.similarity('abcd', 'abce', { threshold: 0.95 })).toBeUndefined()
})

it('rejects an out-of-range prefix weight', () => {
  expect(() =>
    JaroWinkler.similarity('milyarder', 'milyarderlik', { prefixWeight: -0.1 }),
  ).toThrow('prefix_weight has to be in the range 0.0 - 1.0')

  expect(() =>
    JaroWinkler.similarity('milyarder', 'milyarderlik', { prefixWeight: 1.1 }),
  ).toThrow('prefix_weight has to be in the range 0.0 - 1.0')
})

it('handles the edge case lengths found by fuzzing', () => {
  expect(JaroWinkler.similarity('', '')).toBeCloseTo(1, 6)
  expect(JaroWinkler.similarity('0', '0')).toBeCloseTo(1, 6)
  expect(JaroWinkler.similarity('00', '00')).toBeCloseTo(1, 6)
  expect(JaroWinkler.similarity('0', '00')).toBeCloseTo(0.85, 6)

  expect(JaroWinkler.similarity('0'.repeat(65), '0'.repeat(65))).toBeCloseTo(1, 6)
  expect(JaroWinkler.similarity('0'.repeat(64), '0'.repeat(65))).toBeCloseTo(0.996923, 5)
  expect(JaroWinkler.similarity('0'.repeat(63), '0'.repeat(65))).toBeCloseTo(0.993846, 5)

  expect(JaroWinkler.similarity('000000001', '0000010')).toBeCloseTo(0.926984, 5)

  expect(
    JaroWinkler.similarity(
      '10000000000000000000000000000000000000000000000000000000000000020',
      '00000000000000000000000000000000000000000000000000000000000000000',
    ),
  ).toBeCloseTo(0.979487, 5)

  expect(
    JaroWinkler.similarity(
      '0000000000000000000000000000000000000000000000000000000000000000000000000000001',
      '00000000000000100000000000000000000000010000000000000000000000000',
    ),
  ).toBeCloseTo(0.95334, 5)

  expect(
    JaroWinkler.similarity(
      '010000000000000000000000000000000000000000000000000000000000000000' +
        '00000000000000000000000000000000000000000000000000000000000000',
      '00000000000000000000000000000000000000000000000000000000000000000',
    ),
  ).toBeCloseTo(0.852344, 6)
})

it('compares normalized text case-insensitively', () => {
  expect(
    JaroWinkler.similarity(
      defaultProcess('new york mets'),
      defaultProcess('new YORK mets'),
    ),
  ).toBeCloseTo(1, 6)
})

// Not ported — upstream's `prefix_weight` is a keyword argument checked once
// per call, where here it is also parsed when a query is prepared, and the
// cutoff it derives has a case upstream's tests never reach.
it('refuses a prefix weight outside 0 to 1', () => {
  expect(() => jaroWinklerSimilarity('abcd', 'abce', { prefixWeight: 1.5 })).toThrow(
    RangeError,
  )
  expect(() => jaroWinklerSimilarity('abcd', 'abce', { prefixWeight: -0.1 })).toThrow(
    RangeError,
  )
})

it('refuses a prefix weight the prepared path cannot use either', () => {
  const prepare = prepareScorerOf(jaroWinklerSimilarity)

  expect(() => prepare('abcd', { prefixWeight: 1.5 })).toThrow(RangeError)
  expect(() => prepare('abcd', { prefixWeight: 'a lot' })).toThrow(TypeError)
  expect(prepare('abcd', { prefixWeight: 0.2 })('abce', null)).toBeCloseTo(
    jaroWinklerSimilarity('abcd', 'abce', { prefixWeight: 0.2 }),
    12,
  )
})

// Four matching characters at a weight of 0.25 is a full point of bonus, so
// the Jaro cutoff the score has to clear collapses to the 0.7 floor rather
// than being solved for.
it('handles a prefix bonus that covers the whole score', () => {
  const options = { prefixWeight: 0.25, threshold: 0.9 }
  expect(jaroWinklerSimilarity('abcdx', 'abcdy', options)).toBeCloseTo(1, 12)
  expect(
    scoreMatrix(['abcdx'], ['abcdy'], {
      scorer: createScorer(jaroWinklerMetric, { prefixWeight: 0.25 }),
    }).at(0, 0),
  ).toBeCloseTo(1, 12)
  expect(
    createMatcher(['abcdy'], {
      scorer: createScorer(jaroWinklerMetric, { prefixWeight: 0.25 }),
    }).best('abcdx', { threshold: 0.9 })?.score,
  ).toBeCloseTo(1, 12)
})
