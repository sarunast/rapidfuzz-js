import { describe, expect, it } from 'vitest'

import {
  exhaustive,
  exhaustiveScan,
  indexOf,
  LIMITS,
  METRICS,
  pairs,
  THRESHOLDS,
} from '../../../../../testing/invertedIndex.js'

describe('choices and queries with no grams', () => {
  it('scores an equal gramless pair 1 and everything else 0', () => {
    const choices = ['ab', '', 'ab', 'zz', '']
    for (const metric of METRICS) {
      const index = indexOf(metric, 3, choices)
      for (const threshold of THRESHOLDS) {
        for (const limit of LIMITS) {
          expect(pairs(index.select('ab', threshold, limit))).toEqual(
            exhaustive(metric, 3, choices, 'ab', threshold, limit),
          )
        }
        expect(pairs(index.scan('ab', threshold))).toEqual(
          exhaustiveScan(metric, 3, choices, 'ab', threshold),
        )
      }
    }
  })

  it('answers a gramless query against a corpus that has grams', () => {
    const choices = ['abcd', 'abce']
    for (const metric of METRICS) {
      const index = indexOf(metric, 3, choices)
      expect(pairs(index.select('x', null, null))).toEqual(
        exhaustive(metric, 3, choices, 'x', null, null),
      )
      expect(pairs(index.select('x', 0.5, null))).toEqual(
        exhaustive(metric, 3, choices, 'x', 0.5, null),
      )
    }
  })

  it('scores a gramless choice 0 rather than dividing by nothing', () => {
    // `''` has no grams and no norm; a Cosine score of `0/0` clamped to 1 was a
    // real bug, and only a dense list reaches such a choice at all.
    const choices = ['😀c', '😀c', '']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      expect(pairs(index.select('😀c', null, null))).toEqual(
        exhaustive(metric, 2, choices, '😀c', null, null),
      )
    }
  })
})
