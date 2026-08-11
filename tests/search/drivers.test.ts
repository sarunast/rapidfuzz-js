import { describe, expect, it } from 'vitest'

import { bestDistance } from '../../src/search/internal/bestDistance.js'
import { bestSimilarity } from '../../src/search/internal/bestSimilarity.js'
import { topDistance } from '../../src/search/internal/topDistance.js'
import { topSimilarity } from '../../src/search/internal/topSimilarity.js'

function numeric(value: unknown): number {
  if (typeof value !== 'number') throw new TypeError('expected numeric prepared data')
  return value
}

const items = [
  { item: 'first', key: 0, sequence: 'first', prepared: 2 },
  { item: 'second', key: 1, sequence: 'second', prepared: 5 },
  { item: 'tie', key: 2, sequence: 'tie', prepared: 5 },
  { item: 'last', key: 3, sequence: 'last', prepared: 1 },
  { item: 'last tie', key: 4, sequence: 'last tie', prepared: 1 },
]

describe('specialized search drivers', () => {
  it('selects similarities with and without thresholds and trusted optima', () => {
    expect(bestSimilarity([], numeric, null, 5)).toBeUndefined()
    expect(bestSimilarity(items, numeric, 3, null)).toEqual({
      item: 'second',
      key: 1,
      score: 5,
    })
    expect(bestSimilarity(items, numeric, null, 5)).toEqual({
      item: 'second',
      key: 1,
      score: 5,
    })
    expect(bestSimilarity(items, numeric, 9, null)).toBeUndefined()
  })

  it('selects distances with and without thresholds and trusted optima', () => {
    expect(bestDistance([], numeric, null, 0)).toBeUndefined()
    expect(bestDistance(items, numeric, 3, null)).toEqual({
      item: 'last',
      key: 3,
      score: 1,
    })
    expect(bestDistance(items, numeric, null, 2)).toEqual({
      item: 'first',
      key: 0,
      score: 2,
    })
    expect(bestDistance(items, numeric, 0, null)).toBeUndefined()
  })

  it('ranks stable top results in both directions', () => {
    expect(topSimilarity(items, numeric, null, null).map((entry) => entry.key)).toEqual([
      1, 2, 0, 3, 4,
    ])
    expect(topSimilarity(items, numeric, 3, 1)).toEqual([
      { item: 'second', key: 1, score: 5 },
    ])
    expect(topDistance(items, numeric, null, null).map((entry) => entry.key)).toEqual([
      3, 4, 0, 1, 2,
    ])
    expect(topDistance(items, numeric, 2, 1)).toEqual([
      { item: 'last', key: 3, score: 1 },
    ])
  })
})
