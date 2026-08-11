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
  { item: 'first', key: 0, prepared: 2 },
  { item: 'second', key: 1, prepared: 5 },
  { item: 'tie', key: 2, prepared: 5 },
  { item: 'last', key: 3, prepared: 1 },
  { item: 'last tie', key: 4, prepared: 1 },
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
    expect(
      topSimilarity(items, numeric, null, null, null).map((entry) => entry.key),
    ).toEqual([1, 2, 0, 3, 4])
    expect(topSimilarity(items, numeric, 3, 1, null)).toEqual([
      { item: 'second', key: 1, score: 5 },
    ])
    expect(
      topDistance(items, numeric, null, null, null).map((entry) => entry.key),
    ).toEqual([3, 4, 0, 1, 2])
    expect(topDistance(items, numeric, 2, 1, null)).toEqual([
      { item: 'last', key: 3, score: 1 },
    ])
    expect(topSimilarity(items, numeric, 9, 1, null)).toEqual([])
    expect(topDistance(items, numeric, 0, 1, null)).toEqual([])
    expect(
      topSimilarity(items, numeric, 3, null, null).map((entry) => entry.key),
    ).toEqual([1, 2])
    expect(topDistance(items, numeric, 2, null, null).map((entry) => entry.key)).toEqual([
      3, 4, 0,
    ])
    expect(topSimilarity(items, numeric, null, 0, null)).toEqual([])
    expect(topDistance(items, numeric, null, 0, null)).toEqual([])
    expect(topSimilarity(items, numeric, 9, 2, null)).toEqual([])
    expect(topDistance(items, numeric, 0, 2, null)).toEqual([])
  })

  it('keeps a bounded heap and tightens the active scorer cutoff', () => {
    const similarityCutoffs: Array<number | null> = []
    const similarity = (value: unknown, cutoff: number | null): number => {
      similarityCutoffs.push(cutoff)
      return numeric(value)
    }
    expect(
      topSimilarity(items, similarity, null, 2, null).map((entry) => entry.key),
    ).toEqual([1, 2])
    expect(similarityCutoffs).toEqual([null, null, 2, 5, 5])

    const distanceCutoffs: Array<number | null> = []
    const distance = (value: unknown, cutoff: number | null): number => {
      distanceCutoffs.push(cutoff)
      return numeric(value)
    }
    expect(topDistance(items, distance, null, 2, null).map((entry) => entry.key)).toEqual(
      [3, 4],
    )
    expect(distanceCutoffs).toEqual([null, null, 5, 5, 2])
  })

  it('stops once every retained finite result is optimal', () => {
    const perfectSimilarities = [
      { item: 'first', key: 0, prepared: 5 },
      { item: 'second', key: 1, prepared: 5 },
      { item: 'unread', key: 2, prepared: 4 },
    ]
    let similarityCalls = 0
    const similarity = (value: unknown): number => {
      similarityCalls++
      return numeric(value)
    }
    expect(topSimilarity(perfectSimilarities, similarity, null, 2, 5)).toHaveLength(2)
    expect(similarityCalls).toBe(2)

    const perfectDistances = [
      { item: 'first', key: 0, prepared: 0 },
      { item: 'second', key: 1, prepared: 0 },
      { item: 'unread', key: 2, prepared: 1 },
    ]
    let distanceCalls = 0
    const distance = (value: unknown): number => {
      distanceCalls++
      return numeric(value)
    }
    expect(topDistance(perfectDistances, distance, null, 2, 0)).toHaveLength(2)
    expect(distanceCalls).toBe(2)

    expect(
      topSimilarity(
        [
          { item: 'low', key: 0, prepared: 4 },
          { item: 'high', key: 1, prepared: 5 },
          { item: 'replacement', key: 2, prepared: 5 },
        ],
        numeric,
        null,
        2,
        5,
      ),
    ).toHaveLength(2)
    expect(
      topDistance(
        [
          { item: 'high', key: 0, prepared: 1 },
          { item: 'low', key: 1, prepared: 0 },
          { item: 'replacement', key: 2, prepared: 0 },
        ],
        numeric,
        null,
        2,
        0,
      ),
    ).toHaveLength(2)
  })
})
