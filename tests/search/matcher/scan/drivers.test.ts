import { describe, expect, it } from 'vitest'

import { bestDistance } from '../../../../src/search/matcher/scan/bestDistance.js'
import { bestSimilarity } from '../../../../src/search/matcher/scan/bestSimilarity.js'
import { topDistance } from '../../../../src/search/matcher/scan/topDistance.js'
import { topSimilarity } from '../../../../src/search/matcher/scan/topSimilarity.js'

function numeric(value: unknown): number {
  if (typeof value !== 'number') throw new TypeError('expected numeric prepared data')
  return value
}

// Ids are positions in this array: 0 scores 2, 1 and 2 tie at 5, 3 and 4 tie at 1.
const prepared = [2, 5, 5, 1, 1]

describe('specialized search drivers', () => {
  it('selects similarities with and without thresholds and trusted optima', () => {
    expect(bestSimilarity([], numeric, null, 5)).toBeUndefined()
    expect(bestSimilarity(prepared, numeric, 3, null)).toEqual({ id: 1, score: 5 })
    expect(bestSimilarity(prepared, numeric, null, 5)).toEqual({ id: 1, score: 5 })
    expect(bestSimilarity(prepared, numeric, 9, null)).toBeUndefined()
  })

  it('selects distances with and without thresholds and trusted optima', () => {
    expect(bestDistance([], numeric, null, 0)).toBeUndefined()
    expect(bestDistance(prepared, numeric, 3, null)).toEqual({ id: 3, score: 1 })
    expect(bestDistance(prepared, numeric, null, 2)).toEqual({ id: 0, score: 2 })
    expect(bestDistance(prepared, numeric, 0, null)).toBeUndefined()
  })

  it('ranks stable top results in both directions', () => {
    expect(topSimilarity(prepared, numeric, null, null, null).map((e) => e.id)).toEqual([
      1, 2, 0, 3, 4,
    ])
    expect(topSimilarity(prepared, numeric, 3, 1, null)).toEqual([{ id: 1, score: 5 }])
    expect(topDistance(prepared, numeric, null, null, null).map((e) => e.id)).toEqual([
      3, 4, 0, 1, 2,
    ])
    expect(topDistance(prepared, numeric, 2, 1, null)).toEqual([{ id: 3, score: 1 }])
    expect(topSimilarity(prepared, numeric, 9, 1, null)).toEqual([])
    expect(topDistance(prepared, numeric, 0, 1, null)).toEqual([])
    expect(topSimilarity(prepared, numeric, 3, null, null).map((e) => e.id)).toEqual([
      1, 2,
    ])
    expect(topDistance(prepared, numeric, 2, null, null).map((e) => e.id)).toEqual([
      3, 4, 0,
    ])
    expect(topSimilarity(prepared, numeric, null, 0, null)).toEqual([])
    expect(topDistance(prepared, numeric, null, 0, null)).toEqual([])
    expect(topSimilarity(prepared, numeric, 9, 2, null)).toEqual([])
    expect(topDistance(prepared, numeric, 0, 2, null)).toEqual([])
  })

  it('keeps a bounded heap and tightens the active scorer cutoff', () => {
    const similarityCutoffs: Array<number | null> = []
    const similarity = (value: unknown, cutoff: number | null): number => {
      similarityCutoffs.push(cutoff)
      return numeric(value)
    }
    expect(topSimilarity(prepared, similarity, null, 2, null).map((e) => e.id)).toEqual([
      1, 2,
    ])
    expect(similarityCutoffs).toEqual([null, null, 2, 5, 5])

    const distanceCutoffs: Array<number | null> = []
    const distance = (value: unknown, cutoff: number | null): number => {
      distanceCutoffs.push(cutoff)
      return numeric(value)
    }
    expect(topDistance(prepared, distance, null, 2, null).map((e) => e.id)).toEqual([
      3, 4,
    ])
    expect(distanceCutoffs).toEqual([null, null, 5, 5, 2])
  })

  it('stops once every retained finite result is optimal', () => {
    let similarityCalls = 0
    const similarity = (value: unknown): number => {
      similarityCalls++
      return numeric(value)
    }
    expect(topSimilarity([5, 5, 4], similarity, null, 2, 5)).toHaveLength(2)
    expect(similarityCalls).toBe(2)

    let distanceCalls = 0
    const distance = (value: unknown): number => {
      distanceCalls++
      return numeric(value)
    }
    expect(topDistance([0, 0, 1], distance, null, 2, 0)).toHaveLength(2)
    expect(distanceCalls).toBe(2)

    expect(topSimilarity([4, 5, 5], numeric, null, 2, 5)).toHaveLength(2)
    expect(topDistance([1, 0, 0], numeric, null, 2, 0)).toHaveLength(2)
  })
})
