// The q-gram shortlist's own proof: the (2q - 1)K necessary condition it
// prunes with, tested against the real Indel scorer rather than against the
// index, and the exact candidate output that final validation makes possible.
import { describe, expect, it } from 'vitest'

import { createScorer, scorerCompilation } from '#core/scoring/scorer.js'

import { maximumQualifyingDistance } from './candidateIndex.js'
import {
  distance as indelDistance,
  normalizedSimilarity as indelSimilarity,
} from './index.js'

function stringsThrough(maximum: number): string[] {
  const values = ['']
  for (let length = 1; length <= maximum; length++) {
    const previous = values.filter((value) => value.length === length - 1)
    for (const prefix of previous)
      for (const suffix of ['a', 'b', 'c']) values.push(prefix + suffix)
  }
  return values
}

function adjacent(value: number, direction: -1 | 1): number {
  if (!Number.isFinite(value)) return value
  if (value === 0) return direction < 0 ? -Number.MIN_VALUE : Number.MIN_VALUE
  const buffer = new ArrayBuffer(8)
  const view = new DataView(buffer)
  view.setFloat64(0, value)
  let bits = view.getBigUint64(0)
  bits += value > 0 === direction > 0 ? 1n : -1n
  view.setBigUint64(0, bits)
  return view.getFloat64(0)
}

function sharedGrams(left: string, right: string, size: number): number {
  const counts = new Map<string, number>()
  for (let at = 0; at + size <= left.length; at++) {
    const gram = left.slice(at, at + size)
    counts.set(gram, (counts.get(gram) ?? 0) + 1)
  }
  let shared = 0
  for (let at = 0; at + size <= right.length; at++) {
    const gram = right.slice(at, at + size)
    const count = counts.get(gram) ?? 0
    if (count === 0) continue
    shared++
    counts.set(gram, count - 1)
  }
  return shared
}

function sharedCodePointGrams(left: string, right: string, size: number): number {
  const first = Array.from(left)
  const second = Array.from(right)
  const counts = new Map<string, number>()
  for (let at = 0; at + size <= first.length; at++) {
    const gram = JSON.stringify(first.slice(at, at + size))
    counts.set(gram, (counts.get(gram) ?? 0) + 1)
  }
  let shared = 0
  for (let at = 0; at + size <= second.length; at++) {
    const gram = JSON.stringify(second.slice(at, at + size))
    const count = counts.get(gram) ?? 0
    if (count === 0) continue
    shared++
    counts.set(gram, count - 1)
  }
  return shared
}

describe('the normalized-Indel candidate index', () => {
  it('owns choices and returns exact normalized-Indel qualifiers in ascending order', () => {
    const compilation = scorerCompilation(createScorer(indelSimilarity))
    const make = compilation.candidateChoices
    if (make === undefined) throw new Error('missing Indel candidate capability')
    const builder = make()
    const mutable = ['a', 'b']
    builder.add(mutable)
    mutable[0] = 'z'
    const corpus = [...stringsThrough(4), '😀', 'a😀b', 'aaaa', 'abababab']
    for (const value of corpus) builder.add(value)
    const index = builder.seal()

    for (const query of stringsThrough(3)) {
      const scores = corpus.map((choice) => compilation.rawScore(query, choice, null))
      const thresholds = new Set([Number.MIN_VALUE, 0.5, 0.8, 0.9, 1, -1, 2])
      for (const score of scores) {
        thresholds.add(score)
        thresholds.add(adjacent(score, -1))
        thresholds.add(adjacent(score, 1))
      }
      for (const threshold of thresholds) {
        const found = index.candidates(query, threshold)
        const actual = Array.from(found.ids.subarray(0, found.length))
        const expected = [['a', 'b'] as const, ...corpus]
          .map((choice, id) => ({ id, score: compilation.rawScore(query, choice, null) }))
          .filter(({ score }) => score >= threshold)
          .map(({ id }) => id)
        expect(actual).toEqual(expected)
      }
    }
    expect(() => builder.add('late')).toThrow(TypeError)
    expect(() => builder.seal()).toThrow(TypeError)
  })
})

describe('normalized-Indel q-gram proof', () => {
  const values = stringsThrough(5)
  const distance = scorerCompilation(createScorer(indelDistance))
  const similarity = scorerCompilation(createScorer(indelSimilarity))

  it.each([1, 2, 3])(
    'proves the (2q - 1)K condition for q=%i',
    (q) => {
      for (const left of values) {
        for (const right of values) {
          const actual = similarity.rawScore(left, right, null)
          const actualDistance = distance.rawScore(left, right, null)
          const overlap = sharedGrams(left, right, q)
          for (const threshold of [Number.MIN_VALUE, 0.5, 0.8, 0.9, 1, actual]) {
            if (actual < threshold) continue
            const maximum = left.length + right.length
            const k = maximumQualifyingDistance(maximum, threshold)
            const leftGrams = Math.max(left.length - q + 1, 0)
            const rightGrams = Math.max(right.length - q + 1, 0)
            const minimum = Math.max(
              0,
              Math.ceil((leftGrams + rightGrams - (2 * q - 1) * k) / 2),
            )
            expect(overlap).toBeGreaterThanOrEqual(minimum)
            expect(actualDistance).toBeLessThanOrEqual(k)
          }
        }
      }
    },
    30_000,
  )

  it('reports no qualifying distance for a threshold nothing can reach', () => {
    // `candidates` never asks — it refuses a threshold above 1 first — but the
    // helper is total over its own domain and the proof suite calls it directly.
    expect(maximumQualifyingDistance(10, 1.5)).toBe(-1)
    expect(maximumQualifyingDistance(0, 1)).toBe(0)
  })

  it.each([
    [17, 31, 9],
    [2_147_483_647, 4_294_967_295, 4_294_967_295],
  ])('recovers integer Dice overlap at supported bounds', (query, choice, shared) => {
    const boundedShared = Math.min(shared, query, choice)
    const score = (2 * boundedShared) / (query + choice)
    expect(Math.round((score * (query + choice)) / 2)).toBe(boundedShared)
  })

  it.each([1, 2, 3])('proves the same condition over code points for q=%i', (q) => {
    // `sharedGrams` slices UTF-16, so the exhaustive run above only proves the
    // ASCII case. Repeat it over astral and mixed values, counting grams by
    // code point the way the index does.
    const astral = [
      '',
      '😀',
      '😀a',
      'a😀',
      '😀😁',
      '😀a😁',
      'a😀b',
      '😀😀😁',
      '😁😀',
      'ab😀cd',
      '😀b😁d',
    ]
    for (const left of astral) {
      for (const right of astral) {
        const actual = similarity.rawScore(left, right, null)
        const actualDistance = distance.rawScore(left, right, null)
        const overlap = sharedCodePointGrams(left, right, q)
        const leftLength = Array.from(left).length
        const rightLength = Array.from(right).length
        for (const threshold of [Number.MIN_VALUE, 0.5, 0.8, 0.9, 1, actual]) {
          if (actual < threshold) continue
          const k = maximumQualifyingDistance(leftLength + rightLength, threshold)
          const leftGrams = Math.max(leftLength - q + 1, 0)
          const rightGrams = Math.max(rightLength - q + 1, 0)
          const minimum = Math.max(
            0,
            Math.ceil((leftGrams + rightGrams - (2 * q - 1) * k) / 2),
          )
          expect(overlap).toBeGreaterThanOrEqual(minimum)
          expect(actualDistance).toBeLessThanOrEqual(k)
        }
      }
    }
  })
})
