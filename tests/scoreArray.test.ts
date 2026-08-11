// Not ported from RapidFuzz — upstream returns a NumPy array, so the questions
// here are answered by NumPy rather than by its own tests.
//
// `scoreMatrix` returns one typed array with a shape rather than an array of
// arrays, and `into` chooses the element type instead of `dtype` choosing only
// whether to round. The ported assertions go through `matrix.ts`, which
// unwraps to nested arrays; this file is where the wrapper itself is the
// subject.
import { describe, expect, it } from 'vitest'

import { levenshteinDistance } from '../src/distance/levenshtein.js'
import { ratio } from '../src/_fuzz/legacy.js'
import { scoreMatrix, scorePairs } from '../src/search.js'
import { callUntyped } from './common.js'

const QUERIES = ['new york mets', 'chicago cubs']
const CHOICES = ['new york mets', 'atlanta braves', 'chicago cubs']

describe('shape and access', () => {
  it('reports its dimensions', () => {
    const m = scoreMatrix(QUERIES, CHOICES)
    expect(m.rows).toBe(2)
    expect(m.cols).toBe(3)
    expect(m.data.length).toBe(6)
  })

  it('indexes the same scores the scorer produces', () => {
    const m = scoreMatrix(QUERIES, CHOICES)
    for (let i = 0; i < QUERIES.length; i++) {
      for (let j = 0; j < CHOICES.length; j++) {
        expect(m.at(i, j)).toBe(ratio(QUERIES[i], CHOICES[j]))
      }
    }
  })

  it('stores row-major', () => {
    const m = scoreMatrix(QUERIES, CHOICES)
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        expect(m.data[i * m.cols + j]).toBe(m.at(i, j))
      }
    }
  })

  // A `RangeError` rather than `undefined`, which is what lets `at` promise a
  // `number` under `noUncheckedIndexedAccess: false`.
  it('refuses coordinates off the matrix', () => {
    const m = scoreMatrix(QUERIES, CHOICES)
    for (const [row, col] of [
      [-1, 0],
      [0, -1],
      [2, 0],
      [0, 3],
      [0.5, 0],
      [0, Number.NaN],
    ]) {
      expect(() => m.at(row, col)).toThrow(RangeError)
    }
  })

  it('copies to nested arrays', () => {
    const m = scoreMatrix(QUERIES, CHOICES)
    const arrays = m.toArray()
    expect(arrays.length).toBe(2)
    expect(arrays[0].length).toBe(3)
    expect(arrays).toEqual(QUERIES.map((q) => CHOICES.map((c) => ratio(q, c))))
    // A copy, so writing to it cannot reach back into the matrix.
    arrays[0][0] = -1
    expect(m.at(0, 0)).not.toBe(-1)
  })

  it('iterates rows as views over the same buffer', () => {
    const m = scoreMatrix(QUERIES, CHOICES)
    const rows = [...m]
    expect(rows.length).toBe(2)
    expect(Array.from(rows[1])).toEqual(CHOICES.map((c) => ratio(QUERIES[1], c)))
    expect(rows[0].buffer).toBe(m.data.buffer)
  })

  // The point of `into` is a precisely typed store, so row iteration has to
  // keep that type rather than widening to the union. These annotations are the
  // assertion — they stop compiling if the element type is ever lost, which no
  // runtime check would catch.
  it('keeps the element type through data and row iteration', () => {
    const bytes = scoreMatrix(QUERIES, CHOICES, { into: 'u8' })
    const byteData: Uint8Array = bytes.data
    const byteRow: Uint8Array = [...bytes][0]

    const doubles = scoreMatrix(QUERIES, CHOICES)
    const doubleData: Float64Array = doubles.data
    const doubleRow: Float64Array = [...doubles][0]

    expect(byteData.length).toBe(6)
    expect(byteRow.length).toBe(3)
    expect(doubleData.length).toBe(6)
    expect(doubleRow.length).toBe(3)
  })
})

describe('empty inputs', () => {
  it('has no rows for no queries', () => {
    const m = scoreMatrix([], CHOICES)
    expect(m.rows).toBe(0)
    expect(m.toArray()).toEqual([])
  })

  // One row, no columns — the shape upstream produces, and the reason `rows`
  // and `cols` are carried rather than inferred from the data length.
  it('has an empty row per query when there are no choices', () => {
    const m = scoreMatrix(QUERIES, [])
    expect(m.rows).toBe(2)
    expect(m.cols).toBe(0)
    expect(m.toArray()).toEqual([[], []])
  })

  it('is empty both ways', () => {
    expect(scoreMatrix([], []).toArray()).toEqual([])
    expect(Array.from(scorePairs([], []))).toEqual([])
  })
})

describe('the `into` option', () => {
  it('defaults to a double', () => {
    expect(scoreMatrix(QUERIES, CHOICES).data).toBeInstanceOf(Float64Array)
    expect(scorePairs(['ab'], ['ac'])).toBeInstanceOf(Float64Array)
  })

  it('allocates the kind that was asked for', () => {
    expect(scoreMatrix(QUERIES, CHOICES, { into: 'f32' }).data).toBeInstanceOf(
      Float32Array,
    )
    expect(scoreMatrix(QUERIES, CHOICES, { into: 'i32' }).data).toBeInstanceOf(Int32Array)
    expect(scoreMatrix(QUERIES, CHOICES, { into: 'u8' }).data).toBeInstanceOf(Uint8Array)
    expect(scoreMatrix(QUERIES, CHOICES, { into: 'u8c' }).data).toBeInstanceOf(
      Uint8ClampedArray,
    )
    expect(scorePairs(['ab'], ['ac'], { into: 'u16' })).toBeInstanceOf(Uint16Array)
  })

  it('applies to scorePairs as well as scoreMatrix', () => {
    // `cpdist` accepted `dtype` upstream, so dropping the equivalent from
    // `scorePairs` would be a capability lost rather than a simplification.
    const pairs = scorePairs(['test'], ['test2'], {
      scorer: levenshteinDistance,
      into: 'u8',
      scoreMultiplier: 10,
    })
    expect(pairs).toBeInstanceOf(Uint8Array)
    expect(pairs[0]).toBe(10)
  })

  it('rounds for an integral kind and not for a float one', () => {
    const options = { scorer: ratio, scoreMultiplier: 0.01 }
    expect(scoreMatrix(['ab'], ['ac'], { ...options, into: 'f64' }).at(0, 0)).toBe(0.5)
    expect(scoreMatrix(['ab'], ['ac'], { ...options, into: 'i32' }).at(0, 0)).toBe(1)
  })

  // Rounding happens before the store precisely so that the store's own
  // truncation never gets to replace it.
  it('rounds rather than truncating', () => {
    const options = { scorer: ratio, scoreMultiplier: 0.019, into: 'i32' }
    expect(scoreMatrix(['ab'], ['ac'], { ...options, into: 'i32' }).at(0, 0)).toBe(1)
    expect(Math.trunc(50 * 0.019)).toBe(0)
  })

  it('wraps on u8 and saturates on u8c', () => {
    const options = { scorer: ratio, scoreMultiplier: 4 }
    // ratio('ab', 'ab') is 100, so the score is 400 — past a byte either way.
    expect(scoreMatrix(['ab'], ['ab'], { ...options, into: 'u8' }).at(0, 0)).toBe(
      400 % 256,
    )
    expect(scoreMatrix(['ab'], ['ab'], { ...options, into: 'u8c' }).at(0, 0)).toBe(255)
  })

  it('keeps a negative score on a signed kind', () => {
    const options = { scorer: ratio, scoreMultiplier: -0.01 }
    expect(scoreMatrix(['ab'], ['ac'], { ...options, into: 'i32' }).at(0, 0)).toBe(-1)
  })
})

describe('symmetric mirroring writes one rounded value', () => {
  // The mirrored half is a copy of the value already rounded for its own cell,
  // so the two triangles cannot round differently and drift apart.
  it('agrees across the diagonal', () => {
    const strings = ['test', 'test2', 'testing']
    const m = scoreMatrix(strings, strings, {
      scorer: ratio,
      into: 'i32',
      scoreMultiplier: 0.019,
    })
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        expect(m.at(i, j)).toBe(m.at(j, i))
      }
    }
  })
})

// `scorePairs` reads `length` and indexes; it never calls an array method, so a
// JavaScript caller can hand it an array-like whose `length` is not a count. A
// typed-array constructor coerces one rather than refusing it — `1.5` allocates
// a single element — which would report a shorter result than was asked for.
describe('a score count that is not a count', () => {
  it('refuses a fractional length', () => {
    expect(() => callUntyped(scorePairs, { length: 1.5 }, { length: 1.5 })).toThrow(
      RangeError,
    )
  })

  // `NaN` has no test here because it cannot reach the allocation: the two
  // lengths are compared first, and `NaN !== NaN` rejects it as a mismatch.

  // The reachable one, and a different question: two ordinary arrays whose
  // product is larger than any array can hold.
  it('refuses a product no array can hold', () => {
    expect(() =>
      scoreMatrix(new Array<string>(70000).fill('a'), new Array<string>(70000).fill('a')),
    ).toThrow(/more than an array can hold/)
  })
})
