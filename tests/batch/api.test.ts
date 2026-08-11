import { describe, expect, test } from 'vitest'

import * as levenshtein from '../../src/algorithms/levenshtein/index.js'
import {
  allocateScores,
  roundHalfAwayFromZero,
  scoreArrayFactory,
} from '../../src/batch/scoreArray.js'
import { createScorer, scoreMatrix, scorePairs } from '../../src/index.js'

describe('batch scoring', () => {
  test('matrix operations consume Scorer objects', () => {
    const normalized = createScorer(levenshtein.similarity)
    expect(scoreMatrix(['a', 'b'], ['a', 'c'], { scorer: normalized }).toArray()).toEqual(
      [
        [1, 0],
        [0, 0],
      ],
    )
    expect([...scorePairs(['a', 'b'], ['a', 'c'], { scorer: normalized })]).toEqual([
      1, 0,
    ])
    expect(() => scorePairs(['a'], ['a', 'b'], { scorer: normalized })).toThrow(
      RangeError,
    )
    const bytes = scoreMatrix(['a', 'b'], ['a', 'c'], {
      scorer: normalized,
      into: 'u8',
    })
    expect(bytes.data).toBeInstanceOf(Uint8Array)
    expect(bytes.at(0, 0)).toBe(1)
    expect(bytes.toArray()).toEqual([
      [1, 0],
      [0, 0],
    ])
    expect([...bytes].every((row) => row.buffer === bytes.data.buffer)).toBe(true)
    expect(() => bytes.at(-1, 0)).toThrow(RangeError)

    for (const [into, constructor] of [
      ['f64', Float64Array],
      ['f32', Float32Array],
      ['i32', Int32Array],
      ['i16', Int16Array],
      ['i8', Int8Array],
      ['u32', Uint32Array],
      ['u16', Uint16Array],
      ['u8', Uint8Array],
      ['u8c', Uint8ClampedArray],
    ] as const) {
      expect(scorePairs(['a'], ['a'], { scorer: normalized, into })).toBeInstanceOf(
        constructor,
      )
      const matrix = scoreMatrix(['a'], ['a'], { scorer: normalized, into })
      expect([...matrix][0]).toBeInstanceOf(constructor)
    }
    expect(roundHalfAwayFromZero(-0.5)).toBe(-1)
    expect(roundHalfAwayFromZero(-0.1)).toBe(0)
    expect(() => Reflect.apply(scoreArrayFactory, undefined, ['nope'])).toThrow(
      RangeError,
    )
    expect(() => allocateScores('u8', -1, 'test')).toThrow(RangeError)
    expect(() => allocateScores('u8', 2 ** 32, 'test')).toThrow(RangeError)
  })
})
