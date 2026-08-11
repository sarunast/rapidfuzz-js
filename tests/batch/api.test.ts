import { describe, expect, test, vi } from 'vitest'

import * as levenshtein from '../../src/algorithms/levenshtein/index.js'
import {
  allocateScores,
  roundHalfAwayFromZero,
  scoreArrayFactory,
} from '../../src/batch/scoreArray.js'
import { scorerCompilation } from '../../src/core/scorer.js'
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

  test('prepares matrix choices once and mirrors symmetric matrices', () => {
    let calls = 0
    const scorer = createScorer(
      (left, right) => {
        calls++
        return left === right ? 1 : 0
      },
      { direction: 'similarity', bounds: [0, 1], symmetric: true },
    )
    const values = ['a', 'b', 'c']
    expect(scoreMatrix(values, values, { scorer }).toArray()).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ])
    expect(calls).toBe(6)

    const builtIn = createScorer(levenshtein.similarity)
    const compilation = scorerCompilation(builtIn)
    const prepareChoice = vi.spyOn(compilation, 'prepareChoice')
    const prepareQuery = vi.spyOn(compilation, 'prepareQuery')
    scoreMatrix(['a', 'b'], ['a', 'b', 'c'], { scorer: builtIn })
    expect(prepareChoice).toHaveBeenCalledTimes(3)
    expect(prepareQuery).toHaveBeenCalledTimes(2)
  })

  test('scores pairs through the raw pair kernel and normalizes each input once', () => {
    const scorer = createScorer(levenshtein.similarity)
    const compilation = scorerCompilation(scorer)
    const rawScore = vi.spyOn(compilation, 'rawScore')
    const prepareChoice = vi.spyOn(compilation, 'prepareChoice')
    const prepareQuery = vi.spyOn(compilation, 'prepareQuery')
    let normalizations = 0
    const normalize = (value: string | ArrayLike<unknown>) => {
      normalizations++
      return typeof value === 'string' ? value.toLowerCase() : value
    }
    expect([...scorePairs(['A', 'B'], ['a', 'c'], { scorer, normalize })]).toEqual([1, 0])
    expect(normalizations).toBe(4)
    expect(rawScore).toHaveBeenCalledTimes(2)
    expect(prepareChoice).not.toHaveBeenCalled()
    expect(prepareQuery).not.toHaveBeenCalled()

    normalizations = 0
    const shared = ['A', 'B']
    scorePairs(shared, shared, { scorer, normalize })
    expect(normalizations).toBe(2)
    expect(() => scorePairs(['a'], ['a'], { scorer, normalize: () => null })).toThrow(
      TypeError,
    )
  })

  test('normalizes a shared symmetric matrix only once per sequence', () => {
    const scorer = createScorer(levenshtein.similarity)
    const values = ['A', 'B']
    let calls = 0
    const matrix = scoreMatrix(values, values, {
      scorer,
      normalize: (value) => {
        calls++
        return typeof value === 'string' ? value.toLowerCase() : value
      },
    })
    expect(matrix.toArray()).toEqual([
      [1, 0],
      [0, 1],
    ])
    expect(calls).toBe(2)
    expect(() =>
      scoreMatrix(['a'], ['a'], { scorer, normalize: () => undefined }),
    ).toThrow(TypeError)
  })
})
