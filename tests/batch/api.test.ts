import { describe, expect, test, vi } from 'vitest'

import * as indel from '../../src/algorithms/indel/index.js'
import * as levenshtein from '../../src/algorithms/levenshtein/index.js'
import {
  allocateScores,
  buildScoreMatrix,
  roundHalfAwayFromZero,
  scoreArrayFactory,
} from '../../src/batch/storage.js'
import { scorerCompilation } from '../../src/core/scoring/scorer.js'
import { createScorer, scoreMatrix, scorePairs } from '../../src/index.js'
import type { MaybeSequence, Sequence } from '../../src/index.js'

describe('batch scoring', () => {
  test('matrix operations consume Scorer objects', () => {
    const normalized = createScorer(levenshtein.normalizedSimilarity)
    expect(scoreMatrix(['a', 'b'], ['a', 'c'], { scorer: normalized }).toArray()).toEqual(
      [
        [1, 0],
        [0, 0],
      ],
    )
    expect([...scorePairs(['a', 'b'], ['a', 'c'], { scorer: normalized })]).toEqual([
      1, 0,
    ])
    const sharedPairs = ['a', 'b']
    expect([...scorePairs(sharedPairs, sharedPairs, { scorer: normalized })]).toEqual([
      1, 1,
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

    // A dimension is checked on its own: the allocation sees only the product,
    // and `-1 × -1` and `0.5 × 2` are both a length of one.
    const noFill = () => {}
    expect(buildScoreMatrix('f64', 0, 0, 'test', noFill).data.length).toBe(0)
    for (const [rows, cols] of [
      [-1, -1],
      [0.5, 2],
      [2, 0.5],
      [1, -1],
      [Number.NaN, 1],
      [1, Number.POSITIVE_INFINITY],
    ] as const) {
      expect(() => buildScoreMatrix('f64', rows, cols, 'test', noFill)).toThrow(
        RangeError,
      )
    }
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

    const builtIn = createScorer(levenshtein.normalizedSimilarity)
    const compilation = scorerCompilation(builtIn)
    const prepareChoice = vi.spyOn(compilation, 'prepareChoice')
    const prepareQuery = vi.spyOn(compilation, 'prepareQuery')
    scoreMatrix(['a', 'b'], ['a', 'b', 'c'], { scorer: builtIn })
    expect(prepareChoice).toHaveBeenCalledTimes(3)
    expect(prepareQuery).toHaveBeenCalledTimes(2)
  })

  test('scores pairs through the raw pair kernel and normalizes each input once', () => {
    const scorer = createScorer(levenshtein.normalizedSimilarity)
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
    expect([
      ...scorePairs(['A', 'B'], ['a', 'c'], { scorer, normalize, into: 'u8' }),
    ]).toEqual([1, 0])
    expect(normalizations).toBe(4)

    normalizations = 0
    const shared = ['A', 'B']
    scorePairs(shared, shared, { scorer, normalize })
    expect(normalizations).toBe(2)
    expect(() => scorePairs(['a'], ['a'], { scorer, normalize: () => null })).toThrow(
      TypeError,
    )
  })

  test('normalizes a shared symmetric matrix only once per sequence', () => {
    const scorer = createScorer(levenshtein.normalizedSimilarity)
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

  test('applies natural-scale thresholds before explicit score multiplication', () => {
    const scorer = createScorer(levenshtein.normalizedSimilarity)
    expect(
      Array.from(
        scorePairs(['kitten', 'abc'], ['sitting', 'axc'], {
          scorer,
          threshold: 0.6,
          scoreMultiplier: 100,
          into: 'u8',
        }),
      ),
    ).toEqual([0, 67])
    expect(
      scoreMatrix(['kitten'], ['sitting', 'kitten'], {
        scorer,
        threshold: 0.6,
        scoreMultiplier: 100,
        into: 'u8',
      }).toArray(),
    ).toEqual([[0, 100]])

    const custom = createScorer((left, right) => (left === right ? 0.75 : 0.25), {
      direction: 'similarity',
      bounds: [0, 1],
      symmetric: true,
    })
    expect(
      Array.from(
        scorePairs(['a', 'a'], ['a', 'b'], {
          scorer: custom,
          threshold: 0.5,
          scoreMultiplier: -2,
          into: 'i8',
        }),
      ),
    ).toEqual([-2, 0])
    expect(
      scoreMatrix(['a'], ['a', 'b'], {
        scorer: custom,
        threshold: 0.5,
      }).toArray(),
    ).toEqual([[0.75, 0]])

    const customDistance = createScorer((left, right) => (left === right ? 0 : 2), {
      direction: 'distance',
      bounds: [0, 3],
      symmetric: true,
    })
    expect(
      Array.from(
        scorePairs(['a', 'a'], ['a', 'b'], {
          scorer: customDistance,
          threshold: 1,
        }),
      ),
    ).toEqual([0, 3])
    expect(
      scoreMatrix(['a'], ['a', 'b'], {
        scorer: customDistance,
        threshold: 1,
      }).toArray(),
    ).toEqual([[0, 3]])

    for (const scoreMultiplier of [Number.NaN, Infinity, -Infinity]) {
      expect(() => scorePairs(['a'], ['a'], { scorer, scoreMultiplier })).toThrow(
        RangeError,
      )
      expect(() => scoreMatrix(['a'], ['a'], { scorer, scoreMultiplier })).toThrow(
        RangeError,
      )
    }
    expect(() => scorePairs(['a'], ['a'], { scorer, threshold: Infinity })).toThrow(
      RangeError,
    )
  })

  test('a fractional threshold means the same thing to a scorer and to batch', () => {
    // A raw similarity is a count, so 2 does not clear 2.5. The kernels used to
    // truncate the cutoff before deriving their distance budget, so batch —
    // which trusts a built-in kernel to have applied the cutoff itself — stored
    // a score `Scorer.score` rejected.
    for (const metric of [levenshtein.similarity, indel.similarity]) {
      const scorer = createScorer(metric)
      const exact = scorer.score('abc', 'axc')
      expect(exact).toBe(metric === indel.similarity ? 4 : 2)
      for (const offset of [-1.1, -0.1, 0, 0.1, 0.5, 1]) {
        const threshold = (exact ?? 0) + offset
        const qualifies = offset <= 0
        expect(scorer.score('abc', 'axc', { threshold })).toBe(
          qualifies ? exact : undefined,
        )
        expect([...scorePairs(['abc'], ['axc'], { scorer, threshold })]).toEqual([
          qualifies ? exact : 0,
        ])
        expect(scoreMatrix(['abc'], ['axc'], { scorer, threshold }).at(0, 0)).toBe(
          qualifies ? exact : 0,
        )
      }
    }
  })

  test('a threshold nothing can meet rejects every pair', () => {
    // A built-in kernel says "rejected" with `cutoff + 1`, which reads as a
    // rejection only while the cutoff is inside the bounds. Below them it is a
    // real score — `trunc(-0.5) + 1` is 0, a *perfect* distance — so batch used
    // to store an identical pair as having met a threshold `Scorer.score`
    // refuses outright. Out-of-bounds thresholds fall back to the declared
    // bound, which is what a custom scorer's rejection already stored.
    const distance = createScorer(levenshtein.distance)
    for (const threshold of [-0.5, -1, -1e9]) {
      expect(distance.score('abc', 'abc', { threshold })).toBeUndefined()
      expect([...scorePairs(['abc'], ['abc'], { scorer: distance, threshold })]).toEqual([
        Number.POSITIVE_INFINITY,
      ])
      expect(
        scoreMatrix(['abc'], ['abc', 'zzz'], { scorer: distance, threshold }).toArray(),
      ).toEqual([[Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY]])
    }
    // Inside the bounds the kernel's own sentinel still stands, and it is the
    // one `process.cdist` stores: a rejected raw distance is `threshold + 1`.
    expect([
      ...scorePairs(['abc'], ['xyz1234'], { scorer: distance, threshold: 1 }),
    ]).toEqual([2])
    // A bounded similarity has a storable rejection either way, so an
    // impossible threshold is answered rather than refused.
    const bounded = createScorer(levenshtein.normalizedSimilarity)
    expect([...scorePairs(['a'], ['a'], { scorer: bounded, threshold: 1.5 })]).toEqual([
      0,
    ])
    expect(
      scoreMatrix(['a'], ['a'], { scorer: bounded, threshold: 1.5, into: 'u8' }).at(0, 0),
    ).toBe(0)
  })

  test('the scorer is settled before any query or choice is touched', () => {
    // Configuration first, data second — the order `scorePairs` already used.
    // Reaching the scorer only inside the fill meant a `normalize` with a side
    // effect ran, and a whole matrix was allocated, before a scorer this
    // package did not build was refused.
    const normalize = vi.fn((value: Sequence) => value)
    const foreign = { direction: 'similarity', bounds: [0, 100], symmetric: true }
    expect(() =>
      Reflect.apply(scoreMatrix, undefined, [
        ['a'],
        ['b'],
        { scorer: foreign, normalize },
      ]),
    ).toThrow('scorer was not created by createScorer')
    expect(normalize).not.toHaveBeenCalled()

    // No rows means no cell, so the choices are never prepared — but an
    // unusable rejected score is still reported rather than skipped with them.
    const scorer = createScorer(levenshtein.distance)
    const empty = scoreMatrix([], ['a', 'b'], { scorer })
    expect([empty.rows, empty.cols, empty.data.length]).toEqual([0, 2, 0])
    expect(() => scoreMatrix([], ['a'], { scorer, threshold: -1, into: 'u8' })).toThrow(
      RangeError,
    )
  })

  test('a rejected pair a custom scorer cannot express is refused, not stored', () => {
    // `[0, Infinity]` is a legitimate bound for a custom distance and not a
    // storable score: an integer destination turns it into 0 — the best
    // distance there is — and a zero multiplier turns it into NaN.
    const lengthGap = (left: MaybeSequence, right: MaybeSequence): number =>
      Math.abs((left?.length ?? 0) - (right?.length ?? 0))
    const unbounded = createScorer(lengthGap, {
      direction: 'distance',
      bounds: [0, Number.POSITIVE_INFINITY],
      symmetric: true,
    })
    expect([
      ...scorePairs(['abc'], ['abcdefgh'], { scorer: unbounded, threshold: 1 }),
    ]).toEqual([Number.POSITIVE_INFINITY])
    for (const into of ['i32', 'i16', 'u8', 'u8c'] as const) {
      expect(() =>
        scorePairs(['abc'], ['abcdefgh'], { scorer: unbounded, threshold: 1, into }),
      ).toThrow(RangeError)
      expect(() =>
        scoreMatrix(['abc'], ['abcdefgh'], { scorer: unbounded, threshold: 1, into }),
      ).toThrow(RangeError)
    }
    for (const scoreMultiplier of [0, -0]) {
      expect(() =>
        scorePairs(['abc'], ['abcdefgh'], {
          scorer: unbounded,
          threshold: 1,
          scoreMultiplier,
        }),
      ).toThrow(RangeError)
      expect(() =>
        scoreMatrix(['abc'], ['abcdefgh'], {
          scorer: unbounded,
          threshold: 1,
          scoreMultiplier,
        }),
      ).toThrow(RangeError)
    }
    // Without a threshold nothing is ever rejected, so the bound is never read.
    expect([
      ...scorePairs(['abc'], ['abcdefgh'], { scorer: unbounded, into: 'i32' }),
    ]).toEqual([5])
    // A similarity rejects with its lower bound, which is storable either way.
    const unboundedSimilarity = createScorer((left, right) => (left === right ? 1 : 0), {
      direction: 'similarity',
      bounds: [0, Number.POSITIVE_INFINITY],
      symmetric: true,
    })
    expect([
      ...scorePairs(['a'], ['b'], {
        scorer: unboundedSimilarity,
        threshold: 1,
        into: 'i32',
      }),
    ]).toEqual([0])

    // The rejected score is on the scorer's own scale, so the multiplier and
    // the rounding apply to it exactly once, the same way they apply to a real
    // score. A finite bound of 3 at -2 is -6, not -3 and not 12.
    const bounded = createScorer((left, right) => (left === right ? 0 : 2), {
      direction: 'distance',
      bounds: [0, 3],
      symmetric: true,
    })
    expect([
      ...scorePairs(['a', 'a'], ['a', 'b'], {
        scorer: bounded,
        threshold: 1,
        scoreMultiplier: -2,
        into: 'i8',
      }),
    ]).toEqual([0, -6])
    expect(
      scoreMatrix(['a'], ['a', 'b'], {
        scorer: bounded,
        threshold: 1,
        scoreMultiplier: 1.5,
      }).toArray(),
    ).toEqual([[0, 4.5]])
    expect(
      scoreMatrix(['a'], ['a', 'b'], {
        scorer: bounded,
        threshold: 1,
        scoreMultiplier: 1.5,
        into: 'i8',
      }).toArray(),
    ).toEqual([[0, 5]])
  })

  test('a score the element type cannot hold is refused rather than wrapped', () => {
    const lengthGap = (left: MaybeSequence, right: MaybeSequence): number =>
      Math.abs((left?.length ?? 0) - (right?.length ?? 0))
    const unbounded = createScorer(lengthGap, {
      direction: 'distance',
      bounds: [0, Number.POSITIVE_INFINITY],
      symmetric: true,
    })
    // An `Infinity` bound proves nothing, and rejecting the call on it would
    // make every integer distance matrix illegal. What is actually stored is
    // what decides.
    expect([
      ...scorePairs(['abc'], ['abcde'], { scorer: unbounded, into: 'u8' }),
    ]).toEqual([2])
    const long = 'a'.repeat(300)
    for (const [entry, label] of [
      [scoreMatrix, 'scoreMatrix'],
      [scorePairs, 'scorePairs'],
    ] as const) {
      expect(() =>
        Reflect.apply(entry, undefined, [
          [''],
          [long],
          { scorer: unbounded, into: 'u8' },
        ]),
      ).toThrow(`${label} produced the score 300, which 'u8' cannot store`)
      // The low end of the range is its own refusal, and a negative multiplier
      // is the only way a score reaches it.
      expect(() =>
        Reflect.apply(entry, undefined, [
          ['abc'],
          ['abcde'],
          { scorer: unbounded, into: 'u8', scoreMultiplier: -1 },
        ]),
      ).toThrow(`${label} produced the score -2, which 'u8' cannot store`)
    }
    // The multiplier is part of what has to fit: a 0..100 scorer is storable in
    // a u8 at 1 and is not at 3.
    const percent = createScorer((left, right) => (left === right ? 100 : 25), {
      direction: 'similarity',
      bounds: [0, 100],
      symmetric: true,
    })
    expect([...scorePairs(['a'], ['a'], { scorer: percent, into: 'u8' })]).toEqual([100])
    expect(() =>
      scorePairs(['a'], ['a'], { scorer: percent, into: 'u8', scoreMultiplier: 3 }),
    ).toThrow(RangeError)
    expect(() =>
      scorePairs(['a'], ['a'], { scorer: percent, into: 'u8', scoreMultiplier: -1 }),
    ).toThrow(RangeError)
    // Half a unit either way is enough to leave the range, since the score is
    // rounded away from zero before it is stored.
    expect(() =>
      scorePairs(['a'], ['a'], { scorer: percent, into: 'u8', scoreMultiplier: 2.555 }),
    ).toThrow(RangeError)
    // `u8c` saturates by definition, which is how a caller asks for the lossy
    // behaviour on purpose.
    expect([
      ...scorePairs(['a'], ['a'], { scorer: percent, into: 'u8c', scoreMultiplier: 3 }),
    ]).toEqual([255])
    // A float destination holds anything a metric can produce.
    expect([
      ...scorePairs(['a'], ['a'], { scorer: percent, into: 'f32', scoreMultiplier: 3 }),
    ]).toEqual([300])
    // Up to `f32`'s own maximum, past which a finite score stores as `Infinity`
    // — the one way a float destination loses a score outright, as against the
    // precision it is chosen to lose.
    expect([
      ...scorePairs(['a'], ['a'], {
        scorer: percent,
        into: 'f32',
        scoreMultiplier: 3e36,
      }),
    ]).toEqual([Math.fround(3e38)])
    for (const [entry, label] of [
      [scoreMatrix, 'scoreMatrix'],
      [scorePairs, 'scorePairs'],
    ] as const) {
      expect(() =>
        Reflect.apply(entry, undefined, [
          ['a'],
          ['a'],
          { scorer: percent, into: 'f32', scoreMultiplier: 1e38 },
        ]),
      ).toThrow(`${label} produced the score 1e+40, which 'f32' cannot store`)
      // The unproven path reaches the same refusal from an `Infinity` bound.
      expect(() =>
        Reflect.apply(entry, undefined, [
          [''],
          [long],
          { scorer: unbounded, into: 'f32', scoreMultiplier: 1e38 },
        ]),
      ).toThrow(`which 'f32' cannot store`)
    }
    expect([...scorePairs([''], [long], { scorer: unbounded, into: 'f32' })]).toEqual([
      300,
    ])
  })

  test('an option a batch call does not define is refused rather than ignored', () => {
    const scorer = createScorer(levenshtein.normalizedSimilarity)
    // Misspelled, so the threshold it names is silently not applied — the
    // reason the keys are checked at all.
    const misspelled = { scorer, thresold: 0.9 }
    for (const [entry, label] of [
      [scoreMatrix, 'scoreMatrix'],
      [scorePairs, 'scorePairs'],
    ] as const) {
      expect(() => Reflect.apply(entry, undefined, [['a'], ['b'], misspelled])).toThrow(
        `unknown ${label} option 'thresold'`,
      )
      expect(() => Reflect.apply(entry, undefined, [['a'], ['b'], null])).toThrow(
        `${label} options must be an object`,
      )
    }
    // Checked before the lengths, so a caller learns about the option they got
    // wrong rather than about the arrays they did not.
    expect(() =>
      Reflect.apply(scorePairs, undefined, [['a'], ['a', 'b'], misspelled]),
    ).toThrow("unknown scorePairs option 'thresold'")
  })
})
