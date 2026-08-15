import { describe, expect, test } from 'vitest'

import {
  convSequence,
  isSequence,
  normalizeSequence,
  snapshotSequence,
  validatePair,
  validateSequence,
} from './sequence.js'
import type { Sequence } from './types.js'

describe('the sequence boundary', () => {
  test('accepts strings and array-likes with a representable length', () => {
    expect(isSequence('abc')).toBe(true)
    expect(isSequence('')).toBe(true)
    expect(isSequence(['a', 'b'])).toBe(true)
    expect(isSequence(new Uint8Array(3))).toBe(true)
    // Duck typing on purpose: a sparse array-like reads `undefined` at every
    // index, and proving otherwise would make validation O(n).
    expect(isSequence({ length: 3 })).toBe(true)
    expect(isSequence({ length: 0 })).toBe(true)
    // The longest array JavaScript can hold: representable, if not affordable.
    expect(isSequence({ length: 0xffff_ffff })).toBe(true)
  })

  test('refuses lengths no array could have', () => {
    for (const length of [
      -1,
      1.5,
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Number.MAX_SAFE_INTEGER,
      0x1_0000_0000,
      '3',
    ]) {
      expect(isSequence({ length })).toBe(false)
      expect(() => validateSequence({ length })).toThrow(
        'expected a string or an array-like sequence',
      )
    }
    // Without the ceiling, this is the error a caller got instead — ours,
    // raised from `snapshotSequence` long after validation passed.
    expect(() => new Array(Number.MAX_SAFE_INTEGER)).toThrow(RangeError)
  })

  test('refuses values that are not array-like at all', () => {
    for (const value of [null, undefined, 42, true, Symbol('s'), {}, new Map()]) {
      expect(isSequence(value)).toBe(false)
    }
    // A function is structurally an array-like and deliberately not a
    // sequence: accepting one would score a misplaced argument.
    expect(isSequence(() => 1)).toBe(false)
    const callableArrayLike = Object.defineProperties(() => 1, {
      0: { value: 'a' },
      length: { value: 1 },
    })
    expect(isSequence(callableArrayLike)).toBe(false)
  })

  test('pairs follow the direction and the missing policy', () => {
    expect(validatePair('a', 'b', 'similarity', 'compatible')).toEqual(['a', 'b'])
    expect(validatePair(null, 'b', 'similarity', 'compatible')).toBeNull()
    expect(validatePair('a', undefined, 'similarity', 'compatible')).toBeNull()
    for (const [a, b, direction, missing] of [
      [null, 'b', 'similarity', 'throw'],
      [null, 'b', 'distance', 'compatible'],
      ['a', null, 'distance', 'throw'],
    ] as const) {
      expect(() => validatePair(a, b, direction, missing)).toThrow(
        'missing sequences are not supported by this scorer',
      )
    }
    expect(() =>
      Reflect.apply(validatePair, undefined, ['a', 42, 'similarity', 'compatible']),
    ).toThrow('expected a string or an array-like sequence')
  })

  test('snapshots array-likes and passes strings through', () => {
    const source = 'abc'
    expect(snapshotSequence(source)).toBe(source)

    const mutable = ['a', 'b']
    const owned = snapshotSequence(mutable)
    mutable[0] = 'z'
    expect(owned).toEqual(['a', 'b'])

    // The length is read once, so a growing array-like yields the size first
    // observed.
    const growing = { length: 2, 0: 'a', 1: 'b', 2: 'c' }
    expect(snapshotSequence(growing)).toEqual(['a', 'b'])
  })

  test('holds a normalizer to returning a usable sequence', () => {
    const calls: Sequence[] = []
    const upper = (value: Sequence): Sequence => {
      calls.push(value)
      return String(value).toUpperCase()
    }
    expect(normalizeSequence('abc', upper)).toBe('ABC')
    // Once per sequence: the callers that skip normalization skip the call,
    // and the ones that normalize do not normalize twice.
    expect(calls).toEqual(['abc'])

    for (const missing of [null, undefined]) {
      expect(() => normalizeSequence('abc', () => missing)).toThrow(
        'normalize returned a missing value',
      )
    }
    // From JavaScript, where the normalizer's return type proves nothing.
    expect(() => Reflect.apply(normalizeSequence, undefined, ['abc', () => 42])).toThrow(
      'expected a string or an array-like sequence',
    )
  })

  test('converts single-character elements without changing longer strings', () => {
    expect(convSequence(['a', '😀', 'ab'])).toEqual([97, 0x1f600, 'ab'])
  })

  test('uses narrow string storage until a code point needs promotion', () => {
    expect(convSequence('abc')).toBeInstanceOf(Uint16Array)
    expect(convSequence('a😀')).toBeInstanceOf(Uint32Array)
  })
})
