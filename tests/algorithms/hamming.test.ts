// Ported from RapidFuzz tests/distance/test_Hamming.py
import { expect, it } from 'vitest'

import {
  hammingEditops,
  hammingOpcodes,
} from '../../src/algorithms/hamming/implementation.js'
import { normalizeText as defaultProcess } from '../../src/core/normalize.js'
import { editopTuples, opcodeTuples } from '../support/common.js'
import { Hamming } from '../support/scorers.js'

it('handles the basic cases', () => {
  expect(Hamming.distance('', '')).toBe(0)
  expect(Hamming.distance('test', 'test')).toBe(0)
  expect(Hamming.distance('aaaa', 'bbbb')).toBe(4)
  expect(Hamming.distance('aaaa', 'aaaaa')).toBe(1)
})

it('rejects differing lengths when padding is disabled', () => {
  expect(Hamming.distance('', '', { pad: false })).toBe(0)
  expect(Hamming.distance('test', 'test', { pad: false })).toBe(0)
  expect(Hamming.distance('aaaa', 'bbbb', { pad: false })).toBe(4)

  expect(() => Hamming.distance('aaaa', 'aaaaa', { pad: false })).toThrow(
    'Sequences are not the same length.',
  )
})

it('bounds arbitrary array-like sequences without scanning the remainder', () => {
  expect(Hamming.distance([1, 2, 3], [1, 9, 8], { threshold: 1 })).toBeUndefined()
  expect(Hamming.distance([1, 2, 3], [1, 9, 8], { threshold: 2 })).toBe(2)
  expect(Hamming.distance([1], [1, 2, 3], { threshold: 1 })).toBeUndefined()
  expect(Hamming.distance([1, 2], [1, 9])).toBe(1)
})

it('applies native distance thresholds', () => {
  expect(Hamming.distance('South Korea', 'North Korea')).toBe(2)
  expect(Hamming.distance('South Korea', 'North Korea', { threshold: 4 })).toBe(2)
  expect(Hamming.distance('South Korea', 'North Korea', { threshold: 3 })).toBe(2)
  expect(Hamming.distance('South Korea', 'North Korea', { threshold: 2 })).toBe(2)
  expect(Hamming.distance('South Korea', 'North Korea', { threshold: 1 })).toBeUndefined()
  expect(Hamming.distance('South Korea', 'North Korea', { threshold: 0 })).toBeUndefined()
})

it('is case insensitive with the default processor', () => {
  expect(
    Hamming.distance(defaultProcess('new york mets'), defaultProcess('new YORK mets')),
  ).toBe(0)
})

it('reports the edit operations', () => {
  expect(editopTuples(hammingEditops('0', ''))).toEqual([['delete', 0, 0]])
  expect(editopTuples(hammingEditops('', '0'))).toEqual([['insert', 0, 0]])

  expect(editopTuples(hammingEditops('00', '0'))).toEqual([['delete', 1, 1]])
  expect(editopTuples(hammingEditops('0', '00'))).toEqual([['insert', 1, 1]])

  expect(editopTuples(hammingEditops('qabxcd', 'abycdf'))).toEqual([
    ['replace', 0, 0],
    ['replace', 1, 1],
    ['replace', 2, 2],
    ['replace', 3, 3],
    ['replace', 4, 4],
    ['replace', 5, 5],
  ])

  const ops = hammingEditops('aaabaaa', 'abbaaabba')
  expect(ops.srcLen).toBe(7)
  expect(ops.destLen).toBe(9)
})

it('rejects editops on differing lengths when padding is disabled', () => {
  expect(() => hammingEditops('aaaa', 'aaaaa', { pad: false })).toThrow(
    'Sequences are not the same length.',
  )
})

it('expresses the same alignment as blocks', () => {
  expect(opcodeTuples(hammingOpcodes('qabxcd', 'abycdf'))).toEqual(
    opcodeTuples(hammingEditops('qabxcd', 'abycdf').toOpcodes()),
  )
})
