// Ported from RapidFuzz tests/distance/test_Indel.py
import { expect, it } from 'vitest'

import { editopTuples } from '../../../testing/common.js'
import { Indel } from '../../../testing/scorers.js'
import { normalizeText as defaultProcess } from '../../core/normalize.js'
import { indelEditops } from './implementation.js'

it('handles the basic cases', () => {
  expect(Indel.distance('', '')).toBe(0)
  expect(Indel.distance('test', 'test')).toBe(0)
  expect(Indel.distance('aaaa', 'bbbb')).toBe(8)
})

it('applies native distance thresholds (issue 196)', () => {
  expect(Indel.distance('South Korea', 'North Korea')).toBe(4)
  expect(Indel.distance('South Korea', 'North Korea', { threshold: 4 })).toBe(4)
  expect(Indel.distance('South Korea', 'North Korea', { threshold: 3 })).toBeUndefined()
  expect(Indel.distance('South Korea', 'North Korea', { threshold: 2 })).toBeUndefined()
  expect(Indel.distance('South Korea', 'North Korea', { threshold: 1 })).toBeUndefined()
  expect(Indel.distance('South Korea', 'North Korea', { threshold: 0 })).toBeUndefined()
})

it('applies normalized similarity thresholds to direct scoring', () => {
  expect(Indel.normalizedSimilarity('abcd', 'abce')).toBe(0.75)
  expect(Indel.normalizedSimilarity('abcd', 'abce', { threshold: 0.8 })).toBeUndefined()
})

it('compares normalized text case-insensitively', () => {
  expect(
    Indel.distance(defaultProcess('new york mets'), defaultProcess('new YORK mets')),
  ).toBe(0)
})

it('reports the edit operations', () => {
  expect(editopTuples(indelEditops('0', ''))).toEqual([['delete', 0, 0]])
  expect(editopTuples(indelEditops('', '0'))).toEqual([['insert', 0, 0]])

  expect(editopTuples(indelEditops('00', '0'))).toEqual([['delete', 1, 1]])
  expect(editopTuples(indelEditops('0', '00'))).toEqual([['insert', 1, 1]])

  expect(editopTuples(indelEditops('qabxcd', 'abycdf'))).toEqual([
    ['delete', 0, 0],
    ['insert', 3, 2],
    ['delete', 3, 3],
    ['insert', 6, 5],
  ])
  expect(editopTuples(indelEditops('Lorem ipsum.', 'XYZLorem ABC iPsum'))).toEqual([
    ['insert', 0, 0],
    ['insert', 0, 1],
    ['insert', 0, 2],
    ['insert', 6, 9],
    ['insert', 6, 10],
    ['insert', 6, 11],
    ['insert', 6, 12],
    ['insert', 7, 14],
    ['delete', 7, 15],
    ['delete', 11, 18],
  ])

  const ops = indelEditops('aaabaaa', 'abbaaabba')
  expect(ops.srcLen).toBe(7)
  expect(ops.destLen).toBe(9)
})
