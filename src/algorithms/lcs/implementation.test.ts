// Ported from RapidFuzz tests/distance/test_LCSseq.py
import { expect, it } from 'vitest'

import { editopTuples, opcodeTuples } from '../../../testing/common.js'
import { LCSseq } from '../../../testing/scorers.js'
import { normalizeText as defaultProcess } from '../../core/normalize.js'
import { lcsSeqEditops, lcsSeqOpcodes } from './implementation.js'

it('handles the basic cases', () => {
  expect(LCSseq.distance('', '')).toBe(0)
  expect(LCSseq.distance('test', 'test')).toBe(0)
  expect(LCSseq.distance('aaaa', 'bbbb')).toBe(4)
})

it('does not invert the normalized results against a single empty string', () => {
  expect(LCSseq.distance('abc', '')).toBe(3)
  expect(1 - (LCSseq.normalizedSimilarity('abc', '') ?? 0)).toBe(1)
  expect(LCSseq.normalizedSimilarity('abc', '')).toBe(0)

  expect(1 - (LCSseq.normalizedSimilarity('', '') ?? 0)).toBe(0)
  expect(LCSseq.normalizedSimilarity('', '')).toBe(1)
})

it('compares normalized text case-insensitively', () => {
  expect(
    LCSseq.distance(defaultProcess('new york mets'), defaultProcess('new YORK mets')),
  ).toBe(0)
})

it('reports the edit operations', () => {
  expect(editopTuples(lcsSeqEditops('0', ''))).toEqual([['delete', 0, 0]])
  expect(editopTuples(lcsSeqEditops('', '0'))).toEqual([['insert', 0, 0]])

  expect(editopTuples(lcsSeqEditops('00', '0'))).toEqual([['delete', 1, 1]])
  expect(editopTuples(lcsSeqEditops('0', '00'))).toEqual([['insert', 1, 1]])

  expect(editopTuples(lcsSeqEditops('qabxcd', 'abycdf'))).toEqual([
    ['delete', 0, 0],
    ['insert', 3, 2],
    ['delete', 3, 3],
    ['insert', 6, 5],
  ])
  expect(editopTuples(lcsSeqEditops('Lorem ipsum.', 'XYZLorem ABC iPsum'))).toEqual([
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

  const ops = lcsSeqEditops('aaabaaa', 'abbaaabba')
  expect(ops.srcLen).toBe(7)
  expect(ops.destLen).toBe(9)
})

it('does not recover a NaN-to-NaN match from the alignment matrix', () => {
  const source = [NaN]
  const destination = [NaN]
  const ops = lcsSeqEditops(source, destination)

  expect(editopTuples(ops)).toEqual([
    ['insert', 0, 0],
    ['delete', 0, 1],
  ])
})

it('returns an exact successful score from the direct multiword band', () => {
  const left = `${'a'.repeat(198)}xyz`
  const right = `uvw${'a'.repeat(198)}`
  expect(LCSseq.distance(left, right, { threshold: 3 })).toBe(3)
})

// Not ported — upstream tests `opcodes` through the shared `GenericScorer`
// helpers, which this port's editops suite covers for Levenshtein alone.
it('expresses the same alignment as blocks', () => {
  expect(opcodeTuples(lcsSeqOpcodes('aaabaaa', 'abbaaabba'))).toEqual(
    opcodeTuples(lcsSeqEditops('aaabaaa', 'abbaaabba').toOpcodes()),
  )
})
