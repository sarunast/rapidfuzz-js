// Ported from RapidFuzz tests/distance/test_LCSseq.py
import { expect, it } from 'vitest'

import { lcsSeqEditops, lcsSeqOpcodes } from '../../src/algorithms/lcs/implementation.js'
import { normalizeText as defaultProcess } from '../../src/core/normalize.js'
import { editopTuples, opcodeTuples } from '../common.js'
import { LCSseq } from './scorers.js'

it('handles the basic cases', () => {
  expect(LCSseq.distance('', '')).toBe(0)
  expect(LCSseq.distance('test', 'test')).toBe(0)
  expect(LCSseq.distance('aaaa', 'bbbb')).toBe(4)
})

it('does not invert the normalized results against a single empty string', () => {
  expect(LCSseq.distance('abc', '')).toBe(3)
  expect(LCSseq.normalizedDistance('abc', '')).toBe(1)
  expect(LCSseq.normalizedSimilarity('abc', '')).toBe(0)
  expect(LCSseq.normalizedDistance('abc', '', { scoreCutoff: 0.5 })).toBe(1)

  expect(LCSseq.normalizedDistance('', '')).toBe(0)
  expect(LCSseq.normalizedSimilarity('', '')).toBe(1)
})

it('is case insensitive with the default processor', () => {
  expect(
    LCSseq.distance('new york mets', 'new YORK mets', { processor: defaultProcess }),
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

// Not ported — upstream tests `opcodes` through the shared `GenericScorer`
// helpers, which this port's editops suite covers for Levenshtein alone.
it('expresses the same alignment as blocks', () => {
  expect(opcodeTuples(lcsSeqOpcodes('aaabaaa', 'abbaaabba'))).toEqual(
    opcodeTuples(lcsSeqEditops('aaabaaa', 'abbaaabba').toOpcodes()),
  )
})
