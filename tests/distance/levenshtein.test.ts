// Ported from RapidFuzz tests/distance/test_Levenshtein.py
import { createHash } from 'node:crypto'

import { expect, it } from 'vitest'

import { scorerFlagsOf } from '../../src/_common.js'
import { configure } from '../../src/configure.js'
import {
  levenshteinDistance,
  levenshteinEditops,
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
  levenshteinOpcodes,
  levenshteinSimilarity,
  type LevenshteinWeights,
} from '../../src/distance/levenshtein.js'
import { defaultProcess } from '../../src/utils.js'
import { editopTuples, opcodeTuples } from '../common.js'
import { matrixScores } from '../matrix.js'
import { Levenshtein } from './scorers.js'

it('treats two empty strings as a perfect match under any weights', () => {
  expect(Levenshtein.distance('', '')).toBe(0)
  expect(Levenshtein.distance('', '', { weights: [1, 1, 0] })).toBe(0)
  expect(Levenshtein.distance('', '', { weights: [1, 1, 2] })).toBe(0)
  expect(Levenshtein.distance('', '', { weights: [1, 1, 5] })).toBe(0)
  expect(Levenshtein.distance('', '', { weights: [3, 7, 5] })).toBe(0)
})

it('does not overflow on a huge score_cutoff', () => {
  expect(Levenshtein.distance('', '')).toBe(0)
  expect(Levenshtein.distance('', '', { scoreCutoff: 2 ** 63 })).toBe(0)
})

it('interprets strings and sequences the same way', () => {
  expect(Levenshtein.distance('aaaa', 'aaaa')).toBe(0)
  expect(Levenshtein.distance('aaaa', ['a', 'a', 'a', 'a'])).toBe(0)
  expect(Levenshtein.distance([0, -1], [0, -2])).toBe(1)
})

it('can express a word error rate over token sequences', () => {
  expect(Levenshtein.distance(['aaaaa', 'bbbb'], ['aaaaa', 'bbbb'])).toBe(0)
  expect(Levenshtein.distance(['aaaaa', 'bbbb'], ['aaaaa', 'cccc'])).toBe(1)
})

it('handles unicode with weighted operations', () => {
  const s1 = 'ÁÄ'
  const s2 = 'ABCD'

  expect(Levenshtein.distance(s1, s2)).toBe(4) // 2 sub + 2 ins
  expect(Levenshtein.distance(s1, s2, { weights: [1, 1, 0] })).toBe(2)
  expect(Levenshtein.distance(s1, s2, { weights: [1, 1, 2] })).toBe(6)
  expect(Levenshtein.distance(s1, s2, { weights: [1, 1, 5] })).toBe(6) // 2 del + 4 ins
  expect(Levenshtein.distance(s1, s2, { weights: [1, 7, 5] })).toBe(12) // 2 sub + 2 ins
  expect(Levenshtein.distance(s2, s1, { weights: [1, 7, 5] })).toBe(24) // 2 sub + 2 del

  expect(Levenshtein.distance(s1, s1)).toBe(0)
  expect(Levenshtein.distance(s1, s1, { weights: [1, 1, 0] })).toBe(0)
  expect(Levenshtein.distance(s1, s1, { weights: [1, 1, 2] })).toBe(0)
  expect(Levenshtein.distance(s1, s1, { weights: [1, 1, 5] })).toBe(0)
  expect(Levenshtein.distance(s1, s1, { weights: [3, 7, 5] })).toBe(0)
})

it('uses scaled Indel for expensive replacements and trims weighted affixes', () => {
  const prefix = 'a'.repeat(2048)
  const suffix = 'z'.repeat(2048)
  const source = `${prefix}abc${suffix}`
  const inserted = `${prefix}axbc${suffix}`

  expect(Levenshtein.distance(source, inserted, { weights: [2, 2, 5] })).toBe(2)
  expect(
    Levenshtein.distance(source, inserted, {
      weights: [2, 2, 5],
      scoreCutoff: 1,
    }),
  ).toBe(2)
  expect(Levenshtein.distance(source, inserted, { weights: [3, 7, 5] })).toBe(3)
  expect(Levenshtein.distance(inserted, source, { weights: [3, 7, 5] })).toBe(7)
})

it('does not regress on the cached mbleven implementation', () => {
  expect(Levenshtein.distance('0', '101', { scoreCutoff: 1 })).toBe(2)
  expect(Levenshtein.distance('0', '101', { scoreCutoff: 2 })).toBe(2)
  expect(Levenshtein.distance('0', '101', { scoreCutoff: 3 })).toBe(2)
})

it('is case insensitive with the default processor', () => {
  expect(
    Levenshtein.distance('new york mets', 'new YORK mets', { processor: defaultProcess }),
  ).toBe(0)
})

it('reports the edit operations', () => {
  expect(editopTuples(levenshteinEditops('0', ''))).toEqual([['delete', 0, 0]])
  expect(editopTuples(levenshteinEditops('', '0'))).toEqual([['insert', 0, 0]])

  expect(editopTuples(levenshteinEditops('00', '0'))).toEqual([['delete', 1, 1]])
  expect(editopTuples(levenshteinEditops('0', '00'))).toEqual([['insert', 1, 1]])

  expect(editopTuples(levenshteinEditops('qabxcd', 'abycdf'))).toEqual([
    ['delete', 0, 0],
    ['replace', 3, 2],
    ['insert', 6, 5],
  ])
  expect(editopTuples(levenshteinEditops('Lorem ipsum.', 'XYZLorem ABC iPsum'))).toEqual([
    ['insert', 0, 0],
    ['insert', 0, 1],
    ['insert', 0, 2],
    ['insert', 6, 9],
    ['insert', 6, 10],
    ['insert', 6, 11],
    ['insert', 6, 12],
    ['replace', 7, 14],
    ['delete', 11, 18],
  ])

  const ops = levenshteinEditops('aaabaaa', 'abbaaabba')
  expect(ops.srcLen).toBe(7)
  expect(ops.destLen).toBe(9)
})

it('keeps matrix ordering through the long Hirschberg path', () => {
  let state = 0x1234_5678
  const make = (length: number): string => {
    let value = ''
    for (let i = 0; i < length; i++) {
      state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0
      value += String.fromCharCode(97 + (state % 26))
    }
    return value
  }
  const source = make(4096)
  const destination = make(4096)
  const ops = levenshteinEditops(source, destination)
  const digest = createHash('sha256')
    .update(JSON.stringify(editopTuples(ops)))
    .digest('hex')

  // Recorded from the former full-matrix reconstruction for the same inputs.
  expect(digest).toBe('eea64c30c31f91d6519694039f71e34ed6560b4160c3548203eb0806ac3316c0')
  expect(ops.operations.length).toBe(Levenshtein.distance(source, destination))
  expect(ops.apply(source, destination)).toBe(destination)
  expect(ops.toOpcodes().apply(source, destination)).toBe(destination)
})

it('uses Hirschberg for long similar inputs whose full matrix exceeds the limit', () => {
  const source = 'a'.repeat(4096)
  const destination = Array.from(source, (value, index) =>
    index % 256 === 0 ? 'b' : value,
  ).join('')
  const ops = levenshteinEditops(source, destination)

  expect(ops.operations.length).toBe(16)
  expect(ops.apply(source, destination)).toBe(destination)
})

it('recovers a replacement for NaN values that are unequal under `===`', () => {
  const source = [NaN]
  const destination = [NaN]
  const ops = levenshteinEditops(source, destination)

  expect(editopTuples(ops)).toEqual([['replace', 0, 0]])
})

it('reports the opcodes', () => {
  expect(opcodeTuples(levenshteinOpcodes('', 'abc'))).toEqual([['insert', 0, 0, 0, 3]])
})

// Not ported — upstream takes only the positional tuple, because that is what
// its C++ signature takes. Nothing at a call site says which of `[1, 1, 2]` is
// the substitution, so the named form exists here as well; these pin that the
// two spellings are the same option and not two options that drift.
it('accepts named costs as well as the positional tuple', () => {
  const s1 = 'aaabbb'
  const s2 = 'ABCD'

  for (const [insertion, deletion, substitution] of [
    [1, 1, 0],
    [1, 1, 2],
    [1, 7, 5],
    [3, 7, 5],
  ]) {
    const named = { insertion, deletion, substitution }
    const positional: LevenshteinWeights = [insertion, deletion, substitution]

    expect(Levenshtein.distance(s1, s2, { weights: named })).toBe(
      Levenshtein.distance(s1, s2, { weights: positional }),
    )
    expect(Levenshtein.normalizedSimilarity(s1, s2, { weights: named })).toBe(
      Levenshtein.normalizedSimilarity(s1, s2, { weights: positional }),
    )
  }
})

it('reads named costs on the prepared path and in the flags', () => {
  const named = { insertion: 1, deletion: 2, substitution: 1 }
  const positional: LevenshteinWeights = [1, 2, 1]
  const queries = ['abc', 'ab']

  // Asymmetric either way round, so the matrix may not mirror its triangle.
  expect(
    matrixScores(queries, queries, {
      scorer: configure(levenshteinDistance, { weights: named }),
    }),
  ).toEqual(
    matrixScores(queries, queries, {
      scorer: configure(levenshteinDistance, { weights: positional }),
    }),
  )
  expect(
    scorerFlagsOf(configure(levenshteinDistance, { weights: named })).symmetric,
  ).toBe(false)
})

it('keeps fractional weighted cutoffs on the score lattice', () => {
  const weights: LevenshteinWeights = [0.5, 0.5, 0.5]
  const cases = [
    [levenshteinDistance, 0.5, 0.49, 0.5, 1.49],
    [levenshteinSimilarity, 0.5, 0.51, 0.5, 0],
    [levenshteinNormalizedDistance, 0.5, 0.49, 0.5, 1],
    [levenshteinNormalizedSimilarity, 0.5, 0.51, 0.5, 0],
  ] as const

  for (const [scorer, accepted, rejected, score, sentinel] of cases) {
    expect(scorer('ab', 'ac', { weights, scoreCutoff: accepted })).toBe(score)
    expect(scorer('ab', 'ac', { weights, scoreCutoff: rejected })).toBe(sentinel)

    const prepared = configure(scorer, { weights })
    expect(
      matrixScores(['ab'], ['ac'], { scorer: prepared, scoreCutoff: accepted })[0][0],
    ).toBe(score)
    expect(
      matrixScores(['ab'], ['ac'], { scorer: prepared, scoreCutoff: rejected })[0][0],
    ).toBe(sentinel)
  }
})

it('validates named costs the same way as positional ones', () => {
  expect(() =>
    Levenshtein.distance('abc', 'abd', {
      weights: { insertion: 1, deletion: 1, substitution: -1 },
    }),
  ).toThrow(TypeError)
})
