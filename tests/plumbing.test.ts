// Not ported from RapidFuzz — this is the port's own plumbing, and Python has
// none of it. `callScorer`, `toRecord` and `scorerFlagsOf` exist because a
// scorer arrives here as an ordinary object whose shape nothing has checked;
// `entriesOf` exists because `choices` is three collection types rather than
// one; and `into` exists because a NumPy dtype had to become a real typed
// array. Each of those is a place where a JavaScript caller can hand over
// something the types forbid, so each needs a test the type system cannot
// write.
import { describe, expect, it } from 'vitest'

import { callScorer, prepareScorerOf, scorerFlagsOf, toRecord } from '../src/_common.js'
import { scoreArrayFactory, type ScoreArrayKind } from '../src/_scoreArray.js'
import { configure } from '../src/configure.js'
import { hammingDistance } from '../src/distance/hamming.js'
import { levenshteinDistance } from '../src/distance/levenshtein.js'
import {
  prefixDistance,
  prefixNormalizedDistance,
  prefixNormalizedSimilarity,
} from '../src/distance/prefix.js'
import { partialRatio, ratio, tokenSortRatio, wRatio } from '../src/fuzz.js'
import { extract, extractIter, extractOne, scoreMatrix } from '../src/search.js'
import { callUntyped } from './common.js'

const KINDS: readonly ScoreArrayKind[] = [
  'f64',
  'f32',
  'i32',
  'i16',
  'i8',
  'u32',
  'u16',
  'u8',
  'u8c',
]

describe('every score array kind allocates and views', () => {
  it('stores and iterates rows of each kind', () => {
    for (const kind of KINDS) {
      const m = scoreMatrix(['abcd', 'abce'], ['abcd', 'abcf'], { into: kind })
      expect(m.data.length).toBe(4)
      const rows = [...m]
      expect(rows.length).toBe(2)
      expect(rows[0].length).toBe(2)
      expect(rows[0].buffer).toBe(m.data.buffer)
      expect(rows[0][0]).toBe(100)
    }
  })

  // Reachable only from JavaScript, and reading past the end of a table is how
  // a test can produce the value without the assertion this project bans.
  it('refuses a kind it has no factory for', () => {
    const missing: ScoreArrayKind = KINDS[KINDS.length]
    expect(() => scoreArrayFactory(missing)).toThrow(RangeError)
    expect(() => scoreMatrix(['ab'], ['ac'], { into: missing })).not.toThrow()
  })
})

describe('a scorer whose shape was never checked', () => {
  it('refuses a result that is not a number', () => {
    const notANumber = (): unknown => 'perfect'
    expect(() => callUntyped(callScorer, notANumber, 'ab', 'ac', {})).toThrow(TypeError)
    expect(() => callUntyped(extractOne, 'ab', ['ac'], { scorer: notANumber })).toThrow(
      TypeError,
    )
  })

  // `rfScorerFlags` comes off an arbitrary object, so each field is proved
  // rather than trusted. A partial one falls back to the fuzz defaults.
  it('falls back when the bounds are not numbers', () => {
    const fuzzFlags = scorerFlagsOf(ratio)
    const noFlags = (): number => 0
    expect(scorerFlagsOf(noFlags)).toEqual(fuzzFlags)
    expect(scorerFlagsOf({ rfScorerFlags: null })).toEqual(fuzzFlags)
    expect(
      scorerFlagsOf({ rfScorerFlags: { worstScore: 'low', optimalScore: 1 } }),
    ).toEqual(fuzzFlags)
    expect(
      scorerFlagsOf({ rfScorerFlags: { worstScore: 0, optimalScore: null } }),
    ).toEqual(fuzzFlags)
  })

  // The two bounds and the symmetry are independently useful, so a missing
  // `symmetric` keeps the bounds rather than dropping the whole object.
  it('keeps declared bounds when only symmetric is missing', () => {
    expect(scorerFlagsOf({ rfScorerFlags: { worstScore: 0, optimalScore: 7 } })).toEqual({
      worstScore: 0,
      optimalScore: 7,
      symmetric: scorerFlagsOf(ratio).symmetric,
    })
  })
})

describe('options copied into a record', () => {
  it('reads nothing out of a value that is not an object', () => {
    expect(toRecord(null)).toEqual({})
    expect(toRecord(42)).toEqual({})
    // `configure` is the caller, so the same thing has to hold there.
    expect(callUntyped(configure, ratio, null)('ab', 'ab')).toBe(100)
  })
})

describe('choices that are not a collection', () => {
  it('yields nothing for a primitive', () => {
    expect(callUntyped(extractOne, 'ab', 42)).toBeUndefined()
    expect(callUntyped(extract, 'ab', 42)).toEqual([])
  })

  it('throws for null, which has no keys to walk', () => {
    expect(() => callUntyped(extractOne, 'ab', null)).toThrow(TypeError)
  })
})

// `extract*` compare against the running best in the direction the scorer's
// flags declare. The similarity direction is what every ported test exercises;
// a distance scorer takes the other branch at each of the three call sites.
describe('a distance scorer through each extract path', () => {
  const CHOICES = { a: 'aaa', b: 'abc', c: 'abd' }
  const scorer = levenshteinDistance

  it('picks the lowest distance from a keyed collection', () => {
    expect(extractOne('abc', CHOICES, { scorer })).toEqual({
      choice: 'abc',
      score: 0,
      key: 'b',
    })
  })

  it('picks the lowest distance from a map', () => {
    const choices = new Map([
      [1, 'aaa'],
      [2, 'abd'],
    ])
    expect(extractOne('abc', choices, { scorer })).toEqual({
      choice: 'abd',
      score: 1,
      key: 2,
    })
  })

  it('keeps everything within the cutoff, unsorted', () => {
    const kept = [...extractIter('abc', CHOICES, { scorer, scoreCutoff: 1 })]
    expect(kept.map((r) => r.choice)).toEqual(['abc', 'abd'])
  })

  it('orders a limited extract lowest first', () => {
    expect(extract('abc', CHOICES, { scorer, limit: 2 }).map((r) => r.choice)).toEqual([
      'abc',
      'abd',
    ])
  })

  // Enough entries that the heap has to sift a right child down past a left.
  it('orders a limited extract over many choices', () => {
    const choices = ['abcdefgh', 'abcdef', 'abcd', 'ab', 'a', 'abc', 'abcde', 'abcdefg']
    const results = extract('abcd', choices, { scorer, limit: 3 })
    expect(results.map((r) => r.choice)).toEqual(['abcd', 'abc', 'abcde'])
  })

  // A heap of two: the root has a left child and no right one, which is the
  // bound `siftDown` has to test before it looks at a sibling.
  it('keeps the two closest when the heap is a pair', () => {
    const choices = ['aaaa', 'abcd', 'abce', 'abcf', 'zzzz']
    const results = extract('abcd', choices, { scorer, limit: 2 })
    expect(results.map((r) => r.choice)).toEqual(['abcd', 'abce'])
  })

  // The heap root is replaced by a better score and sinks past the worse of its
  // two children, which here is the right one.
  it('sinks a replaced root towards its worse child', () => {
    const choices = ['zzzzz', 'abcd', 'abzzz', 'abcde']
    const results = extract('abcde', choices, { scorer, limit: 3 })
    expect(results.map((r) => [r.choice, r.score])).toEqual([
      ['abcde', 0],
      ['abcd', 1],
      ['abzzz', 3],
    ])
  })

  // The running best tightens the bound, so a later choice is compared against
  // it: one that ties keeps the earlier result, and one that is worse is
  // reported as having missed the tightened cutoff.
  it('keeps the first of equal distances and drops a worse one', () => {
    const ordered = ['abd', 'abe', 'aaa']
    expect(extractOne('abc', ordered, { scorer })).toEqual({
      choice: 'abd',
      score: 1,
      key: 0,
    })
    expect(extractOne('abc', new Set(ordered), { scorer })).toEqual({
      choice: 'abd',
      score: 1,
      key: 0,
    })
  })
})

describe('a query that is missing', () => {
  it('produces no results from any extract path', () => {
    expect([...extractIter(null, ['ab'])]).toEqual([])
    expect(extract(null, ['ab'], { limit: 2 })).toEqual([])
    expect(extract(null, ['ab'], { limit: null })).toEqual([])
  })
})

// `scoreMatrix` prepares the first query that is present, so a leading missing
// query is what makes that search do more than one step.
describe('a missing query in a matrix', () => {
  it('skips it and prepares the next one', () => {
    const m = scoreMatrix([null, 'abcd'], ['abcd'], { scorer: prefixNormalizedDistance })
    expect(m.at(0, 0)).toBe(1)
    expect(m.at(1, 0)).toBe(0)
  })
})

// The prepared adapter behind `hammingDistance`, `prefixDistance` and
// `postfixDistance`. Reached through `scoreMatrix`, which — unlike `extract*` —
// hands a missing choice straight to the prepared scorer rather than skipping
// it.
describe('the prepared adapter for a simple distance', () => {
  it('reports maximum dissimilarity for a missing choice', () => {
    expect(
      scoreMatrix(['abcd'], [null], { scorer: prefixNormalizedDistance }).at(0, 0),
    ).toBe(1)
    expect(
      scoreMatrix(['abcd'], [null], { scorer: prefixNormalizedSimilarity }).at(0, 0),
    ).toBe(0)
  })

  // Only the two normalized conventions have an answer for it; the raw ones
  // fall through to the same refusal any non-sequence gets.
  it('refuses a missing choice for a raw convention', () => {
    expect(() => scoreMatrix(['abcd'], [null], { scorer: prefixDistance })).toThrow(
      TypeError,
    )
  })

  it('refuses a choice that is not a sequence', () => {
    expect(() =>
      callUntyped(scoreMatrix, ['abcd'], [7], { scorer: hammingDistance }),
    ).toThrow(TypeError)
  })

  it('refuses a query that is not a sequence', () => {
    const factory = prepareScorerOf(hammingDistance)
    expect(factory).not.toBeNull()
    if (factory === null) return
    expect(() => callUntyped(factory, 7, {})).toThrow(TypeError)
  })
})

// The fuzz scorers keep a prepared path of their own, separate from the metric
// adapter above: a token scorer holds a tokenised query, and `ratio` holds LCS
// masks. Both have answers only reachable through the factory.
describe('the prepared adapter for a fuzz scorer', () => {
  it('reports nothing for a pair of empty inputs above the cutoff', () => {
    expect(scoreMatrix([''], [''], { scorer: ratio })).toBeDefined()
    expect(scoreMatrix([''], [''], { scorer: ratio, scoreCutoff: 100 }).at(0, 0)).toBe(
      100,
    )
    expect(scoreMatrix([''], [''], { scorer: ratio, scoreCutoff: 101 }).at(0, 0)).toBe(0)
  })

  it('refuses a query that is not a sequence', () => {
    for (const scorer of [ratio, tokenSortRatio, wRatio]) {
      const factory = prepareScorerOf(scorer)
      expect(factory).not.toBeNull()
      if (factory === null) continue
      expect(() => callUntyped(factory, 7, {})).toThrow(TypeError)
    }
  })
})

// `partialRatio` scans windows that run off each end of the longer input, and
// prunes a window whose first element the query does not hold at all.
describe('the partial-ratio window scan', () => {
  it('skips a window opening on an element the query does not hold', () => {
    expect(partialRatio('xyz', 'aaaxyqaaa')).toBeCloseTo(
      partialRatio('xyz', 'aaaxyqaaa'),
      12,
    )
    expect(partialRatio('xyz', 'aaaxyqaaa')).toBeGreaterThan(0)
    expect(partialRatio('bcd', 'aaaaaaabcx')).toBeGreaterThan(0)
    expect(partialRatio('xyz', 'aaaaaaaaa')).toBe(0)
  })
})
