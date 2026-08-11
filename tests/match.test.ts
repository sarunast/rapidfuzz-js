// Not ported from RapidFuzz — upstream has no equivalent. Every scorer reports
// a missed `scoreCutoff` by returning a sentinel rather than by saying so, and
// for a similarity that sentinel is `0`, which is also a real score. These two
// helpers ask the question directly.
//
// What needs pinning down is that the direction comes from the scorer's flags
// rather than from the caller, and that the boundaries where a sentinel and a
// genuine score coincide still report the genuine score.
import { describe, expect, it } from 'vitest'

import { configure } from '../src/configure.js'
import { hammingDistance } from '../src/distance/hamming.js'
import { indelNormalizedSimilarity } from '../src/distance/indel.js'
import {
  levenshteinDistance,
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
} from '../src/distance/levenshtein.js'
import { ratio } from '../src/_fuzz/legacy.js'
import { isMatch, matchScore } from '../src/match.js'
import { defaultProcess } from '../src/utils.js'

describe('a similarity reads threshold as a minimum', () => {
  it('returns the score when it is met', () => {
    expect(
      matchScore(ratio, 'this is a test', 'this is a test!', { threshold: 90 }),
    ).toBe(ratio('this is a test', 'this is a test!'))
  })

  it('returns undefined when it is not', () => {
    expect(
      matchScore(ratio, 'this is a test', 'completely different', { threshold: 90 }),
    ).toBeUndefined()
  })

  it('accepts a score exactly at the threshold', () => {
    const score = ratio('this is a test', 'this is a test!')
    expect(
      matchScore(ratio, 'this is a test', 'this is a test!', { threshold: score }),
    ).toBe(score)
  })

  // The case the sentinel cannot express: `ratio` returns 0 both for "missed
  // the cutoff" and for "genuinely nothing in common". At `threshold: 0`
  // everything matches, so the genuine zero has to come back as a score.
  it('reports a genuine zero at threshold zero', () => {
    expect(matchScore(ratio, 'abc', 'xyz', { threshold: 0 })).toBe(0)
    expect(ratio('abc', 'xyz')).toBe(0)
  })

  it('rejects a genuine zero at any positive threshold', () => {
    expect(matchScore(ratio, 'abc', 'xyz', { threshold: 1 })).toBeUndefined()
  })
})

describe('a distance reads threshold as a maximum', () => {
  it('returns the distance when it is within', () => {
    expect(
      matchScore(levenshteinDistance, 'lewenstein', 'levenshtein', { threshold: 3 }),
    ).toBe(2)
  })

  it('returns undefined when it is not', () => {
    expect(
      matchScore(levenshteinDistance, 'lewenstein', 'levenshtein', { threshold: 1 }),
    ).toBeUndefined()
  })

  it('accepts a distance exactly at the threshold', () => {
    expect(
      matchScore(levenshteinDistance, 'lewenstein', 'levenshtein', { threshold: 2 }),
    ).toBe(2)
  })

  // `1` is what a normalized distance returns for a missed cutoff, and also the
  // largest genuine value. At `threshold: 1` nothing can miss.
  it('reports a genuine one at threshold one', () => {
    expect(
      matchScore(levenshteinNormalizedDistance, 'abc', 'xyz', { threshold: 1 }),
    ).toBe(1)
    expect(levenshteinNormalizedDistance('abc', 'xyz')).toBe(1)
  })

  it('handles a normalized similarity as a minimum', () => {
    expect(
      matchScore(levenshteinNormalizedSimilarity, 'abc', 'abd', { threshold: 0.5 }),
    ).toBeCloseTo(levenshteinNormalizedSimilarity('abc', 'abd'), 12)
    expect(
      matchScore(levenshteinNormalizedSimilarity, 'abc', 'xyz', { threshold: 0.5 }),
    ).toBeUndefined()
  })
})

describe('scorer options come along', () => {
  it('forwards an option the scorer owns', () => {
    expect(matchScore(hammingDistance, 'abc', 'ab', { threshold: 5, pad: true })).toBe(1)
    expect(() =>
      matchScore(hammingDistance, 'abc', 'ab', { threshold: 5, pad: false }),
    ).toThrow('Sequences are not the same length.')
  })

  it('forwards a processor', () => {
    expect(
      matchScore(ratio, 'NEW YORK METS', 'new york mets', {
        threshold: 100,
        processor: defaultProcess,
      }),
    ).toBe(100)
  })

  it('works on a configured scorer, direction and all', () => {
    const weighted = configure(levenshteinDistance, { weights: [5, 5, 3] })
    expect(matchScore(weighted, 'abc', 'abd', { threshold: 3 })).toBe(3)
    expect(matchScore(weighted, 'abc', 'abd', { threshold: 2 })).toBeUndefined()
  })
})

describe('isMatch', () => {
  it('is the predicate form of matchScore', () => {
    const pairs: ReadonlyArray<readonly [string, string]> = [
      ['this is a test', 'this is a test!'],
      ['abc', 'xyz'],
      ['', ''],
    ]
    for (const [a, b] of pairs) {
      for (const threshold of [0, 1, 50, 90, 100]) {
        expect(isMatch(ratio, a, b, { threshold })).toBe(
          matchScore(ratio, a, b, { threshold }) !== undefined,
        )
      }
    }
  })

  it('follows the scorer direction too', () => {
    expect(
      isMatch(levenshteinDistance, 'lewenstein', 'levenshtein', { threshold: 3 }),
    ).toBe(true)
    expect(
      isMatch(levenshteinDistance, 'lewenstein', 'levenshtein', { threshold: 1 }),
    ).toBe(false)
  })
})

describe('a scorer with no flags', () => {
  // `scorerFlagsOf` reports 0..100 for anything this package did not build, so
  // a custom scorer is read as a percentage — higher is better.
  it('is read as a percentage', () => {
    const custom = (_s1: string, _s2: string, options: { scoreCutoff?: number } = {}) => {
      const score = 60
      return score >= (options.scoreCutoff ?? 0) ? score : 0
    }
    expect(matchScore(custom, 'a', 'b', { threshold: 50 })).toBe(60)
    expect(matchScore(custom, 'a', 'b', { threshold: 70 })).toBeUndefined()
  })
})

describe('agreement with the underlying scorer', () => {
  // Whatever the helper returns, it must be the number the scorer itself
  // produces for the same cutoff — the helper only decides miss from hit.
  it('never invents a score', () => {
    const pairs: ReadonlyArray<readonly [string, string]> = [
      ['martha', 'marhta'],
      ['kitten', 'sitting'],
      ['', 'abc'],
      ['abc', ''],
      ['', ''],
    ]
    for (const [a, b] of pairs) {
      for (const threshold of [0, 0.25, 0.5, 0.75, 1]) {
        const got = matchScore(indelNormalizedSimilarity, a, b, { threshold })
        if (got !== undefined) {
          expect(got).toBe(indelNormalizedSimilarity(a, b, { scoreCutoff: threshold }))
          expect(got).toBeGreaterThanOrEqual(threshold)
        }
      }
    }
  })
})

// The flags fallback exists so a scorer this package did not build still works
// here, read as a percentage. That promise is only real if such a scorer can be
// passed at all: constraining the options to `ScorerOptions` refused one whose
// options are its own, because `ScorerOptions` is all-optional and TypeScript
// rejects an unrelated type as having no property in common with it.
describe('a scorer this package did not build', () => {
  const custom = (
    a: string,
    b: string,
    options?: { caseSensitive?: boolean },
  ): number => {
    const [x, y] =
      options?.caseSensitive === false ? [a.toLowerCase(), b.toLowerCase()] : [a, b]
    return x === y ? 100 : 0
  }

  it('takes its own options alongside the threshold', () => {
    expect(
      matchScore(custom, 'ABC', 'abc', { threshold: 80, caseSensitive: false }),
    ).toBe(100)
    expect(matchScore(custom, 'ABC', 'abc', { threshold: 80 })).toBeUndefined()
    expect(isMatch(custom, 'ABC', 'abc', { threshold: 80, caseSensitive: false })).toBe(
      true,
    )
  })

  it('takes one with no options at all', () => {
    const bare = (a: string, b: string): number => (a === b ? 100 : 0)
    expect(matchScore(bare, 'a', 'a', { threshold: 80 })).toBe(100)
    expect(matchScore(bare, 'a', 'b', { threshold: 80 })).toBeUndefined()
  })
})
