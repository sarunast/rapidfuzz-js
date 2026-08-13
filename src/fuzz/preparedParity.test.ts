// Not ported from RapidFuzz — upstream has no prepared/unprepared split to
// compare, so this is the invariant that split creates: a prepared query must
// answer exactly what the raw scorer answers, for every cutoff, down to the last
// ULP.
//
// It failed once. `ratio` scaled by 100 and *then* compared with the percentage
// cutoff, where the raw path divides the cutoff by 100 and compares in `[0, 1]`.
// The two are the same algebra and different floating point: 'ceaece' against
// 'caecec' scores 83.33333333333334, whose division back to 0.8333333333333335
// lands one ULP above the 0.8333333333333334 the score normalises to. Best-match
// search raises its cutoff to the running best, so a six-character pair was
// enough to make `extract` disagree with `ratio`.
import { describe, expect, it } from 'vitest'

import { prepareFuzz } from './internal/prepared.js'
import { partialRatio } from './partial.js'
import { prepareSimilarity, ratio } from './similarity.js'
import { tokenRatio } from './token.js'
import { tokenSetRatio } from './tokenSet.js'
import { wRatio } from './weighted.js'

const preparedScore = (
  factory: ReturnType<typeof prepareSimilarity>,
  query: string,
  choice: string,
  cutoff: number,
): number => {
  const preparation = factory({})
  return preparation.prepareQuery(query)(preparation.prepareChoice(choice), cutoff)
}

describe('prepared scorers agree with raw ones', () => {
  it('accepts and rejects the same score at a cutoff of its own value', () => {
    const factory = prepareSimilarity()
    // The pair that found this: six characters each, sharing five of them.
    const score = ratio('ceaece', 'caecec')
    expect(score).toBe(83.33333333333334)
    expect(ratio('ceaece', 'caecec', { scoreCutoff: score })).toBe(0)
    expect(preparedScore(factory, 'ceaece', 'caecec', score)).toBe(0)
  })

  // Constructed rather than drawn at random, because what separates the two
  // arithmetics is only ever the pair `(maximum, lcs)` — the score is
  // `200 * lcs / maximum`, and the question is whether that value survives a
  // round trip through a division by 100. Sharing a prefix of `k` and then
  // diverging into two disjoint alphabets pins the LCS at exactly `k`, so this
  // walks every such pair up to length 24 instead of hoping to draw one. Random
  // sampling does find them, but rarely: one draw in twenty thousand.
  it('matches the raw ratio at a cutoff of the score itself, for every score', () => {
    const factory = prepareSimilarity()

    for (let lengthA = 1; lengthA <= 24; lengthA++) {
      for (let lengthB = 1; lengthB <= 24; lengthB++) {
        for (let shared = 0; shared <= Math.min(lengthA, lengthB); shared++) {
          const a = 'a'.repeat(shared) + 'b'.repeat(lengthA - shared)
          const b = 'a'.repeat(shared) + 'c'.repeat(lengthB - shared)
          const base = ratio(a, b)

          for (const cutoff of [0, base - 1e-13, base, base + 1e-13, 100]) {
            expect(preparedScore(factory, a, b, cutoff), `${a} / ${b} @ ${cutoff}`).toBe(
              ratio(a, b, { scoreCutoff: cutoff }),
            )
          }
        }
      }
    }
  })

  // The length ratios wRatio branches on — under 1.5, exactly 1.5, up to 8 and
  // past it — crossed with the cutoffs its scale factors turn into impossible
  // ones. A cutoff of 90 divided by the 0.9 partial scale is exactly 100, and
  // one of 95 divided by the 0.95 token scale is exactly 100: both sit on the
  // `> 100` exits rather than inside them, which is where an off-by-one in a
  // shortcut would show.
  it('matches the raw composite scorers across branches and cutoffs', () => {
    const pairs: readonly (readonly [string, string])[] = [
      ['new york mets', 'new york mets'],
      ['new york mets', 'new YORK mets'],
      ['mets new york', 'new york mets'],
      ['alphabetagamma', 'alphabetagammz'],
      ['ceaece', 'caecec'],
      ['abc', 'abcabc'],
      ['abcd', 'abcdef'],
      ['a b', 'a b c d e f g h i'],
      ['grizzly', 'grizzly bear roaming the wide open tundra all afternoon long'],
      ['😀😀 alpha', '😀 alpha beta'],
      ['😀', '😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀'],
      ['x', 'x'.repeat(200)],
      ['', 'abc'],
      ['abc', ''],
    ]
    const cutoffs = [0, 60, 90, 90 + 1e-13, 95, 95 + 1e-13, 100, 100 + 1e-13]
    const scorers = [
      ['wRatio', wRatio, prepareFuzz('wRatio')],
      ['partialRatio', partialRatio, prepareFuzz('partialRatio')],
      ['tokenRatio', tokenRatio, prepareFuzz('tokenRatio')],
      ['tokenSetRatio', tokenSetRatio, prepareFuzz('tokenSetRatio')],
    ] as const

    for (const [name, raw, factory] of scorers) {
      for (const [a, b] of pairs) {
        for (const cutoff of cutoffs) {
          const expected = raw(a, b, { scoreCutoff: cutoff })
          expect(
            preparedScore(factory, a, b, cutoff),
            `${name}: ${a} / ${b} @ ${cutoff}`,
          ).toBe(expected)
          // Both orders: wRatio's branches are asymmetric in which side is the
          // needle, and the prepared path holds only one of the two.
          expect(
            preparedScore(factory, b, a, cutoff),
            `${name}: ${b} / ${a} @ ${cutoff}`,
          ).toBe(raw(b, a, { scoreCutoff: cutoff }))
        }
      }
    }
  })
})
