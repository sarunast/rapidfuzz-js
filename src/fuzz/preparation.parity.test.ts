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

import { fuzzPartialRatio } from './partialRatio.js'
import { prepareFuzz } from './preparation.js'
import { prepareRatio, fuzzRatio } from './ratio.js'
import { fuzzPartialTokenRatio } from './token/partialTokenRatio.js'
import { fuzzPartialTokenSetRatio } from './token/partialTokenSetRatio.js'
import { fuzzPartialTokenSortRatio } from './token/partialTokenSortRatio.js'
import { fuzzTokenRatio } from './token/tokenRatio.js'
import { fuzzTokenSetRatio } from './token/tokenSetRatio.js'
import { prepareTokenSort, fuzzTokenSortRatio } from './token/tokenSortRatio.js'
import { fuzzWeightedRatio } from './weightedRatio.js'

const preparedScore = (
  factory: ReturnType<typeof prepareRatio>,
  query: string,
  choice: string,
  cutoff: number,
): number => {
  const preparation = factory({})
  return preparation.prepareQuery(query)(preparation.prepareChoice(choice), cutoff)
}

describe('prepared scorers agree with raw ones', () => {
  it('accepts and rejects the same score at a cutoff of its own value', () => {
    const factory = prepareRatio()
    // The pair that found this: six characters each, sharing five of them.
    const score = fuzzRatio('ceaece', 'caecec')
    expect(score).toBe(83.33333333333334)
    expect(fuzzRatio('ceaece', 'caecec', { scoreCutoff: score })).toBe(0)
    expect(preparedScore(factory, 'ceaece', 'caecec', score)).toBe(0)
  })

  // Constructed rather than drawn at random, because what separates the two
  // arithmetics is only ever the pair `(maximum, lcs)` — the score is
  // `200 * lcs / maximum`, and the question is whether that value survives a
  // round trip through a division by 100. Sharing a prefix of `k` and then
  // diverging into two disjoint alphabets pins the LCS at exactly `k`, so this
  // walks every such pair up to length 24 instead of hoping to draw one. Random
  // sampling does find them, but rarely: one draw in twenty thousand.
  it('matches the raw fuzzRatio at a cutoff of the score itself, for every score', () => {
    const factory = prepareRatio()

    for (let lengthA = 1; lengthA <= 24; lengthA++) {
      for (let lengthB = 1; lengthB <= 24; lengthB++) {
        for (let shared = 0; shared <= Math.min(lengthA, lengthB); shared++) {
          const a = 'a'.repeat(shared) + 'b'.repeat(lengthA - shared)
          const b = 'a'.repeat(shared) + 'c'.repeat(lengthB - shared)
          const base = fuzzRatio(a, b)

          for (const cutoff of [0, base - 1e-13, base, base + 1e-13, 100]) {
            expect(preparedScore(factory, a, b, cutoff), `${a} / ${b} @ ${cutoff}`).toBe(
              fuzzRatio(a, b, { scoreCutoff: cutoff }),
            )
          }
        }
      }
    }
  })

  // Prepared token-set scoring takes a separate route when the two token sets
  // share nothing: both sides collapse to their sorted unique joins and the
  // query's bit-parallel pattern is reused across choices. These pairs walk
  // that route's branches — the bounded kernel above 128 joined elements at a
  // cutoff of 70, its unreachable-requirement exit, the length gate, the
  // below-cutoff distance exit, and duplicate tokens on either side, which
  // join through the deduplicated path rather than the sorted one.
  it('matches the raw scorer when the token sets are disjoint', () => {
    const factory = prepareFuzz('tokenSetRatio')
    const pairs: readonly (readonly [string, string])[] = [
      [`q${'a'.repeat(70)}`, `c${'a'.repeat(70)}`],
      ['a'.repeat(70), 'z'.repeat(70)],
      ['ab', 'z'.repeat(200)],
      ['foo foo bar', 'baz baz qux'],
      ['alpha beta', '😀 😀 gamma'],
      ['a'.repeat(10), 'z'.repeat(10)],
      ['alpha beta', 'gamma delta epsilon'],
    ]

    for (const [a, b] of pairs) {
      // The score's own value and one ULP either side sit on the exact
      // accept/reject edge — the place a fast path that rebuilt the score
      // through different arithmetic would first disagree.
      const base = fuzzTokenSetRatio(a, b)
      for (const cutoff of [0, 60, 70, 90, 98, 100, base - 1e-13, base, base + 1e-13]) {
        const expected = fuzzTokenSetRatio(a, b, { scoreCutoff: cutoff })
        expect(preparedScore(factory, a, b, cutoff), `${a} / ${b} @ ${cutoff}`).toBe(
          expected,
        )
        expect(preparedScore(factory, b, a, cutoff), `${b} / ${a} @ ${cutoff}`).toBe(
          fuzzTokenSetRatio(b, a, { scoreCutoff: cutoff }),
        )
      }
    }

    // The concrete shape the bounded kernel accepts: disjoint tokens whose
    // characters still align, so the distance stays inside a 70 cutoff.
    expect(preparedScore(factory, `q${'a'.repeat(70)}`, `c${'a'.repeat(70)}`, 70)).toBe(
      fuzzTokenSetRatio(`q${'a'.repeat(70)}`, `c${'a'.repeat(70)}`),
    )
  })

  // A retained choice changes representation as it is reused: its canonical
  // joins build as code-point arrays on the first query, repack into BMP
  // strings on the second, and stay packed from the third on. A Matcher holds
  // its prepared choices for its lifetime, so this is the state machine every
  // matcher corpus walks — and the per-pair helper above never leaves the
  // first state, because it prepares a fresh choice for every assertion. Three
  // passes over one handle pin all three states against the raw scorer, for
  // every token scorer, including the shapes that refuse to pack (astral),
  // deduplicate first (repeated tokens), or chunk the pack (a 1500-element
  // token).
  it('answers identically while a retained choice repacks across queries', () => {
    const scorers = [
      ['tokenSortRatio', fuzzTokenSortRatio, prepareTokenSort()],
      ['tokenSetRatio', fuzzTokenSetRatio, prepareFuzz('tokenSetRatio')],
      ['tokenRatio', fuzzTokenRatio, prepareFuzz('tokenRatio')],
      [
        'partialTokenSortRatio',
        fuzzPartialTokenSortRatio,
        prepareFuzz('partialTokenSortRatio'),
      ],
      [
        'partialTokenSetRatio',
        fuzzPartialTokenSetRatio,
        prepareFuzz('partialTokenSetRatio'),
      ],
      ['partialTokenRatio', fuzzPartialTokenRatio, prepareFuzz('partialTokenRatio')],
      ['weightedRatio', fuzzWeightedRatio, prepareFuzz('weightedRatio')],
    ] as const
    const choices = [
      'grizzly bear roaming the tundra',
      'beta beta alpha',
      '😀😀 alpha beta',
      `${'x'.repeat(1500)} alpha`,
      'delta',
    ]
    const queries = ['bear roaming tundra', 'alpha beta', 'gamma delta epsilon']

    for (const [name, raw, factory] of scorers) {
      const preparation = factory({})
      for (const choiceText of choices) {
        const handle = preparation.prepareChoice(choiceText)
        for (let pass = 1; pass <= 3; pass++) {
          for (const query of queries) {
            const kernel = preparation.prepareQuery(query)
            for (const cutoff of [0, 60, 90]) {
              expect(
                kernel(handle, cutoff),
                `${name}: ${query} / ${choiceText.slice(0, 24)} @ ${cutoff}, pass ${pass}`,
              ).toBe(raw(query, choiceText, { scoreCutoff: cutoff }))
            }
          }
        }
      }
    }
  })

  // The length ratios weightedRatio branches on — under 1.5, exactly 1.5, up to 8 and
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
      ['weightedRatio', fuzzWeightedRatio, prepareFuzz('weightedRatio')],
      ['partialRatio', fuzzPartialRatio, prepareFuzz('partialRatio')],
      ['tokenRatio', fuzzTokenRatio, prepareFuzz('tokenRatio')],
      ['tokenSetRatio', fuzzTokenSetRatio, prepareFuzz('tokenSetRatio')],
    ] as const

    for (const [name, raw, factory] of scorers) {
      for (const [a, b] of pairs) {
        for (const cutoff of cutoffs) {
          const expected = raw(a, b, { scoreCutoff: cutoff })
          expect(
            preparedScore(factory, a, b, cutoff),
            `${name}: ${a} / ${b} @ ${cutoff}`,
          ).toBe(expected)
          // Both orders: weightedRatio's branches are asymmetric in which side is the
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
