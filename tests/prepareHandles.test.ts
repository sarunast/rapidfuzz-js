// Not ported from RapidFuzz — `prepareQuery` and `prepareChoice` have no
// upstream counterpart. RapidFuzz caches the query side of a scorer inside
// `process`; nothing there hands a caller either half.
//
// Both are performance APIs with a correctness contract, and the contract is
// exactly one sentence: a handle scores what the scorer would have scored, in
// the order `scorer(query, choice)`, whichever half is prepared and whichever
// half is raw. So the bulk of this file is a differential against a direct call
// — four call shapes, every scorer shape the package can produce, every
// combination of the two per-call options — and the rest covers what a handle
// refuses.
//
// Two of the scorers below are deliberately **asymmetric**. Without one, every
// assertion here passes against an implementation that swapped its operands,
// which is the one mistake `prepareChoice` is most able to make: it holds the
// side that is conventionally second.
import { describe, expect, it } from 'vitest'

import {
  SIMILARITY_FLAGS,
  withPreparedFlags,
  type PreparedScore,
  type Processor,
  type Sequence,
} from '../src/_common.js'
import { configure } from '../src/configure.js'
import {
  levenshteinDistance,
  levenshteinNormalizedSimilarity,
} from '../src/distance/levenshtein.js'
import {
  partialRatio,
  ratio,
  tokenSetRatio,
  tokenSortRatio,
  wRatio,
} from '../src/fuzz.js'
import { isMatch, matchScore } from '../src/match.js'
import {
  extract,
  extractIter,
  extractOne,
  prepareChoice,
  prepareChoices,
  prepareQuery,
  scoreMatrix,
  scorePairs,
  type PreparedCallOptions,
  type SearchScorer,
} from '../src/search.js'
import { defaultProcess } from '../src/utils.js'
import { callUntyped } from './common.js'

const CHOICES = [
  'new york mets',
  'new YORK mets',
  'the wonders',
  'atlanta braves',
  'dallas cowboys',
  'new york jets',
] as const

const QUERIES = ['new york mets', 'cowboys', 'nothing alike at all', ''] as const

/**
 * Every shape a scorer can arrive in, which is what decides which internal path
 * a handle takes: a factory with a choice hook, a factory without one, no
 * factory at all, and a scorer this package never saw.
 *
 * The third column is a bound in that scorer's own convention, because a cutoff
 * is not one number for all four: a normalized scorer range-checks it against
 * `0.0 - 1.0` and refuses `50` outright.
 */
const SCORERS: readonly (readonly [string, SearchScorer, number])[] = [
  ['wRatio', wRatio, 50],
  ['ratio', ratio, 50],
  ['partialRatio', partialRatio, 50],
  ['tokenSortRatio', tokenSortRatio, 50],
  ['tokenSetRatio', tokenSetRatio, 50],
  ['levenshteinDistance', levenshteinDistance, 8],
  ['levenshteinNormalizedSimilarity', levenshteinNormalizedSimilarity, 0.5],
  // A third-party scorer: no prepared factory, so both halves are held as the
  // processed sequence and the call goes through the generic path.
  ['a plain function', (a: string, b: string): number => (a === b ? 100 : 0), 50],
  // Configured with a baked-in processor, which is the built-in that registers
  // no factory at all.
  ['configured with a processor', configure(ratio, { processor: defaultProcess }), 50],
  // Configured and still preparable, so it keeps the choice hook its factory
  // carries.
  ['configured with weights', configure(levenshteinDistance, { weights: [1, 1, 2] }), 8],
  ['configured twice', configure(configure(tokenSortRatio, {}), {}), 50],
  // Asymmetric, both ways it can happen. `weights: [1, 2, 1]` is insertion 1,
  // deletion 2 — `configuredFlagsOf` resolves that to `symmetric: false`, and it
  // is the only asymmetric scorer this package can build. The plain function is
  // the third-party half of the same question, where nothing declares anything.
  //
  // `weights: [1, 1, 2]` above is **not** a substitute: insertion equals
  // deletion there, so weighted Levenshtein is symmetric and a reversed operand
  // order would score it identically.
  [
    'configured asymmetrically',
    configure(levenshteinDistance, { weights: [1, 2, 1] }),
    8,
  ],
  [
    'an asymmetric plain function',
    (a: string, b: string): number => (a.startsWith(b) ? 100 : 0),
    50,
  ],
]

/**
 * The two bounds a handle still takes per call, plus the two ways of giving
 * none. `undefined` and `{}` are not the same call: the first is the only one
 * that reaches a third-party scorer as a two-argument call.
 */
function optionSets(
  bound: number,
): readonly (readonly [string, PreparedCallOptions | undefined])[] {
  return [
    ['no options', undefined],
    ['empty options', {}],
    ['a cutoff', { scoreCutoff: bound }],
    ['a hint', { scoreHint: bound }],
    ['both', { scoreCutoff: bound, scoreHint: bound }],
  ]
}

/**
 * What a direct call looks like for the same options — the oracle every
 * differential below compares against.
 *
 * Spelled with both keys present, because that is what a handle sends: absence
 * is `undefined` on both sides, so the only difference a scorer could observe
 * is the one the `undefined` row exists to pin — no options argument at all.
 */
function direct(
  scorer: SearchScorer,
  query: Sequence,
  choice: Sequence,
  options: PreparedCallOptions | undefined,
): number {
  if (options === undefined) return callUntyped(scorer, query, choice)
  return callUntyped(scorer, query, choice, {
    scoreCutoff: options.scoreCutoff,
    scoreHint: options.scoreHint,
  })
}

describe('a prepared handle scores what a direct call scores', () => {
  for (const [scorerName, scorer, bound] of SCORERS) {
    it(`${scorerName}, every query, choice and option set`, () => {
      for (const query of QUERIES) {
        const pq = prepareQuery(query, { scorer })
        for (const choice of CHOICES) {
          const pc = prepareChoice(choice, { scorer })
          for (const [, options] of optionSets(bound)) {
            const expected = direct(scorer, query, choice, options)
            // D1: prepared query, raw choice.
            expect(pq(choice, options)).toBe(expected)
            // D2: prepared choice, raw query — the shape that catches a
            // reversed operand order.
            expect(pc(query, options)).toBe(expected)
            // D3 and D4: both halves prepared, composed from either side.
            expect(pq(pc, options)).toBe(expected)
            expect(pc(pq, options)).toBe(expected)
          }
        }
      }
    })
  }
})

describe('a prepared handle applies its processor exactly once', () => {
  for (const [scorerName, scorer, bound] of SCORERS) {
    it(`${scorerName}, on both halves`, () => {
      const options = { scorer, processor: defaultProcess }
      for (const query of QUERIES) {
        const pq = prepareQuery(query, options)
        for (const choice of CHOICES) {
          const pc = prepareChoice(choice, options)
          for (const [, callOptions] of optionSets(bound)) {
            const expected = direct(
              scorer,
              defaultProcess(query),
              defaultProcess(choice),
              callOptions,
            )
            expect(pq(choice, callOptions)).toBe(expected)
            expect(pc(query, callOptions)).toBe(expected)
            expect(pq(pc, callOptions)).toBe(expected)
            expect(pc(pq, callOptions)).toBe(expected)
          }
        }
      }
    })
  }

  // `defaultProcess` is idempotent, so it cannot tell "applied once" from
  // "applied twice". This one can.
  it('and not twice, for a processor that is not idempotent', () => {
    const bang: Processor = (s) => (typeof s === 'string' ? `${s}!` : s)
    const options = { scorer: ratio, processor: bang }
    const pq = prepareQuery('abc', options)
    const pc = prepareChoice('abd', options)
    const expected = ratio('abc!', 'abd!')

    expect(expected).not.toBe(ratio('abc!!', 'abd!!'))
    expect(pq('abd')).toBe(expected)
    expect(pc('abc')).toBe(expected)
    expect(pq(pc)).toBe(expected)
    expect(pc(pq)).toBe(expected)
  })
})

describe('a handle is reusable, and does not drift', () => {
  it('one prepared query over every choice, one prepared choice over every query', () => {
    for (const [, scorer] of SCORERS) {
      const pq = prepareQuery(QUERIES[0], { scorer })
      const pc = prepareChoice(CHOICES[0], { scorer })

      // Twice through, so a handle that mutated its own state on first use
      // would disagree with itself on the second pass.
      for (let pass = 0; pass < 2; pass++) {
        for (const choice of CHOICES) {
          expect(pq(choice)).toBe(direct(scorer, QUERIES[0], choice, undefined))
        }
        for (const query of QUERIES) {
          expect(pc(query)).toBe(direct(scorer, query, CHOICES[0], undefined))
        }
      }
    }
  })
})

describe('a third-party scorer sees the call it would have got', () => {
  // The rule this pins: a handle with no call options is standing in for
  // `scorer(q, c)`, so it makes that call — two arguments, not three with two
  // `undefined` fields. Observable to any scorer that tests its options
  // argument, which is why it is a decision and not an implementation detail.
  const arity = (_a: string, _b: string, options?: PreparedCallOptions): number =>
    options === undefined ? 1 : 2

  it('two arguments when the handle is called with none', () => {
    const pq = prepareQuery('abc', { scorer: arity })
    const pc = prepareChoice('abd', { scorer: arity })

    expect(pq('abd')).toBe(1)
    expect(pc('abc')).toBe(1)
    expect(pq(pc)).toBe(1)
    expect(pc(pq)).toBe(1)
  })

  it('three when it is called with any, including an empty object', () => {
    const pq = prepareQuery('abc', { scorer: arity })
    const pc = prepareChoice('abd', { scorer: arity })

    for (const options of [{}, { scoreCutoff: 50 }, { scoreHint: 50 }]) {
      expect(pq('abd', options)).toBe(2)
      expect(pc('abc', options)).toBe(2)
      expect(pq(pc, options)).toBe(2)
      expect(pc(pq, options)).toBe(2)
    }
  })

  it('and the options object it gets is the one the caller passed', () => {
    // Not just "three arguments": the third one is the object that was passed
    // in, unchanged. A rebuilt `{ scoreCutoff, scoreHint }` would carry two
    // `undefined` fields a caller never wrote, and `Object.keys`, a getter and
    // an identity test can all tell the difference.
    const seen: unknown[] = []
    const record = (_a: string, _b: string, options?: PreparedCallOptions): number => {
      seen.push(options)
      return options === undefined ? 0 : Object.keys(options).length
    }

    const pq = prepareQuery('abc', { scorer: record })
    const pc = prepareChoice('abd', { scorer: record })

    const empty = {}
    expect(pq('abd', empty)).toBe(0)
    expect(seen.at(-1)).toBe(empty)

    const cutoffOnly = { scoreCutoff: 50 }
    expect(pq('abd', cutoffOnly)).toBe(1)
    expect(seen.at(-1)).toBe(cutoffOnly)

    // The same on the choice side and through both composed directions, since
    // each of the four reaches the seam by its own route.
    expect(pc('abc', empty)).toBe(0)
    expect(seen.at(-1)).toBe(empty)
    expect(pq(pc, cutoffOnly)).toBe(1)
    expect(seen.at(-1)).toBe(cutoffOnly)
    expect(pc(pq, empty)).toBe(0)
    expect(seen.at(-1)).toBe(empty)
  })

  it('and a getter on those options is read by the scorer, not by us', () => {
    // The sharpest form of the same rule: a rebuilt object would have called
    // this getter while copying, whether or not the scorer ever looked.
    let reads = 0
    const options = {
      get scoreCutoff(): number {
        reads++
        return 50
      },
    }

    const ignores = (): number => 0
    const pq = prepareQuery('abc', { scorer: ignores })
    expect(pq('abd', options)).toBe(0)
    expect(reads).toBe(0)

    const uses = (_a: string, _b: string, o?: PreparedCallOptions): number =>
      o?.scoreCutoff ?? 0
    expect(prepareQuery('abc', { scorer: uses })('abd', options)).toBe(50)
    expect(reads).toBe(1)
  })

  it('and its result is still checked', () => {
    const notANumber = (): unknown => 'perfect'
    const pq = callUntyped(prepareQuery, 'ab', { scorer: notANumber })
    const pc = callUntyped(prepareChoice, 'ac', { scorer: notANumber })

    // Both arms of the call seam: no options is the two-argument path, options
    // is the three-argument one.
    expect(() => pq('ac')).toThrow(TypeError)
    expect(() => pq('ac', {})).toThrow(TypeError)
    expect(() => pc('ab')).toThrow(TypeError)
    expect(() => pc('ab', {})).toThrow(TypeError)
  })
})

describe('a handle for a scorer that prepares a query but not a choice', () => {
  // Every scorer this package ships attaches a per-choice hook to its prepared
  // factory, so the branch where a factory offers none needs a scorer that does
  // not — the same construction `prepareChoices.test.ts` uses, for the same
  // reason.
  const halves = (a: string, b: string): number => (a === b ? 100 : 0)
  const scorer = withPreparedFlags(halves, SIMILARITY_FLAGS, (query) => {
    const score: PreparedScore = (choice) => (choice === query ? 100 : 0)
    return score
  })

  it('holds the choice unprepared and still scores it', () => {
    const pq = prepareQuery('abc', { scorer })
    const pc = prepareChoice('abc', { scorer })
    const other = prepareChoice('abd', { scorer })

    expect(pq('abc')).toBe(100)
    expect(pq('abd')).toBe(0)
    expect(pc('abc')).toBe(100)
    expect(pq(pc)).toBe(100)
    expect(pq(other)).toBe(0)
    expect(pc(pq)).toBe(100)
  })
})

describe('a handle defaults to what search does', () => {
  it('wRatio and no processor', () => {
    const pq = prepareQuery('new york mets')
    const pc = prepareChoice('new york mets')

    expect(pq.scorer).toBe(wRatio)
    expect(pq.processor).toBeNull()
    expect(pc.scorer).toBe(wRatio)
    expect(pc.processor).toBeNull()
    expect(pq('new york jets')).toBe(wRatio('new york mets', 'new york jets'))
    expect(pc('new york jets')).toBe(wRatio('new york jets', 'new york mets'))
  })

  it('and an omitted processor composes with an explicit undefined one', () => {
    const pq = prepareQuery('abc', { scorer: ratio, processor: undefined })
    const pc = prepareChoice('abd', { scorer: ratio })

    expect(pq.processor).toBeNull()
    expect(pq(pc)).toBe(ratio('abc', 'abd'))
  })
})

describe('a handle refuses a partner it was not built to meet', () => {
  const pq = prepareQuery('abc', { scorer: ratio })
  const pc = prepareChoice('abd', { scorer: ratio })

  it('a different scorer, from either side', () => {
    expect(() => pq(prepareChoice('abd', { scorer: tokenSortRatio }))).toThrow(
      /scorer differs/,
    )
    expect(() => pc(prepareQuery('abc', { scorer: tokenSortRatio }))).toThrow(
      /scorer differs/,
    )
    expect(() => pq(prepareChoice('abd', { scorer: tokenSortRatio }))).toThrow(TypeError)
  })

  it('a different processor, from either side and in both directions', () => {
    const processed = { scorer: ratio, processor: defaultProcess }
    // One side has a processor and the other has none.
    expect(() => pq(prepareChoice('abd', processed))).toThrow(/processor differs/)
    expect(() => pc(prepareQuery('abc', processed))).toThrow(/processor differs/)
    // And the other way round, which the index's own check deliberately
    // tolerates and this one does not: a query normalised by a processor scored
    // against a choice that was not is a wrong number with nothing to see.
    expect(() => prepareQuery('abc', processed)(pc)).toThrow(/processor differs/)
    expect(() => prepareChoice('abd', processed)(pq)).toThrow(/processor differs/)
  })

  it('but takes the same scorer and processor named again', () => {
    const processed = { scorer: ratio, processor: defaultProcess }
    expect(prepareQuery('abc', processed)(prepareChoice('abd', processed))).toBe(
      ratio('abc', 'abd'),
    )
  })
})

describe('a handle that was copied rather than passed', () => {
  // `Object.assign` over a handle copies the brand — symbol-keyed own
  // properties are copied — along with `scorer` and `processor`, so the copy
  // type-checks and passes both identity tests. What it cannot copy is an entry
  // in a table keyed by the original.
  //
  // Note what is *not* asserted here: that calling a copy throws. Calling a copy
  // runs the copy's own body, and no code of this library's runs at all — so the
  // contract is that a forged handle is refused wherever this library consumes
  // one, which is what the two cases below are.
  const pq = prepareQuery('abc', { scorer: ratio })
  const pc = prepareChoice('abd', { scorer: ratio })

  it('is refused as an argument to the real one', () => {
    expect(() => pq(Object.assign(() => 0, pc))).toThrow(/cannot be copied/)
    expect(() => pc(Object.assign(() => 0, pq))).toThrow(/cannot be copied/)
  })

  it('is refused before the forged body could run', () => {
    // The forged query returns a number, so a check that ran after the call
    // would hand back `123` instead of throwing.
    let called = 0
    const forged = Object.assign(() => {
      called++
      return 123
    }, pq)
    expect(() => pc(forged)).toThrow(/cannot be copied/)
    expect(called).toBe(0)
  })

  it('leaves the originals working', () => {
    expect(pq(pc)).toBe(ratio('abc', 'abd'))
  })
})

describe('a handle refuses an operand that is not a sequence', () => {
  const missing = [null, undefined, Number.NaN, 42, {}, () => 0]

  it('at build time, unlike extract, which drops missing values', () => {
    for (const value of missing) {
      expect(() => callUntyped(prepareQuery, value, {})).toThrow(
        /expected a string or an array-like sequence/,
      )
      expect(() => callUntyped(prepareChoice, value, {})).toThrow(TypeError)
    }
  })

  it('at call time too', () => {
    const pq = prepareQuery('abc', { scorer: ratio })
    const pc = prepareChoice('abd', { scorer: ratio })
    for (const value of missing) {
      expect(() => callUntyped(pq, value)).toThrow(TypeError)
      expect(() => callUntyped(pc, value)).toThrow(TypeError)
    }
  })

  it('including a plain function, which has a length and is not a sequence', () => {
    const pq = prepareQuery('abc', { scorer: ratio })
    expect(() => callUntyped(pq, (a: string) => a)).toThrow(
      /expected a string or an array-like sequence/,
    )
  })

  it('but takes an array of values, like every other sequence', () => {
    const pq = prepareQuery([1, 2, 3], { scorer: ratio })
    const pc = prepareChoice([1, 2, 4], { scorer: ratio })
    const expected = ratio([1, 2, 3], [1, 2, 4])

    expect(pq([1, 2, 4])).toBe(expected)
    expect(pc([1, 2, 3])).toBe(expected)
    expect(pq(pc)).toBe(expected)
    expect(pc(pq)).toBe(expected)
  })
})

describe('a handle cannot be edited after it is built', () => {
  const pq = prepareQuery('abc', { scorer: ratio })
  const pc = prepareChoice('abd', { scorer: ratio })

  it('is frozen', () => {
    expect(Object.isFrozen(pq)).toBe(true)
    expect(Object.isFrozen(pc)).toBe(true)
  })

  it('refuses a swapped scorer or processor', () => {
    expect(Reflect.set(pq, 'scorer', tokenSortRatio)).toBe(false)
    expect(Reflect.set(pq, 'processor', defaultProcess)).toBe(false)
    expect(Reflect.set(pc, 'scorer', tokenSortRatio)).toBe(false)
    expect(Reflect.set(pc, 'processor', defaultProcess)).toBe(false)
  })

  it('still reports what it was built for', () => {
    expect(pq.scorer).toBe(ratio)
    expect(pc.scorer).toBe(ratio)
    expect(pq(pc)).toBe(ratio('abc', 'abd'))
  })
})

describe('a handle over a distance scorer keeps the direction and the cutoff', () => {
  const pq = prepareQuery('abc', { scorer: levenshteinDistance })

  it('reports the distance, and the sentinel past the cutoff', () => {
    expect(pq('abc')).toBe(0)
    expect(pq('abcd')).toBe(1)
    expect(pq('xyz')).toBe(3)
    // The cutoff convention is the scorer's own: a distance past the bound
    // comes back as `bound + 1`.
    expect(pq('xyz', { scoreCutoff: 1 })).toBe(2)
    expect(pq('abcd', { scoreCutoff: 1 })).toBe(1)
  })
})

describe('a prepared choice and a prepared index prepare the same way', () => {
  // Not a comparison of `prepareQuery` against `prepareChoices` — those two
  // prepare different halves and are meant to differ. This asserts the narrower
  // thing that has to hold: the singular choice preparation and the index's are
  // one mechanism, so a choice prepared either way scores identically.
  const choices = [...CHOICES]

  for (const [scorerName, scorer] of SCORERS) {
    it(`${scorerName}, singular against the index`, () => {
      for (const processor of [undefined, defaultProcess]) {
        const options = { scorer, processor }
        const index = prepareChoices(choices, options)

        for (const query of QUERIES) {
          const pq = prepareQuery(query, options)
          for (const choice of choices) {
            expect(pq(prepareChoice(choice, options))).toBe(pq(choice))
          }

          // And the index's own scores are those same numbers.
          const scored = new Map(
            extract(query, index, { limit: null }).map((r) => [r.key, r.score]),
          )
          for (const [key, score] of scored) {
            expect(pq(prepareChoice(choices[Number(key)], options))).toBe(score)
          }
        }
      }
    })
  }
})

describe('a prepared handle is not a scorer, and is refused as one', () => {
  // A handle is a function, so it satisfies the erased scorer type
  // contravariantly and type-checks wherever a scorer is expected. Passed
  // there, `callScorer` would hand it the *choice* as its options bag and read
  // its flags off nothing — an exact match scores `0`, silently. So every seam
  // that resolves a scorer refuses one, once per call.
  const pq = prepareQuery('abc', { scorer: ratio })
  const pc = prepareChoice('abc', { scorer: ratio })
  const refused = /cannot be used as a scorer/

  for (const [handleName, handle] of [
    ['a prepared query', pq],
    ['a prepared choice', pc],
  ] as const) {
    it(`${handleName}, at every scorer seam`, () => {
      expect(() => extract('abc', ['abc'], { scorer: handle })).toThrow(refused)
      expect(() => extract('abc', ['abc'], { scorer: handle, limit: null })).toThrow(
        refused,
      )
      expect(() => extract('abc', ['abc'], { scorer: handle, limit: 0 })).toThrow(refused)
      expect(() => extractOne('abc', ['abc'], { scorer: handle })).toThrow(refused)
      expect(() => [...extractIter('abc', ['abc'], { scorer: handle })]).toThrow(refused)
      expect(() => scoreMatrix(['abc'], ['abc'], { scorer: handle })).toThrow(refused)
      expect(() => scorePairs(['abc'], ['abc'], { scorer: handle })).toThrow(refused)
      expect(() => prepareChoices(['abc'], { scorer: handle })).toThrow(refused)
      expect(() => prepareQuery('abc', { scorer: handle })).toThrow(refused)
      expect(() => prepareChoice('abc', { scorer: handle })).toThrow(refused)
    })

    it(`${handleName}, as the query to extract`, () => {
      // Not `expected a string or an array-like sequence`: a handle passed here
      // is a caller who wanted `extract(preparedQuery, choices)` to work, and
      // saying so names the thing to do instead.
      const asQuery = /call the handle/
      expect(() => callUntyped(extract, handle, ['abc'], {})).toThrow(asQuery)
      expect(() => callUntyped(extractOne, handle, ['abc'], {})).toThrow(asQuery)
      expect(() => [...callUntyped(extractIter, handle, ['abc'], {})]).toThrow(asQuery)
    })
  }

  // These two are reached through `callUntyped` because TypeScript already
  // refuses them: both take the scorer by value rather than in an options bag,
  // so `I` is inferred from it and has to serve *both* operands — and a
  // handle's second parameter is its call options, not another sequence. The
  // erased `SearchScorer` in the seams above has no such constraint, which is
  // exactly why they need a run time check and these need one only for a
  // JavaScript caller.
  it('by configure, before it wraps either one', () => {
    expect(() => callUntyped(configure, pq, {})).toThrow(refused)
    expect(() => callUntyped(configure, pc, {})).toThrow(refused)
  })

  it('by matchScore, which isMatch delegates to', () => {
    const threshold = { threshold: 50 }
    expect(() => callUntyped(matchScore, pq, 'abc', 'abc', threshold)).toThrow(refused)
    expect(() => callUntyped(matchScore, pc, 'abc', 'abc', threshold)).toThrow(refused)
    expect(() => callUntyped(isMatch, pq, 'abc', 'abc', threshold)).toThrow(refused)
  })

  it('leaves an ordinary function alone', () => {
    const plain = (a: string, b: string): number => (a === b ? 100 : 0)
    expect(extractOne('abc', ['abc'], { scorer: plain })).toEqual({
      choice: 'abc',
      score: 100,
      key: 0,
    })
    expect(prepareQuery('abc', { scorer: plain })('abc')).toBe(100)
  })
})
