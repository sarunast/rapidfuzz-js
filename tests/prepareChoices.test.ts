// Not ported from RapidFuzz — `prepareChoices` has no upstream counterpart.
// It is a performance API with a correctness contract: an index has to be
// indistinguishable from the collection it was built from, for every scorer,
// every collection shape and every `extract*` entry point. So the bulk of this
// file is one differential that asserts exactly that, and the rest covers the
// two mismatches it refuses and the scorers it cannot prepare.
import { describe, expect, it } from 'vitest'

import {
  withPreparedFlags,
  prepareScorerOf,
  PREPARE_CHOICE,
  SIMILARITY_FLAGS,
  type PreparedScore,
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
import {
  extract,
  extractIter,
  extractOne,
  prepareChoices,
  scoreMatrix,
  type ExtractResult,
  type PrepareOptions,
  type PreparedChoiceIndex,
  type SearchOptions,
  type SearchScorer,
} from '../src/search.js'
import { defaultProcess } from '../src/utils.js'

const CHOICES = [
  'new york mets',
  'new YORK mets',
  null,
  'the wonders',
  'atlanta braves',
  'dallas cowboys',
  'new york jets',
] as const

const QUERIES = ['new york mets', 'cowboys', 'nothing alike at all', '']

const SCORERS: readonly (readonly [string, SearchScorer])[] = [
  ['wRatio', wRatio],
  ['ratio', ratio],
  ['partialRatio', partialRatio],
  ['tokenSortRatio', tokenSortRatio],
  ['tokenSetRatio', tokenSetRatio],
  ['levenshteinDistance', levenshteinDistance],
  ['levenshteinNormalizedSimilarity', levenshteinNormalizedSimilarity],
  // A third-party scorer: no prepared factory, so the index holds the processed
  // choices and nothing else.
  ['a plain function', (a: string, b: string): number => (a === b ? 100 : 0)],
  // Configured with a baked-in processor, which is the built-in that registers
  // no factory at all.
  ['configured with a processor', configure(ratio, { processor: defaultProcess })],
  // Configured and still preparable, which is the case that keeps its factory
  // and has to keep the choice hook the factory carries with it.
  ['configured with weights', configure(levenshteinDistance, { weights: [1, 1, 2] })],
  ['configured twice', configure(configure(tokenSortRatio, {}), {})],
]

/** The same choices in each shape `Choices` admits, with the keys they imply. */
const SHAPES = [
  ['a list', (v: readonly (string | null)[]): (string | null)[] => [...v]],
  [
    'a map',
    (v: readonly (string | null)[]): Map<string, string | null> =>
      new Map(v.map((choice, i) => [`k${i}`, choice])),
  ],
  [
    'an object',
    (v: readonly (string | null)[]): Record<string, string | null> =>
      Object.fromEntries(v.map((choice, i) => [`k${i}`, choice])),
  ],
  ['a set', (v: readonly (string | null)[]): Set<string | null> => new Set(v)],
] as const

/** `toEqual` over results, which are objects and compare field by field. */
function sameResults<T>(a: readonly ExtractResult<T>[], b: readonly ExtractResult<T>[]) {
  expect(b).toEqual(a)
}

describe('an index scores exactly as the collection it was built from', () => {
  for (const [shapeName, build] of SHAPES) {
    for (const [scorerName, scorer] of SCORERS) {
      it(`${shapeName}, ${scorerName}`, () => {
        const choices = build(CHOICES)
        const options: SearchOptions = { scorer }
        const index = prepareChoices(choices, options)

        for (const query of QUERIES) {
          sameResults(
            extract(query, choices, options),
            extract(query, index, { limit: 5 }),
          )
          sameResults(
            extract(query, choices, { ...options, limit: null }),
            extract(query, index, { limit: null }),
          )
          sameResults(
            extract(query, choices, { ...options, limit: 1 }),
            extract(query, index, { limit: 1 }),
          )
          sameResults(
            [...extractIter(query, choices, options)],
            [...extractIter(query, index, {})],
          )
          expect(extractOne(query, index, {})).toEqual(
            extractOne(query, choices, options),
          )
        }
      })
    }
  }
})

describe('an index carries the processor it was built with', () => {
  const RAW = ['  NEW YORK METS!! ', 'the wonders', null, 'DALLAS cowboys']

  for (const [scorerName, scorer] of SCORERS) {
    it(`applies it once, for ${scorerName}`, () => {
      const options: SearchOptions = { scorer, processor: defaultProcess }
      const index = prepareChoices(RAW, options)

      for (const query of ['new york METS', 'cowboys']) {
        sameResults(extract(query, RAW, options), extract(query, index, {}))
        // Naming the same processor again is allowed; it is the same one.
        sameResults(
          extract(query, RAW, options),
          extract(query, index, { processor: defaultProcess }),
        )
      }
    })
  }

  it('applies it through the keyed shapes too', () => {
    const options: SearchOptions = { scorer: ratio, processor: defaultProcess }
    const map = new Map(RAW.map((choice, i) => [`k${i}`, choice]))
    sameResults(
      extract('mets', map, options),
      extract('mets', prepareChoices(map, options), {}),
    )
    const object = Object.fromEntries(RAW.map((choice, i) => [`k${i}`, choice]))
    sameResults(
      extract('mets', object, options),
      extract('mets', prepareChoices(object, options), {}),
    )
  })
})

describe('an index refuses a call it was not prepared for', () => {
  const index = prepareChoices(['abc', 'abd'], { scorer: ratio })

  it('refuses a different scorer', () => {
    expect(() => extract('abc', index, { scorer: tokenSortRatio })).toThrow(TypeError)
    expect(() => extractOne('abc', index, { scorer: tokenSortRatio })).toThrow(
      /prepared for/,
    )
    expect(() => [...extractIter('abc', index, { scorer: wRatio })]).toThrow(TypeError)
    expect(() => extract('abc', index, { scorer: wRatio, limit: null })).toThrow(
      TypeError,
    )
  })

  it('refuses a different processor', () => {
    expect(() => extract('abc', index, { processor: defaultProcess })).toThrow(
      /prepared for/,
    )
  })

  it('refuses both before it looks at the query', () => {
    // A missing query returns empty rather than throwing, so this is the test
    // that the check runs first.
    expect(() => extract(null, index, { scorer: wRatio })).toThrow(TypeError)
  })

  it('accepts the scorer it was prepared for, named again', () => {
    expect(extract('abc', index, { scorer: ratio })).toEqual(extract('abc', index, {}))
  })

  it('refuses a limit that asks for nothing', () => {
    // `limit <= 0` returns an empty array without scoring anything, which is a
    // reason to skip the work and not a reason to accept the call: the options
    // disagree with the index either way.
    for (const limit of [0, -1]) {
      expect(() => extract('abc', index, { scorer: tokenSortRatio, limit })).toThrow(
        /prepared for/,
      )
      expect(() => extract('abc', index, { processor: defaultProcess, limit })).toThrow(
        /prepared for/,
      )
    }
  })

  it('accepts a limit that asks for nothing when the options agree', () => {
    expect(extract('abc', index, { scorer: ratio, limit: 0 })).toEqual([])
    expect(extract('abc', index, { limit: -1 })).toEqual([])
  })

  it('leaves a plain collection alone at the same limits', () => {
    expect(extract('abc', ['abc'], { scorer: tokenSortRatio, limit: 0 })).toEqual([])
    expect(extract('abc', ['abc'], { processor: defaultProcess, limit: -1 })).toEqual([])
  })
})

describe('the renamed types name the same thing', () => {
  // `PrepareChoicesOptions` and `PreparedChoices` were the 0.4.0 names, renamed
  // in 0.5.0 once a query and a single choice could be prepared too — neither
  // old name describes what it holds any more. They were removed rather than
  // aliased: `check-exports.mjs` can only assert runtime values, so a type-only
  // alias has nothing but a test holding it, and this is that test for the names
  // that replaced them.
  it('are assignable from what prepareChoices returns', () => {
    const options: PrepareOptions = { scorer: ratio }
    const index: PreparedChoiceIndex<string, number> = prepareChoices(['abc'], options)

    expect(index.scorer).toBe(ratio)
    expect(extractOne('abc', index, {})).toEqual({ choice: 'abc', score: 100, key: 0 })
  })
})

describe('an index defaults to what extract does', () => {
  it('scores with wRatio when neither names a scorer', () => {
    const index = prepareChoices(['new york mets', 'the wonders'])
    expect(extract('new york mets', index, {})).toEqual(
      extract('new york mets', ['new york mets', 'the wonders'], {}),
    )
    expect(index.scorer).toBe(wRatio)
    expect(index.processor).toBeNull()
  })
})

describe('an index drops missing choices as it builds', () => {
  it('keeps every surviving key and value', () => {
    const index = prepareChoices(['a', null, 'b', undefined, Number.NaN, 'c'])
    expect(index.values).toEqual(['a', 'b', 'c'])
    expect(index.keys).toEqual([0, 2, 5])
    expect(index.size).toBe(3)
  })

  it('keeps map and object keys', () => {
    const fromMap = prepareChoices(
      new Map([
        ['x', 'a'],
        ['y', null],
        ['z', 'c'],
      ]),
    )
    expect(fromMap.keys).toEqual(['x', 'z'])
    const fromObject = prepareChoices({ x: 'a', y: null, z: 'c' })
    expect(fromObject.keys).toEqual(['x', 'z'])
  })

  it('holds an empty index for a collection with nothing in it', () => {
    const index = prepareChoices([])
    expect(index.values).toEqual([])
    expect(extract('abc', index, {})).toEqual([])
    expect(extractOne('abc', index, {})).toBeUndefined()
    expect([...extractIter('abc', index, {})]).toEqual([])
  })
})

describe('an index that was copied rather than passed', () => {
  // A spread carries symbol-keyed own properties, so the copy is branded and
  // reaches the scoring loops — but the prepared state it needs is held in a
  // table keyed by the original. Scoring the copy against state it does not
  // have would be silently wrong, so it says so instead.
  const index = prepareChoices(['abc', 'abd'], { scorer: ratio })
  const copy = { ...index }

  it('is refused by every entry point', () => {
    expect(() => extract('abc', copy, {})).toThrow(/cannot be copied/)
    expect(() => extract('abc', copy, { limit: null })).toThrow(TypeError)
    expect(() => extract('abc', copy, { limit: 1 })).toThrow(TypeError)
    expect(() => extractOne('abc', copy, {})).toThrow(TypeError)
    expect(() => [...extractIter('abc', copy, {})]).toThrow(TypeError)
  })

  it('is refused before the query and before the limit', () => {
    // The two paths that would otherwise return early with nothing to score: a
    // missing query, and a limit asking for no results. Provenance is a fact
    // about the index, so neither is a reason to stop asking about it.
    expect(() => extract(null, copy, {})).toThrow(/cannot be copied/)
    expect(() => extract('abc', copy, { limit: 0 })).toThrow(/cannot be copied/)
    expect(() => extract('abc', copy, { limit: -1 })).toThrow(/cannot be copied/)
    expect(() => extract('abc', copy, { scorer: ratio, limit: 0 })).toThrow(
      /cannot be copied/,
    )
  })

  it('leaves the original working', () => {
    expect(extractOne('abc', index, {})).toEqual({ choice: 'abc', score: 100, key: 0 })
  })
})

describe('configuring a scorer keeps the choice hook its factory carries', () => {
  // White-box, because the thing that breaks is invisible from outside: an
  // index and a matrix over a configured built-in both still return the right
  // answers with no hook at all, just after preparing every choice from scratch.
  // The hook used to hang off the prepared score, so a wrapping factory got it
  // back for free; now it hangs off the factory, and the wrapper has to carry it
  // deliberately. `scoreMatrix` has prepared choices this way since long before
  // `prepareChoices` existed, so dropping it here is a regression in shipped
  // code, not just a weaker new API.
  const hookOf = (scorer: object): unknown => prepareScorerOf(scorer)?.[PREPARE_CHOICE]

  it('on a configured built-in', () => {
    const hook = hookOf(levenshteinDistance)
    expect(hook).toBeTypeOf('function')
    expect(hookOf(configure(levenshteinDistance, { weights: [1, 1, 2] }))).toBe(hook)
  })

  it('on a token scorer, whose hook is its own', () => {
    const hook = hookOf(tokenSortRatio)
    expect(hook).toBeTypeOf('function')
    expect(hook).not.toBe(hookOf(levenshteinDistance))
    expect(hookOf(configure(tokenSortRatio, {}))).toBe(hook)
  })

  it('through a second configure', () => {
    const hook = hookOf(wRatio)
    expect(hook).toBeTypeOf('function')
    expect(hookOf(configure(configure(wRatio, {}), {}))).toBe(hook)
  })

  it('but not on one with a baked processor, which keeps no factory at all', () => {
    expect(prepareScorerOf(configure(ratio, { processor: defaultProcess }))).toBeNull()
    expect(hookOf(configure(ratio, { processor: defaultProcess }))).toBeUndefined()
  })
})

describe('an index cannot be edited after it is built', () => {
  // `readonly` in the type is a promise to the type checker only, and an index
  // outlives the call that made it. A write to `values` or `scorer` would leave
  // the prepared state describing something the index no longer says it holds:
  // scored from the old choice, reported as the new one, with nothing to see.
  const index = prepareChoices(['abc', 'abd'], { scorer: ratio })

  it('is frozen, along with both arrays', () => {
    expect(Object.isFrozen(index)).toBe(true)
    expect(Object.isFrozen(index.values)).toBe(true)
    expect(Object.isFrozen(index.keys)).toBe(true)
  })

  it('refuses a swapped scorer, processor or array', () => {
    expect(Reflect.set(index, 'scorer', tokenSortRatio)).toBe(false)
    expect(Reflect.set(index, 'processor', defaultProcess)).toBe(false)
    expect(Reflect.set(index, 'values', ['zzz', 'zzz'])).toBe(false)
    expect(Reflect.set(index, 'size', 99)).toBe(false)
  })

  it('refuses a rewritten choice or key', () => {
    expect(Reflect.set(index.values, 0, 'completely different')).toBe(false)
    expect(Reflect.set(index.keys, 0, 7)).toBe(false)
  })

  it('still reports what it was built from', () => {
    expect(index.scorer).toBe(ratio)
    expect(index.values).toEqual(['abc', 'abd'])
    expect(extractOne('abc', index, {})).toEqual({ choice: 'abc', score: 100, key: 0 })
  })
})

describe('an index with a missing query yields nothing', () => {
  const index = prepareChoices(['abc', 'abd'])

  it('through every entry point', () => {
    expect(extract(null, index, {})).toEqual([])
    expect(extract(null, index, { limit: null })).toEqual([])
    expect(extractOne(null, index, {})).toBeUndefined()
    expect([...extractIter(null, index, {})]).toEqual([])
  })
})

describe('an index leaves a choice the scorer will refuse alone', () => {
  it('throws where the collection would, and at the same call', () => {
    const withNumber = [42, 'abc']
    const index = prepareChoices(withNumber, { scorer: ratio })
    expect(() => extract('abc', index, {})).toThrow(TypeError)
    expect(() => extract('abc', withNumber, { scorer: ratio })).toThrow(TypeError)
  })

  it('builds without complaint, and defers the refusal to scoring', () => {
    // The point of the pair above: preparing is not the moment to refuse, or
    // the error moves to a call the collection would have got through.
    expect(() => prepareChoices([42, 'abc'], { scorer: ratio })).not.toThrow()
    // And with no sequence at all there is no hook to look up, so the choices
    // are held exactly as they came.
    expect(() => prepareChoices([1, 2, 3], { scorer: ratio })).not.toThrow()
    expect(prepareChoices([1, 2, 3], { scorer: ratio }).size).toBe(3)
  })
})

describe('an index for a scorer that prepares a query but not a choice', () => {
  // Every scorer this package ships attaches a per-choice hook to its prepared
  // *factory*, so the branch where a factory offers none needs a scorer that
  // does not. `withPreparedFlags` is the registration every built-in goes
  // through, and it takes any factory — the hook is optional on `PrepareScorer`
  // because a prepared query is a separate fact from a prepared choice.
  // Left unannotated so `configure` below can infer its argument types from it,
  // rather than from the erased signature `SearchScorer` widens them to.
  const halves = (a: string, b: string): number => (a === b ? 100 : 0)
  const scorer = withPreparedFlags(halves, SIMILARITY_FLAGS, (query) => {
    const score: PreparedScore = (choice) => (choice === query ? 100 : 0)
    return score
  })

  it('holds the choices unprepared and scores them', () => {
    const index = prepareChoices(['abc', 'abd'], { scorer })
    expect(extractOne('abd', index, {})).toEqual({ choice: 'abd', score: 100, key: 1 })
  })

  it('leaves a scoreMatrix over the same scorer unprepared too', () => {
    expect(scoreMatrix(['abc'], ['abc', 'abd'], { scorer }).toArray()).toEqual([[100, 0]])
  })

  it('survives configure, which has nothing to carry over', () => {
    // The other half of the hook propagation: `configure` wraps the factory, so
    // it has to leave the wrapper hookless when the factory it wrapped was.
    const configured = configure(scorer, {})
    expect(prepareScorerOf(configured)).not.toBeNull()
    expect(prepareScorerOf(configured)?.[PREPARE_CHOICE]).toBeUndefined()
    const index = prepareChoices(['abc', 'abd'], { scorer: configured })
    expect(extractOne('abd', index, {})).toEqual({ choice: 'abd', score: 100, key: 1 })
  })
})

describe('an index over a distance scorer keeps the direction', () => {
  const index = prepareChoices(['abc', 'abcd', 'xyz'], { scorer: levenshteinDistance })

  it('reports the smallest distance first and honours a cutoff', () => {
    expect(extract('abc', index, { limit: 2 })).toEqual([
      { choice: 'abc', score: 0, key: 0 },
      { choice: 'abcd', score: 1, key: 1 },
    ])
    expect(extract('abc', index, { scoreCutoff: 1, limit: null })).toEqual([
      { choice: 'abc', score: 0, key: 0 },
      { choice: 'abcd', score: 1, key: 1 },
    ])
    expect(extractOne('abc', index, {})).toEqual({ choice: 'abc', score: 0, key: 0 })
  })
})

describe('an index takes the same per-call options a collection does', () => {
  const choices = ['new york mets', 'new york jets', 'atlanta braves']
  const index = prepareChoices(choices, { scorer: ratio })

  it('cutoff, hint and limit all still apply', () => {
    for (const options of [
      { scoreCutoff: 80 },
      { scoreHint: 90 },
      { limit: 2 },
      { limit: 0 },
      { limit: null, scoreCutoff: 50 },
    ]) {
      sameResults(
        extract('new york mets', choices, { ...options, scorer: ratio }),
        extract('new york mets', index, options),
      )
    }
  })

  it('extractIter drops what the cutoff rejects', () => {
    expect([...extractIter('new york mets', index, { scoreCutoff: 95 })]).toEqual([
      { choice: 'new york mets', score: 100, key: 0 },
    ])
  })
})
