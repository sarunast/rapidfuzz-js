// Ported from RapidFuzz tests/test_process.py
//
// Upstream's numpy-specific assertions (dtype identity, array shapes) and its
// pandas cases have no JavaScript equivalent: scoreMatrix/scorePairs return typed
// arrays here. The behaviour those tests pin down — score multipliers, integral
// rounding, empty inputs, asymmetric scorers — is covered below.
import { runInNewContext } from 'node:vm'

import { describe, expect, it } from 'vitest'

import { configure } from '../src/configure.js'
import {
  damerauLevenshteinDistance,
  damerauLevenshteinNormalizedSimilarity,
} from '../src/distance/damerauLevenshtein.js'
import { hammingDistance, hammingNormalizedSimilarity } from '../src/distance/hamming.js'
import { indelNormalizedSimilarity } from '../src/distance/indel.js'
import { jaroNormalizedSimilarity, jaroSimilarity } from '../src/distance/jaro.js'
import { jaroWinklerNormalizedSimilarity } from '../src/distance/jaroWinkler.js'
import { lcsSeqNormalizedSimilarity } from '../src/distance/lcsSeq.js'
import {
  levenshteinDistance,
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
  levenshteinSimilarity,
} from '../src/distance/levenshtein.js'
import {
  osaDistance,
  osaNormalizedDistance,
  osaNormalizedSimilarity,
  osaSimilarity,
} from '../src/distance/osa.js'
import { postfixSimilarity } from '../src/distance/postfix.js'
import { prefixNormalizedDistance } from '../src/distance/prefix.js'
import {
  partialRatio,
  partialTokenRatio,
  qRatio,
  ratio,
  tokenSetRatio,
  tokenSortRatio,
  tokenRatio,
  wRatio,
  type FuzzInput,
  type FuzzOptions,
} from '../src/fuzz.js'
import {
  extract,
  extractIter,
  extractOne,
  scoreMatrix,
  scorePairs,
} from '../src/search.js'
import { defaultProcess } from '../src/utils.js'
import { callUntyped } from './common.js'
import { matrixScores, pairScores } from './matrix.js'

const BASEBALL_STRINGS = [
  'new york mets vs chicago cubs',
  'chicago cubs vs chicago white sox',
  'philladelphia phillies vs atlanta braves',
  'braves vs mets',
]

/** Upstream's `custom_scorer` — a scorer with no attached flags. */
function customScorer(s1: FuzzInput, s2: FuzzInput, options: FuzzOptions = {}): number {
  return ratio(s1, s2, { scoreCutoff: options.scoreCutoff })
}

/** Upstream's `wrapped` — a scorer that ignores its inputs. */
function wrapped(): number {
  return 100
}

it('rejects non-sequence inputs', () => {
  expect(() => callUntyped(extractOne, 1, [''])).toThrow(TypeError)
  expect(() => extractOne('', [1])).toThrow(TypeError)
  expect(() => extractOne('', { 1: 1 })).toThrow(TypeError)

  expect(() => callUntyped(extract, 1, [''])).toThrow(TypeError)
  expect(() => extract('', [1])).toThrow(TypeError)
  expect(() => extract('', { 1: 1 })).toThrow(TypeError)

  expect(() => [...callUntyped(extractIter, 1, [''])]).toThrow(TypeError)
  expect(() => [...extractIter('', [1])]).toThrow(TypeError)
  expect(() => [...extractIter('', { 1: 1 })]).toThrow(TypeError)
})

describe('picks the best choice', () => {
  const CASES: ReadonlyArray<readonly [string, number]> = [
    ['new york mets at atlanta braves', 3],
    ['philadelphia phillies at atlanta braves', 2],
    ['atlanta braves at philadelphia phillies', 2],
    ['chicago cubs vs new york mets', 0],
  ]

  for (const [query, index] of CASES) {
    it(query, () => {
      expect(extractOne(query, BASEBALL_STRINGS)?.choice).toBe(BASEBALL_STRINGS[index])
      expect(extractOne(query, new Set(BASEBALL_STRINGS))?.choice).toBe(
        BASEBALL_STRINGS[index],
      )
      expect(extract(query, BASEBALL_STRINGS)[0].choice).toBe(BASEBALL_STRINGS[index])
      expect(extract(query, new Set(BASEBALL_STRINGS))[0].choice).toBe(
        BASEBALL_STRINGS[index],
      )
    })
  }
})

it('accepts any type as long as the processor yields a string', () => {
  const events = [
    ['chicago cubs vs new york mets', 'CitiField', '2011-05-11', '8pm'],
    ['new york yankees vs boston red sox', 'Fenway Park', '2011-05-11', '8pm'],
    ['atlanta braves vs pittsburgh pirates', 'PNC Park', '2011-05-11', '8pm'],
  ]
  const query = events[0]
  const first = (event: string | ArrayLike<unknown>): string =>
    String(Array.from(event)[0])

  expect(extractOne(query, events, { processor: first })?.choice).toBe(events[0])
  expect(extract(query, events, { processor: first })[0].choice).toBe(events[0])

  const eventsMap = new Map(events.map((event, i) => [i, event]))
  expect(extractOne(query, eventsMap, { processor: first })?.choice).toBe(events[0])
  expect(extract(query, eventsMap, { processor: first })[0].choice).toBe(events[0])

  const withoutProcessor = extractOne('new york mets', ['new YORK mets'])?.score ?? 0
  expect(withoutProcessor).toBeGreaterThan(72)
  expect(withoutProcessor).toBeLessThan(73)

  expect(extract('new york mets', ['new YORK mets'])[0].score).toBe(withoutProcessor)

  expect(
    extractOne('new york mets', ['new YORK mets'], { processor: defaultProcess })?.score,
  ).toBe(100)
  expect(
    extract('new york mets', ['new YORK mets'], { processor: defaultProcess })[0].score,
  ).toBe(100)
})

it('honours a custom scorer', () => {
  const choices = [
    'new york mets vs chicago cubs',
    'chicago cubs at new york mets',
    'atlanta braves vs pittsbugh pirates',
    'new york yankees vs boston red sox',
  ]
  const choicesMap = new Map(choices.map((choice, i) => [i + 1, choice]))
  const query = 'new york mets at chicago cubs'

  expect(extractOne(query, choices)?.choice).toBe(choices[1])
  expect(extract(query, choices)[0].choice).toBe(choices[1])
  expect(extractOne(query, choicesMap)?.choice).toBe(choicesMap.get(2))
  expect(extract(query, choicesMap)[0].choice).toBe(choicesMap.get(2))

  expect(extractOne(query, choices, { scorer: qRatio })?.choice).toBe(choices[0])
  expect(extract(query, choices, { scorer: qRatio })[0].choice).toBe(choices[0])
  expect(extractOne(query, choicesMap, { scorer: qRatio })?.choice).toBe(
    choicesMap.get(1),
  )
  expect(extract(query, choicesMap, { scorer: qRatio })[0].choice).toBe(choicesMap.get(1))
})

it('applies score_cutoff', () => {
  const choices = [
    'new york mets vs chicago cubs',
    'chicago cubs at new york mets',
    'atlanta braves vs pittsbugh pirates',
    'new york yankees vs boston red sox',
  ]
  const query = 'los angeles dodgers vs san francisco giants'

  expect(extractOne(query, choices, { scoreCutoff: 50 })).toBeUndefined()
  expect(extractOne(query, choices)).not.toBeUndefined()
})

it('handles score_cutoff edge cases', () => {
  const choices = [
    'new york mets vs chicago cubs',
    'chicago cubs at new york mets',
    'atlanta braves vs pittsbugh pirates',
    'new york yankees vs boston red sox',
  ]

  const best = extractOne('new york mets vs chicago cubs', choices, { scoreCutoff: 100 })
  expect(best).not.toBeUndefined()
  expect(best?.choice).toBe(choices[0])

  // A score of 0 is still a result.
  const zero = extractOne('', choices)
  expect(zero).not.toBeUndefined()
  expect(zero?.score).toBe(0)
})

it('skips missing elements but keeps the index', () => {
  expect(extractOne('test', [null, 'tes'])?.key).toBe(1)
  expect(extractOne(null, [null, 'tes'])).toBeUndefined()
  expect(
    extractOne(
      'test',
      new Map([
        [0, null],
        [1, 'tes'],
      ]),
    )?.key,
  ).toBe(1)
  expect(
    extractOne(
      null,
      new Map([
        [0, null],
        [1, 'tes'],
      ]),
    ),
  ).toBeUndefined()

  expect(extractOne('test', [null, 'tes'], { processor: defaultProcess })?.key).toBe(1)
  expect(extractOne(null, [null, 'tes'], { processor: defaultProcess })).toBeUndefined()

  expect(extractOne('test', [Number.NaN, 'tes'])?.key).toBe(1)
  expect(callUntyped(extractOne, Number.NaN, [Number.NaN, 'tes'])).toBeUndefined()
})

it('returns the first of several equally good matches', () => {
  expect(extractOne('test', ['tes', 'tes'])?.key).toBe(0)
  expect(extract('test', ['tes', 'tes'], { limit: 1 })[0].key).toBe(0)
})

it('handles special limits', () => {
  expect(extract('test', ['tes', 'tes'], { limit: 1, scoreCutoff: 100 })).toEqual([])

  expect(
    extract('test', ['te', 'test'], { limit: null, scorer: levenshteinDistance }),
  ).toEqual([
    { choice: 'test', score: 0, key: 1 },
    { choice: 'te', score: 2, key: 0 },
  ])
})

it('skips empty strings', () => {
  const choices = [
    '',
    'new york mets vs chicago cubs',
    'new york yankees vs boston red sox',
    '',
    '',
  ]
  expect(extractOne('new york mets at chicago cubs', choices)?.choice).toBe(choices[1])
})

it('skips missing strings', () => {
  const choices = [
    null,
    'new york mets vs chicago cubs',
    'new york yankees vs boston red sox',
    null,
    null,
  ]
  const query = 'new york mets at chicago cubs'

  expect(extractOne(query, choices)?.choice).toBe(choices[1])

  const bests = extract(query, choices)
  expect(bests[0].choice).toBe(choices[1])
  expect(bests[1].choice).toBe(choices[2])

  const iterated = [...extractIter(query, choices)]
  expect(iterated[0].choice).toBe(choices[1])
  expect(iterated[1].choice).toBe(choices[2])

  const scores = matrixScores([query], choices)
  expect(scores[0][0]).toBe(0)
  expect(scores[0][3]).toBe(0)
  expect(scores[0][4]).toBe(0)

  const pairs = matrixScores(['', null], ['', null], {
    scorer: damerauLevenshteinNormalizedSimilarity,
  })
  expect(pairs[0][0]).toBe(1)
  expect(pairs[0][1]).toBe(0)
  expect(pairs[1][0]).toBe(0)
  expect(pairs[1][1]).toBe(0)

  const pairwise = pairScores(choices, choices)
  expect(pairwise[0]).toBe(0)
  expect(pairwise[3]).toBe(0)
  expect(pairwise[4]).toBe(0)
})

describe('is case sensitive without a processor', () => {
  const scorers = { ratio, customScorer }

  for (const [name, scorer] of Object.entries(scorers)) {
    // Upstream parametrises over `[None, lambda s: s]` — "no processor" and an
    // identity one. `undefined` is how the option types now spell the first.
    for (const [i, processor] of [
      undefined,
      (s: string | ArrayLike<unknown>) => s,
    ].entries()) {
      it(`${name} (processor ${i})`, () => {
        expect(
          extractOne('new york mets', ['new', 'new YORK mets'], {
            processor,
            scorer,
          })?.score,
        ).not.toBe(100)
      })
    }
  }
})

describe('uses the first match', () => {
  const scorers = { ratio, customScorer }

  for (const [name, scorer] of Object.entries(scorers)) {
    it(name, () => {
      expect(
        extractOne('new york mets', ['new york mets', 'new york mets'], { scorer })?.key,
      ).toBe(0)
    })
  }
})

describe('cdist handles empty inputs', () => {
  const scorers = { ratio, wRatio, customScorer }

  for (const [name, scorer] of Object.entries(scorers)) {
    it(name, () => {
      expect(matrixScores([], ['a', 'b'], { scorer })).toEqual([])
      expect(matrixScores(['a', 'b'], [], { scorer })).toEqual([[], []])
    })
  }
})

describe('cpdist handles empty inputs', () => {
  const scorers = { ratio, wRatio, customScorer }

  for (const [name, scorer] of Object.entries(scorers)) {
    it(name, () => {
      expect(pairScores([], [], { scorer })).toEqual([])
    })
  }
})

it('accepts a scorer that ignores its inputs', () => {
  expect(
    callUntyped(scoreMatrix, ['test'], [Number.NaN], { scorer: wrapped }).at(0, 0),
  ).toBe(100)
  expect(matrixScores(['test'], [null], { scorer: wrapped })[0][0]).toBe(100)
  expect(matrixScores(['test'], ['tes'], { scorer: wrapped })[0][0]).toBe(100)
  expect(callUntyped(scorePairs, ['test'], [Number.NaN], { scorer: wrapped })[0]).toBe(
    100,
  )
  expect(pairScores(['test'], [null], { scorer: wrapped })[0]).toBe(100)
  expect(pairScores(['test'], ['tes'], { scorer: wrapped })[0]).toBe(100)
})

it('supports an asymmetric scorer', () => {
  const strings = ['test', 'test2']

  expect(
    matrixScores(strings, strings, {
      scorer: configure(levenshteinDistance, { weights: [1, 2, 1] }),
    }),
  ).toEqual([
    [0, 1],
    [2, 0],
  ])
})

it('requires cpdist inputs of the same length', () => {
  expect(() => pairScores(['a', 'b'], [])).toThrow(
    'Length of queries and choices must be the same!',
  )
  expect(() => pairScores(['a', 'b'], ['f'])).toThrow(
    'Length of queries and choices must be the same!',
  )
})

it('applies a score multiplier in cpdist', () => {
  expect(
    pairScores(['test'], ['test2'], {
      scorer: levenshteinNormalizedSimilarity,
      scoreMultiplier: 255,
      into: 'i32',
    }),
  ).toEqual([204])

  expect(
    pairScores(['test'], ['test2'], {
      scorer: levenshteinNormalizedDistance,
      scoreMultiplier: 255,
      into: 'i32',
    }),
  ).toEqual([51])
})

it('applies a score multiplier in cdist', () => {
  const strings = ['test', 'test2']

  expect(
    matrixScores(strings, strings, {
      scorer: levenshteinNormalizedSimilarity,
      scoreMultiplier: 255,
      into: 'i32',
    }),
  ).toEqual([
    [255, 204],
    [204, 255],
  ])

  expect(
    matrixScores(strings, strings, {
      scorer: levenshteinNormalizedDistance,
      scoreMultiplier: 255,
      into: 'i32',
    }),
  ).toEqual([
    [0, 51],
    [51, 0],
  ])

  expect(
    matrixScores(strings, strings, { scorer: levenshteinSimilarity, scoreMultiplier: 2 }),
  ).toEqual([
    [8, 8],
    [8, 10],
  ])

  expect(
    matrixScores(strings, strings, { scorer: levenshteinDistance, scoreMultiplier: 2 }),
  ).toEqual([
    [0, 2],
    [2, 0],
  ])
})

it('accepts a generator as choices', () => {
  function* generateChoices(): Generator<string> {
    yield* ['a', 'Bb', 'CcC']
  }

  expect(extract('aaa', generateChoices()).length).toBeGreaterThan(0)
})

// Upstream asks `hasattr(choices, "items")`, and its docs promise a pandas
// `Series` works — so "is this a mapping" is a structural question there, and
// the type here says the same thing by accepting `ReadonlyMap`. Getting the
// answer wrong is not a type error but a wrong result: a mapping that falls
// through to the iterable branch has each of its `[key, value]` pairs scored as
// though the pair were the choice, and its keys replaced by positions.
describe('choices that are mappings without being a `Map`', () => {
  const pairs: readonly (readonly [number, string])[] = [
    [10, 'new york mets'],
    [20, 'chicago cubs'],
  ]

  class Lookup implements ReadonlyMap<number, string> {
    readonly #pairs: Map<number, string>

    constructor(entries: readonly (readonly [number, string])[]) {
      this.#pairs = new Map(entries)
    }

    get size(): number {
      return this.#pairs.size
    }
    get(key: number): string | undefined {
      return this.#pairs.get(key)
    }
    has(key: number): boolean {
      return this.#pairs.has(key)
    }
    entries(): MapIterator<[number, string]> {
      return this.#pairs.entries()
    }
    keys(): MapIterator<number> {
      return this.#pairs.keys()
    }
    values(): MapIterator<string> {
      return this.#pairs.values()
    }
    forEach(
      callback: (value: string, key: number, map: ReadonlyMap<number, string>) => void,
    ): void {
      for (const [key, value] of this.#pairs) callback(value, key, this)
    }
    [Symbol.iterator](): MapIterator<[number, string]> {
      return this.#pairs[Symbol.iterator]()
    }
  }

  it('reads one that only satisfies the interface', () => {
    const choices = new Lookup(pairs)
    expect(choices instanceof Map).toBe(false)

    expect(extractOne('new york mets', choices)).toEqual({
      choice: 'new york mets',
      score: 100,
      key: 10,
    })
    expect(extract('chicago cubs', choices)[0].key).toBe(20)
    expect([...extractIter('chicago cubs', choices)].map(({ key }) => key)).toEqual([
      10, 20,
    ])
  })

  // The case `instanceof Map` cannot answer at all: a genuine `Map`, built by
  // the same constructor, from a context whose realm is not this one.
  it('reads a `Map` from another realm', () => {
    const choices: ReadonlyMap<number, string> = runInNewContext(
      'new Map(entries)',
      // Copied in: an array literal from this realm is fine as the argument,
      // but the `Map` has to be the other realm's to be worth testing.
      { entries: pairs.map(([key, value]) => [key, value]) },
    )
    expect(choices instanceof Map).toBe(false)
    expect(choices.get(20)).toBe('chicago cubs')

    expect(extractOne('new york mets', choices)?.key).toBe(10)
    expect(extract('chicago cubs', choices)[0].key).toBe(20)
  })

  // A `Set` has `has`, `entries` and `size` too, and its `entries()` yields
  // `[value, value]` — so reading it as a mapping would report a choice as its
  // own key. `get` is what tells the two apart.
  it('still enumerates a `Set` by position', () => {
    expect(
      extractOne('chicago cubs', new Set(['new york mets', 'chicago cubs'])),
    ).toEqual({ choice: 'chicago cubs', score: 100, key: 1 })
  })
})

// `Object.keys` does not report symbol-keyed properties, so a plain object's
// choices are its string-keyed ones — which is what the result type says too.
it('skips a symbol-keyed property of an object of choices', () => {
  const hidden = Symbol('hidden')
  const choices = { mets: 'new york mets', [hidden]: 'chicago cubs' }

  expect([...extractIter('chicago cubs', choices)].map(({ key }) => key)).toEqual([
    'mets',
  ])
  expect(extractOne('chicago cubs', choices)?.choice).toBe('new york mets')
})

it('keeps stable ordering with a bounded heap', () => {
  const choices = Array.from({ length: 40 }, (_, index) => `same-${index}`)
  const constant = (): number => 50

  expect(extract('query', choices, { scorer: constant, limit: 5 })).toEqual(
    choices.slice(0, 5).map((choice, index) => ({ choice, score: 50, key: index })),
  )
  expect(
    extract('query', choices, {
      scorer: levenshteinDistance,
      limit: 5,
    }),
  ).toEqual(
    extract('query', choices, {
      scorer: levenshteinDistance,
      limit: null,
    }).slice(0, 5),
  )
})

it('does not mirror custom scorers or suppress callback side effects', () => {
  const values = ['a', 'b', 'c']
  let calls = 0
  const observable = (): number => ++calls

  matrixScores(values, values, { scorer: observable })
  expect(calls).toBe(values.length * values.length)
})

it('forwards scoreHint without changing process results', () => {
  const choices = ['sitting', 'kitten', 'bitten', 'written']
  expect(
    extract('kitten', choices, {
      scorer: levenshteinDistance,
      limit: null,
      scoreHint: 0,
    }),
  ).toEqual(extract('kitten', choices, { scorer: levenshteinDistance, limit: null }))
  expect(
    matrixScores(choices, choices, { scorer: levenshteinDistance, scoreHint: 1 }),
  ).toEqual(matrixScores(choices, choices, { scorer: levenshteinDistance }))
})

it('matches direct scoring for BMP, astral, arrays, and typed arrays', () => {
  const queries = ['kitten', '🦊kitten', ['k', 'i', 't'], Uint16Array.of(107, 105, 116)]
  const choices = ['sitting', '🦊sitting', ['s', 'i', 't'], Uint16Array.of(115, 105, 116)]
  const reference = queries.map((query) => choices.map((choice) => ratio(query, choice)))

  expect(matrixScores(queries, choices, { scorer: ratio })).toEqual(reference)
  expect(pairScores(queries, choices, { scorer: ratio })).toEqual(
    queries.map((query, index) => ratio(query, choices[index])),
  )
})

it('keeps prepared OSA exact across score conventions and cutoffs', () => {
  const queries = ['CA', '🦊ab', ['a', 'b', 'c'], Uint16Array.of(97, 98, 99)]
  const choices = ['ABC', '🦊ba', ['a', 'c', 'b'], Uint16Array.of(97, 99, 98)]
  const cases = [
    [osaDistance, 2],
    [osaSimilarity, 1],
    [osaNormalizedDistance, 0.75],
    [osaNormalizedSimilarity, 0.25],
  ] as const

  for (const [scorer, scoreCutoff] of cases) {
    const reference = queries.map((query) =>
      choices.map((choice) => scorer(query, choice, { scoreCutoff })),
    )
    expect(matrixScores(queries, choices, { scorer, scoreCutoff })).toEqual(reference)
  }
})

it('prepares lowercase token scorer names with query token data', () => {
  for (const scorer of [tokenSortRatio, tokenSetRatio]) {
    expect(extractOne('a', ['a'], { scorer })).toEqual({
      choice: 'a',
      score: 100,
      key: 0,
    })
  }
})

it('keeps prepared token sets collision-safe for mixed sequence elements', () => {
  const query = ['aa', 'bb']
  const collidingChoice = ['aa\u0000string:bb']

  expect(extractOne(query, [collidingChoice], { scorer: tokenSetRatio })?.score).toBe(
    tokenSetRatio(query, collidingChoice),
  )
  expect(extractOne(query, [collidingChoice], { scorer: tokenSetRatio })?.score).toBe(0)
})

it('keeps prepared partial ratio identical for equal-length and empty inputs', () => {
  expect(extractOne('aaa', ['aba'], { scorer: partialRatio })?.score).toBe(
    partialRatio('aaa', 'aba'),
  )
  expect(extractOne('', [''], { scorer: partialRatio })).toEqual({
    choice: '',
    score: 100,
    key: 0,
  })
})

it('keeps new prepared distance adapters identical to direct scoring', () => {
  const queries = ['martha', '🦊dwayne', 'abc'.repeat(24)]
  const choices = ['marhta', '🦊duane', 'abd'.repeat(24)]

  expect(
    matrixScores(queries, choices, { scorer: jaroSimilarity, scoreCutoff: 0.8 }),
  ).toEqual(
    queries.map((query) =>
      choices.map((choice) => jaroSimilarity(query, choice, { scoreCutoff: 0.8 })),
    ),
  )
  expect(
    matrixScores(queries, choices, {
      scorer: configure(jaroWinklerNormalizedSimilarity, { prefixWeight: 0.2 }),
      scoreCutoff: 0.85,
    }),
  ).toEqual(
    queries.map((query) =>
      choices.map((choice) =>
        jaroWinklerNormalizedSimilarity(query, choice, {
          scoreCutoff: 0.85,
          prefixWeight: 0.2,
        }),
      ),
    ),
  )
  expect(
    matrixScores(queries, choices, {
      scorer: levenshteinNormalizedSimilarity,
      scoreCutoff: 0.8,
    }),
  ).toEqual(
    queries.map((query) =>
      choices.map((choice) =>
        levenshteinNormalizedSimilarity(query, choice, { scoreCutoff: 0.8 }),
      ),
    ),
  )
  expect(
    matrixScores(queries, choices, {
      scorer: indelNormalizedSimilarity,
      scoreCutoff: 0.8,
    }),
  ).toEqual(
    queries.map((query) =>
      choices.map((choice) =>
        indelNormalizedSimilarity(query, choice, { scoreCutoff: 0.8 }),
      ),
    ),
  )
  expect(
    matrixScores(queries, choices, {
      scorer: lcsSeqNormalizedSimilarity,
      scoreCutoff: 0.8,
    }),
  ).toEqual(
    queries.map((query) =>
      choices.map((choice) =>
        lcsSeqNormalizedSimilarity(query, choice, { scoreCutoff: 0.8 }),
      ),
    ),
  )
  expect(jaroNormalizedSimilarity(null, 'value')).toBe(0)
})

it('keeps cached token choices and short metric adapters identical to direct scoring', () => {
  const queries: Array<string | readonly unknown[]> = [
    'beta alpha beta',
    '🦊 fox\tmoon',
    ['aa', ' ', 'bb', Number.NaN, ' ', 'aa'],
  ]
  const choices: Array<string | readonly unknown[]> = [
    'alpha gamma beta',
    '🦊 moon fox',
    ['aa', ' ', 'cc', Number.NaN],
  ]
  const fuzzScorers = [
    tokenSortRatio,
    tokenSetRatio,
    tokenRatio,
    partialTokenRatio,
    wRatio,
  ]
  for (const scorer of fuzzScorers) {
    expect(matrixScores(queries, choices, { scorer, scoreCutoff: 37 })).toEqual(
      queries.map((query) =>
        choices.map((choice) => scorer(query, choice, { scoreCutoff: 37 })),
      ),
    )
  }

  expect(matrixScores(queries, choices, { scorer: damerauLevenshteinDistance })).toEqual(
    queries.map((query) =>
      choices.map((choice) => damerauLevenshteinDistance(query, choice)),
    ),
  )
  expect(matrixScores(queries, choices, { scorer: prefixNormalizedDistance })).toEqual(
    queries.map((query) =>
      choices.map((choice) => prefixNormalizedDistance(query, choice)),
    ),
  )
  expect(matrixScores(queries, choices, { scorer: postfixSimilarity })).toEqual(
    queries.map((query) => choices.map((choice) => postfixSimilarity(query, choice))),
  )
  expect(matrixScores(queries, choices, { scorer: hammingNormalizedSimilarity })).toEqual(
    queries.map((query) =>
      choices.map((choice) => hammingNormalizedSimilarity(query, choice)),
    ),
  )
  expect(() =>
    matrixScores(['abc'], ['ab'], { scorer: configure(hammingDistance, { pad: false }) }),
  ).toThrow('Sequences are not the same length.')
})

it('preserves custom scorer and processor observability', () => {
  let scorerCalls = 0
  let processorCalls = 0
  const scorer = (
    _first: FuzzInput,
    _second: FuzzInput,
    options: FuzzOptions = {},
  ): number => {
    scorerCalls++
    return options.scoreCutoff ?? 50
  }
  const processor = (value: string | ArrayLike<unknown>): string | ArrayLike<unknown> => {
    processorCalls++
    return value
  }

  matrixScores(['a', 'b'], ['c', 'd', 'e'], { scorer, processor, scoreCutoff: 7 })
  expect(scorerCalls).toBe(6)
  expect(processorCalls).toBe(5)

  scorerCalls = 0
  processorCalls = 0
  pairScores(['a', 'b'], ['c', 'd'], { scorer, processor, scoreCutoff: 7 })
  expect(scorerCalls).toBe(2)
  expect(processorCalls).toBe(4)

  const cutoffs: Array<number | null | undefined> = []
  const cutoffScorer = (
    _first: FuzzInput,
    _second: FuzzInput,
    options: FuzzOptions = {},
  ): number => {
    cutoffs.push(options.scoreCutoff)
    return 50
  }
  extract('a', ['b', 'c', 'd'], { scorer: cutoffScorer, scoreCutoff: 10, limit: 2 })
  expect(cutoffs).toEqual([10, 10, 10])
})

// Not ported. `search` used to substitute the scorer's worst score for a cutoff
// the caller had not given, which for a distance meant telling the scorer to
// bound at `2**63` — a number no caller wrote and no distance can reach. What a
// scorer is told now is what was asked: nothing.
it('tells a scorer nothing when no scoreCutoff was given', () => {
  const cutoffs: Array<number | null | undefined> = []
  const recording = (
    _first: FuzzInput,
    _second: FuzzInput,
    options: FuzzOptions = {},
  ): number => {
    cutoffs.push(options.scoreCutoff)
    return 50
  }

  for (const scorer of [recording, configure(recording, {})]) {
    for (const run of [
      () => extract('a', ['b', 'c'], { scorer, limit: null }),
      () => [...extractIter('a', ['b', 'c'], { scorer })],
    ]) {
      cutoffs.length = 0
      run()
      expect(cutoffs).toEqual([undefined, undefined])
    }

    // `extractOne` still tightens against the running best, so only the call
    // made before there is a best has nothing to be told.
    cutoffs.length = 0
    extractOne('a', ['b', 'c'], { scorer })
    expect(cutoffs).toEqual([undefined, 50])
  }

  // And a score no cutoff can reject is still kept, whichever direction the
  // scorer's flags read in.
  expect(extractOne('abc', ['xyz'], { scorer: levenshteinDistance })?.score).toBe(3)
  expect(extract('abc', ['xyz'], { scorer: levenshteinSimilarity })[0].score).toBe(0)
})

it('rounds when an integral dtype is requested', () => {
  const floatResult = pairScores(['1 2 33 5'], ['1 22 33 55'])[0]
  const intResult = pairScores(['1 2 33 5'], ['1 22 33 55'], { into: 'i32' })[0]

  expect(floatResult % 1).toBeGreaterThanOrEqual(0.5)
  expect(Math.round(floatResult)).toBe(intResult)
})

// An integral dtype stands in for an integer numpy array, which has no signed
// zero — so neither should a rounded score that lands on zero from below.
it('an integral dtype never reports a negative zero', () => {
  const matrix = matrixScores(['a'], ['a'], {
    scorer: ratio,
    into: 'i32',
    scoreMultiplier: -0.001,
  })
  const pairwise = pairScores(['a'], ['a'], {
    scorer: ratio,
    into: 'i32',
    scoreMultiplier: -0.001,
  })

  expect(matrix[0][0]).toBe(0)
  expect(pairwise[0]).toBe(0)
  expect(Object.is(matrix[0][0], -0)).toBe(false)
  expect(Object.is(pairwise[0], -0)).toBe(false)
})
