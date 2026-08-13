import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import {
  distance as cosineDistance,
  similarity as cosineSimilarity,
} from '../../src/algorithms/cosine/index.js'
import {
  distance as diceDistance,
  similarity as diceSimilarity,
} from '../../src/algorithms/dice/index.js'
import { feasibleRadices } from '../../src/algorithms/shared/gramKey.js'
import {
  assertAddressable,
  assertCosineExact,
  assertCosineNormsExact,
  assertQueryIndexable,
  createCosineIndexBuilder,
  createDiceIndexBuilder,
  repackKey,
} from '../../src/algorithms/shared/ngramIndex.js'
import type { ChoiceIndex } from '../../src/core/scoring/compilation.js'
import { createScorer, scorerCompilation } from '../../src/core/scoring/scorer.js'
import { createMatcher } from '../../src/index.js'

type Metric = 'dice' | 'cosine'

function indexOf(
  metric: Metric,
  gramSize: number,
  choices: readonly string[],
): ChoiceIndex {
  const builder =
    metric === 'dice'
      ? createDiceIndexBuilder(gramSize)
      : createCosineIndexBuilder(gramSize)
  for (const choice of choices) builder.add(choice)
  return builder.seal()
}

/** What the exhaustive Matcher answers, as `(id, score)` pairs. */
function exhaustive(
  metric: Metric,
  gramSize: number,
  choices: readonly string[],
  query: string,
  threshold: number | null,
  limit: number | null,
): { id: number; score: number }[] {
  const scorer = createScorer(metric === 'dice' ? diceSimilarity : cosineSimilarity, {
    gramSize,
  })
  const matcher = createMatcher(choices, { scorer })
  const call = threshold === null ? { limit } : { limit, threshold }
  return matcher
    .search(query, call)
    .map((match) => ({ id: Number(match.key), score: match.score }))
}

function pairs(selected: {
  ids: Uint32Array
  scores: Float64Array
  length: number
}): { id: number; score: number }[] {
  const out: { id: number; score: number }[] = []
  for (let at = 0; at < selected.length; at++) {
    out.push({ id: selected.ids[at], score: selected.scores[at] })
  }
  return out
}

/** `searchIter`'s answer: every qualifying match, in collection order. */
function exhaustiveScan(
  metric: Metric,
  gramSize: number,
  choices: readonly string[],
  query: string,
  threshold: number | null,
): { id: number; score: number }[] {
  const scorer = createScorer(metric === 'dice' ? diceSimilarity : cosineSimilarity, {
    gramSize,
  })
  const matcher = createMatcher(choices, { scorer })
  const call = threshold === null ? undefined : { threshold }
  return [...matcher.searchIter(query, call)].map((match) => ({
    id: Number(match.key),
    score: match.score,
  }))
}

const METRICS: readonly Metric[] = ['dice', 'cosine']
const THRESHOLDS: readonly (number | null)[] = [null, 0, 0.5, 0.8, 1]
const LIMITS: readonly (number | null)[] = [1, 3, null]

const CORPORA: readonly (readonly string[])[] = [
  [],
  [''],
  ['a'],
  ['ab', 'ab', 'ba'],
  ['abc', 'abcd', 'ab', 'a', ''],
  ['banana', 'bananas', 'ananab', 'band', 'b'],
  ['😀abc', 'abc😀', '😀abc', 'a\ud800b', '\ud800\ud800'],
  ['aaaa', 'aaaaa', 'aaaaaa', 'aaab'],
  ['zzz', 'yyy', 'xxx'],
  // Every choice shares `no`/`od`, which is what makes a list dense.
  ['node', 'nodes', 'noded', 'nodex', 'nodey', 'nodez', 'qq'],
]

const QUERIES: readonly string[] = [
  '',
  'a',
  'ab',
  'abc',
  'banana',
  '😀abc',
  'aaaa',
  'qqq',
  'node',
]

describe('an indexed search answers what the exhaustive one does', () => {
  it('matches key, score and order across the whole matrix', () => {
    let cases = 0
    for (const metric of METRICS) {
      for (const gramSize of [2, 3]) {
        for (const choices of CORPORA) {
          for (const query of QUERIES) {
            const index = indexOf(metric, gramSize, choices)
            for (const threshold of THRESHOLDS) {
              for (const limit of LIMITS) {
                expect(pairs(index.select(query, threshold, limit))).toEqual(
                  exhaustive(metric, gramSize, choices, query, threshold, limit),
                )
                cases++
              }
              expect(pairs(index.scan(query, threshold))).toEqual(
                exhaustiveScan(metric, gramSize, choices, query, threshold),
              )
              cases++
            }
          }
        }
      }
    }
    expect(cases).toBeGreaterThan(2000)
  })

  it('matches on randomised corpora', () => {
    const letters = fc.constantFrom('a', 'b', 'c', '😀', '\ud800', ' ')
    const text = fc.array(letters, { maxLength: 12 }).map((parts) => parts.join(''))
    fc.assert(
      fc.property(
        fc.array(text, { maxLength: 12 }),
        text,
        fc.constantFrom(...THRESHOLDS),
        fc.constantFrom(...LIMITS),
        fc.constantFrom(2, 3),
        fc.constantFrom(...METRICS),
        (choices, query, threshold, limit, gramSize, metric) => {
          const index = indexOf(metric, gramSize, choices)
          expect(pairs(index.select(query, threshold, limit))).toEqual(
            exhaustive(metric, gramSize, choices, query, threshold, limit),
          )
          expect(pairs(index.scan(query, threshold))).toEqual(
            exhaustiveScan(metric, gramSize, choices, query, threshold),
          )
          return true
        },
      ),
      { numRuns: 400, seed: 0x5eed },
    )
  })

  it('answers nothing when a caller asks for nothing', () => {
    // `limit: 0` is a supported answer rather than an excuse, and it is the one
    // call that leaves selection with no result array to insert into. The dense
    // corpus matters here: it puts every choice into the walk, so the empty
    // room is reached with candidates in hand rather than none.
    for (const metric of METRICS) {
      for (const choices of [
        ['node', 'nodes', 'noded', 'nodex', 'nodey', 'nodez', 'qq'],
        ['abc', 'abd'],
      ]) {
        const index = indexOf(metric, 3, choices)
        for (const threshold of THRESHOLDS) {
          expect(pairs(index.select('node', threshold, 0))).toEqual([])
          expect(pairs(index.select('node', threshold, 0))).toEqual(
            exhaustive(metric, 3, choices, 'node', threshold, 0),
          )
        }
      }
    }
  })
})

describe('the key scheme', () => {
  it('reaches every rung the gram size allows', () => {
    expect(feasibleRadices(1)).toEqual([0x100, 0x1_0000, 0x11_0000])
    expect(feasibleRadices(2)).toEqual([0x100, 0x1_0000, 0x11_0000])
    expect(feasibleRadices(3)).toEqual([0x100, 0x1_0000])
    expect(feasibleRadices(6)).toEqual([0x100])
    // Seven bytes is 2^56, past a safe integer, so no packed rung survives and
    // the joined-string scheme is the only one left.
    expect(feasibleRadices(7)).toEqual([])
  })

  it('widens from a byte to BMP, and again to strings, inside one choice', () => {
    // A byte-radix index, then a lone surrogate that needs BMP, then an astral
    // character that no trigram radix holds at all.
    const choices = ['abc', '\ud800bc', '😀bc']
    for (const metric of METRICS) {
      const index = indexOf(metric, 3, choices)
      for (const query of choices) {
        expect(pairs(index.select(query, 0.99, 1))).toEqual(
          exhaustive(metric, 3, choices, query, 0.99, 1),
        )
      }
    }
  })

  it('keeps a joined-string index exact', () => {
    // Gram size 3 over astral text has no feasible packed radix at all.
    const choices = ['😀😁😂', '😀😁😃', '😀😁😂😄']
    for (const metric of METRICS) {
      const index = indexOf(metric, 3, choices)
      for (const query of [...choices, '😀😁']) {
        expect(pairs(index.select(query, null, null))).toEqual(
          exhaustive(metric, 3, choices, query, null, null),
        )
      }
    }
  })

  it('drops to joined strings for a negative element', () => {
    // An array choice may hold any integer, and positional packing has no room
    // below zero — so a negative element takes the whole index to strings.
    const builder = createDiceIndexBuilder(2)
    for (const choice of [
      [1, 2, 3],
      [-1, 2, 3],
      [1, 2, 3],
    ])
      builder.add(choice)
    const index = builder.seal()
    expect(pairs(index.select([1, 2, 3], 0.99, 3))).toEqual([
      { id: 0, score: 1 },
      { id: 2, score: 1 },
    ])
    expect(pairs(index.select([-1, 2, 3], 0.99, 3))).toEqual([{ id: 1, score: 1 }])
  })

  it('starts on joined strings when no packed rung can hold the gram', () => {
    // Seven bytes is 2^56, past a safe integer, so the ladder is empty and the
    // index is string-keyed from the first choice.
    const builder = createDiceIndexBuilder(7)
    for (const choice of ['abcdefgh', 'abcdefgi']) builder.add(choice)
    const index = builder.seal()
    expect(pairs(index.select('abcdefgh', 0.5, 2))).toEqual(
      exhaustive('dice', 7, ['abcdefgh', 'abcdefgi'], 'abcdefgh', 0.5, 2),
    )
  })

  it('leaves an already-joined key alone when the scheme widens', () => {
    expect(repackKey('1,2,3', 0x100, null, 3)).toBe('1,2,3')
    expect(repackKey(0x616263, 0x100, 0x1_0000, 3)).toBe(
      0x61 * 0x1_0000 * 0x1_0000 + 0x62 * 0x1_0000 + 0x63,
    )
    expect(repackKey(0x616263, 0x100, null, 3)).toBe('97,98,99')
  })

  it('counts a query gram no packed index could hold', () => {
    // The index is byte-keyed; the query's astral grams cannot appear in it, and
    // still have to count toward the query's own gram count and norm.
    const choices = ['abcd', 'abce']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      expect(pairs(index.select('ab😀cd', null, null))).toEqual(
        exhaustive(metric, 2, choices, 'ab😀cd', null, null),
      )
    }
  })
})

describe('the posting representation', () => {
  it('stores no counts when nothing repeats, and widens when it does', () => {
    for (const metric of METRICS) {
      for (const repeats of [1, 2, 300, 70_000]) {
        const choices = ['ab', `${'a'.repeat(repeats + 1)}b`]
        const index = indexOf(metric, 2, choices)
        expect(pairs(index.select('aa', null, 2))).toEqual(
          exhaustive(metric, 2, choices, 'aa', null, 2),
        )
      }
    }
  })

  it('inverts a list that covers most of the corpus', () => {
    // `no` is in six of seven choices, past the two-thirds cutoff.
    const choices = ['node', 'nodes', 'noded', 'nodex', 'nodey', 'nodez', 'qq']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      for (const threshold of THRESHOLDS) {
        expect(pairs(index.select('node', threshold, 3))).toEqual(
          exhaustive(metric, 2, choices, 'node', threshold, 3),
        )
        expect(pairs(index.scan('node', threshold))).toEqual(
          exhaustiveScan(metric, 2, choices, 'node', threshold),
        )
      }
    }
  })

  it('inverts a list when no frequency anywhere exceeds one', () => {
    // Dense with `counts === null`: `ab` is in six of seven choices and no gram
    // repeats within any of them, so an exception can only be an absence.
    const choices = ['abc', 'abd', 'abe', 'abf', 'abg', 'abh', 'xyz']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      for (const threshold of THRESHOLDS) {
        expect(pairs(index.select('abc', threshold, 3))).toEqual(
          exhaustive(metric, 2, choices, 'abc', threshold, 3),
        )
        expect(pairs(index.scan('abc', threshold))).toEqual(
          exhaustiveScan(metric, 2, choices, 'abc', threshold),
        )
      }
    }
  })

  it('walks a sparse list under a scan a dense list already widened', () => {
    // `ab` is dense and `zq` is not, so one query reaches both and the sparse
    // walk runs with the touched set already abandoned.
    const choices = ['abc', 'abd', 'abe', 'abf', 'abg', 'abzq', 'zq']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      expect(pairs(index.select('abzq', null, null))).toEqual(
        exhaustive(metric, 2, choices, 'abzq', null, null),
      )
    }
  })

  it('walks a counted sparse list under a widened scan', () => {
    // The same, with a repeat somewhere so the whole index carries counts.
    const choices = ['abc', 'abd', 'abe', 'abf', 'abg', 'abzqzq', 'zq']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      // Both sides of the shared minimum: the query holds `zq` twice in the
      // first and once in the second, against a choice that holds it twice.
      for (const query of ['abzqzq', 'abzq']) {
        expect(pairs(index.select(query, null, null))).toEqual(
          exhaustive(metric, 2, choices, query, null, null),
        )
      }
    }
  })

  it('inverts a list whose members repeat the gram', () => {
    // Dense with a counts array: most choices hold `aa`, and some hold it twice.
    const choices = ['aab', 'aaab', 'aaac', 'aad', 'aae', 'aaf', 'zz']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      expect(pairs(index.select('aab', null, null))).toEqual(
        exhaustive(metric, 2, choices, 'aab', null, null),
      )
    }
  })

  it('narrows posting ids to the corpus, and stays exact either side of the bound', () => {
    // A wrong bound here does not throw: it wraps an id and answers the wrong
    // choice, so both sizes around 65,536 are pinned with the only match last.
    for (const choiceCount of [0x1_0000, 0x1_0001]) {
      const choices: string[] = new Array<string>(choiceCount).fill('')
      const last = choiceCount - 1
      choices[last] = 'abc'
      const index = createDiceIndexBuilder(3)
      for (const choice of choices) index.add(choice)
      const sealed = index.seal()
      expect(pairs(sealed.select('abc', 0.5, 1))).toEqual([{ id: last, score: 1 }])
    }
  })
})

describe('choices and queries with no grams', () => {
  it('scores an equal gramless pair 1 and everything else 0', () => {
    const choices = ['ab', '', 'ab', 'zz', '']
    for (const metric of METRICS) {
      const index = indexOf(metric, 3, choices)
      for (const threshold of THRESHOLDS) {
        for (const limit of LIMITS) {
          expect(pairs(index.select('ab', threshold, limit))).toEqual(
            exhaustive(metric, 3, choices, 'ab', threshold, limit),
          )
        }
        expect(pairs(index.scan('ab', threshold))).toEqual(
          exhaustiveScan(metric, 3, choices, 'ab', threshold),
        )
      }
    }
  })

  it('answers a gramless query against a corpus that has grams', () => {
    const choices = ['abcd', 'abce']
    for (const metric of METRICS) {
      const index = indexOf(metric, 3, choices)
      expect(pairs(index.select('x', null, null))).toEqual(
        exhaustive(metric, 3, choices, 'x', null, null),
      )
      expect(pairs(index.select('x', 0.5, null))).toEqual(
        exhaustive(metric, 3, choices, 'x', 0.5, null),
      )
    }
  })

  it('scores a gramless choice 0 rather than dividing by nothing', () => {
    // `''` has no grams and no norm; a Cosine score of `0/0` clamped to 1 was a
    // real bug, and only a dense list reaches such a choice at all.
    const choices = ['😀c', '😀c', '']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      expect(pairs(index.select('😀c', null, null))).toEqual(
        exhaustive(metric, 2, choices, '😀c', null, null),
      )
    }
  })
})

describe('ordering', () => {
  it('breaks a tie on the earlier id, whatever order the postings arrive in', () => {
    // The query's grams are `ab` then `bc`, so accumulation reaches choice 1
    // before choice 0 and the touched set is descending. Both score the same, so
    // the winner is decided by the tie rule alone — and dropping it answers
    // choice 1, which is the bug this pins.
    const choices = ['bc', 'ab']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      const top = pairs(index.select('abc', null, 1))
      expect(top).toEqual(exhaustive(metric, 2, choices, 'abc', null, 1))
      expect(top[0].id).toBe(0)
      expect(pairs(index.select('abc', null, 2))).toEqual(
        exhaustive(metric, 2, choices, 'abc', null, 2),
      )
    }
  })

  it('breaks a tie on the earlier id when the result is already full', () => {
    // The same shape with a third, worse choice in the way, so the tie is
    // resolved by displacing the last entry rather than by an empty slot.
    const choices = ['bc', 'ab', 'zz']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      expect(pairs(index.select('abc', null, 1))).toEqual(
        exhaustive(metric, 2, choices, 'abc', null, 1),
      )
    }
  })

  it('interleaves zero-scoring choices by id when scanning', () => {
    // The trap `scan` exists for: ranked order puts the matches first, and
    // collection order puts choice 0 first even though it scores nothing.
    const choices = ['zzzz', 'abcd', 'yyyy', 'abcd']
    for (const metric of METRICS) {
      const index = indexOf(metric, 3, choices)
      const scanned = pairs(index.scan('abcd', null))
      expect(scanned.map((row) => row.id)).toEqual([0, 1, 2, 3])
      expect(scanned).toEqual(exhaustiveScan(metric, 3, choices, 'abcd', null))
    }
  })

  it('confines a scan to the touched choices under a positive threshold', () => {
    const choices = ['zzzz', 'abcd', 'yyyy', 'abcd']
    for (const metric of METRICS) {
      const index = indexOf(metric, 3, choices)
      expect(pairs(index.scan('abcd', 0.5))).toEqual(
        exhaustiveScan(metric, 3, choices, 'abcd', 0.5),
      )
    }
  })
})

describe('what an index refuses', () => {
  it('refuses a choice whose elements are not integers', () => {
    const builder = createDiceIndexBuilder(2)
    expect(() => builder.add([{}, {}, {}])).toThrow(TypeError)
    expect(() => builder.add([{}, {}, {}])).toThrow(/integer elements only/)
  })

  it('refuses a query whose elements are not integers', () => {
    const index = indexOf('dice', 2, ['abc'])
    expect(() => index.select([{}, {}, {}], null, 1)).toThrow(TypeError)
  })

  it('refuses what the fixed-width arrays cannot address', () => {
    expect(() => assertAddressable(0x1_0000_0000, 0, 0)).toThrow(RangeError)
    expect(() => assertAddressable(0x1_0000_0000, 0, 0)).toThrow(/4294967295 choices/)
    expect(() => assertAddressable(1, 0x1_0000_0000, 0)).toThrow(/posting entries/)
    expect(() => assertAddressable(1, 0, 0x1_0000_0000)).toThrow(/grams/)
    expect(() => assertAddressable(0xffff_ffff, 0xffff_ffff, 0xffff_ffff)).not.toThrow()
  })

  it('refuses a query too large for the narrow Dice accumulator', () => {
    expect(() => assertQueryIndexable(0x8000_0000)).toThrow(RangeError)
    expect(() => assertQueryIndexable(0x8000_0000)).toThrow(/2147483647 grams/)
    expect(() => assertQueryIndexable(0x7fff_ffff)).not.toThrow()
  })

  it('refuses a Cosine pair whose dot product would leave the exact integers', () => {
    // The bound is a product, not a length: a query is refused against a corpus
    // holding one enormous choice and accepted against the same query length
    // when nothing in the corpus is long. Above it a dense list's `q·(c-1) + q`
    // and a sparse list's `q·c` stop agreeing — 12358404163972748 against
    // 12358404163972750 at these two counts.
    expect(() => assertCosineExact(116_982_125, 105_643_526)).toThrow(RangeError)
    expect(() => assertCosineExact(116_982_125, 105_643_526)).toThrow(/cosine query/)
    expect(116_982_125 * (105_643_526 - 1) + 116_982_125).not.toBe(
      116_982_125 * 105_643_526,
    )
    expect(() => assertCosineExact(116_982_125, 32)).not.toThrow()
    expect(() => assertCosineExact(0x7fff_ffff, 0x7fff_ffff)).toThrow(RangeError)
    // `MAX_SAFE_INTEGER` is itself safe, so a product landing exactly on it is
    // the last accepted pair rather than the first refused one.
    expect(6361 * 69431 * 20_394_401).toBe(Number.MAX_SAFE_INTEGER)
    expect(() => assertCosineExact(6361 * 69431, 20_394_401)).not.toThrow()
  })

  it('refuses a Cosine pair whose squared norms would leave the exact integers', () => {
    // The second half of the denominator, and a second spelling of the same
    // sum: this file adds `2c + 1` an occurrence, a packed profile adds `c²` a
    // distinct gram. Both are exact while the norm is a safe integer, and one
    // gram repeated 268,435,459 times is where they part.
    const repeated = 268_435_459
    let stepwise = 0
    for (let count = 0; count < repeated; count++) stepwise += 2 * count + 1
    expect(stepwise).not.toBe(repeated * repeated)
    expect(stepwise - repeated * repeated).toBe(-16)

    expect(() => assertCosineNormsExact(repeated * repeated, 4)).toThrow(RangeError)
    expect(() => assertCosineNormsExact(4, repeated * repeated)).toThrow(
      /repeated this often/,
    )
    // A norm, not a length: 100 million distinct grams never come near this,
    // and the bound must not refuse them for being numerous.
    expect(() => assertCosineNormsExact(100_000_000, 100_000_000)).not.toThrow()
    // Each side alone, since either can carry the norm out of range while the
    // other is a single gram: the largest safe norm passes, and the first
    // double above it does not.
    expect(() => assertCosineNormsExact(Number.MAX_SAFE_INTEGER, 1)).not.toThrow()
    expect(() => assertCosineNormsExact(1, Number.MAX_SAFE_INTEGER)).not.toThrow()
    expect(() =>
      assertCosineNormsExact(Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER),
    ).not.toThrow()
    expect(() => assertCosineNormsExact(Number.MAX_SAFE_INTEGER + 1, 1)).toThrow(
      RangeError,
    )
    expect(() => assertCosineNormsExact(1, Number.MAX_SAFE_INTEGER + 1)).toThrow(
      RangeError,
    )
  })

  it('takes a gramless sequence of any element the exhaustive scorer takes', () => {
    // Not an oversight that `add` skips the integer check here: the gramless
    // branch stores elements and compares them, and refusing what the metric
    // itself scores would be the one thing an index may not do — disagree with
    // the scorer it stands in for. Longer sequences still reach the check.
    const scorer = createScorer(diceSimilarity, { gramSize: 3 })
    const builder = createDiceIndexBuilder(3)
    expect(() => builder.add([{}, {}])).not.toThrow()
    expect(builder.seal().select([{}, {}], null, 1).scores[0]).toBe(0)
    expect(scorer.score([{}, {}], [{}, {}])).toBe(0)
    const shared = [{}, {}]
    const sharedBuilder = createDiceIndexBuilder(3)
    sharedBuilder.add(shared)
    expect(sharedBuilder.seal().select(shared, null, 1).scores[0]).toBe(1)
    expect(scorer.score(shared, shared)).toBe(1)
  })

  it('scores nothing when the threshold is past the scale', () => {
    // A gramless pair is the one shape that can reach 1, so a threshold above it
    // has to leave the result empty rather than admitting the equal choice.
    const index = indexOf('dice', 3, ['ab', 'ab'])
    expect(pairs(index.select('ab', 1.5, 2))).toEqual([])
    expect(pairs(index.scan('ab', 1.5))).toEqual([])
  })

  it('is one-shot', () => {
    const builder = createDiceIndexBuilder(2)
    builder.add('abc')
    builder.seal()
    expect(() => builder.add('abcd')).toThrow(TypeError)
    expect(() => builder.add('abcd')).toThrow(/already sealed/)
    expect(() => builder.seal()).toThrow(/already sealed/)
  })
})

describe('the capability a metric declares', () => {
  it('is offered by both similarity metrics and answers like the Matcher', () => {
    const choices = ['abcd', 'abce', 'zzzz']
    for (const metric of METRICS) {
      const scorer = createScorer(metric === 'dice' ? diceSimilarity : cosineSimilarity, {
        gramSize: 3,
      })
      const indexChoices = scorerCompilation(scorer).indexChoices
      expect(indexChoices).toBeTypeOf('function')
      if (indexChoices === undefined) throw new Error('no index capability')
      const builder = indexChoices()
      for (const choice of choices) builder.add(choice)
      expect(pairs(builder.seal().select('abcd', 0.5, 3))).toEqual(
        exhaustive(metric, 3, choices, 'abcd', 0.5, 3),
      )
    }
  })

  it('is offered at the gram size the scorer was configured with', () => {
    const choices = ['abcd', 'abce']
    for (const gramSize of [2, 3, 4]) {
      const indexChoices = scorerCompilation(
        createScorer(diceSimilarity, { gramSize }),
      ).indexChoices
      if (indexChoices === undefined) throw new Error('no index capability')
      const builder = indexChoices()
      for (const choice of choices) builder.add(choice)
      expect(pairs(builder.seal().select('abcd', null, 2))).toEqual(
        exhaustive('dice', gramSize, choices, 'abcd', null, 2),
      )
    }
  })

  it('is absent on the distance direction', () => {
    expect(scorerCompilation(createScorer(diceDistance)).indexChoices).toBeUndefined()
    expect(scorerCompilation(createScorer(cosineDistance)).indexChoices).toBeUndefined()
  })
})

describe('reuse', () => {
  it('answers repeated queries from the same scratch', () => {
    const choices = ['node', 'nodes', 'noded', 'nodex', 'nodey', 'nodez', 'qq']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      for (const query of ['node', 'qq', 'nodes', 'node', 'zzzz']) {
        expect(pairs(index.select(query, null, 3))).toEqual(
          exhaustive(metric, 2, choices, query, null, 3),
        )
      }
      // A dense query, then a sparse one, then a dense one again: the sparse
      // walk has to see an accumulator the dense scan left clean.
      for (const query of ['node', 'qq', 'node']) {
        expect(pairs(index.scan(query, 0.1))).toEqual(
          exhaustiveScan(metric, 2, choices, query, 0.1),
        )
      }
    }
  })
})
