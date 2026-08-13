import { describe, expect, it } from 'vitest'

import { indexOf, pairs } from '../../../../../testing/invertedIndex.js'
import { createScorer } from '../../../../core/scoring/scorer.js'
import { similarity as diceSimilarity } from '../../../dice/index.js'
import { assertAddressable } from './builder.js'
import { assertCosineExact, assertCosineNormsExact } from './cosine.js'
import { assertDiceAccumulatorExact, createDiceIndexBuilder } from './dice.js'

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
    expect(() => assertDiceAccumulatorExact(0x8000_0000)).toThrow(RangeError)
    expect(() => assertDiceAccumulatorExact(0x8000_0000)).toThrow(/2147483647 grams/)
    expect(() => assertDiceAccumulatorExact(0x7fff_ffff)).not.toThrow()
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
