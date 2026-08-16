// Not ported from RapidFuzz — the shared bitmask builders are ours, and so is
// the question these cover: which of the two owners each thing a build produces
// belongs to.
//
// The module owns what it can bound and reuse — the direct symbol table, the
// row vectors, a mask pool up to `RETAINED_MASK_WORDS`, and the one-entry memo
// over them. The operation owns the rest: a mask pool too big to keep, and any
// lookup holding elements the caller handed us, which may be objects of any
// size. Everything operation-owned is reachable only through the object a build
// returns, so it goes when the kernel that read it does.
//
// The results themselves are covered wherever the kernels are; what is only
// visible here is where each buffer ends up afterwards.
import { describe, expect, it } from 'vitest'

import {
  blockMasksFor,
  buildWordMasks,
  maskPoolOf,
  resetBitVectorScratch,
} from './blockMasks.js'
import { wordCount } from './words.js'

const RETAINED_MASK_WORDS = 1 << 19
const MASK_PATTERN_LIMIT = 4096

/** Distinct objects, so every element takes a block of its own. */
function unique(count: number, tag: string): ReadonlyArray<unknown> {
  return Array.from({ length: count }, (_, index) => ({ tag, index }))
}

/** Distinct characters, so a string pattern can outgrow the pool as well. */
function distinctText(count: number): string {
  return Array.from({ length: count }, (_, index) =>
    String.fromCharCode(0x4e00 + index),
  ).join('')
}

function blocksOf(pattern: ArrayLike<unknown>): ReturnType<typeof blockMasksFor> {
  return blockMasksFor(pattern, 0, pattern.length, wordCount(pattern.length))
}

describe('what a build leaves with the module', () => {
  it('builds a pool within the cap into the module’s own', () => {
    resetBitVectorScratch()

    const first = blocksOf('abcdefghij'.repeat(10))
    const second = blocksOf('klmnopqrst'.repeat(10))

    expect(second.pool).toBe(maskPoolOf())
    expect(second.pool).toBe(first.pool)
    // The same empty map both times: an ordinary build allocates no lookup.
    expect(second.wide).toBe(first.wide)
    expect(second.wide.size).toBe(0)
  })

  it('keeps a pool past the cap to the build that needed it', () => {
    resetBitVectorScratch()

    const built = blocksOf(unique(MASK_PATTERN_LIMIT + 1, 'past'))

    expect(built.pool.length).toBeGreaterThan(RETAINED_MASK_WORDS)
    expect(maskPoolOf().length).toBe(RETAINED_MASK_WORDS)
    expect(maskPoolOf()).not.toBe(built.pool)
  })

  // A memo entry names masks the module still holds. One recorded for a build
  // that kept its pool would send the next identical call to a pool holding
  // some other pattern's masks. No string the memo accepts can reach that
  // state — which is the whole reason the cap is the memo's size — so the
  // guard is only reachable from the sequences the memo never takes.
  it('does not memoise a build whose pool it does not have', () => {
    resetBitVectorScratch()

    const pattern = unique(MASK_PATTERN_LIMIT + 1, 'unmemoised')
    const first = blocksOf(pattern)
    const second = blocksOf(pattern)

    expect(second.stamp).not.toBe(first.stamp)
    expect(second.pool).not.toBe(first.pool)
  })

  // The pairing the cap exists for: the largest pattern the memo accepts, with
  // every element distinct, is still a pattern the module can hold the masks
  // for. Lower the cap or raise `MASK_PATTERN_LIMIT` and this is what breaks.
  it('holds the masks for the largest pattern the memo takes', () => {
    resetBitVectorScratch()

    const pattern = distinctText(MASK_PATTERN_LIMIT)
    const first = blocksOf(pattern)
    const second = blocksOf(pattern)

    expect(first.pool).toBe(maskPoolOf())
    expect(second.stamp).toBe(first.stamp)
  })

  // Either side of the boundary itself, in elements rather than characters:
  // 4096 distinct take 128 words each and land exactly on the cap, 4097 take
  // 129 and are handed twice it, which is the first size the module refuses.
  it('draws the line one element past the pattern limit', () => {
    resetBitVectorScratch()

    const held = blocksOf(unique(MASK_PATTERN_LIMIT, 'held'))
    expect(held.pool).toBe(maskPoolOf())
    expect(held.pool.length).toBe(RETAINED_MASK_WORDS)

    const spilled = blocksOf(unique(MASK_PATTERN_LIMIT + 1, 'spilled'))
    expect(spilled.pool.length).toBe(2 * RETAINED_MASK_WORDS)
    expect(maskPoolOf().length).toBe(RETAINED_MASK_WORDS)
    expect(maskPoolOf()).not.toBe(spilled.pool)

    // And again, to show the local pool is rebuilt exactly while the retained
    // prefix underneath it is the one that gets reused.
    const again = blocksOf(unique(MASK_PATTERN_LIMIT + 1, 'spilled'))
    expect(again.pool.length).toBe(spilled.pool.length)
    expect(maskPoolOf().length).toBe(RETAINED_MASK_WORDS)
  })

  it('still memoises a repeat it can serve from the pool it kept', () => {
    resetBitVectorScratch()

    const pattern = 'abcdefghij'.repeat(10)
    const first = blocksOf(pattern)
    const second = blocksOf(pattern)

    expect(second.stamp).toBe(first.stamp)
    expect(second.pool).toBe(maskPoolOf())
  })
})

describe('what a build leaves holding the caller elements', () => {
  it('gives a wide build a lookup of its own', () => {
    resetBitVectorScratch()

    const elements = unique(100, 'a')
    const built = blocksOf(elements)

    expect(built.wide.size).toBe(100)
    expect(built.wide.has(elements[0])).toBe(true)
  })

  it('holds no element of an earlier build, at any size', () => {
    resetBitVectorScratch()

    const small = unique(100, 'small')
    const large = unique(MASK_PATTERN_LIMIT + 1, 'large')
    const held = [blocksOf(small).wide, blocksOf(large).wide]

    const after = blocksOf('abcdefghij'.repeat(10))

    expect(after.wide.size).toBe(0)
    expect(held[0].has(small[0])).toBe(true)
    expect(held[1].has(large[0])).toBe(true)
    expect(held[0]).not.toBe(held[1])
  })

  // The single-word builder writes bit masks rather than pool offsets, and had
  // a lookup of its own to clear. A pattern of one object is enough to prove
  // the ownership, and a byte count would not have caught it.
  it('holds no element of a single-word build either', () => {
    resetBitVectorScratch()

    const elements = unique(4, 'word')
    const built = buildWordMasks(elements, 0, elements.length)

    expect(built.wide.size).toBe(4)

    const after = buildWordMasks('abcd', 0, 4)
    expect(after.wide.size).toBe(0)
    expect(after.wide).not.toBe(built.wide)
  })

  it('allocates no lookup when a single-word build has nothing wide', () => {
    resetBitVectorScratch()

    const first = buildWordMasks('abcd', 0, 4)
    const second = buildWordMasks('efgh', 0, 4)

    expect(second.wide).toBe(first.wide)
    expect(second.pool.length).toBe(0)
  })
})
