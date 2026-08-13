import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import { cosineSimilarity } from '../../src/algorithms/cosine/implementation.js'
import { diceSimilarity } from '../../src/algorithms/dice/implementation.js'
import { canonicalRadix } from '../../src/algorithms/shared/gramKey.js'
import {
  buildProfile,
  directSharedFrequency,
  dotProduct,
  dotProductKernel,
  elementsEqual,
  NGramProfile,
  packedProfile,
  profileOfElements,
  sharedFrequency,
  sharedFrequencyKernel,
  trieProfile,
} from '../../src/algorithms/shared/ngram.js'
import { Cosine, Dice } from '../support/scorers.js'

describe('the profile at every depth', () => {
  it('counts, norms and intersects a repeated gram', () => {
    // `aaaa` is one distinct bigram with a count of 3.
    const four = buildProfile('aaaa', 2)
    const three = buildProfile('aaa', 2)
    expect(four.gramCount).toBe(3)
    expect(four.squaredNorm).toBe(9)
    expect(three.gramCount).toBe(2)
    expect(three.squaredNorm).toBe(4)
    expect(sharedFrequency(four, three)).toBe(2)
    expect(dotProduct(four, three)).toBe(6)
    // Dice reads the counts, cosine reads the direction — and these two
    // frequency vectors are parallel.
    expect(diceSimilarity('aaaa', 'aaa')).toBeCloseTo(0.8, 12)
    expect(cosineSimilarity('aaaa', 'aaa')).toBe(1)
  })

  it('specializes gram sizes 1 to 3 without changing the answer', () => {
    // gramSize 1, 2 and 3 take literal loops; 4 and up take the explicit stack.
    // The four paths have to agree wherever they overlap.
    for (const gramSize of [1, 2, 3, 4, 9]) {
      const a = buildProfile('abcabcab', gramSize)
      const b = buildProfile('abcabc', gramSize)
      expect(sharedFrequency(a, b), `gramSize ${gramSize}`).toBe(sharedFrequency(b, a))
      expect(dotProduct(a, b), `gramSize ${gramSize}`).toBe(dotProduct(b, a))
    }
    expect(sharedFrequency(buildProfile('aba', 1), buildProfile('ab', 1))).toBe(2)
    expect(sharedFrequency(buildProfile('aba', 2), buildProfile('ab', 2))).toBe(1)
    expect(sharedFrequency(buildProfile('abcd', 3), buildProfile('abc', 3))).toBe(1)
  })

  it('leaves a trigram unmatched at whichever level first diverges', () => {
    // The literal trigram loop misses at two levels, and each has its own exit:
    // an absent first element never reaches the second, and an absent second
    // never reaches the counts.
    const abc = buildProfile('abc', 3)
    for (const other of ['xbc', 'axc']) {
      const diverging = buildProfile(other, 3)
      expect(sharedFrequency(abc, diverging), other).toBe(0)
      expect(dotProduct(abc, diverging), other).toBe(0)
    }
    // Both levels match and only the last element differs, which is where the
    // counts of two reachable leaves are compared.
    expect(sharedFrequency(abc, buildProfile('abx', 3))).toBe(0)
    expect(dotProduct(buildProfile('abcabc', 3), buildProfile('abcab', 3))).toBe(4)
  })

  it('has nothing to intersect when a profile has no grams', () => {
    for (const gramSize of [1, 2, 3, 5]) {
      const empty = buildProfile('', gramSize)
      const full = buildProfile('abcdef', gramSize)
      expect(empty.gramCount, `gramSize ${gramSize}`).toBe(0)
      expect(empty.squaredNorm, `gramSize ${gramSize}`).toBe(0)
      expect(sharedFrequency(empty, full), `gramSize ${gramSize}`).toBe(0)
      expect(sharedFrequency(full, empty), `gramSize ${gramSize}`).toBe(0)
      expect(dotProduct(empty, full), `gramSize ${gramSize}`).toBe(0)
      expect(dotProduct(empty, empty), `gramSize ${gramSize}`).toBe(0)
    }
    expect(sharedFrequency(buildProfile('ab', 3), buildProfile('abcd', 3))).toBe(0)
  })

  it('retains the elements only when there is nothing else to compare', () => {
    expect(buildProfile('ab', 3).elements).not.toBeNull()
    expect(buildProfile('abcd', 2).elements).toBeNull()
  })
})

describe('element equality', () => {
  it('treats a gram holding NaN as unmatchable at every window', () => {
    // Unmatchable grams still count toward `gramCount` and `squaredNorm`, so
    // the denominators stay right and cosine never divides by zero.
    const single = buildProfile([Number.NaN], 1)
    expect(single.gramCount).toBe(1)
    expect(single.squaredNorm).toBe(1)
    expect(sharedFrequency(single, single)).toBe(0)
    expect(dotProduct(single, single)).toBe(0)

    const pair = buildProfile([Number.NaN, Number.NaN], 1)
    expect(pair.gramCount).toBe(2)
    expect(pair.squaredNorm).toBe(2)
    expect(sharedFrequency(pair, pair)).toBe(0)

    // Both bigram windows of `[1, NaN, 1]` contain the NaN; at gramSize 1 only
    // the middle element does, and the two `1`s still match.
    const bigrams = buildProfile([1, Number.NaN, 1], 2)
    expect(bigrams.gramCount).toBe(2)
    expect(bigrams.squaredNorm).toBe(2)
    expect(sharedFrequency(bigrams, bigrams)).toBe(0)

    const unigrams = buildProfile([1, Number.NaN, 1], 1)
    expect(unigrams.gramCount).toBe(3)
    expect(unigrams.squaredNorm).toBe(5)
    expect(sharedFrequency(unigrams, unigrams)).toBe(2)
    expect(
      Dice.similarity([1, Number.NaN, 1], [1, Number.NaN, 1], { gramSize: 1 }),
    ).toBeCloseTo((2 * 2) / (3 + 3), 12)

    // A trailing NaN takes out the windows that reach it and no others.
    const trailing = buildProfile(['a', 'b', 'c', Number.NaN], 2)
    expect(trailing.gramCount).toBe(3)
    expect(sharedFrequency(trailing, buildProfile(['a', 'b', 'c'], 2))).toBe(2)
  })

  it('reaches back a whole window for a leading NaN at depth 3', () => {
    // The depth-3 builder unrolls the scan of the elements before the first
    // window, so which of the two it finds — and that a later one wins — is
    // worth pinning: an off-by-one here is a wrong `squaredNorm`, not a crash.
    const second = buildProfile(['a', Number.NaN, 'c', 'd', 'e'], 3)
    expect(second.gramCount).toBe(3)
    // Two windows reach the NaN and count only toward the norm; `cde` is the
    // one gram inserted.
    expect(second.squaredNorm).toBe(3)
    expect(sharedFrequency(second, buildProfile(['x', 'y', 'c', 'd', 'e'], 3))).toBe(1)

    const first = buildProfile([Number.NaN, 'b', 'c', 'd'], 3)
    expect(first.gramCount).toBe(2)
    expect(first.squaredNorm).toBe(2)
    expect(sharedFrequency(first, buildProfile(['b', 'c', 'd'], 3))).toBe(1)
  })

  it('matches negative zero to positive zero', () => {
    // `===` and SameValueZero agree here, so no special handling is needed and
    // none is present — this pins that.
    expect(elementsEqual([-0], [0])).toBe(true)
    expect(sharedFrequency(buildProfile([-0], 1), buildProfile([0], 1))).toBe(1)
    expect(Dice.similarity([-0, 1], [0, 1], { gramSize: 1 })).toBe(1)
    expect(Cosine.similarity([-0, 1], [0, 1], { gramSize: 1 })).toBe(1)
  })

  it('compares objects by identity and strings by value', () => {
    const token = {}
    expect(sharedFrequency(buildProfile([token], 1), buildProfile([token], 1))).toBe(1)
    expect(sharedFrequency(buildProfile([token], 1), buildProfile([{}], 1))).toBe(0)
    expect(sharedFrequency(buildProfile(['ab'], 1), buildProfile(['ab'], 1))).toBe(1)
  })

  it('agrees between a string and its code points', () => {
    const text = 'the wonderful 😀 mets'
    const codePoints = Array.from(text)
    expect(diceSimilarity(text, codePoints)).toBe(1)
    expect(cosineSimilarity(text, codePoints)).toBe(1)
    expect(sharedFrequency(buildProfile(text, 2), buildProfile(codePoints, 2))).toBe(
      buildProfile(text, 2).gramCount,
    )
  })
})

describe('the bounded kernel', () => {
  it('carries a suffix total past 2^31 without turning it negative', () => {
    // A sequence may be 0xffff_ffff long, so the frequency still to come can
    // pass 2^31. Held signed it wraps negative, the bound reads as unreachable
    // and the kernel abandons a candidate that shares everything it needs.
    // Built here rather than measured: the sequence itself would not fit.
    const counts = new Map<unknown, number>([
      ['common', 0x8000_0000],
      ['rare', 1],
    ])
    const query = new NGramProfile(
      1,
      0x8000_0001,
      0,
      { kind: 'trie', root: { children: null, counts } },
      null,
    )
    const choice = buildProfile(['rare'], 1)

    expect(sharedFrequencyKernel(query)(choice, 1)).toBe(1)
  })

  it('stops early only once the minimum is genuinely out of reach', () => {
    for (const gramSize of [1, 2, 3, 4]) {
      const query = buildProfile('abcabcab', gramSize)
      const kernel = sharedFrequencyKernel(query)
      const exact = sharedFrequency(query, buildProfile('abcabc', gramSize))
      // Asking for no more than the intersection holds must not cut the walk.
      expect(
        kernel(buildProfile('abcabc', gramSize), exact),
        `gramSize ${gramSize}`,
      ).toBe(exact)
      expect(kernel(buildProfile('abcabc', gramSize), 0), `gramSize ${gramSize}`).toBe(
        exact,
      )
      // Above it the answer may be any count below the minimum, never at it.
      expect(
        kernel(buildProfile('abcabc', gramSize), exact + 1),
        `gramSize ${gramSize}`,
      ).toBeLessThan(exact + 1)
    }
  })
})

/** The two representations of the same input, for a differential comparison. */
function bothStorages(
  elements: readonly unknown[],
  gramSize: number,
): { packed: NGramProfile; trie: NGramProfile } | null {
  if (elements.length < gramSize) return null
  const packed = packedProfile(elements, gramSize)
  if (packed === null) return null
  return { packed, trie: trieProfile(elements, gramSize) }
}

describe('packed storage', () => {
  it('packs what the canonical radix holds, and no more', () => {
    // The boundary each depth turns on, either side of it. A `<=` where `<`
    // belongs moves a profile to the other representation and nothing else
    // would say so.
    const cases: Array<[number, number]> = [
      [2, 0x11_0000],
      [3, 0x1_0000],
      [6, 0x100],
    ]
    for (const [gramSize, radix] of cases) {
      expect(canonicalRadix(gramSize), `gramSize ${gramSize}`).toBe(radix)
      const inside = new Array<number>(gramSize).fill(radix - 1)
      const outside = new Array<number>(gramSize).fill(radix)
      expect(
        profileOfElements(inside, gramSize).storage.kind,
        `gramSize ${gramSize} at radix - 1`,
      ).toBe('packed')
      expect(
        profileOfElements(outside, gramSize).storage.kind,
        `gramSize ${gramSize} at radix`,
      ).toBe('trie')
      expect(profileOfElements([0, 0, 0, 0, 0, 0], gramSize).storage.kind).toBe('packed')
    }
    // No rung reaches seven elements, so depth alone decides.
    expect(profileOfElements([1, 1, 1, 1, 1, 1, 1], 7).storage.kind).toBe('trie')
  })

  it('falls back for every element a key cannot hold', () => {
    const unpackable: Array<readonly unknown[]> = [
      [1, -1, 2],
      [1, 1.5, 2],
      [1, Number.NaN, 2],
      [1, Number.POSITIVE_INFINITY, 2],
      [1, {}, 2],
      [1, null, 2],
      [1, true, 2],
      // A mixed domain: packing would make `'b'` and `98` the same gram.
      [97, 'b', 99],
      ['a', 98, 'c'],
      // Two UTF-16 units are one astral element, which no digit holds.
      ['a', '😀', 'c'],
      ['a', 'bc', 'd'],
    ]
    for (const elements of unpackable) {
      expect(profileOfElements(elements, 2).storage.kind, JSON.stringify(elements)).toBe(
        'trie',
      )
    }
    // Astral at trigram depth exceeds a BMP digit; at bigram depth it fits.
    expect(profileOfElements([0x1_f600, 97, 98], 3).storage.kind).toBe('trie')
    expect(profileOfElements([0x1_f600, 97, 98], 2).storage.kind).toBe('packed')
  })

  it('refuses an element a rolling window meets late as readily as one it meets first', () => {
    // Depths 2 and 3 carry the window forward rather than reading every element
    // through a digit array, so *where* an unpackable element sits decides which
    // read finds it — the seeding reads before the loop, the loop's own read, or
    // the final iteration's. A generic path that validated everything up front
    // could not tell these apart; this one has to.
    for (const gramSize of [2, 3, 4]) {
      for (const bad of [Number.NaN, -1, Number.POSITIVE_INFINITY, 1.5, {}, 'bc']) {
        for (let position = 0; position < 6; position++) {
          const elements: unknown[] = [10, 11, 12, 13, 14, 15]
          elements[position] = bad
          expect(
            profileOfElements(elements, gramSize).storage.kind,
            `gramSize ${gramSize}, ${String(bad)} at ${position}`,
          ).toBe('trie')
        }
      }
    }
    // The last element belongs to no gram at depth 6 over six elements minus
    // one — every element is read regardless, because a sequence packs whole.
    expect(profileOfElements([10, 11, 12, 13, 14, Number.NaN], 5).storage.kind).toBe(
      'trie',
    )
  })

  it('refuses a character the narrow radices cannot hold', () => {
    // Depth 4 and up pack at 0x100, so a Latin-1 letter fits and `ā` does not —
    // the char domain's own radix boundary, which the number domain reaches
    // through a different comparison.
    expect(profileOfElements(['a', 'b', 'c', 'd', 'e', 'f'], 6).storage.kind).toBe(
      'packed',
    )
    expect(profileOfElements(['a', 'b', 'c', 'd', 'e', 'ā'], 6).storage.kind).toBe('trie')
  })

  it('walks a trie query against a trie choice at every depth', () => {
    // An object element keeps both sides off the packed path, which is where
    // the flattened query arms still do the work.
    const odd = {}
    // Run both orientations: `min` takes the query's count one way and the
    // choice's the other, and only a pair whose counts differ tells them apart.
    const left = [1, 2, odd, 3]
    const right = [1, 2, odd, 1, 2, odd, 3]
    for (const gramSize of [1, 2, 3, 4]) {
      for (const [a, b] of [
        [left, right],
        [right, left],
      ]) {
        const query = profileOfElements(a, gramSize)
        const choice = profileOfElements(b, gramSize)
        const where = `gramSize ${gramSize}, ${a.length} against ${b.length}`
        expect(query.storage.kind, where).toBe('trie')
        expect(choice.storage.kind, where).toBe('trie')
        expect(sharedFrequencyKernel(query)(choice, 0), where).toBe(
          referenceShared(a, b, gramSize),
        )
        expect(dotProductKernel(query)(choice), where).toBe(referenceDot(a, b, gramSize))
      }
    }
  })

  it('shares nothing between two depths', () => {
    // Packing flattens structural identity into numeric identity: the unigram
    // `[97]` and the bigram `[0, 97]` both key to 97, where a depth-1 trie and
    // a depth-2 one could never line up. A scorer only ever compares profiles
    // of its own gram size; this is what keeps a stray internal call honest.
    const unigram = profileOfElements([97], 1)
    const bigram = profileOfElements([0, 97], 2)
    expect(unigram.storage.kind).toBe('packed')
    expect(bigram.storage.kind).toBe('packed')
    expect(sharedFrequency(unigram, bigram)).toBe(0)
    expect(sharedFrequency(bigram, unigram)).toBe(0)
    expect(dotProduct(unigram, bigram)).toBe(0)
  })

  it('spells its keys at the canonical radix for its depth', () => {
    for (const gramSize of [1, 2, 3, 4, 6]) {
      const storage = profileOfElements([1, 1, 1, 1, 1, 1], gramSize).storage
      expect(storage.kind, `gramSize ${gramSize}`).toBe('packed')
      if (storage.kind !== 'packed') continue
      expect(storage.radix, `gramSize ${gramSize}`).toBe(canonicalRadix(gramSize))
    }
  })

  it('reuses one compiled query form across a mixed corpus', () => {
    // A prepared kernel meets both storages in one search, repeatedly. The
    // query side is compiled on the first candidate of a kind and reused for
    // every later one, so the answers have to be identical across the run.
    const astral = [97, 98, 99, 0x1_f600]
    const query = profileOfElements(astral, 3)
    expect(query.storage.kind).toBe('trie')
    const kernel = sharedFrequencyKernel(query)
    const dot = dotProductKernel(query)
    const corpus = [
      profileOfElements([97, 98, 99, 100], 3),
      profileOfElements(['a', 'b', 'c', 'd'], 3),
      profileOfElements([97, 98, 99, 0x1_f600], 3),
      profileOfElements([97, 98, 99, 101], 3),
      profileOfElements(['a', 'b', 'c', 'e'], 3),
      profileOfElements([100, 101, 102, 0x1_f600], 3),
    ]
    const kinds = corpus.map((choice) => choice.storage.kind)
    expect(kinds).toEqual(['packed', 'packed', 'trie', 'packed', 'packed', 'trie'])
    // `abc` is the only gram the query and a numeric packed choice can share;
    // the char-domain choices share nothing with a numeric query.
    expect(corpus.map((choice) => kernel(choice, 0))).toEqual([1, 0, 2, 1, 0, 0])
    expect(corpus.map((choice) => dot(choice))).toEqual([1, 0, 2, 1, 0, 0])
    // Same again, now that every form is compiled.
    expect(corpus.map((choice) => kernel(choice, 0))).toEqual([1, 0, 2, 1, 0, 0])
    expect(corpus.map((choice) => dot(choice))).toEqual([1, 0, 2, 1, 0, 0])

    // And from the other side: a packed query over a corpus holding both.
    const packedQuery = profileOfElements([97, 98, 99, 100], 3)
    expect(packedQuery.storage.kind).toBe('packed')
    const fromPacked = sharedFrequencyKernel(packedQuery)
    const dotFromPacked = dotProductKernel(packedQuery)
    expect(corpus.map((choice) => fromPacked(choice, 0))).toEqual([2, 0, 1, 1, 0, 0])
    expect(corpus.map((choice) => fromPacked(choice, 0))).toEqual([2, 0, 1, 1, 0, 0])
    expect(corpus.map((choice) => dotFromPacked(choice))).toEqual([2, 0, 1, 1, 0, 0])
    expect(corpus.map((choice) => dotFromPacked(choice))).toEqual([2, 0, 1, 1, 0, 0])
  })

  it('projects a trie query into whichever domain the candidate packed', () => {
    // A query trie can hold grams of both domains at once; each packed
    // candidate sees the projection of its own, and neither borrows the other's.
    const mixed = [97, 98, 99, {}, 'a', 'b', 'c']
    const query = profileOfElements(mixed, 3)
    expect(query.storage.kind).toBe('trie')
    const kernel = sharedFrequencyKernel(query)
    expect(kernel(profileOfElements([97, 98, 99, 97], 3), 0)).toBe(1)
    expect(kernel(profileOfElements(['a', 'b', 'c', 'a'], 3), 0)).toBe(1)
    expect(kernel(profileOfElements([100, 101, 102], 3), 0)).toBe(0)
  })

  it('searches into a much longer side instead of walking it', () => {
    // Past a length ratio of 8 the walk switches to a binary search per query
    // gram. Same answers, and the bound still applies — the query drives either
    // way, so its suffix totals stay meaningful.
    const long: number[] = []
    for (let at = 0; at < 400; at++)
      long.push(97 + (at % 26), 98 + (at % 7), 99 + (at % 11))
    const choice = profileOfElements(long, 3)
    expect(choice.storage.kind).toBe('packed')
    for (const query of [
      long.slice(0, 6), // shares its grams with the head of the choice
      [1, 2, 3, 4, 5, 6], // shares nothing, so every search misses
      long.slice(1197), // shares only at the very end
    ]) {
      const profile = profileOfElements(query, 3)
      const exact = referenceShared(query, long, 3)
      expect(sharedFrequency(profile, choice), JSON.stringify(query.slice(0, 3))).toBe(
        exact,
      )
      // Both operand orders: unbounded, the shorter side drives whichever way
      // round it was passed.
      expect(sharedFrequency(choice, profile)).toBe(exact)
      expect(dotProduct(profile, choice)).toBe(referenceDot(query, long, 3))
      expect(dotProduct(choice, profile)).toBe(referenceDot(query, long, 3))
      expect(sharedFrequencyKernel(profile)(choice, 0)).toBe(exact)
      expect(dotProductKernel(profile)(choice)).toBe(referenceDot(query, long, 3))
      // Bounded, above what the pair can reach, it may stop short — never at or
      // above the minimum it was given.
      expect(sharedFrequencyKernel(profile)(choice, exact + 1)).toBeLessThan(exact + 1)
      expect(sharedFrequencyKernel(profile)(choice, exact)).toBe(exact)
    }

    // `min` has to read whichever side is smaller, and a probe meets both
    // orders: three of a gram against one occurrence, and one against three.
    const repeated = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    const onceElements = [...long, 1, 2, 3]
    const manyElements = [...long, ...repeated]
    const once = profileOfElements(onceElements, 3)
    const many = profileOfElements(manyElements, 3)
    expect(sharedFrequency(profileOfElements(repeated, 3), once)).toBe(
      referenceShared(repeated, onceElements, 3),
    )
    expect(dotProduct(profileOfElements(repeated, 3), once)).toBe(
      referenceDot(repeated, onceElements, 3),
    )
    expect(sharedFrequency(profileOfElements([1, 2, 3], 3), many)).toBe(
      referenceShared([1, 2, 3], manyElements, 3),
    )
  })

  it('keeps a profile with no grams on the trie, holding its elements', () => {
    const empty = profileOfElements(['a'], 3)
    expect(empty.storage.kind).toBe('trie')
    expect(empty.elements).not.toBeNull()
  })

  it('answers the same whether a representation is forced or chosen', () => {
    // The forced builders exist for the differential tests, so they have to be
    // semantic equivalents of `profileOfElements` — including the zero-gram
    // rule, which no packed profile can express: there is no first element to
    // take a domain from, and the elements have to survive for
    // `zeroGramSimilarity` to compare them.
    for (const [elements, gramSize] of [
      [[], 2],
      [['a'], 2],
      [['a', 'b'], 3],
      [[1, 2, 3], 9],
    ] as const) {
      const where = `${JSON.stringify(elements)} at gramSize ${gramSize}`
      expect(packedProfile(elements, gramSize), where).toBeNull()
      const forced = trieProfile(elements, gramSize)
      const chosen = profileOfElements(elements, gramSize)
      expect(forced.gramCount, where).toBe(0)
      expect(forced.elements, where).toBe(elements)
      expect(chosen.elements, where).toBe(elements)
      expect(forced.storage.kind, where).toBe('trie')
    }
    // Two such profiles compare as their sequences do, whichever builder made
    // them — which is the behaviour a packed zero-gram profile would have lost.
    expect(Dice.similarity(['a'], ['a'], { gramSize: 3 })).toBe(1)
    expect(Dice.similarity(['a'], ['b'], { gramSize: 3 })).toBe(0)
  })

  it('keeps the element domains apart', () => {
    // `convPair` aligns the public metrics, so a string and its code points
    // agree there. At this layer they are different elements, and packing both
    // to 97 would silently merge them.
    const chars = profileOfElements('abc', 1)
    const numbers = profileOfElements([97, 98, 99], 1)
    expect(chars.storage.kind).toBe('packed')
    expect(numbers.storage.kind).toBe('packed')
    expect(sharedFrequency(chars, numbers)).toBe(0)
    expect(sharedFrequency(numbers, chars)).toBe(0)
    expect(dotProduct(chars, numbers)).toBe(0)
    expect(sharedFrequencyKernel(chars)(numbers, 0)).toBe(0)
    expect(dotProductKernel(chars)(numbers)).toBe(0)
    // The public metrics are unchanged by that: they convert both sides first.
    expect(diceSimilarity('abc', [97, 98, 99])).toBe(1)
    expect(cosineSimilarity('abc', [97, 98, 99])).toBe(1)
  })

  it('answers a packed profile against a trie one in both orientations', () => {
    // Reachable through the public metrics: one astral code point at trigram
    // depth is a trie, and the BMP string it is compared against is packed.
    const packed = profileOfElements([97, 98, 99, 100], 3)
    const trie = profileOfElements([97, 98, 99, 0x1_f600], 3)
    expect(packed.storage.kind).toBe('packed')
    expect(trie.storage.kind).toBe('trie')
    expect(sharedFrequency(packed, trie)).toBe(1)
    expect(sharedFrequency(trie, packed)).toBe(1)
    expect(dotProduct(packed, trie)).toBe(1)
    expect(dotProduct(trie, packed)).toBe(1)
    expect(sharedFrequencyKernel(packed)(trie, 0)).toBe(1)
    expect(sharedFrequencyKernel(trie)(packed, 0)).toBe(1)
    expect(dotProductKernel(packed)(trie)).toBe(1)
    expect(dotProductKernel(trie)(packed)).toBe(1)
    // A decoded key has to become the element the trie is keyed by, not the
    // digit: a char-domain profile against a char-domain trie.
    const chars = profileOfElements('abcd', 3)
    const charTrie = trieProfile(['a', 'b', 'c', 'd'], 3)
    expect(chars.storage.kind).toBe('packed')
    expect(sharedFrequency(chars, charTrie)).toBe(2)
    expect(dotProduct(charTrie, chars)).toBe(2)
  })
})

/**
 * A profile the trie is checked against: one bucket per gram, keyed by a
 * position-aware structure rather than a serialized string, and using the
 * library's `===` so a gram holding `NaN` matches nothing — including itself.
 */
function referenceGrams(
  elements: readonly unknown[],
  gramSize: number,
): Array<{ readonly gram: readonly unknown[]; count: number }> {
  const buckets: Array<{ readonly gram: readonly unknown[]; count: number }> = []
  for (let start = 0; start + gramSize <= elements.length; start++) {
    const gram = elements.slice(start, start + gramSize)
    if (gram.some((element) => typeof element === 'number' && Number.isNaN(element))) {
      continue
    }
    const existing = buckets.find(
      (bucket) =>
        bucket.gram.length === gram.length &&
        bucket.gram.every((element, index) => element === gram[index]),
    )
    if (existing === undefined) buckets.push({ gram, count: 1 })
    else existing.count++
  }
  return buckets
}

function referenceShared(
  a: readonly unknown[],
  b: readonly unknown[],
  gramSize: number,
): number {
  const right = referenceGrams(b, gramSize)
  let shared = 0
  for (const bucket of referenceGrams(a, gramSize)) {
    const match = right.find((other) =>
      other.gram.every((element, index) => element === bucket.gram[index]),
    )
    if (match !== undefined) shared += Math.min(bucket.count, match.count)
  }
  return shared
}

function referenceDot(
  a: readonly unknown[],
  b: readonly unknown[],
  gramSize: number,
): number {
  const right = referenceGrams(b, gramSize)
  let product = 0
  for (const bucket of referenceGrams(a, gramSize)) {
    const match = right.find((other) =>
      other.gram.every((element, index) => element === bucket.gram[index]),
    )
    if (match !== undefined) product += bucket.count * match.count
  }
  return product
}

describe('the two storages answer alike', () => {
  // Only what the packed builder accepts: one domain, and every element inside
  // the canonical radix. The values it refuses are group B's job below, where
  // `profileOfElements` picks the storage and the oracle checks the answer.
  const packableNumbers = fc.array(fc.integer({ min: 0, max: 0xff }), { maxLength: 16 })
  const packableChars = fc
    .array(fc.constantFrom('a', 'b', 'c', 'd'), { maxLength: 16 })
    .map((letters) => letters.slice())
  const packable = fc.oneof(packableNumbers, packableChars)
  const packableGramSizes = fc.integer({ min: 1, max: 6 })

  it('agrees on counts, norms and both intersections', () => {
    fc.assert(
      fc.property(packable, packable, packableGramSizes, (left, right, gramSize) => {
        const a = bothStorages(left, gramSize)
        const b = bothStorages(right, gramSize)
        if (a === null || b === null) return
        expect(a.packed.gramCount).toBe(a.trie.gramCount)
        expect(a.packed.squaredNorm).toBe(a.trie.squaredNorm)
        expect(sharedFrequency(a.packed, b.packed)).toBe(sharedFrequency(a.trie, b.trie))
        expect(dotProduct(a.packed, b.packed)).toBe(dotProduct(a.trie, b.trie))
        // Every mixed orientation reaches the same number as either pure one.
        expect(sharedFrequency(a.packed, b.trie)).toBe(sharedFrequency(a.trie, b.trie))
        expect(sharedFrequency(a.trie, b.packed)).toBe(sharedFrequency(a.trie, b.trie))
        expect(dotProduct(a.packed, b.trie)).toBe(dotProduct(a.trie, b.trie))
        expect(dotProduct(a.trie, b.packed)).toBe(dotProduct(a.trie, b.trie))
      }),
      { numRuns: 3000 },
    )
  })

  it('agrees through the kernels, at every minimum a caller may ask for', () => {
    fc.assert(
      fc.property(packable, packable, packableGramSizes, (left, right, gramSize) => {
        const a = bothStorages(left, gramSize)
        const b = bothStorages(right, gramSize)
        if (a === null || b === null) return
        const exact = sharedFrequency(a.trie, b.trie)
        expect(dotProductKernel(a.packed)(b.packed)).toBe(dotProduct(a.trie, b.trie))
        expect(dotProductKernel(a.packed)(b.trie)).toBe(dotProduct(a.trie, b.trie))
        expect(dotProductKernel(a.trie)(b.packed)).toBe(dotProduct(a.trie, b.trie))
        for (const query of [a.packed, a.trie]) {
          for (const choice of [b.packed, b.trie]) {
            // At or below the true count the walk may not stop short of it.
            expect(sharedFrequencyKernel(query)(choice, 0)).toBe(exact)
            expect(sharedFrequencyKernel(query)(choice, exact)).toBe(exact)
            // Above it any count is allowed, as long as it stays below what was
            // asked for — that is the whole of the bounded contract.
            expect(sharedFrequencyKernel(query)(choice, exact + 1)).toBeLessThan(
              exact + 1,
            )
          }
        }
      }),
      { numRuns: 1500 },
    )
  })

  it('agrees when only one side can be packed', () => {
    // The mixed dispatch, driven from both sides: an element the canonical
    // radix cannot hold puts one profile on the trie while the other packs.
    const unpackableElement = fc.constantFrom(-1, 1.5, Number.NaN, 0x11_0000, 'ab', {})
    fc.assert(
      fc.property(
        packableNumbers.filter((elements) => elements.length >= 2),
        unpackableElement,
        fc.integer({ min: 1, max: 3 }),
        fc.integer({ min: 1, max: 3 }),
        (elements, odd, at, gramSize) => {
          const spoiled: unknown[] = elements.slice()
          spoiled.splice(Math.min(at, spoiled.length), 0, odd)
          const packed = profileOfElements(elements, gramSize)
          const trie = profileOfElements(spoiled, gramSize)
          if (packed.storage.kind !== 'packed' || trie.storage.kind !== 'trie') return
          const shared = referenceShared(elements, spoiled, gramSize)
          const product = referenceDot(elements, spoiled, gramSize)
          expect(sharedFrequency(packed, trie)).toBe(shared)
          expect(sharedFrequency(trie, packed)).toBe(shared)
          expect(dotProduct(packed, trie)).toBe(product)
          expect(dotProduct(trie, packed)).toBe(product)
          expect(sharedFrequencyKernel(packed)(trie, 0)).toBe(shared)
          expect(sharedFrequencyKernel(trie)(packed, 0)).toBe(shared)
          expect(dotProductKernel(packed)(trie)).toBe(product)
          expect(dotProductKernel(trie)(packed)).toBe(product)
        },
      ),
      { numRuns: 2000 },
    )
  })
})

describe('the transient direct counter', () => {
  // The counter returns a *count*, so it is checked against the profiles as an
  // integer rather than through a score within a tolerance: a rounding-sized
  // disagreement would be a wrong multiset, and `toBe` is what says so.
  const packableNumbers = fc.array(fc.integer({ min: 0, max: 0xff }), { maxLength: 24 })
  const packableChars = fc.array(fc.constantFrom('a', 'b', 'c', 'd'), { maxLength: 24 })

  it('counts exactly what the two profiles intersect to', () => {
    fc.assert(
      fc.property(
        fc.oneof(packableNumbers, packableChars),
        fc.oneof(packableNumbers, packableChars),
        fc.integer({ min: 1, max: 6 }),
        (left, right, gramSize) => {
          const counted = directSharedFrequency(left, right, gramSize)
          if (counted === null) return
          expect(counted).toBe(
            sharedFrequency(
              profileOfElements(left, gramSize),
              profileOfElements(right, gramSize),
            ),
          )
          // Symmetric, and the tally side is chosen by length rather than by
          // argument order — so both orientations must reach one number.
          expect(directSharedFrequency(right, left, gramSize)).toBe(counted)
        },
      ),
      { numRuns: 4000 },
    )
  })

  it('declines a pair of mixed domains that still shares a gram, rather than answering zero', () => {
    // The domains come from the *first* element of each side, so a pair can be
    // refused here and still have grams in common: both of these hold `[1, 2]`.
    // Turning this decline into a `0` — which a fully homogeneous mixed pair
    // would deserve — silently loses that gram.
    const withChar = ['x', 1, 2]
    const withNumber = [9, 1, 2]
    expect(directSharedFrequency(withChar, withNumber, 2)).toBeNull()
    expect(directSharedFrequency(withNumber, withChar, 2)).toBeNull()
    expect(
      sharedFrequency(profileOfElements(withChar, 2), profileOfElements(withNumber, 2)),
    ).toBe(1)
    expect(Dice.similarity(withChar, withNumber, { gramSize: 2 })).toBeCloseTo(0.5, 12)
  })

  it('counts a repeated gram only as often as the smaller side holds it', () => {
    // `min(3, 7)`: the larger side keeps meeting a gram whose count has already
    // run out, which is the decrement's whole purpose.
    expect(directSharedFrequency('aaaa', 'aaaaaaaa', 2)).toBe(3)
    expect(directSharedFrequency('aaaaaaaa', 'aaaa', 2)).toBe(3)
    // And a gram the smaller side never held at all.
    expect(directSharedFrequency('abab', 'cdcdcdcd', 2)).toBe(0)
  })

  it('declines every shape it cannot pack, leaving the profiles to answer', () => {
    // No rung reaches seven elements.
    expect(
      directSharedFrequency([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], 7),
    ).toBeNull()
    // Nothing to count on one side or the other.
    expect(directSharedFrequency([1], [1, 2, 3], 2)).toBeNull()
    expect(directSharedFrequency([1, 2, 3], [1], 2)).toBeNull()
    // Two domains: `'b'` and `98` would become one key.
    expect(directSharedFrequency(['a', 'b', 'c'], [97, 98, 99], 2)).toBeNull()
    expect(directSharedFrequency([97, 98, 99], ['a', 'b', 'c'], 2)).toBeNull()
    // An unpackable element, on the side that is tallied and on the side that is
    // spent — the two sides are packed by separate calls.
    const long = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    const short = ['a', '😀', 'c']
    expect(directSharedFrequency(short, long, 3)).toBeNull()
    expect(directSharedFrequency(long, short, 3)).toBeNull()
    expect(directSharedFrequency(long.concat(short), long, 3)).toBeNull()
  })

  it('is invisible to the public score, on either side of the threshold', () => {
    // 512 grams is where Dice routes bigrams and trigrams to the counter, and
    // every other depth stays on the profiles however long the input is. The
    // score may not notice either way, so all six are checked at a length past
    // the threshold — which depth is routed is a benchmark's business.
    for (const gramSize of [1, 2, 3, 4, 5, 6]) {
      for (const length of [511, 512, 513, 700]) {
        const text = 'abcdefghij'.repeat(Math.ceil(length / 10)).slice(0, length)
        const edited = `${text.slice(0, 100)}z${text.slice(101)}`
        expect(
          Dice.similarity(text, edited, { gramSize }),
          `gramSize ${gramSize}, ${length} chars`,
        ).toBe(
          (2 *
            sharedFrequency(
              profileOfElements(Array.from(text), gramSize),
              profileOfElements(Array.from(edited), gramSize),
            )) /
            (text.length - gramSize + 1 + (edited.length - gramSize + 1)),
        )
      }
    }
    // Above the threshold and unpackable: the counter declines and the profiles
    // answer, which the score cannot tell from never having tried.
    const astral = `${'ab'.repeat(400)}😀`
    expect(Dice.similarity(astral, astral, { gramSize: 3 })).toBe(1)
    // 801 code points, so 799 trigrams, and only the last of them reaches the
    // emoji: the other 798 all match, against 798 on the plain side.
    expect(Dice.similarity(astral, 'ab'.repeat(400), { gramSize: 3 })).toBeCloseTo(
      (2 * 798) / (799 + 798),
      12,
    )
  })
})

describe('properties', () => {
  const sequences = fc.string({ maxLength: 24 })
  const gramSizes = fc.integer({ min: 1, max: 5 })
  // Every element kind the trie has to keep apart, including the two the `Map`
  // and `===` disagree about.
  const shared = { unused: 0 }
  const elements = fc.oneof(
    fc.constantFrom('a', 'b', 'ab', '', 'a,b'),
    fc.constantFrom(0, -0, 1, 2, Number.NaN),
    fc.constantFrom(true, false, null, undefined),
    fc.constantFrom(shared, { unused: 0 }),
  )

  it('agrees with a reference n-gram counter over arbitrary elements', () => {
    fc.assert(
      fc.property(
        fc.array(elements, { maxLength: 14 }),
        fc.array(elements, { maxLength: 14 }),
        gramSizes,
        (left, right, gramSize) => {
          const a = buildProfile(left, gramSize)
          const b = buildProfile(right, gramSize)
          expect(sharedFrequency(a, b)).toBe(referenceShared(left, right, gramSize))
          expect(dotProduct(a, b)).toBe(referenceDot(left, right, gramSize))
        },
      ),
      { numRuns: 500 },
    )
  })

  it('intersects symmetrically and within the smaller profile', () => {
    fc.assert(
      fc.property(sequences, sequences, gramSizes, (left, right, gramSize) => {
        const a = buildProfile(left, gramSize)
        const b = buildProfile(right, gramSize)
        expect(sharedFrequency(a, b)).toBe(sharedFrequency(b, a))
        expect(dotProduct(a, b)).toBe(dotProduct(b, a))
        expect(sharedFrequency(a, b)).toBeLessThanOrEqual(
          Math.min(a.gramCount, b.gramCount),
        )
        expect(dotProduct(a, b)).toBeGreaterThanOrEqual(0)
        expect(sharedFrequency(a, a)).toBe(a.gramCount)
        expect(dotProduct(a, a)).toBe(a.squaredNorm)
      }),
    )
  })
})
