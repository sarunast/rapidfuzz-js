// Crossing between the two key representations. An index keys elements
// directly until one arrives that no integer scheme can spell, and then keys
// ordinals — in both directions, without ever re-reading a choice.
import { describe, expect, it } from 'vitest'

import type { ChoiceIndex } from '#core/scoring/choiceIndex.js'
import type { Sequence } from '#core/types.js'

import {
  exhaustive,
  exhaustiveScan,
  indexOf,
  pairs,
  REPRESENTATION_SPECS,
} from '../../../../testing/invertedIndex.js'
import { NGramIndexBuilder, type SealedIndex } from './builder.js'

const OBJECT_A = { name: 'react' }

// A builder hands its sealed structure to the callback rather than returning it,
// so the representation it settled on is reached by capturing a real seal — the
// same walk `overlap.test.ts` takes.
const stubIndex: ChoiceIndex = {
  select() {
    throw new Error('the captured index answers no queries')
  },
  scan() {
    throw new Error('the captured index answers no queries')
  },
}

function sealedOf(gramSize: number, choices: readonly Sequence[]): SealedIndex<null> {
  let captured: SealedIndex<null> | undefined
  const builder = new NGramIndexBuilder<null>(
    gramSize,
    () => null,
    (sealed) => {
      captured = sealed
      return stubIndex
    },
  )
  for (const choice of choices) builder.add(choice)
  builder.seal()
  if (captured === undefined) throw new Error('the builder did not seal')
  return captured
}

function agree(
  gramSize: number,
  choices: readonly unknown[][],
  query: readonly unknown[],
): void {
  for (const spec of REPRESENTATION_SPECS) {
    const index = indexOf(spec, gramSize, choices)
    expect(pairs(index.select(query, null, null))).toEqual(
      exhaustive(spec, gramSize, choices, query, null, null),
    )
    expect(pairs(index.scan(query, null))).toEqual(
      exhaustiveScan(spec, gramSize, choices, query, null),
    )
  }
}

describe('the move to ordinal keys', () => {
  it('re-keys a packed index without re-reading a choice', () => {
    // The first two choices are keyed by code point; the third cannot be, and
    // every posting already recorded has to survive the move.
    const choices = [
      ['a', 'b', 'c', 'd', 'e', 'f'],
      ['a', 'b', 'c', 'x', 'y', 'z'],
      ['react', 'typescript', 'node'],
    ]
    agree(2, choices, ['a', 'b', 'c', 'd', 'e', 'f'])
    agree(3, choices, ['react', 'typescript', 'node'])
    expect(sealedOf(2, choices).elementOrdinals).not.toBeNull()
  })

  it('re-keys a joined-string index too', () => {
    // A negative element has no packed rung, so the index is string-keyed
    // before the tokens arrive — the other spelling the decoder has to read.
    const choices = [
      [-1, 2, 3],
      [-1, 2, 4],
      ['react', 'typescript', 'node'],
    ]
    agree(2, choices, [-1, 2, 3])
    agree(2, choices, ['react', 'typescript'])
  })

  it('re-keys an index whose gram size has no packed rung at all', () => {
    // Seven elements a gram is past a safe integer in every radix, so both
    // spellings before and after the move are joined strings.
    const choices = [
      ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
      ['react', 'typescript', 'node', 'vite', 'esbuild', 'rollup', 'swc', 'oxc'],
    ]
    agree(7, choices, ['react', 'typescript', 'node', 'vite', 'esbuild', 'rollup', 'swc'])
  })

  it('narrows the radix a text corpus had been forced to widen', () => {
    // Code points past a byte force the wide rung; three distinct tokens fit a
    // byte again, and the answers have to be identical either way.
    const choices = [
      ['😀', '😁', '😂'],
      ['react', 'typescript', 'node'],
    ]
    agree(2, choices, ['react', 'typescript', 'node'])
    // Six distinct elements between them, so the byte rung holds the whole
    // corpus again — narrowing is the reason the transition assigns ordinals
    // densely rather than keeping the elements it decoded.
    expect(sealedOf(2, choices).radix).toBe(0x100)
  })

  it('widens an ordinal radix when the corpus outgrows it', () => {
    const many = Array.from({ length: 300 }, (_, at) => `token-${at}`)
    const choices = [many.slice(0, 6), many, many.slice(100, 140)]
    agree(2, choices, many.slice(100, 140))
    expect(sealedOf(2, choices).radix).toBe(0x1_0000)
  })

  it('keeps dense ordinal postings exact', () => {
    // `shared` is in every choice but the last, which is what a dense list is.
    const choices = [
      ['shared', 'one'],
      ['shared', 'two'],
      ['shared', 'three'],
      ['shared', 'four'],
      ['shared', 'five'],
      ['other', 'six'],
    ]
    agree(1, choices, ['shared', 'two'])
    agree(1, choices, ['shared'])
  })

  it('leaves a gramless arbitrary choice keyed by nothing', () => {
    // A choice too short to make a gram never reaches a key, so it must not
    // force the move — and must still score exactly once a later choice does.
    const choices = [[OBJECT_A], ['senior', 'software', 'engineer']]
    agree(3, choices, [OBJECT_A])
    agree(3, choices, ['senior', 'software', 'engineer'])
    // Beside choices the direct scheme can spell, the same gramless choice must
    // leave the index in direct mode: `elementOrdinals === null` says the keys
    // are direct, not that every element in the corpus is an integer.
    expect(sealedOf(3, [[OBJECT_A], 'abcdef', [1, 2, 3, 4]]).elementOrdinals).toBeNull()
  })

  it('moves an index that has no element to give an ordinal to', () => {
    // Both windows are unmatchable, so the move happens with an empty table and
    // no largest ordinal to size a radix from. The choice still carries its
    // gram count and norm, exactly as the exhaustive trie does.
    agree(
      2,
      [
        [NaN, NaN],
        ['a', 'b', 'c'],
      ],
      [NaN, NaN],
    )
    agree(
      2,
      [
        [NaN, NaN],
        ['react', 'typescript'],
      ],
      ['react', 'typescript'],
    )
  })
})

describe('a direct index taking an arbitrary query', () => {
  it('matches the grams it still can and counts the rest', () => {
    const choices = [
      [1, 2, 3, 4],
      [9, 9, 9, 9],
    ]
    agree(2, choices, [1, 2, 'unknown', 4])
  })

  it('counts a repeated unknown shingle once', () => {
    // A fallback that miscounted the unknown grams would change Cosine's norm
    // without changing which choices match.
    const choices = [
      [1, 2, 3, 4],
      [1, 2, 3, 5],
    ]
    agree(2, choices, [1, 2, 'u', 'v', 'u', 'v', 3, 4])
  })

  it('takes an unmatchable element in a query the corpus never needed one for', () => {
    agree(
      2,
      [
        ['a', 'b', 'c'],
        ['a', 'b', 'd'],
      ],
      ['a', NaN, 'b', 'c'],
    )
  })

  it('drops an integer gram the index could not have keyed either', () => {
    // 300 is past the byte rung this corpus settled on, so its grams miss the
    // same way the direct extractor's joined fallback would have missed.
    agree(
      2,
      [
        [1, 2, 3],
        [1, 2, 4],
      ],
      [1, 2, 300, 'token'],
    )
  })

  it('takes an arbitrary query against a joined-string index', () => {
    agree(
      2,
      [
        [-1, 2, 3],
        [-1, 2, 4],
      ],
      [-1, 2, 'token', 3],
    )
  })

  it('reuses its scratch between an ordinary query and an arbitrary one', () => {
    // The direct extraction that fails has already half-filled the key scratch,
    // so the ordinary query after the arbitrary one is the one that breaks.
    const choices = [
      [1, 2, 3, 4],
      [9, 9, 9, 9],
    ]
    for (const spec of REPRESENTATION_SPECS) {
      const index = indexOf(spec, 2, choices)
      const ordinary = exhaustive(spec, 2, choices, [1, 2, 3, 4], null, null)
      expect(pairs(index.select([1, 2, 3, 4], null, null))).toEqual(ordinary)
      expect(pairs(index.select([1, 2, 'unknown', 4], null, null))).toEqual(
        exhaustive(spec, 2, choices, [1, 2, 'unknown', 4], null, null),
      )
      expect(pairs(index.select([1, 2, 3, 4], null, null))).toEqual(ordinary)
    }
  })

  it('takes a query of nothing but unmatchable elements', () => {
    agree(2, [['a', 'b', 'c']], [NaN, NaN, NaN])
  })
})

describe('an ordinal index taking a query it has no ordinals for', () => {
  it('keys an unknown element past its radix without widening for it', () => {
    // The corpus settles on the byte rung with 254 ordinals, so a query's own
    // unknown elements run past it. Widening the sealed index for a query would
    // be wrong, so those grams take the joined spelling and simply miss.
    const corpus = Array.from({ length: 254 }, (_, at) => `token-${at}`)
    const unknown = Array.from({ length: 8 }, (_, at) => `absent-${at}`)
    agree(2, [corpus, corpus.slice(0, 10)], [...corpus.slice(0, 4), ...unknown])
  })
})
