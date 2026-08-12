// Not ported from RapidFuzz — upstream cannot express these inputs at all. A
// Python sequence element is a character, so there is no upstream answer for a
// token holding an object, a function or a symbol. This port accepts them, which
// means it has to say what identity and ordering mean for them.
//
// `UniqueTokenSet` hashes a token to a bucket and then settles equality with
// `===`. The hash therefore has to agree with `===`: the same object must always
// reach the same bucket, or one identity is counted as two tokens.
import { describe, expect, it } from 'vitest'

import { tokenSetRatio } from '../../src/fuzz/tokenSet.js'
import { tokenSortRatio } from '../../src/fuzz/tokenSort.js'
import { wRatio } from '../../src/fuzz/weighted.js'
import { matrixScores } from '../support/matrix.js'

describe('tokens holding objects', () => {
  // Regression: the hash used to be `String(x)`, which runs the caller's
  // `toString`. An object whose text changed between two calls landed in two
  // buckets, so one identity counted twice and this scored 50.
  it('counts one object identity once, however its text behaves', () => {
    let n = 0
    const shifty = {
      toString() {
        return String(++n)
      },
    }

    expect(tokenSetRatio([shifty, ' ', shifty], [shifty])).toBe(100)
  })

  // Regression: `String(Object.create(null))` throws `TypeError: Cannot convert
  // object to primitive value`, which used to crash the scorer outright.
  it('accepts an element with no prototype', () => {
    const bare: object = Object.create(null)

    expect(tokenSetRatio([bare], [bare])).toBe(100)
    expect(tokenSetRatio([bare], [{}])).toBe(0)
  })

  it('keeps two distinct objects distinct even when their text matches', () => {
    const text = {
      toString() {
        return 'same'
      },
    }
    const twin = {
      toString() {
        return 'same'
      },
    }

    // Two tokens on the left, one on the right, and the right one is shared.
    expect(tokenSetRatio([text, ' ', twin], [text])).toBe(100)
    // Nothing shared: distinct identities must not collapse onto one bucket.
    expect(tokenSetRatio([text], [twin])).toBe(0)
  })

  it('sorts a bag of objects into the same order whichever way it is given', () => {
    const x = {}
    const y = {}

    expect(tokenSortRatio([x, ' ', y], [y, ' ', x])).toBe(100)
  })

  it('does not run a getter or coercion hook while hashing', () => {
    let coercions = 0
    const watched = {
      toString() {
        coercions++
        return 'watched'
      },
      [Symbol.toPrimitive]() {
        coercions++
        return 'watched'
      },
    }

    tokenSetRatio([watched, ' ', watched], [watched])
    expect(coercions).toBe(0)
  })
})

describe('tokens holding symbols', () => {
  it('separates two symbols that share a description', () => {
    const a = Symbol('dup')
    const b = Symbol('dup')

    expect(tokenSetRatio([a], [b])).toBe(0)
    expect(tokenSetRatio([a], [a])).toBe(100)
  })

  it('sorts symbols into a stable order', () => {
    const a = Symbol('dup')
    const b = Symbol('dup')

    expect(tokenSortRatio([a, ' ', b], [b, ' ', a])).toBe(100)
  })

  // Descriptions separate two symbols where they differ; identity is only the
  // tie-break for the ones they do not.
  it('sorts symbols by description before falling back to identity', () => {
    const early = Symbol('aaa')
    const late = Symbol('zzz')

    expect(tokenSortRatio([early, ' ', late], [late, ' ', early])).toBe(100)
  })
})

// The comparator behind `tokenSortRatio` has to be a total order over whatever
// a JavaScript array can hold, because `Array.prototype.sort` with a
// contradictory comparator sorts `[x, y]` and `[y, x]` differently — and
// agreeing on those two is the whole of what `tokenSortRatio` claims.
describe('ordering tokens of every element type', () => {
  const SPACE = ' '
  const SYM = Symbol('dup')
  const TWIN = Symbol('dup')
  const OBJ = { name: 'a' }
  const OTHER = { name: 'b' }

  // No single-character strings: `convElement` turns those into code points,
  // which is the branch the numbers below already cover.
  const BAG: readonly unknown[] = [
    undefined,
    null,
    false,
    true,
    2,
    1,
    10n,
    2n,
    'cd',
    'ab',
    SYM,
    TWIN,
    OBJ,
    OTHER,
  ]

  /** One element per token, so the sort is a sort of the bag itself. */
  function tokens(items: readonly unknown[]): unknown[] {
    const out: unknown[] = []
    for (const item of items) {
      if (out.length > 0) out.push(SPACE)
      out.push(item)
    }
    return out
  }

  it('sorts the same bag into the same order whichever way it arrives', () => {
    expect(tokenSortRatio(tokens(BAG), tokens([...BAG].reverse()))).toBe(100)
    expect(tokenSetRatio(tokens(BAG), tokens([...BAG].reverse()))).toBe(100)
  })

  it('orders two tokens by the first element that differs', () => {
    expect(
      tokenSortRatio([OBJ, OBJ, SPACE, OBJ, OTHER], [OBJ, OTHER, SPACE, OBJ, OBJ]),
    ).toBe(100)
  })

  // `NaN` is unordered by `<` and unequal to itself, so it needs a place of its
  // own in the order — and two tokens holding one are two tokens, not one.
  it('gives NaN a place past every real number', () => {
    const forward = [
      Number.NaN,
      SPACE,
      5,
      SPACE,
      Number.NaN,
      SPACE,
      2,
      SPACE,
      Number.NaN,
      SPACE,
      9,
    ]
    const backward = [
      9,
      SPACE,
      Number.NaN,
      SPACE,
      2,
      SPACE,
      Number.NaN,
      SPACE,
      5,
      SPACE,
      Number.NaN,
    ]

    expect(tokenSortRatio(forward, backward)).toBe(tokenSortRatio(backward, forward))
    expect(tokenSetRatio(forward, backward)).toBe(tokenSetRatio(backward, forward))
  })

  // The whitespace scan reads an element that is not a code point through its
  // general form rather than the numeric one.
  it('finds no whitespace among elements that are not code points', () => {
    expect(wRatio([OBJ, OTHER], [OBJ, OTHER])).toBe(100)
    expect(wRatio([OBJ, OTHER], [OBJ, SYM])).toBeLessThan(100)
  })
})

// `scoreMatrix` converts every choice once through the token preparer, which
// has to hand back anything that is not a sequence untouched for the scorer to
// answer for.
it('passes a missing choice through the token preparer', () => {
  expect(matrixScores(['fuzzy wuzzy'], [null], { scorer: tokenSortRatio })).toEqual([[0]])
})
