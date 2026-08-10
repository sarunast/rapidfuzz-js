// Not ported from RapidFuzz — upstream cannot express these inputs at all. A
// Python sequence element is a character, so there is no upstream answer for a
// token holding an object, a function or a symbol. This port accepts them, which
// means it has to say what identity and ordering mean for them.
//
// `UniqueTokenSet` hashes a token to a bucket and then settles equality with
// `===`. The hash therefore has to agree with `===`: the same object must always
// reach the same bucket, or one identity is counted as two tokens.
import { describe, expect, it } from 'vitest'

import { tokenSetRatio, tokenSortRatio } from '../src/fuzz.js'

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
})
