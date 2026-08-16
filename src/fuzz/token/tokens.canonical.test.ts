// Not ported from RapidFuzz — upstream has no canonical-form cache to pin.
// The sorted and unique joins build as code-point arrays, and a second access
// repacks them into a BMP string: only a retained choice is ever read twice,
// so one-shot scoring keeps the array cost while matcher corpora trade one
// packing pass for the string kernels and ~4x smaller retained joins. Astral
// and non-numeric tokens must refuse the pack once and stay arrays forever.
import { describe, expect, it } from 'vitest'

import { prepareTokenChoice, sortedOf, uniqueJoinedOf } from './tokens.js'

describe('canonical joins repack on the second access', () => {
  it('returns an array first, the packed string from then on', () => {
    const choice = prepareTokenChoice('beta alpha')

    const first = sortedOf(choice)
    expect(Array.isArray(first)).toBe(true)

    const second = sortedOf(choice)
    expect(second).toBe('alpha beta')
    expect(sortedOf(choice)).toBe(second)
  })

  it('keeps astral joins as arrays, and only attempts the pack once', () => {
    const choice = prepareTokenChoice('😀😀 alpha')

    const first = sortedOf(choice)
    expect(sortedOf(choice)).toBe(first)
    expect(sortedOf(choice)).toBe(first)
    expect(Array.isArray(first)).toBe(true)
  })

  it('chunks the pack for joins past 1024 elements', () => {
    const long = 'x'.repeat(1500)
    const choice = prepareTokenChoice(`${long} alpha`)

    sortedOf(choice)
    expect(sortedOf(choice)).toBe(`alpha ${long}`)
  })

  it('follows sortedOf when the tokens are unique, so both share one upgrade', () => {
    const choice = prepareTokenChoice('beta alpha')

    expect(Array.isArray(uniqueJoinedOf(choice))).toBe(true)
    expect(uniqueJoinedOf(choice)).toBe('alpha beta')
    expect(sortedOf(choice)).toBe('alpha beta')
  })

  it('deduplicates before joining when tokens repeat, with its own upgrade', () => {
    const choice = prepareTokenChoice('beta beta alpha')

    const first = uniqueJoinedOf(choice)
    expect(Array.isArray(first)).toBe(true)
    expect(first).toHaveLength('alpha beta'.length)

    expect(uniqueJoinedOf(choice)).toBe('alpha beta')
    expect(sortedOf(choice)).not.toBe('alpha beta')
  })

  it('keeps a duplicated astral join as an array', () => {
    const choice = prepareTokenChoice('😀 😀 alpha')

    const first = uniqueJoinedOf(choice)
    expect(uniqueJoinedOf(choice)).toBe(first)
    expect(uniqueJoinedOf(choice)).toBe(first)
    expect(Array.isArray(first)).toBe(true)
  })
})
