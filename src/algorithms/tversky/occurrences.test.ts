// The occurrence walk and the distinct-element view built on it: who is an
// occurrence, who is an entry, and what mass can never be matched at all.
import { describe, expect, it } from 'vitest'

import { compileElementWeights } from '../ngram/weightedProfile.js'
import {
  canonicalElements,
  elementTableOf,
  fuzzyOperand,
  occurrencesOf,
} from './occurrences.js'

describe('occurrencesOf', () => {
  it('walks a string by code point, keeping the character as the raw value', () => {
    const occurrences = occurrencesOf('a😀', null)
    expect(occurrences.map((one) => one.raw)).toEqual(['a', '😀'])
    expect(occurrences.map((one) => one.canonical)).toEqual([97, 128_512])
    expect(occurrences.map((one) => one.index)).toEqual([0, 1])
  })

  it('walks an array by index, canonicalizing each element', () => {
    const occurrences = occurrencesOf(['ab', 'a', 97], null)
    expect(occurrences.map((one) => one.canonical)).toEqual(['ab', 97, 97])
    expect(occurrences.map((one) => one.weight)).toEqual([1, 1, 1])
  })

  it('prices each occurrence from the compiled table', () => {
    const weights = compileElementWeights(new Map([['swisscom', 5]]), 0.1)
    const occurrences = occurrencesOf(['swisscom', 'ag'], weights)
    expect(occurrences.map((one) => one.weight)).toEqual([5, 0.1])
  })
})

describe('canonicalElements', () => {
  it('projects the canonical values in occurrence order', () => {
    expect(canonicalElements(occurrencesOf(['ab', 'a'], null))).toEqual(['ab', 97])
  })
})

describe('fuzzyOperand', () => {
  it('accepts a string', () => {
    expect(fuzzyOperand('swisscom')).toBe('swisscom')
  })

  // Every single code point canonicalizes to a number, astral included, so a
  // one-character token never reaches an element scorer.
  it.each([
    ['a code point', 97],
    ['an astral code point', 128_512],
    ['a plain number', 12_345],
    ['an object', { token: 'ab' }],
    ['an array', ['a', 'b']],
    ['a symbol', Symbol('ab')],
    ['a nullish element', undefined],
  ])('refuses %s', (_label, value) => {
    expect(fuzzyOperand(value)).toBeNull()
  })
})

describe('elementTableOf', () => {
  it('collapses repeats and keeps first-occurrence order', () => {
    const table = elementTableOf(occurrencesOf(['react', 'vue', 'react'], null))
    expect(table.entries.map((entry) => entry.canonical)).toEqual(['react', 'vue'])
    expect(table.entries.map((entry) => entry.count)).toEqual([2, 1])
    expect(table.indexOf.get('react')).toBe(0)
    expect(table.indexOf.get('vue')).toBe(1)
  })

  it('carries an operand only for a string element', () => {
    const table = elementTableOf(occurrencesOf(['react', 'a', 97], null))
    expect(table.entries.map((entry) => entry.operand)).toEqual(['react', null])
  })

  it('holds unmatchable mass out of the entries', () => {
    const table = elementTableOf(occurrencesOf(['react', Number.NaN, Number.NaN], null))
    expect(table.entries.map((entry) => entry.canonical)).toEqual(['react'])
    expect(table.unmatchableMass).toBe(2)
  })

  it('prices unmatchable mass by its own weight', () => {
    const weights = compileElementWeights(new Map<unknown, number>([[Number.NaN, 4]]), 1)
    const table = elementTableOf(occurrencesOf(['react', Number.NaN], weights))
    expect(table.unmatchableMass).toBe(4)
  })

  it('drops a weightless element from entries and from the unmatchable mass', () => {
    const weights = compileElementWeights(new Map([['react', 2]]), 0)
    const table = elementTableOf(occurrencesOf(['react', 'vue', Number.NaN], weights))
    expect(table.entries.map((entry) => entry.canonical)).toEqual(['react'])
    expect(table.entries[0].weight).toBe(2)
    expect(table.unmatchableMass).toBe(0)
  })

  it('is empty for a sequence with nothing to match', () => {
    const table = elementTableOf(occurrencesOf([], null))
    expect(table.entries).toEqual([])
    expect(table.indexOf.size).toBe(0)
    expect(table.unmatchableMass).toBe(0)
  })
})
