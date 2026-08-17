import { describe, expect, test } from 'vitest'

import { convSequence } from '#core/sequence.js'

import {
  compileElementWeights,
  type CompiledElementWeights,
  preparedWeightedProfile,
  weightedComponents,
  weightedProfile,
  weightedQueryGroups,
  zeroMassSimilarity,
} from './weightedProfile.js'

function weightsOf(entries: readonly (readonly [unknown, number])[], fallback?: number) {
  return compileElementWeights(new Map(entries), fallback)
}

// Every caller converts first — the weight keys are canonical elements, so a
// sequence that skipped conversion would look up `'a'` against a table of `97`.
function profileOf(sequence: readonly unknown[], weights: CompiledElementWeights) {
  return weightedProfile(convSequence(sequence), weights)
}

function queryOf(sequence: readonly unknown[], weights: CompiledElementWeights) {
  return weightedQueryGroups(convSequence(sequence), weights)
}

function weightOfGroup(weights: CompiledElementWeights, element: unknown): number {
  const group = weights.groupOf.get(element)
  return group === undefined ? Number.NaN : weights.groupWeights[group]
}

function componentsOf(
  a: readonly unknown[],
  b: readonly unknown[],
  entries: readonly (readonly [unknown, number])[],
  fallback?: number,
): { shared: number; firstOnly: number; secondOnly: number } {
  const weights = weightsOf(entries, fallback)
  const parts = new Float64Array(3)
  weightedComponents(queryOf(a, weights), profileOf(b, weights), weights, parts)
  return { shared: parts[0], firstOnly: parts[1], secondOnly: parts[2] }
}

describe('compiled element weights', () => {
  test('reserves group 0 for weight zero, whatever the configuration holds', () => {
    const weights = weightsOf([
      ['a', 5],
      ['b', 1],
      ['c', 3],
    ])
    expect([...weights.groupWeights]).toEqual([0, 1, 3, 5])
  })

  test('canonicalizes a single-character key to its code point', () => {
    const weights = weightsOf([['a', 4]])
    expect(weightOfGroup(weights, 97)).toBe(4)
    expect(weights.groupOf.get('a')).toBeUndefined()
    // The same element written both ways, agreeing, is one entry.
    expect(
      weightsOf([
        ['a', 4],
        [97, 4],
      ]).groupOf.size,
    ).toBe(1)
  })

  test('refuses one element two weights rather than letting order decide', () => {
    expect(() =>
      weightsOf([
        ['a', 2],
        [97, 4],
      ]),
    ).toThrow('one element two weights')
  })

  test('defaults every unnamed element to weight 1', () => {
    const weights = weightsOf([['ag', 0.5]])
    expect(weights.groupWeights[weights.defaultGroup]).toBe(1)
    expect(weightOfGroup(weights, 'ag')).toBe(0.5)
  })

  test('places an explicit zero default in the ignored group', () => {
    const weights = compileElementWeights(new Map([['react', 2]]), 0)
    expect(weights.defaultGroup).toBe(0)
    expect(weightOfGroup(weights, 'react')).toBe(2)
  })

  test('gives a zero-weight element the ignored group', () => {
    const weights = weightsOf([['ag', 0]])
    expect(weights.groupOf.get('ag')).toBe(0)
    expect([...weights.groupWeights]).toEqual([0, 1])
  })

  test('decides at compile time whether the weighting prices anything', () => {
    // One positive weight, held by the default too, is a constant factor over
    // all three components and cancels from the ratio.
    expect(weightsOf([['ag', 1]]).uniformPositive).toBe(true)
    expect(compileElementWeights(undefined, 7).uniformPositive).toBe(true)
    expect(
      compileElementWeights(
        new Map([
          ['ag', 7],
          ['gmbh', 7],
        ]),
        7,
      ).uniformPositive,
    ).toBe(true)
    // Two positive weights price something; so does a default outside the one
    // group, and so does an element the table ignores.
    expect(weightsOf([['ag', 0.5]]).uniformPositive).toBe(false)
    expect(compileElementWeights(new Map([['ag', 2]]), 0).uniformPositive).toBe(false)
    expect(weightsOf([['ag', 0]]).uniformPositive).toBe(false)
  })

  test('answers that for a large vocabulary without walking it again', () => {
    // The adversarial shape: one positive group, a default inside it, and a
    // single ignored element at the very end. Deciding this by walking the table
    // would be right and would cost the whole vocabulary per score, so the
    // answer is a field the compiler filled in.
    const entries: [unknown, number][] = Array.from({ length: 20_000 }, (_, at) => [
      `token${at}`,
      1,
    ])
    expect(compileElementWeights(new Map(entries), undefined).uniformPositive).toBe(true)
    entries.push(['ag', 0])
    const mixed = compileElementWeights(new Map(entries), undefined)
    expect(mixed.uniformPositive).toBe(false)
    expect([...mixed.groupWeights]).toEqual([0, 1])
    expect(mixed.defaultGroup).toBe(1)
  })

  test('accepts a weights map only as something with entries and get', () => {
    for (const value of [1, 'ab', null, [5, 10], new Set([1])]) {
      expect(() => compileElementWeights(value, undefined)).toThrow(
        'elementWeights must be a map',
      )
    }
    // Map-like rather than a Map: no `instanceof`, so a cross-realm map works.
    const mapLike = {
      entries: () => [['react', 3]] as const,
      get: () => undefined,
    }
    expect(weightOfGroup(compileElementWeights(mapLike, undefined), 'react')).toBe(3)
  })

  test('refuses weights that are not finite non-negative numbers', () => {
    for (const weight of [Number.NaN, Infinity, -Infinity, -1]) {
      expect(() => weightsOf([['react', weight]])).toThrow(RangeError)
    }
    for (const weight of [null, undefined, '1', {}]) {
      expect(() =>
        compileElementWeights(new Map([['react', weight]]), undefined),
      ).toThrow('must be a number')
    }
    expect(() => compileElementWeights(undefined, null)).toThrow(
      'defaultElementWeight must be a number',
    )
    expect(() => compileElementWeights(undefined, -1)).toThrow(RangeError)
  })

  test('scales a weight no mass could hold by a power of two, losslessly', () => {
    const weights = weightsOf([['huge', Number.MAX_VALUE]])
    const scaled = weightOfGroup(weights, 'huge')
    expect(scaled).toBeLessThan(Number.MAX_VALUE)
    // A power of two divided out changes no mantissa, so the scaling is exact.
    const scale = Number.MAX_VALUE / scaled
    expect(Math.log2(scale) % 1).toBe(0)
    expect(scaled * scale).toBe(Number.MAX_VALUE)
    // The unnamed default rode the same scale, so their ratio is untouched.
    expect(weights.groupWeights[weights.defaultGroup] * scale).toBe(1)
  })

  test('refuses a weight span too wide to scale rather than flushing one away', () => {
    expect(() =>
      weightsOf([
        ['huge', Number.MAX_VALUE],
        ['tiny', Number.MIN_VALUE],
      ]),
    ).toThrow('too wide to represent')
  })
})

describe('weighted profiles', () => {
  test('counts elements and totals occurrences per weight group', () => {
    const weights = weightsOf([
      ['react', 3],
      ['ag', 0.5],
    ])
    const profile = profileOf(['react', 'react', 'ag', 'other'], weights)
    expect(profile.counts.get('react')).toBe(2)
    expect(profile.counts.get('ag')).toBe(1)
    expect([...profile.groupIds]).toEqual([1, 2, 3])
    // Ascending by weight: 0.5 (ag), 1 (other), 3 (react x2).
    expect([...profile.groupTotals]).toEqual([1, 1, 2])
    expect(profile.hasUnmatchable).toBe(false)
  })

  test('counts an unmatchable element toward its group but never toward counts', () => {
    const profile = profileOf([Number.NaN, 97], weightsOf([]))
    expect(profile.counts.size).toBe(1)
    expect([...profile.groupTotals]).toEqual([2])
    expect(profile.hasUnmatchable).toBe(true)
  })

  test('has no positive group when every element is ignored', () => {
    const profile = profileOf(['ag', 'gmbh'], compileElementWeights(undefined, 0))
    expect(profile.groupIds.length).toBe(0)
    expect(profile.counts.size).toBe(2)
  })

  test('refuses a prepared value that is not a weighted profile', () => {
    for (const value of [null, undefined, 'ab', {}, []]) {
      expect(() => preparedWeightedProfile(value)).toThrow(
        'invalid prepared weighted profile',
      )
    }
    const profile = weightedProfile(['x'], weightsOf([]))
    expect(preparedWeightedProfile(profile)).toBe(profile)
  })

  test('groups a query by weight, keeping the ignored group aside', () => {
    const weights = weightsOf([
      ['ag', 0],
      ['react', 3],
    ])
    const query = queryOf(['react', 'ag', 'node', 'react'], weights)
    expect([...query.groupIds]).toEqual([1, 2])
    expect([...query.groupStart]).toEqual([0, 1, 2])
    expect(query.elements).toEqual(['node', 'react'])
    expect([...query.counts]).toEqual([1, 2])
    expect(query.zeroElements).toEqual(['ag'])
    expect([...query.zeroCounts]).toEqual([1])
  })
})

describe('weighted components', () => {
  test('weights each occurrence of a shared element', () => {
    expect(componentsOf(['react', 'react'], ['react'], [['react', 3]])).toEqual({
      shared: 3,
      firstOnly: 3,
      secondOnly: 0,
    })
  })

  test('prices what each side alone carries', () => {
    expect(
      componentsOf(
        ['google', 'deepmind', 'ag'],
        ['google', 'ag'],
        [
          ['google', 4],
          ['deepmind', 5],
          ['ag', 0.1],
        ],
      ),
    ).toEqual({ shared: 4.1, firstOnly: 5, secondOnly: 0 })
  })

  test('accumulates a real unmatched occurrence a mass would have absorbed', () => {
    // `massA` would fold 1e16 + 1 back to 1e16, reporting nothing unmatched.
    expect(
      componentsOf(
        ['x', 'y'],
        ['x'],
        [
          ['x', 1e16],
          ['y', 1],
        ],
      ),
    ).toEqual({ shared: 1e16, firstOnly: 1, secondOnly: 0 })
  })

  test('reads the same for either order of the same multiset', () => {
    const entries = [
      ['x', 1e16],
      ['y', 1],
      ['z', 0.1],
    ] as const
    const forward = componentsOf(['y', 'z', 'x'], ['x', 'y', 'z'], entries)
    const reversed = componentsOf(['x', 'y', 'z'], ['z', 'x', 'y'], entries)
    expect(forward).toEqual(reversed)
    expect(forward.firstOnly).toBe(0)
    expect(forward.secondOnly).toBe(0)
    expect(forward.shared).toBe(reversed.shared)
  })

  test('ignores a zero-weight element on either side', () => {
    expect(
      componentsOf(
        ['google', 'ag'],
        ['google', 'gmbh'],
        [
          ['ag', 0],
          ['gmbh', 0],
        ],
      ),
    ).toEqual({ shared: 1, firstOnly: 0, secondOnly: 0 })
  })

  test('prices a group only the choice carries', () => {
    expect(componentsOf(['a'], ['a', 'b'], [['b', 7]])).toEqual({
      shared: 1,
      firstOnly: 0,
      secondOnly: 7,
    })
  })
})

describe('zero mass', () => {
  const ignored = compileElementWeights(undefined, 0)

  function zeroScore(a: readonly unknown[], b: readonly unknown[], weights = ignored) {
    return zeroMassSimilarity(queryOf(a, weights), profileOf(b, weights))
  }

  test('is 1 for equal multisets, whatever their order', () => {
    expect(zeroScore(['ag'], ['ag'])).toBe(1)
    expect(zeroScore(['ag', 'gmbh'], ['gmbh', 'ag'])).toBe(1)
    expect(zeroScore(['ag', 'ag'], ['ag', 'ag'])).toBe(1)
    expect(zeroScore([], [])).toBe(1)
  })

  test('is 0 for different content, lengths, or repeat counts', () => {
    expect(zeroScore(['ag'], ['gmbh'])).toBe(0)
    expect(zeroScore(['ag'], ['ag', 'gmbh'])).toBe(0)
    expect(zeroScore(['ag', 'ag'], ['ag'])).toBe(0)
    expect(zeroScore(['ag', 'gmbh'], ['ag', 'ag'])).toBe(0)
  })

  test('is 0 whenever either side holds an unmatchable element', () => {
    expect(zeroScore([Number.NaN], [Number.NaN])).toBe(0)
  })

  test('is 0 when either side carries weighted mass', () => {
    const weights = weightsOf([['ag', 0]])
    expect(zeroScore(['ag'], ['google'], weights)).toBe(0)
    expect(zeroScore(['google'], ['ag'], weights)).toBe(0)
  })
})
