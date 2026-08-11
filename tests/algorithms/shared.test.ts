import { expect, it } from 'vitest'

import { prefixDistance } from '../../src/algorithms/prefix/implementation.js'
import { sharesAffix } from '../../src/algorithms/shared/affix.js'
import {
  PREPARE_SCORER,
  prepareMetric,
  preparedChoiceSequence,
  prepareChoiceSequence,
} from '../../src/algorithms/shared/preparation.js'
import {
  configurationCanonicalizerOf,
  configurationSymmetryOf,
  DISTANCE_FLAGS,
  withPreparedFlags,
} from '../../src/algorithms/shared/scorerMetadata.js'
import { convSequence } from '../../src/algorithms/shared/sequence.js'

it('treats mixed string and array representations as worth aligning', () => {
  const text = 'a'.repeat(64)
  const elements = new Array(64).fill('a')
  expect(sharesAffix(text, elements)).toBe(true)
  expect(sharesAffix(elements, text)).toBe(true)
})

it('answers a free true below the probe minimum length', () => {
  expect(sharesAffix('ab', 'zz')).toBe(true)
})

it('accepts a probe-width prefix in either representation', () => {
  const text = 'a'.repeat(64)
  expect(sharesAffix(text, `${'a'.repeat(56)}${'z'.repeat(8)}`)).toBe(true)
  const codes = new Uint32Array(64).fill(97)
  const prefixed = new Uint32Array(64).fill(97)
  prefixed.fill(122, 56)
  expect(sharesAffix(codes, prefixed)).toBe(true)
})

it('converts single-character sequence elements without changing longer strings', () => {
  expect(convSequence(['a', '😀', 'ab'])).toEqual([97, 0x1f600, 'ab'])
})

it('proves prepared representations before reading them', () => {
  expect(preparedChoiceSequence(prepareChoiceSequence('abc'))).toBe('abc')
  expect(() => preparedChoiceSequence({ value: 'abc' })).toThrow(TypeError)
  expect(() => preparedChoiceSequence(42)).toThrow(TypeError)
  expect(() => preparedChoiceSequence(null)).toThrow(TypeError)
})

it('scores a string query the same across choice representations', () => {
  const preparation = prefixDistance[PREPARE_SCORER]({})
  const prepared = preparation.prepareQuery('abc')
  const arrayChoice = preparation.prepareChoice(['a', 'b', 'z'])
  expect(prepared(arrayChoice, null)).toBe(1)
  expect(prepared(arrayChoice, null)).toBe(1)
  expect(prepared(preparation.prepareChoice('abz'), null)).toBe(1)
  const arrayPrepared = preparation.prepareQuery(['a', 'b', 'c'])
  expect(arrayPrepared(preparation.prepareChoice('abz'), null)).toBe(1)
})

it('replaces an earlier registration instead of merging into it', () => {
  const prepare = prepareMetric(
    'distance',
    () => 0,
    () => 0,
  )
  const implementation = (): number => 0
  withPreparedFlags(implementation, DISTANCE_FLAGS, prepare, {
    configurationSymmetry: () => false,
    configurationCanonicalizer: (options) => options,
  })
  expect(configurationSymmetryOf(implementation)).not.toBeNull()
  expect(configurationCanonicalizerOf(implementation)).not.toBeNull()
  withPreparedFlags(implementation, DISTANCE_FLAGS, prepare)
  expect(configurationSymmetryOf(implementation)).toBeNull()
  expect(configurationCanonicalizerOf(implementation)).toBeNull()
})

it('reports no registration for an implementation it never saw', () => {
  const stranger = (): number => 0
  expect(configurationSymmetryOf(stranger)).toBeNull()
  expect(configurationCanonicalizerOf(stranger)).toBeNull()
})
