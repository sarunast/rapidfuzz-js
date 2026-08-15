import { expect, it } from 'vitest'

import { prefixDistance } from '#algorithms/prefix/implementation.js'

import {
  PREPARE_SCORER,
  prepareChoiceSequence,
  preparedChoiceSequence,
} from './preparation.js'

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
