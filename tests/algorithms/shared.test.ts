import { expect, it } from 'vitest'

import { sharesAffix } from '../../src/algorithms/shared/affix.js'
import {
  preparedScorerSequence,
  prepareScorerChoice,
} from '../../src/algorithms/shared/preparation.js'
import { convSequence } from '../../src/algorithms/shared/sequence.js'

it('treats mixed string and array representations as worth aligning', () => {
  const text = 'a'.repeat(64)
  const elements = new Array(64).fill('a')
  expect(sharesAffix(text, elements)).toBe(true)
  expect(sharesAffix(elements, text)).toBe(true)
})

it('converts single-character sequence elements without changing longer strings', () => {
  expect(convSequence(['a', '😀', 'ab'])).toEqual([97, 0x1f600, 'ab'])
})

it('proves opaque prepared sequences before reading them', () => {
  expect(preparedScorerSequence(prepareScorerChoice('abc'))).toBe('abc')
  expect(() => preparedScorerSequence({ value: 'abc' })).toThrow(TypeError)
})
