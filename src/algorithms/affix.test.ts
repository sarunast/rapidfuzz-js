import { expect, it } from 'vitest'

import { sharesAffix } from './affix.js'

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
