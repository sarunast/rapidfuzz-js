import { expect, it } from 'vitest'

import { passesAffixProbe, passesWideAffixProbe } from './affix.js'

it('probes mixed representations directly through code points', () => {
  const text = 'a'.repeat(64)
  const codes = new Uint32Array(64).fill(97)
  expect(passesAffixProbe(text, codes)).toBe(true)
  expect(passesAffixProbe(codes, text)).toBe(true)
  const other = new Uint32Array(64).fill(122)
  expect(passesAffixProbe(text, other)).toBe(false)
  expect(passesAffixProbe(other, text)).toBe(false)
})

it('rejects a mixed pair whose elements are not code points', () => {
  // Objects and multi-character strings never equal a `charCodeAt` value, so
  // an unconverted pair fails the probe and stays on the prepared kernel.
  expect(passesAffixProbe('a'.repeat(64), new Array(64).fill('a'))).toBe(false)
})

it('answers a free true below the probe minimum length', () => {
  expect(passesAffixProbe('ab', 'zz')).toBe(true)
})

it('accepts a probe-width prefix in either representation', () => {
  const text = 'a'.repeat(64)
  expect(passesAffixProbe(text, `${'a'.repeat(56)}${'z'.repeat(8)}`)).toBe(true)
  const codes = new Uint32Array(64).fill(97)
  const prefixed = new Uint32Array(64).fill(97)
  prefixed.fill(122, 56)
  expect(passesAffixProbe(codes, prefixed)).toBe(true)
})

it('accepts a probe-width suffix in either representation', () => {
  const text = `${'z'.repeat(56)}${'a'.repeat(8)}`
  expect(passesAffixProbe('a'.repeat(64), text)).toBe(true)
  const codes = new Uint32Array(64).fill(97)
  const suffixed = new Uint32Array(64).fill(122)
  suffixed.fill(97, 56)
  expect(passesAffixProbe(codes, suffixed)).toBe(true)
})

it('rejects a pair with neither a probe-width prefix nor suffix', () => {
  expect(passesAffixProbe('a'.repeat(64), 'z'.repeat(64))).toBe(false)
  const codes = new Uint32Array(64).fill(97)
  const other = new Uint32Array(64).fill(122)
  expect(passesAffixProbe(codes, other)).toBe(false)
})

it('rejects an affix one element short of the probe width', () => {
  // 64 shared elements probe 8 deep on each side.
  const shortPrefix = `${'a'.repeat(7)}${'z'.repeat(57)}`
  expect(passesAffixProbe('a'.repeat(64), shortPrefix)).toBe(false)
  const shortSuffix = `${'z'.repeat(57)}${'a'.repeat(7)}`
  expect(passesAffixProbe('a'.repeat(64), shortSuffix)).toBe(false)
})

it('caps the probe at 32 elements however long the pair grows', () => {
  // A 1024-element pair would probe 128 deep uncapped; 32 suffice.
  const long = 'a'.repeat(1024)
  expect(passesAffixProbe(long, `${'a'.repeat(32)}${'z'.repeat(992)}`)).toBe(true)
  expect(passesAffixProbe(long, `${'a'.repeat(31)}${'z'.repeat(993)}`)).toBe(false)
})

it('probes a quarter of the shorter side in the wide variant', () => {
  // 64 elements probe 16 deep, twice the narrow variant's 8.
  const text = 'a'.repeat(64)
  expect(passesWideAffixProbe(text, `${'a'.repeat(16)}${'z'.repeat(48)}`)).toBe(true)
  expect(passesWideAffixProbe(text, `${'a'.repeat(15)}${'z'.repeat(49)}`)).toBe(false)
  expect(passesWideAffixProbe(text, `${'z'.repeat(48)}${'a'.repeat(16)}`)).toBe(true)
})

it('caps the wide probe at 64 elements', () => {
  const long = 'a'.repeat(1024)
  expect(passesWideAffixProbe(long, `${'a'.repeat(64)}${'z'.repeat(960)}`)).toBe(true)
  expect(passesWideAffixProbe(long, `${'a'.repeat(63)}${'z'.repeat(961)}`)).toBe(false)
})
