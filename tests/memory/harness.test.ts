import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import {
  median,
  parseCli,
  assertCliOptions,
  numberOption,
  recoveryDelta,
  retainedTheilSen,
  runIsolated,
  type MemorySample,
} from '../../bench/memory/harness.ts'

function sample(retained: number): MemorySample {
  return {
    heapUsed: retained,
    heapTotal: retained + 1,
    arrayBuffers: 0,
    external: 2,
    rss: 3,
    retained,
  }
}

describe('the Node memory harness', () => {
  it('calculates medians without mutating the input', () => {
    const values = [9, 1, 5, 3]
    expect(median(values)).toBe(4)
    expect(values).toEqual([9, 1, 5, 3])
  })

  it('uses every pairwise slope over only the final ten samples', () => {
    const readings = [
      sample(1_000),
      ...Array.from({ length: 10 }, (_, i) => sample(7 * i)),
    ]
    expect(retainedTheilSen(readings)).toBe(7)
    expect(retainedTheilSen(Array.from({ length: 20 }, (_, i) => sample(i)))).toBe(1)
  })

  it('clamps recovery deltas at zero', () => {
    expect(recoveryDelta([sample(100), sample(120)], [sample(90), sample(100)])).toBe(0)
    expect(recoveryDelta([sample(100)], [sample(145)])).toBe(45)
  })

  it('parses flags and assigned values and rejects positional arguments', () => {
    expect([...parseCli(['--snapshot', '--scenario=steady'])]).toEqual([
      ['snapshot', true],
      ['scenario', 'steady'],
    ])
    expect(() => parseCli(['steady'])).toThrow(/unexpected argument/)
    expect(() => parseCli(['--scenario='])).toThrow(/invalid option/)
    expect(() => parseCli(['--snapshot', '--snapshot=/tmp'])).toThrow(/duplicate option/)
  })

  it('rejects unknown options and missing numeric values', () => {
    const options = parseCli(['--choies=100000'])
    expect(() => assertCliOptions(options, ['choices'])).toThrow(/unknown option/)
    expect(() => numberOption(parseCli(['--choices']), 'choices', 100_000)).toThrow(
      /requires a value/,
    )
  })

  it('runs children with exposed GC and propagates malformed output', () => {
    const fixture = fileURLToPath(new URL('../fixtures/memoryChild.ts', import.meta.url))
    expect(runIsolated<{ gc: string; child: boolean }>(fixture, [])).toEqual({
      gc: 'function',
      child: true,
    })
    expect(() => runIsolated(fixture, ['--invalid-json'])).toThrow(/invalid JSON/)
    expect(() => runIsolated(fixture, ['--fail'])).toThrow(/Command failed|child failure/)
  })
})
