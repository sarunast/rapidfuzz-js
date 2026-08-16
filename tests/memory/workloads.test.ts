import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import {
  LATE_INVALID_ELEMENTS,
  runLateInvalidQuery,
} from '../../bench/memory/workloads/exceptionRecovery.ts'
import {
  QUERY_PROFILE_ELEMENTS,
  runQueryProfileSpike,
} from '../../bench/memory/workloads/queryProfile.ts'
import {
  lowercaseBigramCorpus,
  ordinaryQuery,
  type IndexedMatcherWorkload,
  unrelatedQuery,
} from '../../bench/memory/workloads/shared.ts'
import { emptySteadyTally, runSteadyBatch } from '../../bench/memory/workloads/steady.ts'
import {
  lowercaseBigramQuery,
  runTouchedSetSpike,
  validateTouchedCorpus,
} from '../../bench/memory/workloads/touchedSet.ts'

function stub(
  onQuery: (query: ArrayLike<unknown> | string) => void,
): IndexedMatcherWorkload {
  return {
    size: 100_000,
    best(query) {
      onQuery(query)
      return undefined
    },
    search(query) {
      onQuery(query)
      return []
    },
  }
}

describe('runtime-neutral memory workloads', () => {
  it('generates deterministic, distinct queries from the operation index', () => {
    const first = Array.from({ length: 10_000 }, (_, index) => ordinaryQuery(index))
    expect(new Set(first).size).toBe(first.length)
    expect(ordinaryQuery(1234)).toBe(ordinaryQuery(1234))
    expect(unrelatedQuery(1234)).not.toMatch(/[a-z]/)
  })

  it('executes the exact steady-state operation mix', () => {
    const tally = runSteadyBatch(
      stub(() => {}),
      0,
      5_000,
      emptySteadyTally(),
    )
    expect(tally).toEqual({
      best: 2_105,
      limit5: 1_579,
      limit100: 789,
      unrelated: 526,
      unlimited: 1,
    })
  })

  it('preserves the steady operation mix for custom batch sizes', () => {
    const tally = runSteadyBatch(
      stub(() => {}),
      0,
      1_000,
    )
    expect(tally).toEqual({
      best: 421,
      limit5: 316,
      limit100: 157,
      unrelated: 105,
      unlimited: 1,
    })
  })

  it('creates the giant profile query inside the operation helper', () => {
    let observedLength = 0
    runQueryProfileSpike(
      stub((query) => {
        observedLength = query.length
        expect(query[0]).toBe(0)
        expect(query[QUERY_PROFILE_ELEMENTS - 1]).toBe(QUERY_PROFILE_ELEMENTS - 1)
      }),
    )
    expect(observedLength).toBe(QUERY_PROFILE_ELEMENTS)
  })

  it('reaches the final element before rejecting the late-invalid query', () => {
    let sawFinal = false
    const matcher = stub((query) => {
      expect(query[0]).toBe(0)
      expect(query[LATE_INVALID_ELEMENTS - 2]).toBe(LATE_INVALID_ELEMENTS - 2)
      expect(String(query[LATE_INVALID_ELEMENTS - 1])).toBe('late-invalid-sentinel')
      sawFinal = true
      throw new TypeError(
        'an indexed choice holds integer elements only, and one of them is late-invalid-sentinel',
      )
    })
    runLateInvalidQuery(matcher)
    expect(sawFinal).toBe(true)
  })

  it('covers every lowercase bigram without a dense posting', () => {
    const query = lowercaseBigramQuery()
    const grams = new Set(
      Array.from({ length: query.length - 1 }, (_, i) => query.slice(i, i + 2)),
    )
    expect(query).toHaveLength(26 * 26 + 1)
    expect(grams.size).toBe(26 * 26)

    const corpus = lowercaseBigramCorpus(100_000)
    expect(validateTouchedCorpus(corpus)).toMatchObject({ touched: 100_000, fraction: 1 })
    let observed = ''
    runTouchedSetSpike(stub((value) => (observed = String(value))))
    expect(observed).toBe(query)
  })

  it('does not import Node runtime mechanics from workload modules', () => {
    const directory = new URL('../../bench/memory/workloads/', import.meta.url)
    for (const name of [
      'shared.ts',
      'steady.ts',
      'queryProfile.ts',
      'touchedSet.ts',
      'exceptionRecovery.ts',
    ]) {
      const source = readFileSync(fileURLToPath(new URL(name, directory)), 'utf8')
      expect(source).not.toMatch(/node:|child_process|globalThis\.gc|process\./)
    }
  })
})
