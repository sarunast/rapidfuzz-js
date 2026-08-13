/**
 * Not ported from RapidFuzz — upstream does not agree with itself on these inputs,
 * so there is no single upstream answer to port. Each fixture entry records what the
 * C++ extension says, what the pure-Python fallback says, which side is *correct* and
 * the defect that makes the other one wrong.
 *
 * The verdict is never a preference. Every entry carries `evidence`: spellings of the
 * same comparison on which both backends agree. That agreement is what identifies the
 * defective answer, and the tests below assert we match the evidence as well as the
 * verdict — being consistent across the equivalent spellings is the whole claim.
 *
 * See tests/fuzz/tokenWhitespace.test.ts for the prose behind the token-splitting pair.
 */

import { describe, expect, test } from 'vitest'

import * as jaroWinkler from '../../src/algorithms/jaroWinkler/index.js'
import * as levenshtein from '../../src/algorithms/levenshtein/index.js'
import { scorePairs } from '../../src/batch/scorePairs.js'
import { createScorer } from '../../src/core/scoring/scorer.js'
import type { Sequence } from '../../src/core/types.js'
import * as fuzz from '../../src/fuzz/index.js'
import fixture from '../fixtures/rapidfuzz-3.14.5.json' with { type: 'json' }

type Observed = number | readonly number[] | undefined

const DEFECTS = [
  'self-contradiction',
  'out-of-domain-value',
  'leaked-internal-error',
  'both-defective',
] as const

function isSequence(value: unknown): value is Sequence {
  return typeof value === 'string' || Array.isArray(value)
}

function sequenceField(entry: object, key: string): Sequence {
  const value = Reflect.get(entry, key)
  if (!isSequence(value)) throw new TypeError(`divergence ${key} must be a sequence`)
  return value
}

function stringsField(entry: object, key: string): string[] {
  const value = Reflect.get(entry, key)
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    throw new TypeError(`divergence ${key} must be a list of strings`)
  }
  return value.filter((item) => typeof item === 'string')
}

function numberField(entry: object, key: string): number {
  const value = Reflect.get(entry, key)
  if (typeof value !== 'number') throw new TypeError(`divergence ${key} must be numeric`)
  return value
}

function optionalField(entry: object, key: string): unknown {
  return Reflect.get(entry, key)
}

/** What this port answers for the surface an entry names. */
function observe(entry: object, surface: string): Observed {
  if (surface === 'fuzz.tokenSortSimilarity') {
    return fuzz.tokenSortSimilarity(
      sequenceField(entry, 'left'),
      sequenceField(entry, 'right'),
    )
  }
  if (surface === 'jaroWinkler.similarity') {
    const threshold = optionalField(entry, 'threshold')
    const left = sequenceField(entry, 'left')
    const right = sequenceField(entry, 'right')
    const scorer = createScorer(jaroWinkler.similarity)
    return threshold === undefined
      ? scorer.score(left, right)
      : scorer.score(left, right, { threshold: numberField(entry, 'threshold') })
  }
  if (surface === 'scorePairs(levenshtein.distance)') {
    return Array.from(
      scorePairs(stringsField(entry, 'queries'), stringsField(entry, 'choices'), {
        scorer: createScorer(levenshtein.distance),
        threshold: numberField(entry, 'threshold'),
      }),
    )
  }
  throw new TypeError(`unknown divergence surface ${surface}`)
}

function agrees(actual: Observed, expected: unknown): boolean {
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length) return false
    return expected.every((value, index) => {
      const ours = actual[index]
      return (
        typeof value === 'number' &&
        typeof ours === 'number' &&
        Math.abs(ours - value) < 1e-12
      )
    })
  }
  return (
    typeof expected === 'number' &&
    typeof actual === 'number' &&
    Math.abs(actual - expected) < 1e-12
  )
}

/**
 * The value an `ours` verdict is held to: what both backends answer for every
 * equivalent spelling. A verdict with no such agreement has nothing behind it.
 */
function agreedEvidenceValue(entry: {
  readonly id: string
  readonly evidence: unknown[]
}): number {
  const values = entry.evidence.map((item) => {
    if (typeof item !== 'object' || item === null) {
      throw new TypeError(`${entry.id} evidence must be an object`)
    }
    const cpp = Reflect.get(item, 'cpp')
    const py = Reflect.get(item, 'py')
    if (typeof cpp !== 'number' || typeof py !== 'number' || cpp !== py) {
      throw new TypeError(`${entry.id} evidence does not agree across backends`)
    }
    return cpp
  })
  const [first] = values
  if (first === undefined || values.some((value) => value !== first)) {
    throw new TypeError(`${entry.id} evidence does not agree with itself`)
  }
  return first
}

describe(`RapidFuzz ${fixture.rapidfuzzVersion} intentional divergences`, () => {
  test('gives every divergence a verdict and a named defect', () => {
    expect(fixture.divergences.length).toBeGreaterThan(0)
    for (const entry of fixture.divergences) {
      expect(['cpp', 'py', 'ours'], `${entry.id} verdict`).toContain(entry.correct)
      expect(DEFECTS, `${entry.id} defect`).toContain(entry.defect)
      expect(entry.evidence.length, `${entry.id} evidence`).toBeGreaterThan(0)
      expect(entry.reason.length, `${entry.id} reason`).toBeGreaterThan(0)
    }
  })

  test('answers the correct side, and differs from the defective one', () => {
    for (const entry of fixture.divergences) {
      const actual = observe(entry, entry.surface)
      const correct =
        entry.correct === 'cpp'
          ? entry.cpp
          : entry.correct === 'py'
            ? entry.py
            : agreedEvidenceValue(entry)
      const rejected = entry.correct === 'cpp' ? entry.py : entry.cpp

      expect(agrees(actual, correct), `${entry.id} answers the correct side`).toBe(true)

      // `ours` means neither backend was right on this input, so there is no single
      // rejected value; and a backend that raises has no value to differ from.
      if (entry.correct !== 'ours' && rejected !== null) {
        expect(
          agrees(actual, rejected),
          `${entry.id} still differs from the defect`,
        ).toBe(false)
      }
    }
  })

  test('answers every equivalent spelling the same way both backends do', () => {
    for (const entry of fixture.divergences) {
      for (const item of entry.evidence) {
        const cpp = optionalField(item, 'cpp')
        const py = optionalField(item, 'py')
        // Only the spellings the two backends agree on carry an oracle, and only
        // those that name their own inputs can be replayed.
        if (cpp !== py) continue
        if (!isSequence(optionalField(item, 'left'))) continue

        const actual = observe(item, entry.surface)
        expect(
          agrees(actual, cpp),
          `${entry.id} evidence: ${optionalField(item, 'note')}`,
        ).toBe(true)
      }
    }
  })
})
