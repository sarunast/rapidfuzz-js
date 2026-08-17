import { describe, expect, expectTypeOf, it } from 'vitest'

import { callUntyped } from '../../../testing/untyped.js'
import type { Sequence } from '../types.js'
import {
  NORMALIZED_SIMILARITY_FLAGS,
  withPreparedFlags,
} from './builtIn/implementation.js'
import {
  builtInMetric,
  type BuiltInMetric,
  type ExplainableBuiltInMetric,
} from './builtIn/metric.js'
import type { PreparationFactory } from './builtIn/preparation.js'
import {
  createScorer,
  scorerCompilation,
  type ExplainableScorer,
  type Scorer,
} from './scorer.js'

interface ProbeConfiguration {
  readonly depth?: number | undefined
}

interface ProbeExplainConfiguration extends ProbeConfiguration {
  readonly depth: 1
}

interface ProbeEvidence {
  readonly score: number
  readonly firstLength: number
  readonly secondLength: number
}

function probeScore(first: Sequence, second: Sequence): number {
  return first.length === second.length ? 1 : 0
}

function probeSequence(value: unknown): Sequence {
  if (typeof value !== 'string' && !Array.isArray(value)) {
    throw new TypeError('invalid prepared probe')
  }
  return value
}

function probeKernels(): {
  prepareQuery: (query: Sequence) => (choice: unknown) => number
  prepareChoice: (choice: Sequence) => Sequence
} {
  return {
    prepareQuery: (query) => (choice) => probeScore(query, probeSequence(choice)),
    prepareChoice: (choice) => choice,
  }
}

const explainingPreparation: PreparationFactory<ProbeEvidence> = (options) => ({
  ...probeKernels(),
  explain:
    Reflect.get(options, 'depth') === 1
      ? (first, second) => ({
          score: probeScore(first, second),
          firstLength: first.length,
          secondLength: second.length,
        })
      : undefined,
})

const plainPreparation: PreparationFactory = () => probeKernels()

// Two distinct function objects: `withPreparedFlags` installs the preparation
// on the function it is handed, so one shared body would give both metrics the
// second registration.
const explainable: ExplainableBuiltInMetric<
  'probe.explainable',
  'similarity',
  ProbeConfiguration,
  ProbeExplainConfiguration,
  ProbeEvidence
> = builtInMetric({
  implementation: withPreparedFlags(
    (first: Sequence, second: Sequence) => probeScore(first, second),
    NORMALIZED_SIMILARITY_FLAGS,
    explainingPreparation,
  ),
  direction: 'similarity',
  bounds: [0, 1],
  configurationKeys: ['depth'],
})

const plain: BuiltInMetric<'probe.plain', 'similarity', ProbeConfiguration> =
  builtInMetric({
    implementation: withPreparedFlags(
      (first: Sequence, second: Sequence) => probeScore(first, second),
      NORMALIZED_SIMILARITY_FLAGS,
      plainPreparation,
    ),
    direction: 'similarity',
    bounds: [0, 1],
    configurationKeys: ['depth'],
  })

describe('an explanation capability', () => {
  it('reaches the scorer as a method when the preparation declares one', () => {
    const scorer = createScorer(explainable, { depth: 1 })

    expect(scorer.explain('abc', 'xyz')).toEqual({
      score: 1,
      firstLength: 3,
      secondLength: 3,
    })
    expect(scorer.explain('abc', 'xy')).toEqual({
      score: 0,
      firstLength: 3,
      secondLength: 2,
    })
    expect(scorer.score('abc', 'xyz')).toBe(1)
    expect(Object.isFrozen(scorer)).toBe(true)
    expect(scorerCompilation(scorer).explain).toBeTypeOf('function')
  })

  it('refuses an operand it cannot explain', () => {
    const scorer = createScorer(explainable, { depth: 1 })

    expect(() => callUntyped(scorer.explain, null, 'abc')).toThrow(TypeError)
    expect(() => callUntyped(scorer.explain, 'abc', undefined)).toThrow(TypeError)
    expect(() => callUntyped(scorer.explain, 42, 'abc')).toThrow(TypeError)
  })

  it('leaves an ordinary scorer without the property at all', () => {
    const scorer = createScorer(plain, { depth: 1 })

    expect('explain' in scorer).toBe(false)
    expect(scorerCompilation(scorer).explain).toBeUndefined()
    expect(scorer.score('abc', 'xyz')).toBe(1)
    expect(Object.isFrozen(scorer)).toBe(true)
  })

  it('is decided by the configuration, not by the metric alone', () => {
    const configured = createScorer(explainable, { depth: 2 })

    expect('explain' in configured).toBe(false)
    expect(configured.score('abc', 'xyz')).toBe(1)
  })
})

// `toEqualTypeOf` rather than `not.toHaveProperty`: the latter compiles either
// way here, so it would pin nothing. These are checked by `typecheck:dev`,
// which is the config that compiles `*.test.ts` — `typecheck:library` excludes
// them.
type ExplainableProbe = ExplainableScorer<
  'similarity',
  'probe.explainable',
  ProbeEvidence
>

describe('the capability in the type system', () => {
  it('is granted by a configuration literal that unlocks it', () => {
    expectTypeOf(
      createScorer(explainable, { depth: 1 }),
    ).toEqualTypeOf<ExplainableProbe>()
    expectTypeOf(
      createScorer(explainable, { depth: 1 }).explain('abc', 'xyz'),
    ).toEqualTypeOf<ProbeEvidence>()
  })

  it('is withheld from a configuration that does not, and from a plain metric', () => {
    expectTypeOf(createScorer(explainable, { depth: 2 })).toEqualTypeOf<
      Scorer<'similarity', 'probe.explainable'>
    >()
    expectTypeOf(createScorer(plain, { depth: 1 })).toEqualTypeOf<
      Scorer<'similarity', 'probe.plain'>
    >()
  })

  it('is withheld from a configuration hoisted without its literal type', () => {
    const widened = { depth: 1 }
    const preserved = { depth: 1 } satisfies ProbeExplainConfiguration

    expectTypeOf(createScorer(explainable, widened)).toEqualTypeOf<
      Scorer<'similarity', 'probe.explainable'>
    >()
    expectTypeOf(createScorer(explainable, preserved)).toEqualTypeOf<ExplainableProbe>()
  })
})
