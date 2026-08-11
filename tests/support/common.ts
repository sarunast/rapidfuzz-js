/**
 * Port of RapidFuzz's `tests/common.py` + `tests/distance/common.py`.
 *
 * The Python original cross-checks the pure-Python scorer against the C++ one
 * and against `process.extractOne`/`extract`/`cdist`. We have a single
 * implementation and no `process` module yet, so what carries over is the part
 * that actually pins down each metric's contract:
 *
 *   - `distance == maximum - similarity`
 *   - `normalizedDistance == distance / maximum`
 *   - `normalizedSimilarity == similarity / maximum`
 *   - both are 0.0 / 1.0 respectively when `maximum` is 0
 *   - symmetric metrics give the same score with the arguments swapped
 *
 * Every assertion in the ported test files runs through here, so a metric that
 * gets one of its four entry points wrong fails even when the test only asserts
 * on `distance`.
 */
import { expect } from 'vitest'

import type { LevenshteinWeightsInput } from '../../src/algorithms/levenshtein/metric.js'
import type {
  EditopTag,
  Editops,
  MatchingBlock,
  OpcodeTag,
  Opcodes,
} from '../../src/algorithms/shared/editops/index.js'
import type {
  ScorerOptions,
  Sequence,
} from '../../src/algorithms/shared/scorerSupport.js'

/** Union of every scorer-specific option, so one harness covers all metrics. */
export interface TestOptions extends ScorerOptions {
  weights?: LevenshteinWeightsInput | undefined
  pad?: boolean | undefined
  prefixWeight?: number | undefined
}

type ScorerFn = (s1: Sequence, s2: Sequence, options?: TestOptions) => number

/** The four internal functions used to verify each algorithm implementation. */
export interface ScorerFns {
  distance: ScorerFn
  similarity: ScorerFn
  normalizedDistance: ScorerFn
  normalizedSimilarity: ScorerFn
}

export interface ScorerFlags {
  /** Score of a maximally dissimilar pair — the denominator for normalisation. */
  maximum: number
  symmetric: boolean
}

/**
 * Call something with an argument its parameter types do not admit.
 *
 * Two ported behaviours need this. `NaN` is still "missing" at runtime, for
 * parity with Python's `float("nan")`, but it is deliberately not in
 * `MaybeSequence` — see the comment there. And a non-`NaN` number is the case
 * upstream raises `TypeError` on, which is now also a compile error, so the
 * test that pins the throw cannot spell the call directly.
 *
 * Routing through `Reflect.apply` rather than asserting a type is the same
 * trick `callScorer` in `src/process.ts` uses at the one place a scorer's
 * concrete inputs meet a caller that only knows it holds *some* scorer. The
 * rest parameter is `never[]` so that every function is assignable to it
 * contravariantly; `R` still comes back precisely, so a result can be indexed
 * or awaited as usual.
 */
export function callUntyped<R>(fn: (...args: never[]) => R, ...args: unknown[]): R {
  return Reflect.apply(fn, undefined, args)
}

/** Mirrors `pytest.approx`, which compares relatively rather than absolutely. */
function approxEqual(a: number, b: number): boolean {
  if (a === b) return true
  return Math.abs(a - b) <= 1e-6 * Math.max(Math.abs(a), Math.abs(b))
}

function expectApprox(actual: number, expected: number, what: string): void {
  expect(
    approxEqual(actual, expected),
    `${what}: expected ${actual} to approximately equal ${expected}`,
  ).toBe(true)
}

export class GenericScorer {
  readonly #fns: ScorerFns
  readonly #getFlags: (s1: Sequence, s2: Sequence, options: TestOptions) => ScorerFlags

  constructor(
    fns: ScorerFns,
    getFlags: (s1: Sequence, s2: Sequence, options: TestOptions) => ScorerFlags,
  ) {
    this.#fns = fns
    this.#getFlags = getFlags
  }

  /** Run `fn` both ways round and assert the metric is symmetric, as declared. */
  #call(fn: ScorerFn, s1: Sequence, s2: Sequence, options: TestOptions): number {
    const score = fn(s1, s2, options)

    if (this.#getFlags(s1, s2, options).symmetric) {
      expectApprox(fn(s2, s1, options), score, 'symmetry')
    }

    return score
  }

  /** The invariant block from `GenericScorer._validate`. */
  #validate(
    s1: Sequence,
    s2: Sequence,
    options: TestOptions,
  ): [number, number, number, number] {
    const base: TestOptions = { ...options, scoreCutoff: undefined }
    const { maximum } = this.#getFlags(s1, s2, base)

    const dist = this.#call(this.#fns.distance, s1, s2, base)
    const sim = this.#call(this.#fns.similarity, s1, s2, base)
    const normDist = this.#call(this.#fns.normalizedDistance, s1, s2, base)
    const normSim = this.#call(this.#fns.normalizedSimilarity, s1, s2, base)

    expectApprox(dist, maximum - sim, 'distance == maximum - similarity')

    if (maximum !== 0) {
      expectApprox(normDist, dist / maximum, 'normalizedDistance == distance / maximum')
      expectApprox(normSim, sim / maximum, 'normalizedSimilarity == similarity / maximum')
    } else {
      expectApprox(normDist, 0, 'normalizedDistance of two empty inputs')
      expectApprox(normSim, 1, 'normalizedSimilarity of two empty inputs')
    }

    return [dist, sim, normDist, normSim]
  }

  distance(s1: Sequence, s2: Sequence, options: TestOptions = {}): number {
    const [dist] = this.#validate(s1, s2, options)
    return options.scoreCutoff == null
      ? dist
      : this.#call(this.#fns.distance, s1, s2, options)
  }

  similarity(s1: Sequence, s2: Sequence, options: TestOptions = {}): number {
    const [, sim] = this.#validate(s1, s2, options)
    return options.scoreCutoff == null
      ? sim
      : this.#call(this.#fns.similarity, s1, s2, options)
  }

  normalizedDistance(s1: Sequence, s2: Sequence, options: TestOptions = {}): number {
    const [, , normDist] = this.#validate(s1, s2, options)
    return options.scoreCutoff == null
      ? normDist
      : this.#call(this.#fns.normalizedDistance, s1, s2, options)
  }

  normalizedSimilarity(s1: Sequence, s2: Sequence, options: TestOptions = {}): number {
    const [, , , normSim] = this.#validate(s1, s2, options)
    return options.scoreCutoff == null
      ? normSim
      : this.#call(this.#fns.normalizedSimilarity, s1, s2, options)
  }
}

/**
 * `max(|s1|, |s2|)` — the `maximum` used by most metrics.
 *
 * Measured the way a scorer measures, which for a string means code points:
 * `conv` expands an astral one before anything counts it, so `'\u{1F600}a'` is two
 * elements to every metric and three to JavaScript. Reading `.length` here
 * instead would not fail a comparison against a scorer — it would quietly
 * expect the wrong normalised value for any input above the BMP.
 */
export function maxLen(s1: Sequence, s2: Sequence): number {
  return Math.max(scorerLength(s1), scorerLength(s2))
}

function scorerLength(s: Sequence): number {
  return typeof s === 'string' ? [...s].length : s.length
}

/**
 * Upstream's `as_list()`, kept in the harness rather than on the collections.
 *
 * `Editops` and `Opcodes` hold records here — the Python list protocol and the
 * tuples that came with it are not part of this library's API. Every expected
 * value below is still transcribed from RapidFuzz's own suite, where an
 * operation *is* a tuple, so the conversion has to live somewhere; putting it
 * here keeps those tables comparable to upstream's line for line, and keeps
 * the API from carrying a method that exists for the tests.
 */
export function editopTuples(ops: Editops): Array<[EditopTag, number, number]> {
  return ops.operations.map((op) => [op.tag, op.srcPos, op.destPos])
}

export function opcodeTuples(
  ops: Opcodes,
): Array<[OpcodeTag, number, number, number, number]> {
  return ops.operations.map((op) => [
    op.tag,
    op.srcStart,
    op.srcEnd,
    op.destStart,
    op.destEnd,
  ])
}

export function blockTuples(
  blocks: readonly MatchingBlock[],
): Array<[number, number, number]> {
  return blocks.map((block) => [block.srcStart, block.destStart, block.length])
}
