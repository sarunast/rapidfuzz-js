/**
 * One set of measurement options for every case, and the state resets that stop
 * a case's number depending on which case ran before it.
 *
 * ## Why not the stock options
 *
 * The runner's defaults are tuned for benchmarks read once, off a terminal.
 * These are read against a stored baseline months later, where the question is
 * whether a few percent moved — so both windows are longer: enough warmup that
 * a case is measured after V8 has settled on a tier rather than during the
 * change, and enough measured time that the median has a few hundred samples
 * to choose from rather than a few dozen.
 *
 * Note what these options deliberately do *not* try to fix. A case whose body
 * is tens of milliseconds is not rescued by demanding more samples of it —
 * every one of those samples is a wide enough window to catch a collection or
 * a preemption, so the median has nothing clean left to pick. The fix for that
 * is a smaller body, and `compare.mjs` checks the sizes it actually measured
 * and names the cases outside the envelope. This file's job is to give a
 * properly sized case enough warmup and enough time.
 *
 * ## Why the resets
 *
 * The kernels reuse module-level buffers that grow on demand and never shrink,
 * and the shared symbol table grows permanently. So a case that ran after a
 * 16,384-element pair never paid for the allocation that pair paid for, and
 * measured faster for it. Clearing the buffers before each warmup gives every
 * case the same starting state; the measured phase still runs warm, because the
 * warmup regrows what it needs.
 *
 * ## `pnpm bench` is not `pnpm bench:compare`
 *
 * Collecting between cases needs `--expose-gc`, which `compare.mjs` passes and
 * a bare `vitest bench` does not — under it, {@link collectGarbage} is a no-op
 * and one case's garbage is collected on the next one's time. That is fine for
 * a quick look, which is what `pnpm bench` is for. Numbers meant to be compared
 * against a baseline come from `pnpm bench:compare`.
 */

import { bench } from 'vitest'

import { resetDamerauScratch } from '../src/algorithms/damerauLevenshtein/implementation.js'
import { resetJaroScratch } from '../src/algorithms/jaro/implementation.js'
import { resetWeightedScratch } from '../src/algorithms/levenshtein/internal/scratch.js'
import { resetOsaScratch } from '../src/algorithms/osa/internal/kernel.js'
import { resetBitVectorScratch } from '../src/algorithms/shared/bitmask/blockMasks.js'

/**
 * The subset of tinybench's options a case may override.
 *
 * Spelled out here rather than imported: `tinybench` is vitest's dependency,
 * not ours, and under pnpm's layout it is not resolvable from this file.
 */
export interface MeasureOptions {
  /** Milliseconds to keep sampling for, once `iterations` is satisfied. */
  time?: number
  /** Fewest samples to take, even if `time` has elapsed. */
  iterations?: number
  /** Milliseconds of discarded samples before measurement starts. */
  warmupTime?: number
  /** Fewest discarded samples before measurement starts. */
  warmupIterations?: number
}

/**
 * `gc` exists only under `--expose-gc`, which `bench/compare.mjs` passes and a
 * bare `vitest bench` does not. Collecting before a case starts keeps the
 * previous case's garbage from being collected on this case's time.
 */
function collectGarbage(): void {
  globalThis.gc?.()
}

function beforeCase(_task: unknown, mode: 'warmup' | 'run'): void {
  if (mode === 'warmup') {
    resetBitVectorScratch()
    resetOsaScratch()
    resetWeightedScratch()
    resetJaroScratch()
    resetDamerauScratch()
  }
  collectGarbage()
}

/**
 * The two floors are not the same number because they answer different
 * questions.
 *
 * `iterations` is how many samples the median gets to choose from, so it is a
 * statistical floor. `warmupIterations` only has to be enough executions for
 * V8 to have finished tiering, and at the sample sizes this suite aims for the
 * 300 ms window already supplies a few hundred of them — the floor matters
 * only for a case whose body is slow enough that the window buys very few, and
 * there it is pure cost. Sixteen is enough to reach the optimising tier;
 * matching it to `iterations` would just charge the slowest cases for samples
 * that get discarded.
 */
const DEFAULTS = {
  warmupTime: 300,
  warmupIterations: 16,
  time: 1000,
  iterations: 32,
  setup: beforeCase,
}

/**
 * Shortened windows for `--quick`, set by `compare.mjs` in the child's
 * environment.
 *
 * A tenth of the measured time and a third of the warmup, which turns a
 * ten-minute comparison into under a minute. Both halves of that cost
 * something. Fewer samples make each median coarser, and a shorter warmup can
 * leave a case measured before V8 has settled on its final tier — which biases
 * a case and its controls by different amounts, since different code tiers up
 * at different rates, so it does not cancel in the ratio.
 *
 * The result answers "did I break something" and not "is this 4% faster".
 * `compare.mjs` widens its threshold to match and refuses to record a baseline
 * from it.
 */
const QUICK = {
  warmupTime: 100,
  warmupIterations: 8,
  time: 100,
  iterations: 8,
  setup: beforeCase,
}

const BASE = process.env['BENCH_QUICK'] === '1' ? QUICK : DEFAULTS

/**
 * Register a benchmark case.
 *
 * A drop-in for vitest's `bench` that supplies {@link DEFAULTS}. Cases should
 * only override them with a reason — the point of one shared set is that two
 * cases are comparable, and {@link MeasureOptions} deliberately offers no way
 * to replace {@link beforeCase} and opt out of the resets.
 *
 * `void` here does not forbid a body that returns something: TypeScript ignores
 * the return type of a function assigned where `void` is expected, so
 * `measure('ratio', () => ratio(a, b))` is as valid as the braced form. It says
 * only that nothing reads the value.
 */
export function measure(
  name: string,
  fn: () => void,
  overrides: MeasureOptions = {},
): void {
  // A case's own overrides exist to buy back samples its size costs it, which
  // is the opposite of what quick mode is for — honouring them there would make
  // the slowest cases most of a run that exists to be fast.
  bench(name, fn, BASE === QUICK ? BASE : { ...BASE, ...overrides })
}
