/**
 * The benchmark registry, the sampling loop, and the state resets that stop a
 * case's number depending on which case ran before it.
 *
 * ## Why no vitest
 *
 * These cases used to run through `vitest bench`, which meant vite's SSR
 * transform sat between the measured code and V8: every ESM import rewritten
 * into namespace property access, constant folding defeated, and roughly 2.5x
 * added to every case body — asymmetrically, so an A/B across module layouts
 * lied. `bench/harness/runner.ts` now bundles a run once with esbuild and executes it
 * in bare `node`: plain statically compiled JS with no runtime transform.
 * (Not the shipped `dist/` shape — tsdown emits that unbundled — but a stable
 * artifact free of the distortion, which is what regression numbers need.)
 * This file supplies what vitest used to: `describe`, `measure`, and the loop
 * that turns a case into a median.
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
 * ## `pnpm bench:quick` is not `pnpm bench:compare`
 *
 * Quick mode shortens every window to a tenth; see {@link QUICK}. Short windows
 * can measure a case before V8 has settled on a tier, so quick numbers answer
 * "is it roughly where I left it", never "is this 4% faster". `compare.ts`
 * widens its threshold to match and refuses to record a baseline from them.
 */

import { writeFileSync } from 'node:fs'
import process from 'node:process'

import { resetBitVectorScratch } from '../../src/algorithms/bitmask/blockMasks.js'
import { resetDamerauScratch } from '../../src/algorithms/damerauLevenshtein/implementation.js'
import { resetJaroScratch } from '../../src/algorithms/jaro/implementation.js'
import { resetWeightedScratch } from '../../src/algorithms/levenshtein/internal/scratch.js'
import { resetOsaScratch } from '../../src/algorithms/osa/internal/kernel.js'

/** The subset of measurement options a case may override. */
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

interface RegisteredCase {
  readonly file: string
  readonly group: string
  readonly name: string
  readonly fn: () => unknown
  readonly options: RunWindows
}

/** A measurement mode: everything a case may override, plus the floor the
 * adaptive stop must reach before it may end sampling early. */
interface RunWindows extends Required<MeasureOptions> {
  readonly minTime: number
}

/**
 * The two floors are not the same number because they answer different
 * questions.
 *
 * `iterations` is how many samples the median gets to choose from, so it is a
 * statistical floor. `warmupIterations` only prevents a very slow case from
 * receiving one or two warmup executions when the time window buys so few;
 * the time window is the primary warmup mechanism, because V8's tiering
 * depends on too much — version, call-site shape, OSR, deopts — for any
 * fixed execution count to be a guarantee.
 *
 * `warmupTime` stays fixed while `time` is only a ceiling (see the adaptive
 * stop in {@link runCase}): a case that looks stable during warmup may merely
 * be stable in a lower tier, so elapsed time is the only safe proxy there,
 * whereas a settled *measured* median is the quantity the comparison reads.
 * A well-behaved case exits at `minTime`. `iterations` outranks the ceiling:
 * a 100 ms body takes 2.4 seconds to reach 24 samples, and does.
 */
const DEFAULTS: RunWindows = {
  warmupTime: 150,
  warmupIterations: 8,
  minTime: 200,
  time: 750,
  iterations: 24,
}

/**
 * Shortened windows for `--quick`, set by the spawning script in this
 * process's environment.
 *
 * Short enough that a full quick pass answers "did I break something" in
 * seconds. The cost is real: fewer samples make the median coarser, and a
 * short warmup can leave a case measured before V8 has settled on its final
 * tier — which biases a case and its controls by different amounts, since
 * different code tiers up at different rates, so it does not cancel in the
 * ratio.
 */
const QUICK: RunWindows = {
  warmupTime: 75,
  warmupIterations: 8,
  minTime: 100,
  time: 100,
  iterations: 8,
}

/**
 * Widened windows for `--confirm`, for re-measuring the one case a normal
 * comparison flagged. Deliberately past the point of diminishing returns:
 * this mode runs a handful of cases, not the suite, so its job is precision.
 * It answers "is that 3-5% move really there", which a slightly longer normal
 * run cannot.
 */
const CONFIRM: RunWindows = {
  warmupTime: 300,
  warmupIterations: 16,
  minTime: 1000,
  time: 2000,
  iterations: 64,
}

const QUICK_MODE = process.env['BENCH_QUICK'] === '1'
const BASE = QUICK_MODE
  ? QUICK
  : process.env['BENCH_CONFIRM'] === '1'
    ? CONFIRM
    : DEFAULTS

const registry: RegisteredCase[] = []
let currentFile = '(unknown file)'
let currentGroup = ''

/**
 * Name the bench file whose top-level `describe` calls come next.
 *
 * Called by the generated entry stub in `runner.ts` before each bench module
 * is imported, because a bundled module cannot ask where its source lived and
 * the reported case names have to start with the project-relative path the
 * baseline stores — `bench/fuzz.bench.ts > partialRatio > …`.
 */
export function setCurrentFile(file: string): void {
  currentFile = file
}

/**
 * Register a named group of cases. The body runs immediately, at module load,
 * exactly as vitest's `describe` did; only registration happens here, no
 * measurement.
 */
export function describe(name: string, body: () => void): void {
  const previous = currentGroup
  currentGroup = previous === '' ? name : `${previous} > ${name}`
  body()
  currentGroup = previous
}

/**
 * Register a benchmark case.
 *
 * Cases should only override the defaults with a reason — the point of one
 * shared set is that two cases are comparable, and {@link MeasureOptions}
 * deliberately offers no way to opt out of the per-case state resets.
 *
 * `unknown` rather than `void`, because the value is read: every call's result
 * is written to a sink so V8 cannot delete the work that produced it. Either
 * spelling accepts `measure('ratio', () => ratio(a, b))`, but only this one
 * says what happens to the ratio.
 */
export function measure(
  name: string,
  fn: () => unknown,
  overrides: MeasureOptions = {},
): void {
  registry.push({
    file: currentFile,
    group: currentGroup,
    name,
    fn,
    // A case's own overrides exist to buy back samples its size costs it,
    // which is the opposite of what quick mode is for — honouring them there
    // would make the slowest cases most of a run that exists to be fast.
    options: QUICK_MODE ? BASE : { ...BASE, ...overrides },
  })
}

/**
 * `gc` exists only under `--expose-gc`, which `runner.ts` always passes to
 * this process. It is called once per case, between the warmup and the timed
 * phase: everything the previous case and this case's own warmup left behind
 * is collected there, rather than during a sample.
 */
function collectGarbage(): void {
  globalThis.gc?.()
}

function resetScratchState(): void {
  resetBitVectorScratch()
  resetOsaScratch()
  resetWeightedScratch()
  resetJaroScratch()
  resetDamerauScratch()
}

interface CaseResult {
  readonly file: string
  readonly group: string
  readonly name: string
  readonly median: number
  readonly min: number
  readonly sampleCount: number
  /** Calls per timed sample; median and min are per call regardless. */
  readonly batch: number
  /**
   * What the adaptive stop did, which is otherwise invisible — and the only
   * evidence for whether these windows are the right ones. `stability` is the
   * last block spread measured, null if the case never completed
   * {@link STABLE_BLOCKS} blocks.
   */
  readonly measuredTime: number
  readonly stoppedStable: boolean
  readonly stability: number | null
}

/**
 * The adaptive stop's block structure.
 *
 * Sampling is divided into ~{@link BLOCK_TIME} ms blocks, each with its own
 * median; sampling may end once the case has run for its mode's `minTime`,
 * has its `iterations` floor, and the last {@link STABLE_BLOCKS} block
 * medians sit within {@link STABILITY} of each other. Independent block
 * medians rather than one running median, because a running median over a
 * growing window barely moves by construction — it would call a case stable
 * while its recent samples were still drifting. The comparison's own floor is
 * ±3%, so agreement to 1% leaves the measurement well inside the threshold
 * it feeds.
 *
 * Four blocks rather than three, at the same 50 ms each: three consecutive
 * windows can agree inside a temporary plateau of a case that is still
 * trending, and a fourth is nearly free — 4 x 50 ms is the 200 ms `minTime`
 * already requires, so the stronger condition costs a well-behaved case
 * nothing.
 */
const BLOCK_TIME = 50
const STABLE_BLOCKS = 4
const STABILITY = 0.01

/**
 * How long one timed sample should be, at minimum, for a body measured in
 * single microseconds.
 *
 * Below {@link BATCH_THRESHOLD} ms per call, one sample becomes a small batch
 * of calls timed together and reported per call: at that size the two
 * `hrtime.bigint()` reads and the sample-array growth are a visible share of
 * a single call, and batching to ~{@link BATCH_TARGET} ms drops that share
 * to noise. Longer bodies are never batched — a multi-millisecond sample is
 * one a garbage collection hides in, which is exactly what the median exists
 * to reject.
 */
const BATCH_THRESHOLD = 0.1
const BATCH_TARGET = 0.15
const BATCH_LIMIT = 4096

/** The middle of an already-ascending list. */
function medianOfSorted(values: readonly number[]): number {
  const middle = values.length >> 1
  return values.length % 2 === 1
    ? values[middle]
    : (values[middle - 1] + values[middle]) / 2
}

function medianOf(values: readonly number[]): number {
  return medianOfSorted([...values].sort((left, right) => left - right))
}

/**
 * Where every case's return value goes, so V8 cannot prove the computation
 * dead. The write is unconditionally live — the object outlives every case —
 * which is what denies the optimiser permission to delete increasingly large
 * parts of a body whose result nothing read. Benchmark authors get this for
 * free rather than each remembering their own `sink`.
 */
const sink: { value: unknown } = { value: undefined }

function isThenable(value: unknown): boolean {
  if (value === null) return false
  // A function can carry a `then` too, and a body returning the async
  // function it meant to call is not the far-fetched case it sounds like.
  if (typeof value !== 'object' && typeof value !== 'function') return false
  return 'then' in value && typeof Reflect.get(value, 'then') === 'function'
}

/**
 * Warm a case until V8 has settled, then time executions until the window and
 * the sample floor are both satisfied.
 *
 * One timed sample is `batch` consecutive executions — one, for anything but
 * a body measured in single microseconds — with nothing between the two
 * `hrtime.bigint` reads but the body itself, and the duration divided by the
 * batch so every reported figure stays per call. No promise, no callback:
 * that is the whole reason this loop exists instead of a benchmark library's.
 * At the ~1 ms bodies this suite aims for, per-sample harness overhead is
 * measurable, and overhead that differs between two compared runs is
 * indistinguishable from a regression.
 */
function runCase(entry: RegisteredCase): CaseResult {
  const { fn, options } = entry
  // No collection before the warmup: whatever the previous case left behind is
  // collected during the discarded warmup phase for free. The collection that
  // matters is the one after it, so warmup garbage is not collected inside
  // the timed phase.
  resetScratchState()

  const warmupBudget = BigInt(Math.round(options.warmupTime * 1e6))
  const warmupStart = process.hrtime.bigint()

  // An async body satisfies `() => unknown` as readily as a synchronous one,
  // and the loop below would then time promise construction rather than the
  // work. Caught on the first execution, before a single sample is taken.
  const first: unknown = fn()
  sink.value = first
  if (isThenable(first)) {
    throw new TypeError(
      `${entry.group} > ${entry.name} returned a promise — benchmark bodies ` +
        `must be synchronous, or the harness times promise construction`,
    )
  }

  // Inside the timed window, not before it: `perCall` below divides by
  // `warmups`, so a first execution counted in one and not the other
  // understates the estimate — by the slowest call of the run, since that
  // first one is the coldest. The extra clock read per iteration is paid in
  // discarded warmup only.
  let warmups = 1
  let warmupEnd = process.hrtime.bigint()
  while (warmups < options.warmupIterations || warmupEnd - warmupStart < warmupBudget) {
    sink.value = fn()
    warmups++
    warmupEnd = process.hrtime.bigint()
  }

  // The batch decision needs a per-call estimate before measurement starts,
  // and the warmup produced one — but a warmup call is a call plus a clock
  // read, and the clock read is a meaningful share of exactly the microsecond
  // bodies this decides the batch for. So the warmup figure only decides
  // whether the question is worth asking, and a case anywhere near the
  // threshold is then timed the way it will actually be measured: one clock
  // read either side of a batch of calls. It runs after the warmup and before
  // the collection, so it is neither cold nor part of any sample.
  let perCall = Number(warmupEnd - warmupStart) / 1e6 / warmups
  if (perCall < BATCH_THRESHOLD * 2) {
    const probe = Math.min(BATCH_LIMIT, Math.max(8, Math.ceil(BATCH_TARGET / perCall)))
    const probeStart = process.hrtime.bigint()
    for (let call = 0; call < probe; call++) sink.value = fn()
    perCall = Number(process.hrtime.bigint() - probeStart) / 1e6 / probe
  }
  // Rounded up, because {@link BATCH_TARGET} is the floor of how much work
  // should sit between two clock reads rather than a figure to land near.
  const batch =
    perCall < BATCH_THRESHOLD
      ? Math.min(BATCH_LIMIT, Math.max(1, Math.ceil(BATCH_TARGET / perCall)))
      : 1

  collectGarbage()

  // `options.time` is a cap, not a target: see {@link BLOCK_TIME}. A settled
  // case exits at `minTime`; only one the machine keeps disturbing pays the
  // whole cap. Stability of the *median* is the right stopping signal because
  // the median is the exact statistic everything downstream reads.
  const budget = BigInt(Math.round(options.time * 1e6))
  const minimum = BigInt(Math.round(Math.min(options.minTime, options.time) * 1e6))
  const blockBudget = BigInt(Math.round(BLOCK_TIME * 1e6))
  const start = process.hrtime.bigint()
  const samples: number[] = []
  const blockMedians: number[] = []
  let blockStart = start
  let blockFrom = 0
  let stoppedStable = false
  let stability: number | null = null
  while (true) {
    const before = process.hrtime.bigint()
    for (let call = 0; call < batch; call++) sink.value = fn()
    const after = process.hrtime.bigint()
    samples.push(Number(after - before) / 1e6 / batch)

    if (after - blockStart >= blockBudget) {
      blockMedians.push(medianOf(samples.slice(blockFrom)))
      blockStart = after
      blockFrom = samples.length
    }

    const elapsed = after - start
    if (elapsed >= budget && samples.length >= options.iterations) break
    if (
      elapsed >= minimum &&
      samples.length >= options.iterations &&
      blockMedians.length >= STABLE_BLOCKS
    ) {
      const recent = blockMedians.slice(-STABLE_BLOCKS)
      const centre = medianOf(recent)
      stability = (Math.max(...recent) - Math.min(...recent)) / centre
      if (stability <= STABILITY) {
        stoppedStable = true
        break
      }
    }
  }
  const measuredTime = Number(process.hrtime.bigint() - start) / 1e6

  // Sorted once, here, and read as sorted below: `medianOf` would copy and
  // sort a second time, and at a few thousand samples a case that is pure
  // overhead.
  samples.sort((left, right) => left - right)

  return {
    file: entry.file,
    group: entry.group,
    name: entry.name,
    median: medianOfSorted(samples),
    min: samples[0],
    sampleCount: samples.length,
    batch,
    measuredTime,
    stoppedStable,
    stability,
  }
}

/**
 * The report shape `compare.ts` reads: the same nesting vitest's
 * `--outputJson` produced, so the baseline format and everything downstream of
 * it survived the runner swap unchanged. `fullName` carries the file and the
 * group; each leaf carries the case.
 */
interface Report {
  readonly files: readonly {
    readonly groups: readonly {
      readonly fullName: string
      readonly benchmarks: readonly {
        readonly name: string
        readonly median: number
        readonly min: number
        readonly sampleCount: number
        readonly batch: number
        readonly measuredTime: number
        readonly stoppedStable: boolean
        readonly stability: number | null
      }[]
    }[]
  }[]
}

function reportOf(results: readonly CaseResult[]): Report {
  const groups = new Map<string, CaseResult[]>()
  for (const result of results) {
    const key = `${result.file} > ${result.group}`
    const bucket = groups.get(key)
    if (bucket === undefined) groups.set(key, [result])
    else bucket.push(result)
  }
  return {
    files: [
      {
        groups: [...groups.entries()].map(([fullName, cases]) => ({
          fullName,
          benchmarks: cases.map((result) => ({
            name: result.name,
            median: result.median,
            min: result.min,
            sampleCount: result.sampleCount,
            batch: result.batch,
            measuredTime: result.measuredTime,
            stoppedStable: result.stoppedStable,
            stability: result.stability,
          })),
        })),
      },
    ],
  }
}

function formatMs(value: number): string {
  return value >= 100
    ? value.toFixed(1)
    : value >= 10
      ? value.toFixed(2)
      : value.toFixed(4)
}

/**
 * How the adaptive stop ended, and at what batch size — both worth seeing per
 * case. A case that crosses the batching threshold between two versions has
 * changed measurement regime, which reads as a suspicious move otherwise.
 */
function howItStopped(result: CaseResult): string {
  const spread =
    result.stability === null ? '' : ` ±${(result.stability * 100).toFixed(2)}%`
  return `${result.stoppedStable ? 'stable' : 'window'}${spread}`
}

function printTable(results: readonly CaseResult[]): void {
  let group = ''
  const nameWidth = Math.max(...results.map((result) => result.name.length)) + 2
  for (const result of results) {
    const heading = `${result.file} > ${result.group}`
    if (heading !== group) {
      group = heading
      process.stdout.write(`\n  ${heading}\n`)
    }
    process.stdout.write(
      `    ${result.name.padEnd(nameWidth)}` +
        `${formatMs(result.median).padStart(9)}ms median` +
        `${formatMs(result.min).padStart(9)}ms min` +
        `${String(result.sampleCount).padStart(7)} samples` +
        `${(result.batch === 1 ? '' : `x${result.batch}`).padStart(6)}` +
        `${String(Math.round(result.measuredTime)).padStart(6)}ms  ` +
        `${howItStopped(result)}\n`,
    )
  }
  process.stdout.write('\n')
}

/**
 * Run every registered case and report.
 *
 * Called by the generated entry stub after all bench modules have loaded.
 * `BENCH_FILTER` carries the `-t` regexp and is matched against
 * `group > name`, which is what vitest matched its name pattern against.
 * `BENCH_OUTPUT` names the JSON file `compare.ts` reads; without it the
 * results print as a table, which is what `pnpm bench` and `pnpm bench:quick`
 * show.
 */
export function runRegisteredBenchmarks(): void {
  const filterSource = process.env['BENCH_FILTER']
  const filter = filterSource === undefined ? null : new RegExp(filterSource)
  // Space-joined, without the file, which is the string vitest matched `-t`
  // against — `taskName` in `compare.ts` predicts matches with the same rule,
  // and the two must not disagree about which cases a pattern selects.
  const chosen = registry.filter(
    (entry) =>
      filter === null ||
      filter.test(`${entry.group.replaceAll(' > ', ' ')} ${entry.name}`),
  )
  // `compare.ts` alternates this between repeats. Cases in one process are
  // never independent — each inherits the optimisation state, inline caches
  // and heap the ones before it produced, and a machine drifting across the
  // pass biases early and late cases in opposite directions. Reversing every
  // other repeat makes both effects push each case both ways instead of the
  // same way twice.
  if (process.env['BENCH_REVERSE'] === '1') chosen.reverse()

  const duplicates = new Set<string>()
  const seen = new Set<string>()
  for (const entry of chosen) {
    const name = `${entry.file} > ${entry.group} > ${entry.name}`
    if (seen.has(name)) duplicates.add(name)
    seen.add(name)
  }
  if (duplicates.size > 0) {
    throw new Error(`duplicate benchmark name: ${[...duplicates].join(', ')}`)
  }

  const verbose = process.env['BENCH_PROGRESS'] === '1'
  const results: CaseResult[] = []
  for (const entry of chosen) {
    if (verbose) {
      process.stderr.write(`  … ${entry.group} > ${entry.name}\n`)
    }
    results.push(runCase(entry))
  }

  const output = process.env['BENCH_OUTPUT']
  if (output !== undefined) {
    writeFileSync(output, JSON.stringify(reportOf(results)))
  }
  if (output === undefined || verbose) printTable(results)
}
