/**
 * Running the suite and turning what came back into stored cases.
 *
 * The measurement half: which files a pass covers, spawning the runner children
 * around the controls, and normalising each repeat against its own machine
 * factor. What the resulting numbers *mean* against history is `report.ts`.
 */

import { spawnSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

import { CONTROL_FILE } from '../harness/discovery.ts'
import type { CaseRecord } from './baseline.ts'
import { environment, fileOf, fingerprint, isControl } from './baseline.ts'
import {
  geometricMedian,
  machineRatio,
  median,
  MIN_CONTROLS,
  relativeSpread,
  sameControls,
} from './statistics.ts'
import { dim, out } from './terminal.ts'

const REGRESSION_DIR = dirname(fileURLToPath(import.meta.url))
const BENCH_DIR = dirname(REGRESSION_DIR)
const PROJECT_DIR = dirname(BENCH_DIR)

/** The measurement child: bundles with esbuild, measures in bare node. */
const RUNNER = join(BENCH_DIR, 'harness', 'runner.ts')

/**
 * What a run needs from the parsed command line. Narrower than the CLI's own
 * options object: the command line belongs to `compare.ts`.
 */
export interface MeasurementOptions {
  quick: boolean
  confirm: boolean
  name: string | null
  verbose: boolean
}

/** One case, as the runner reported it. */
export interface Timing {
  /** Milliseconds per call. */
  median: number
  /** Milliseconds per call. */
  min: number
  samples: number
  /** Calls per timed sample. */
  batch: number
  /** Milliseconds spent in the timed phase. */
  measuredTime: number
  /** Ended on agreeing blocks, rather than on the window. */
  stoppedStable: boolean
  /** The block spread it ended at. */
  stability: number | null
}

/** The runner's `--outputJson` leaf. */
export interface Benchmark {
  name: string
  median: number
  min: number
  sampleCount: number
  batch: number
  measuredTime: number
  stoppedStable: boolean
  stability: number | null
}

export interface RunnerReport {
  files: { groups: { fullName: string; benchmarks: Benchmark[] }[] }[]
}

/** One repeat: controls, suite, controls. */
export interface Pass {
  report: RunnerReport
  /** Per control, sqrt(pre × post) in milliseconds. */
  controls: Record<string, number>
  /** post / pre, i.e. drift across the pass. */
  slope: number
}

/** What the adaptive stop did, over a run. */
export interface MeasurementStats {
  /** Case-repeats measured. */
  cases: number
  /** Fraction that stopped on agreeing blocks. */
  stable: number
  /** Median block spread at the stop. */
  spread: number | null
  /** The 95th percentile of that spread. */
  worst: number | null
  /** Seconds spent inside timed phases. */
  timed: number
}

/** What one runner child needs. */
export interface RunOptions {
  quick: boolean
  confirm: boolean
  reverse: boolean
  name: string | null
  verbose: boolean
  bundleDir: string
}

/**
 * Build every bundle the run needs, once, before anything is timed.
 *
 * esbuild is CPU work, and CPU work inside a pass heats the machine the
 * controls exist to describe. Doing it here also means repeat 1 and repeat 2
 * execute literally the same bytes rather than two builds that merely should
 * be identical.
 */
export function prebundle(bundleDir: string, files: readonly string[]): void {
  const result = spawnSync(
    process.execPath,
    [RUNNER, '--prepare', `--bundleDir=${bundleDir}`, CONTROL_FILE, ...files],
    { cwd: PROJECT_DIR, encoding: 'utf8' },
  )
  if (result.error !== undefined) throw result.error
  if (result.status !== 0) {
    process.stderr.write(result.stdout ?? '')
    process.stderr.write(result.stderr ?? '')
    throw new Error(
      result.status === null
        ? `bundling was killed by signal ${result.signal}`
        : `bundling exited with status ${result.status}`,
    )
  }
}

/**
 * One full pass over the suite, returning the parsed `--outputJson` report.
 *
 * The measurement options come in as a record rather than as positional flags
 * because the controls run through this same function, and the one thing that
 * must differ between them and the suite is the name filter. A `null` in
 * argument four would be easy to add by accident and impossible to read.
 */
export function runSuite(
  outputPath: string,
  files: readonly string[],
  { quick, confirm, reverse, name, verbose, bundleDir }: RunOptions,
): RunnerReport {
  const result = spawnSync(
    process.execPath,
    [
      RUNNER,
      `--outputJson=${outputPath}`,
      // Modes travel as flags rather than environment variables so the runner
      // clears anything a parent shell left set — a full run is the only kind
      // `--record` will store, and an env var nobody passed is exactly how a
      // quick one would masquerade as it.
      ...(quick ? ['--quick'] : []),
      ...(confirm ? ['--confirm'] : []),
      ...(reverse ? ['--reverse'] : []),
      // This run prepared that directory moments ago, in `prebundle`, and owns
      // it until the scratch tree is deleted — the one situation in which
      // trusting a bundle that is already there is a claim anyone can make.
      `--bundleDir=${bundleDir}`,
      '--reusePrepared',
      ...(verbose ? ['--progress'] : []),
      // One token, so a pattern with a space in it survives on every platform.
      ...(name === null ? [] : [`--testNamePattern=${name}`]),
      ...files,
    ],
    {
      cwd: PROJECT_DIR,
      encoding: 'utf8',
      // Piped, the child says nothing until it exits — which for a full pass is
      // a minute of a terminal that looks hung. The data still comes from
      // the JSON either way; `--verbose` only decides whether the runner's own
      // progress reaches the terminal while it works. Inheriting also means
      // there is nothing left to re-print on failure, because it was printed.
      stdio: verbose ? 'inherit' : 'pipe',
      // A growing suite can overrun the default buffer, and spawnSync reports
      // that as a failure with no explanation.
      maxBuffer: 64 * 1024 * 1024,
    },
  )

  // A spawn that failed outright never produced a status; surfacing the OS
  // error beats reporting "exited with status null".
  if (result.error !== undefined) throw result.error
  if (result.status !== 0) {
    // Under `--verbose` these are null: the child wrote straight to the
    // terminal, so the failure is already above this line.
    process.stderr.write(result.stdout ?? '')
    process.stderr.write(result.stderr ?? '')
    throw new Error(
      result.status === null
        ? `bench runner was killed by signal ${result.signal}`
        : `bench runner exited with status ${result.status}`,
    )
  }

  const report: RunnerReport = JSON.parse(readFileSync(outputPath, 'utf8'))
  return report
}

/**
 * One repeat: time the controls, measure the suite, time the controls again.
 *
 * Per control, the yardstick entry is the geometric mean of the two, so it
 * describes the middle of the window the cases were measured in rather than
 * one end of it. `slope` is what the two ends say about each other — a machine
 * that warms up over a multi-minute pass shows here and nowhere else, because
 * it does the same thing every repeat and so leaves the factors themselves in
 * perfect agreement.
 */
export function runPass(
  scratch: string,
  bundleDir: string,
  index: number,
  files: readonly string[],
  options: MeasurementOptions,
): Pass {
  // Every second repeat runs its files, and the cases inside them, in reverse.
  // Cases in one process are never independent — each inherits the V8 state
  // the ones before it produced — and a machine drifting across a pass biases
  // early and late cases in opposite directions. Alternating the order makes
  // both effects push each case both ways instead of the same way twice, so
  // they largely cancel in the median across repeats.
  const reverse = index % 2 === 0

  // Unfiltered on purpose, and not an oversight to be tidied up later. `-t`
  // chooses what is being judged; the controls are the yardstick it is judged
  // against, and there is no pattern for which "the machine was this fast"
  // means something different. A pattern that reached them would filter them to
  // nothing and leave `controlVector` with no yardstick to build.
  const control = (label: string): Map<string, number> =>
    controlVector(
      collect(
        runSuite(join(scratch, `${label}-${index}.json`), [CONTROL_FILE], {
          quick: options.quick,
          confirm: options.confirm,
          reverse,
          name: null,
          verbose: options.verbose,
          bundleDir,
        }),
      ),
    )

  // `spawnSync` blocks, so nothing can be reported from inside a child. Naming
  // each phase before starting it is the whole of what this can do: a pass is
  // three children and the middle one is minutes long, and without this the
  // difference between "measuring" and "hung" is invisible.
  const phase = (label: string): void =>
    out(options.verbose ? `\n  ${label} …\n` : dim(` ${label}`))

  phase('anchor')
  const before = control('pre')
  phase('suite')
  const report = runSuite(join(scratch, `run-${index}.json`), files, {
    quick: options.quick,
    confirm: options.confirm,
    reverse,
    name: options.name,
    verbose: options.verbose,
    bundleDir,
  })
  phase('anchor')
  const after = control('post')

  // Per control, the midpoint of its two observations describes the middle of
  // the window the suite was measured in; the slope is what the two ends say
  // about each other, taken as the middle of the per-control ratios so one
  // control's bad run cannot masquerade as machine drift.
  const controls: Record<string, number> = {}
  const slopes: number[] = []
  // Set equality, not merely "everything before it also ran after": a control
  // that appeared only in the second timing would leave the pass anchored to a
  // yardstick built from a different set of workloads than the one it reports.
  for (const name of sameControls([...before.keys()], [...after.keys()])) {
    const pre = before.get(name)
    const post = after.get(name)
    if (pre === undefined || post === undefined) {
      throw new Error(`control ${name} lost a timing between the two anchors`)
    }
    controls[name] = Math.sqrt(pre * post)
    slopes.push(post / pre)
  }
  return { report, controls, slope: geometricMedian(slopes) }
}

/**
 * Flatten a runner report to `name -> timing`.
 *
 * `group.fullName` already reads `bench/suites/distance.bench.ts > indelDistance`, so
 * it carries the file without the absolute path the report's `filepath` has.
 */
export function collect(report: RunnerReport): Map<string, Timing> {
  const timings = new Map<string, Timing>()
  for (const file of report.files) {
    for (const group of file.groups) {
      for (const benchmark of group.benchmarks) {
        const name = `${group.fullName} > ${benchmark.name}`
        // Two cases with one name would silently overwrite each other here, and
        // then compare against one baseline entry between them.
        if (timings.has(name)) throw new Error(`duplicate benchmark name: ${name}`)
        timings.set(name, {
          median: benchmark.median,
          min: benchmark.min,
          samples: benchmark.sampleCount,
          batch: benchmark.batch,
          measuredTime: benchmark.measuredTime,
          stoppedStable: benchmark.stoppedStable,
          stability: benchmark.stability,
        })
      }
    }
  }
  return timings
}

/**
 * The machine's yardstick: each control's median, in milliseconds, by name.
 *
 * Named rather than collapsed to one number, because the controls are four
 * *different* workloads. A median of their absolute times is really "whichever
 * control happens to sit in the middle of the ordering", and one control
 * having a bad run can change which one that is — the anchor then jumps by
 * the gap between adjacent controls, which has nothing to do with how noisy
 * any of them was. Meaningful robustness comes later, from taking the middle
 * of the per-control *ratios* between two of these vectors: dimensionless,
 * same workload over same workload, where one bad control is one outvoted
 * ratio.
 */
export function controlVector(timings: Map<string, Timing>): Map<string, number> {
  const controls = new Map<string, number>(
    [...timings]
      .filter(([name]) => isControl(name))
      .map(([name, timing]) => [name, timing.median]),
  )
  if (controls.size < MIN_CONTROLS) {
    throw new Error(
      `only ${controls.size} control(s) ran, fewer than the ${MIN_CONTROLS} a ` +
        `yardstick needs. Nothing would anchor these numbers to the machine.`,
    )
  }
  return controls
}

/**
 * What the adaptive stop did across every case of every repeat.
 *
 * The stop rule cannot be tuned by reasoning about it — whether `minTime` is
 * generous or the window is doing all the work is a property of these cases on
 * this machine. This is the evidence: how often a case ended because its block
 * medians agreed, how closely they agreed, and how much of the wall clock went
 * into timed phases at all.
 */
export function measurementStats(
  collected: readonly Map<string, Timing>[],
): MeasurementStats {
  const timings = collected.flatMap((repeat) => [...repeat.values()])
  const spreads = timings
    .map((timing) => timing.stability)
    .filter((value) => value !== null)
    .sort((a, b) => a - b)
  return {
    cases: timings.length,
    stable:
      timings.length === 0
        ? 0
        : timings.filter((timing) => timing.stoppedStable).length / timings.length,
    spread: spreads.length === 0 ? null : median(spreads),
    // Nearest-rank, so it names a spread one case actually reported rather
    // than an interpolation between two.
    worst:
      spreads.length === 0
        ? null
        : spreads[Math.min(spreads.length - 1, Math.ceil(0.95 * spreads.length) - 1)],
    timed: timings.reduce((total, timing) => total + timing.measuredTime, 0) / 1000,
  }
}

/**
 * Normalise each repeat against its own machine factor, then take the middle
 * of the repeats.
 *
 * The session's yardstick is the per-control middle across the passes; a
 * pass's factor is the middle of its per-control *ratios* against that
 * yardstick, which is dimensionless and outvotes one control having a bad
 * pass. A case's normalised value is therefore its median in milliseconds, as
 * it would have measured on the session's average machine.
 */
export function aggregate(passes: readonly Pass[]): {
  cases: Record<string, CaseRecord>
  measurement: MeasurementStats
  anchorNoise: { between: number; within: number }
} {
  const collected = passes.map((pass) => collect(pass.report))

  const observed = new Map<string, number[]>()
  for (const pass of passes) {
    for (const [name, value] of Object.entries(pass.controls)) {
      const values = observed.get(name) ?? []
      values.push(value)
      observed.set(name, values)
    }
  }
  const machine: Record<string, number> = {}
  for (const [name, values] of observed) {
    if (values.length !== passes.length) {
      throw new Error(
        `control ${name} ran in ${values.length} of ${passes.length} repeats`,
      )
    }
    machine[name] = geometricMedian(values)
  }
  const factors = passes.map((pass) => machineRatio(pass.controls, machine))

  const names = new Set(collected.flatMap((timings) => [...timings.keys()]))
  const cases: Record<string, CaseRecord> = {}
  const stamp = {
    ...environment(),
    recordedAt: new Date().toISOString(),
    controls: fingerprint(CONTROL_FILE),
  }

  for (const name of names) {
    const found = collected.map((repeat) => repeat.get(name))
    // A case missing from one repeat would get a narrower band than the rest
    // and no indication of it. Filtered rather than tested, so that what the
    // rest of this loop reads is the array proven to have no holes in it.
    const timings = found.filter((timing) => timing !== undefined)
    if (timings.length !== found.length) {
      throw new Error(
        `${name} ran in ${timings.length} of ${passes.length} ` +
          `repeats; the suite must be identical across repeats`,
      )
    }

    const raw = timings.map((timing) => timing.median)
    const normalised = raw.map((value, i) => value / factors[i])
    const centre = median(normalised)

    cases[name] = {
      normalised: centre,
      noise: relativeSpread(normalised, centre),
      // Kept in milliseconds for the sample-size envelope and for display; the
      // comparison never reads it.
      median: median(raw),
      machine,
      min: Math.min(...timings.map((timing) => timing.min)),
      samples: Math.min(...timings.map((timing) => timing.samples)),
      // Kept so a case that crosses the batching threshold between two runs
      // can be pointed out: its measurement regime changed, not just its time.
      batch: Math.min(...timings.map((timing) => timing.batch)),
      // Taken per repeat and then reduced, so the figure the envelope judges
      // is one a repeat actually measured — a median from one repeat times a
      // batch from another is a sample nothing took.
      sample: Math.min(...timings.map((timing) => timing.median * timing.batch)),
      repeats: passes.length,
      source: fingerprint(fileOf(name)),
      ...stamp,
    }
  }
  // Two different ways the machine can fail to hold still. Each repeat is
  // corrected by its own factor, so neither is an error by itself — they are
  // the evidence for whether that factor described the repeat it came from.
  //
  // `between` is drift from one repeat to the next. `within` is the machine
  // moving *during* a repeat, which the factors alone cannot show: a run that
  // starts cool and heats up does so identically every repeat, so the factors
  // agree perfectly while the cases that ran first were measured on a different
  // machine from the ones that ran last. It is visible only because the
  // controls are timed both before and after each pass.
  const between = (Math.max(...factors) - Math.min(...factors)) / median(factors)
  const within = Math.max(...passes.map((pass) => Math.abs(pass.slope - 1)))

  return {
    cases,
    measurement: measurementStats(collected),
    anchorNoise: { between, within },
  }
}
