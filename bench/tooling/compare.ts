#!/usr/bin/env node
/**
 * Run the benchmark suite several times and compare the result against
 * `bench/tooling/baseline.json`.
 *
 * ## What a baseline stores
 *
 * Not a bare time. Every repeat brackets the suite with
 * `bench/control.bench.ts` — four workloads this library cannot change — and
 * stores what each of them measured, by name. That vector is the session's
 * yardstick, and a case's stored number is its median divided by how fast the
 * machine was while it ran: *this case, in units of that machine*.
 *
 * Two sessions are compared by dividing their yardsticks control by control
 * and taking the middle ratio. Per control, and dimensionless: the alternative
 * — one number per session, the median of four absolute timings — is really
 * "whichever control happens to sit in the middle of the ordering", so one
 * control having a bad run changes which one that is and the anchor jumps by
 * the gap between two unrelated workloads. Same workload over same workload,
 * one bad control is one outvoted ratio.
 *
 * That one decision is what makes the rest safe:
 *
 * - **A run on a slower day is still comparable.** Thermal state, background
 *   load and CPU frequency shift a whole run at once; one measured run was 4%
 *   slower than baseline in every single case. The yardstick moves with it.
 * - **Repeats are corrected individually.** If repeat 2 was the slow one, it is
 *   divided by repeat 2's factor, not by an average of both.
 * - **A pass that drifts while it runs is visible.** The controls are timed
 *   before *and* after each pass. A machine that warms up over a multi-minute
 *   suite does so identically every repeat, so the factors compared only
 *   against each other agree perfectly while the cases that ran first were
 *   measured on a different machine from the ones that ran last.
 * - **A partial re-record cannot lie.** Recording one file measures a new
 *   yardstick. If the untouched cases were normalised later, against that
 *   yardstick, they would each shift by however much the machine differed that
 *   day — an unchanged case reporting as several percent faster. Each case
 *   instead carries the yardstick from its own recording session, so
 *   re-recording one file cannot move another file's numbers. Controls never
 *   become cases: they are measured in their own passes and survive only as
 *   those yardsticks.
 *
 * The suite's own geometric mean is still computed, and reported as how far the
 * whole suite moved. Estimating drift from *that* — the obvious approach —
 * cannot distinguish "the machine was 10% slower" from "every kernel got 10%
 * slower", and divides out the second along with the first. Measured against
 * the controls it stays visible, and fails `--fail-on-regression`.
 *
 * ## The rest of what this does that a stock benchmark runner does not
 *
 * - **Compares medians, not means.** One garbage collection in a thousand
 *   samples moved a case's mean by 43% in testing. Its median did not move.
 * - **Measures the noise before judging a move.** Each case is timed in every
 *   repeat; the deviation across those repeats is that case's noise band, and a
 *   move inside its own band is not reported as a change.
 * - **Keys cases by name.** Pairing a run against a baseline by position means
 *   inserting a case silently compares every later one against the wrong
 *   entry. Names survive reordering.
 * - **Refuses to compare what is not comparable.** A baseline recorded on a
 *   different Node, under different flags, on another machine, or against a
 *   different version of the benchmark definitions is reported as such rather
 *   than diffed anyway.
 *
 * ## Usage
 *
 * See {@link USAGE}, or `node bench/tooling/compare.ts --help`.
 */

import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { cpus, tmpdir } from 'node:os'
import { dirname, join, relative, resolve, sep } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const TOOLING_DIR = dirname(fileURLToPath(import.meta.url))
const BENCH_DIR = dirname(TOOLING_DIR)
const PROJECT_DIR = dirname(BENCH_DIR)
const DEFAULT_BASELINE = join(TOOLING_DIR, 'baseline.json')

/** The measurement child: bundles with esbuild, measures in bare node. */
const RUNNER = join(TOOLING_DIR, 'runner.ts')

interface Options {
  /** Write the baseline instead of comparing. */
  record: boolean
  /** Short windows; see `harness.ts`. */
  quick: boolean
  /** Widened windows; see `harness.ts`. */
  confirm: boolean
  /** Passes over the suite. */
  repeats: number
  failOnRegression: boolean
  allowEnvironmentChange: boolean
  baseline: string
  /** Canonical bench paths; empty means all. */
  files: string[]
  /** `-t` regexp source, or null for no filter. */
  name: string | null
  help: boolean
  /** Let the child's reporter reach the terminal. */
  verbose: boolean
}

/** One case, as the runner reported it. */
interface Timing {
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

/** Everything outside the code under test that changes what a number means. */
interface Environment {
  node: string
  esbuild: string
  platform: string
  cpu: string
  nodeOptions: string
  /** {@link MEASUREMENT_VERSION} */
  measurement: number
}

/**
 * One entry in `baseline.json`, which is a file on disk that outlives any one
 * run — this interface is the only description of what is in it, and a field
 * quietly added or dropped here is a baseline a later version reads
 * differently.
 *
 * `normalised` is the case's median in milliseconds, corrected to its
 * session's average machine speed. `machine` is that session's yardstick: the
 * per-control median, in milliseconds, keyed by control name.
 */
interface CaseRecord extends Environment {
  normalised: number
  noise: number
  median: number
  machine: Record<string, number>
  min: number
  samples: number
  batch: number
  /** The shortest timed sample any repeat took, in milliseconds. */
  sample: number
  repeats: number
  source: string
  controls: string
  recordedAt: string
}

interface Baseline {
  cases: Record<string, CaseRecord>
}

/** The runner's `--outputJson` leaf. */
interface Benchmark {
  name: string
  median: number
  min: number
  sampleCount: number
  batch: number
  measuredTime: number
  stoppedStable: boolean
  stability: number | null
}

interface RunnerReport {
  files: { groups: { fullName: string; benchmarks: Benchmark[] }[] }[]
}

/** One repeat: controls, suite, controls. */
interface Pass {
  report: RunnerReport
  /** Per control, sqrt(pre × post) in milliseconds. */
  controls: Record<string, number>
  /** post / pre, i.e. drift across the pass. */
  slope: number
}

/** What the adaptive stop did, over a run. */
interface MeasurementStats {
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
interface RunOptions {
  quick: boolean
  confirm: boolean
  reverse: boolean
  name: string | null
  verbose: boolean
  bundleDir: string
}

const USAGE = `
  node bench/tooling/compare.ts [options] [file …]

  Compare the benchmark suite against bench/tooling/baseline.json. A file may
  be named by any substring of its path. With no file, the whole suite runs;
  bench/control.bench.ts is the anchor and always runs either way, whatever is
  filtered.

    --record                     (re)write the baseline instead of comparing
    --quick                      short windows, ±15% threshold, 1 repeat
    --confirm                    widened windows, ±1.5% floor, 4 repeats — for
                                 re-measuring the case a normal run flagged
    --repeat=N                   passes over the suite (default 2; 1 quick)
    -t, --name=<pattern>         only cases whose name matches this regexp
    --baseline=<path>            compare against or record to another file
    --fail-on-regression         exit non-zero on a regression or a bad run
    --allow-environment-change   compare across Node/CPU/esbuild versions anyway
    --verbose                    stream the runner's per-case progress while it runs
    -h, --help                   this

  Examples

    node bench/tooling/compare.ts fuzz                 one file, by substring
    node bench/tooling/compare.ts -t 'partialRatio'    one group, any file
    node bench/tooling/compare.ts --quick -t '128 chars'  the fast look
    node bench/tooling/compare.ts --record bench/fuzz.bench.ts

  For the edit loop, \`pnpm bench:quick\` skips the comparison entirely.
`

/** The file whose cases anchor every other case. See its header. */
const CONTROL_FILE = 'bench/control.bench.ts'

/**
 * Smallest move worth reporting when the measured noise is smaller than this.
 * Nothing this suite optimises for is a 2% change, and a floor keeps a
 * suspiciously quiet run from flagging every case.
 */
const FLOOR = 0.03

/**
 * The same floor under `--quick`.
 *
 * Shorter windows make each median coarser and can measure a case before V8 has
 * finished tiering, which biases cases and controls by different amounts. A
 * threshold that pretended to the usual resolution would spend the run flagging
 * its own measurement error. At this width it still catches the thing quick
 * mode is for: something that got noticeably, obviously worse.
 */
const QUICK_FLOOR = 0.15

/** Beyond this much machine drift the run is too unlike the baseline's run to
 *  mean anything, whatever the yardstick corrects for. */
const DRIFT_LIMIT = 0.1

/**
 * How far the machine may move between repeats, or across a single pass.
 *
 * Each repeat is corrected by its own factor, so drift between repeats is
 * already handled — this is the point past which the correction stops being
 * believable. Drift across one pass is the more dangerous of the two: the
 * factor is a single number for a window the cases were spread across, and if
 * the machine changed inside that window it describes none of them well.
 *
 * Tighter than {@link DRIFT_LIMIT} because both are within one run, rather than
 * a difference between two sessions on different days.
 */
const RUN_INSTABILITY_LIMIT = 0.05

/**
 * Fewest controls a yardstick may be built from.
 *
 * The whole case for taking the middle of the per-control ratios rather than
 * their mean is that one control having a bad run cannot drag it. With two
 * there is no middle, and with one there is nothing to be robust about.
 */
const MIN_CONTROLS = 3

/**
 * What a stored number means.
 *
 * The fingerprints below cover what was measured. This covers how: the
 * yardstick, the aggregation, the flags the child runs under. Moving from raw
 * medians to per-repeat normalised medians changed every stored value's
 * meaning without changing a single benchmark, and nothing would have caught
 * it. Increment this when measurement or normalisation semantics change — not
 * for output format.
 */
const MEASUREMENT_VERSION = 3

/** How far the suite may move past the controls before that is a regression. */
const BROAD_MOVE_LIMIT = 0.03

/**
 * The same limit under `--quick`.
 *
 * Averaging over cases cancels the noise in each of them only if that noise is
 * independent, and quick mode's is not: a short warmup leaves cases measured
 * mid-tiering, which pushes a whole run the same way. So the geometric mean
 * keeps the bias instead of dividing it out, and at the full limit quick mode
 * announced a "broad improvement" of +5.4% on an unmodified tree. Matching
 * {@link QUICK_FLOOR} says the same thing the per-case bands do: this mode
 * resolves nothing finer than 15%.
 */
const QUICK_BROAD_MOVE_LIMIT = QUICK_FLOOR

/**
 * The `--confirm` floor. Confirm mode exists to re-measure the one case a
 * normal comparison flagged, with windows past the point of diminishing
 * returns and more repeats, so its verdict is allowed to be finer than the
 * everyday ±3%.
 */
const CONFIRM_FLOOR = 0.015

/**
 * Fewest cases before "the suite as a whole moved" means anything.
 *
 * The broad detector has no noise band — it exists to catch a change too even
 * to trip any single case's. Over one filtered case that is not a second
 * opinion, it is the same measurement judged by a stricter rule, and it would
 * fail a run that the case's own band correctly called noise.
 */
const MIN_BROAD_CASES = 5

/**
 * A sample should be tens of microseconds to two milliseconds: short enough
 * that a garbage collection lands in some samples rather than all of them,
 * long enough that the harness's own cost per sample is not part of the
 * answer. Every claim this script makes rests on that, and nothing in
 * `measure()` can enforce it — the body decides how much work a sample is. So
 * it is checked here, against what was actually measured, and outside this
 * looser envelope the case is called out rather than quietly trusted.
 *
 * The lower bound was 0.25 ms under tinybench, whose async per-sample
 * machinery cost single microseconds. `harness.ts` now spends two
 * `hrtime.bigint()` reads and one array push per sample — comfortably under
 * 200 ns — so a 20 µs sample keeps the harness below one percent of the
 * number.
 *
 * A *sample*, not a call: the harness batches a body under 0.1 ms into a
 * sample of several calls for this exact reason, so what is checked here is
 * `median × batch`. Checking the per-call figure would report every batched
 * case as too short, which is the arrangement that fixed it.
 */
const SAMPLE_TOO_SHORT = 0.02
const SAMPLE_TOO_LONG = 5

/**
 * Everything a case's timing depends on besides `src` and its own bench file.
 *
 * `bench/tooling/runner.ts` is in here because the bundling options are decided
 * there, and changing any of them changes what a stored number means
 * without touching a benchmark. {@link MEASUREMENT_VERSION} covers a deliberate
 * change of method; this covers forgetting to bump it.
 */
const SHARED_SOURCES = [
  'bench/tooling/corpus.ts',
  'bench/tooling/harness.ts',
  'bench/tooling/runner.ts',
]

/**
 * A path argument in the one spelling everything downstream compares against.
 *
 * Stored case names begin with the path the runner reports — `bench/fuzz.bench.ts`,
 * project-relative and slash-separated — and that string is the identity behind
 * three separate things: which baseline entries a partial `--record` replaces,
 * which ones `candidates` considers, and which file gets fingerprinted. Left
 * raw, `./bench/fuzz.bench.ts` and an absolute path each name the same file and
 * match none of them, so a re-record would quietly keep the stale entries it
 * was run to remove. Normalising here means there is only ever one namespace.
 */
function canonicalFile(arg: string): string {
  // Resolved against the project, not the process's cwd: `bench/fuzz.bench.ts`
  // has to name the same file whichever directory the command was typed in,
  // and the runner canonicalises its own arguments the same way.
  return relative(PROJECT_DIR, resolve(PROJECT_DIR, arg)).split(sep).join('/')
}

function parseArgs(argv: readonly string[]): Options {
  // `repeats` is the one field that is not final until parsing is done: its
  // default depends on `--quick`, which may appear after `--repeat=`. Null
  // means "not asked for", which `--repeat=0` is not — that is an error, and
  // a sentinel like 0 could not tell the two apart.
  const options: Omit<Options, 'repeats'> & { repeats: number | null } = {
    record: false,
    quick: false,
    confirm: false,
    repeats: null,
    failOnRegression: false,
    allowEnvironmentChange: false,
    baseline: DEFAULT_BASELINE,
    files: [],
    // A regexp over case names, handed to the runner as `-t`. Null rather than an
    // empty string: an empty pattern matches everything, which is what no
    // filter means, but the two have to stay distinguishable — the checks
    // below are about whether a filter was *asked for*.
    name: null,
    help: false,
    // Let the child's own reporter reach the terminal. Off by default because
    // a pass spawns three of them and a full run nine, and the comparison
    // below is the output that matters.
    verbose: false,
  }

  // Indexed rather than `for…of` because `-t` takes the argument after it.
  // Spelling it `-t` is not a free choice: it is the spelling every bench
  // script here has always taken, and muscle memory is part of the interface.
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--record') options.record = true
    else if (arg === '--quick') options.quick = true
    else if (arg === '--confirm') options.confirm = true
    else if (arg === '--fail-on-regression') options.failOnRegression = true
    else if (arg === '--allow-environment-change') options.allowEnvironmentChange = true
    else if (arg === '-h' || arg === '--help') options.help = true
    else if (arg === '--verbose') options.verbose = true
    else if (arg === '-t') {
      const pattern = argv[++i]
      if (pattern === undefined) throw new Error('-t needs a pattern after it')
      // The one hazard of a two-token option: `-t bench/fuzz.bench.ts` is a
      // plausible typo, and it would eat the file as a pattern and then
      // silently measure every file instead of the one asked for. A pattern
      // that names a bench file is never what was meant.
      if (pattern.endsWith('.bench.ts')) {
        throw new Error(
          `-t takes a case-name pattern, not a file. Drop the -t to measure ` +
            `${pattern}, or use --name= if that really is the pattern.`,
        )
      }
      options.name = pattern
    } else if (arg.startsWith('--name=')) options.name = arg.slice(7)
    else if (arg.startsWith('--repeat=')) options.repeats = Number(arg.slice(9))
    else if (arg.startsWith('--baseline=')) options.baseline = resolve(arg.slice(11))
    else if (arg.startsWith('-')) throw new Error(`unknown option: ${arg} (see --help)`)
    else options.files.push(canonicalFile(arg))
  }
  if (options.help) return { ...options, repeats: 1 }

  // Two, for recording and for comparing alike.
  //
  // The argument for a third was that a baseline's noise band is frozen into
  // every later comparison, so it deserves more care than the runs against it.
  // Measured, it does not pay: over the fuzz suite at three repeats, 18 of 19
  // cases reported a spread under the ±3% floor, and a band is
  // `max(current + baseline, FLOOR)`. The third repeat refines a number the
  // floor then discards in almost every case.
  //
  // What it does cost is the whole suite again. The run has to be cheap enough
  // to sit through, or it stops being run at all.
  //
  // Two, not one: a single repeat measures no spread, which is stored as a zero
  // band and called out as stale on every later comparison. Two is the fewest
  // that still measures one — and it measures it from two samples, so a case
  // whose band matters is one to re-run, or to record on its own with
  // `--repeat=`.
  //
  // Confirm mode exists to spend precision on one flagged case, so it takes
  // more repeats by default; everything else stays at two.
  const repeats = options.repeats ?? (options.quick ? 1 : options.confirm ? 4 : 2)

  if (options.quick && options.confirm) {
    throw new Error('--quick and --confirm are opposites')
  }
  // Confirm windows are wider than the ones the baseline was recorded under;
  // the adaptive stop converges on the same median, but a baseline should
  // only ever hold numbers taken one way.
  if (options.confirm && options.record) {
    throw new Error('--confirm is for re-measuring against a baseline, not recording one')
  }

  // Quick numbers come from shorter windows than the baseline's, so recording
  // them would store values that every later full run disagrees with — and
  // nothing downstream could tell, because the stored fields look identical.
  if (options.quick && options.record) {
    throw new Error('--quick measures too coarsely to record a baseline from')
  }
  // A name filter narrows to cases; a fingerprint is per *file*. So recording
  // through one always leaves the rest of that file's cases stored against the
  // file hash they had before the edit that prompted the re-record — and every
  // later run reports them as "definition changed" until the file is recorded
  // whole. There is no filtered re-record that ends in a consistent baseline,
  // so the narrowest useful unit to record is the file.
  if (options.record && options.name !== null) {
    throw new Error(
      'a fingerprint covers a whole bench file, so recording a name-filtered ' +
        "subset of one would leave that file's other cases stale. Record the " +
        'file: --record <file>',
    )
  }
  if (options.name !== null) {
    // The runner would report this from inside the measured child, seconds and
    // one stack trace later.
    try {
      new RegExp(options.name)
    } catch (error) {
      throw new Error(
        `-t is a regexp and ${JSON.stringify(options.name)} is not one: ` +
          `${error instanceof Error ? error.message : error}`,
      )
    }
  }
  if (!Number.isInteger(repeats) || repeats < 1) {
    throw new Error(`--repeat must be a positive integer, got ${options.repeats}`)
  }
  return { ...options, repeats }
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = sorted.length >> 1
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2
}

function geometricMean(ratios: readonly number[]): number {
  if (ratios.length === 0) return 1
  return Math.exp(ratios.reduce((total, r) => total + Math.log(r), 0) / ratios.length)
}

/** The middle of a set of ratios, which one bad member cannot drag. */
function geometricMedian(ratios: readonly number[]): number {
  return Math.exp(median(ratios.map(Math.log)))
}

/**
 * How far the repeats sit from their middle, as a fraction of it.
 *
 * The median absolute deviation, scaled so that it estimates the same quantity
 * a standard deviation would on normal data. The obvious alternative — the
 * furthest repeat from the median — is not an estimator of anything: drawing
 * more samples can only find more extreme ones, so it grows with the repeat
 * count. `--repeat=9` would then widen every band and make the gate *less*
 * sensitive, which is the opposite of what asking for more repeats means. This
 * converges on the real spread instead.
 */
const MAD_TO_SIGMA = 1.4826
function relativeSpread(values: readonly number[], centre: number): number {
  if (centre <= 0) return 0
  const deviations = values.map((value) => Math.abs(value - centre))
  return (MAD_TO_SIGMA * median(deviations)) / centre
}

/**
 * Identify the definition a case was measured under.
 *
 * A case is named, not numbered, which survives reordering — but a name says
 * nothing about the fixture behind it. Widening `128 chars` to 256 without
 * renaming it would otherwise report as a 2x regression. The fingerprint covers
 * the bench file and everything it shares, so any such edit is visible as an
 * edit rather than as a result.
 */
const fingerprints = new Map<string, string>()
function fingerprint(benchFile: string): string {
  const cached = fingerprints.get(benchFile)
  if (cached !== undefined) return cached

  const hash = createHash('sha256')
  for (const name of [benchFile, ...SHARED_SOURCES]) {
    const path = join(PROJECT_DIR, name)
    // Treating a missing file as empty content would fingerprint every case
    // by the shared helpers alone, and then editing a fixture would no longer
    // invalidate anything. If the path a case's name implies is not there,
    // the assumption behind these names has broken and the fingerprint is
    // worth nothing.
    if (!existsSync(path)) {
      throw new Error(`cannot fingerprint benchmark source: ${path}`)
    }
    hash.update(readFileSync(path))
  }
  const digest = hash.digest('hex').slice(0, 12)
  fingerprints.set(benchFile, digest)
  return digest
}

/**
 * The version actually installed, not the range asked for.
 *
 * A declaration can say `^0.28.0` while two checkouts run 0.28.1 and 0.28.4,
 * and the measurement layer is exactly where that difference shows up: esbuild
 * decides the shape of the bundle every case executes as.
 */
function installedVersion(name: string): string {
  const path = join(PROJECT_DIR, 'node_modules', name, 'package.json')
  if (!existsSync(path)) return 'unknown'
  const manifest: { version: string } = JSON.parse(readFileSync(path, 'utf8'))
  return manifest.version
}

/**
 * `NODE_OPTIONS` is inherited from whoever invoked this, and a `--jitless` or a
 * heap flag in there would produce a baseline that looks like any other. It is
 * recorded rather than stripped, so a run under deliberate flags is still
 * possible and still labelled.
 */
function environment(): Environment {
  return {
    node: process.version,
    esbuild: installedVersion('esbuild'),
    platform: `${process.platform}-${process.arch}`,
    cpu: cpus()[0]?.model ?? 'unknown',
    nodeOptions: (process.env['NODE_OPTIONS'] ?? '').trim(),
    measurement: MEASUREMENT_VERSION,
  }
}

/**
 * The fields `--allow-environment-change` may waive.
 *
 * {@link MEASUREMENT_VERSION} is deliberately not among them, though
 * {@link environment} still stamps it onto every case. The others describe the
 * machine a number was taken on, and comparing across them is a judgement call
 * a person is entitled to make — a newer patch of Node, the same laptop after
 * an OS update. A measurement version says the stored numbers *mean* something
 * else, and no amount of willingness makes two different quantities
 * comparable. So it is checked separately, and refused unconditionally.
 */
const ENVIRONMENT_KEYS: (keyof Environment)[] = [
  'node',
  'esbuild',
  'platform',
  'cpu',
  'nodeOptions',
]
const describeEnvironment = (record: Partial<Environment>): string =>
  ENVIRONMENT_KEYS.map((key) => `${key}=${record[key] ?? '?'}`).join(' ')

/**
 * The bench files a pass should measure — everything asked for, never the
 * controls.
 *
 * Resolved to real paths, rather than passed through. A positional argument is
 * a *substring* to the runner, so `compare.ts fuzz` runs `bench/fuzz.bench.ts`
 * perfectly well — and then every later use of that argument is comparing the
 * string "fuzz" against a stored case named `bench/fuzz.bench.ts > …`, which
 * matches nothing. Recording is where that bites: the entries the re-record
 * exists to replace would be kept, because they did not look like they
 * belonged to the file being recorded. Matching the filter against the
 * directory here means one namespace downstream, the same one the stored names
 * are in.
 *
 * The controls are timed on their own either side of the pass rather than
 * inside it, so they are never in this list.
 */
function suiteFiles(filters: readonly string[]): string[] {
  const all = readdirSync(BENCH_DIR)
    .filter((entry) => entry.endsWith('.bench.ts'))
    // Interpolated, not `join`ed. What is being built here is an identity — the
    // same string the runner puts in a case's name and `baseline.json` stores — and
    // identities do not have a platform. `join` would spell it
    // `bench\fuzz.bench.ts` on Windows, where it would match neither
    // {@link CONTROL_FILE} on the next line, so the anchor would be measured as
    // a subject, nor anything {@link canonicalFile} produced, so a filter naming
    // a real file would be rejected as matching none.
    .map((entry) => `bench/${entry}`)
    .filter((file) => file !== CONTROL_FILE)

  if (all.length === 0) {
    throw new Error(`nothing to measure: ${CONTROL_FILE} is the anchor, not a subject`)
  }
  if (filters.length === 0) return all

  const chosen = all.filter((file) => filters.some((filter) => file.includes(filter)))
  // Reported here rather than as an empty run: a filter matching no file would
  // otherwise surface minutes later as a report with no cases in it.
  if (chosen.length === 0) {
    throw new Error(
      `no benchmark file matched ${filters.join(', ')}. Known files: ${all.join(', ')}`,
    )
  }
  return chosen
}

/**
 * Build every bundle the run needs, once, before anything is timed.
 *
 * esbuild is CPU work, and CPU work inside a pass heats the machine the
 * controls exist to describe. Doing it here also means repeat 1 and repeat 2
 * execute literally the same bytes rather than two builds that merely should
 * be identical.
 */
function prebundle(bundleDir: string, files: readonly string[]): void {
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
function runSuite(
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
function runPass(
  scratch: string,
  bundleDir: string,
  index: number,
  files: readonly string[],
  options: Options,
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
 * `group.fullName` already reads `bench/distance.bench.ts > indelDistance`, so
 * it carries the file without the absolute path the report's `filepath` has.
 */
function collect(report: RunnerReport): Map<string, Timing> {
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

const isControl = (name: string): boolean => name.startsWith(`${CONTROL_FILE} > `)

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
function controlVector(timings: Map<string, Timing>): Map<string, number> {
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
 * The middle of the per-control ratios between two yardsticks: how much
 * slower (>1) or faster (<1) the machine described by `numerator` was than
 * the one described by `denominator`.
 */
function machineRatio(
  numerator: Record<string, number>,
  denominator: Record<string, number>,
): number {
  // The same controls on both sides, or none of it means anything. An
  // intersection would still produce a number — a plausible one, from three
  // controls where four were expected — and nothing downstream could tell it
  // from a yardstick. The fingerprint catches a changed *definition*; this
  // catches the runtime data disagreeing with it.
  const names = sameControls(Object.keys(numerator), Object.keys(denominator))
  return geometricMedian(names.map((name) => numerator[name] / denominator[name]))
}

/**
 * The control names two yardsticks agree on, or a refusal to compare them.
 */
function sameControls(left: readonly string[], right: readonly string[]): string[] {
  const ours = [...left].sort()
  const theirs = [...right].sort()
  if (
    ours.length !== theirs.length ||
    ours.some((name, index) => name !== theirs[index])
  ) {
    throw new Error(
      `control yardsticks name different controls: ${ours.join(', ')} ` +
        `vs ${theirs.join(', ')}`,
    )
  }
  return ours
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
function measurementStats(collected: readonly Map<string, Timing>[]): MeasurementStats {
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
function aggregate(passes: readonly Pass[]): {
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

function percent(value: number): string {
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`
}

const COLOUR = process.stdout.isTTY && !process.env['NO_COLOR']
const paint = (code: number, text: string): string =>
  COLOUR ? `\u001B[${code}m${text}\u001B[0m` : text
const dim = (text: string): string => paint(2, text)
const green = (text: string): string => paint(32, text)
const red = (text: string): string => paint(31, text)
const yellow = (text: string): string => paint(33, text)
const out = (text: string): void => {
  process.stdout.write(text)
}

/**
 * The baseline entries this invocation could possibly compare against.
 *
 * Selected by the same two filters the run itself applies, so this picks
 * exactly the cases a filtered run will produce — which is what lets the
 * environment be checked before spending minutes measuring, without rejecting
 * a run over some unrelated file recorded on another Node.
 *
 * `files` is the resolved list from {@link suiteFiles}, not the raw arguments.
 * It is only consulted when a filter was actually given: with none, every
 * stored case is a candidate, including one whose bench file has been deleted
 * outright. That case cannot run, and saying so is the point — a benchmark
 * that quietly stopped being measured is the failure this is here to catch.
 *
 * Controls are excluded deliberately. A run needs them measured, but their
 * *baseline* entries belong to whichever session re-recorded them last, which
 * need not be the session that recorded the cases under test. Judging a
 * comparison by them would let a partial re-record on another Node reject a
 * pairing that is perfectly valid — each case already carries the environment
 * and the yardstick it was measured under.
 */
function candidates(
  baseline: Baseline,
  options: Options,
  files: readonly string[],
): string[] {
  let names = Object.keys(baseline.cases).filter((name) => !isControl(name))
  if (options.files.length > 0) {
    const selected = new Set(files)
    names = names.filter((name) => selected.has(fileOf(name)))
  }
  if (options.name !== null) {
    // No flags, because the runner builds its own the same way.
    const pattern = new RegExp(options.name)
    names = names.filter((name) => pattern.test(taskName(name)))
  }
  return names
}

/**
 * The string the `-t` filter matches against.
 *
 * Not the name this script stores. The filter sees a case's group path and its
 * name with single spaces and never sees the file, so `bench/fuzz.bench.ts >
 * ratio > 8 chars` is `ratio 8 chars` to a pattern. Predicting the filter with
 * the stored name instead would quietly disagree with the run — `-t '^ratio'`
 * matches every case here and none there.
 */
function taskName(name: string): string {
  return name.slice(name.indexOf(' > ') + 3).replaceAll(' > ', ' ')
}

/** The bench file a stored case came from — the first segment of its name. */
function fileOf(name: string): string {
  return name.slice(0, name.indexOf(' > '))
}

/**
 * Refuse to compare a baseline recorded somewhere else.
 *
 * A V8 release moves individual microbenchmarks by different amounts, which no
 * yardstick can repair, and a different CPU is not a comparison at all.
 */
function checkEnvironment(
  baseline: Baseline,
  names: readonly string[],
  allow: boolean,
  quiet = false,
): void {
  // Checked first, and not waivable. A stored number under another measurement
  // version is a different quantity — a raw median where this expects one in
  // units of the machine, say — and diffing the two produces a percentage that
  // looks exactly like every other percentage here. `--allow-environment-change`
  // is for a machine that differs, not for arithmetic that does not apply.
  const versions = new Set<number | string>()
  for (const name of names) {
    const record = baseline.cases[name]
    if (record !== undefined && record.measurement !== MEASUREMENT_VERSION) {
      versions.add(record.measurement ?? 'unstamped')
    }
  }
  if (versions.size > 0) {
    throw new Error(
      `baseline measurement version ${[...versions].join(', ')} cannot be ` +
        `compared with ${MEASUREMENT_VERSION}: the stored numbers mean ` +
        `something different. Re-record with --record.`,
    )
  }

  const now = describeEnvironment(environment())
  const mismatched = new Set<string>()
  for (const name of names) {
    const record = baseline.cases[name]
    if (record !== undefined && describeEnvironment(record) !== now) {
      mismatched.add(describeEnvironment(record))
    }
  }
  if (mismatched.size === 0) return

  const detail =
    `${mismatched.size} baseline environment(s) differ from this run.\n` +
    `  this run: ${now}\n` +
    [...mismatched].map((value) => `  baseline: ${value}`).join('\n')
  if (!allow) {
    throw new Error(`${detail}\n  pass --allow-environment-change to compare anyway`)
  }
  // The preflight and the report both call this; only one of them should say so.
  if (!quiet) out(`  ${yellow('!')} ${detail}\n`)
}

/**
 * Name the cases whose samples are the wrong size to reason about.
 *
 * A case far above the envelope gives a disturbance — a collection, a
 * preemption — a long enough window to land in most samples, leaving the median
 * little that is clean to pick. One far below it spends a visible share of its
 * time in the harness rather than the work. Either way the noise band that case
 * reports describes something other than what it claims to, which is worth
 * knowing before believing a ratio next to it.
 *
 * The figure judged is the timed sample, not the per-call median, because
 * batching a fast body up to a workable sample is exactly the thing that puts
 * it inside the envelope.
 */
function reportSampleSizes(current: Record<string, CaseRecord>): void {
  const long: [string, number][] = []
  const short: [string, number][] = []
  for (const [name, record] of Object.entries(current)) {
    if (record.sample > SAMPLE_TOO_LONG) long.push([name, record.sample])
    else if (record.sample < SAMPLE_TOO_SHORT) short.push([name, record.sample])
  }

  const outsized: [string, [string, number][]][] = [
    [
      `over ${SAMPLE_TOO_LONG}ms — long enough that a scheduler or collector` +
        ` pause is hard to isolate from the work`,
      long,
    ],
    [
      `under ${SAMPLE_TOO_SHORT}ms — short enough that the harness's own` +
        ` per-sample overhead is a visible share of the number`,
      short,
    ],
  ]

  for (const [label, entries] of outsized) {
    if (entries.length === 0) continue
    out(`\n  ${yellow('!')} ${entries.length} sample(s) ${label}:\n`)
    for (const [name, value] of entries) {
      out(`    ${dim(`${value.toFixed(4)}ms  ${name}`)}\n`)
    }
  }
}

/**
 * How the adaptive stop behaved over the whole run.
 *
 * Printed on every run, recording included, because it is the only evidence
 * for whether the windows in `harness.ts` are the right ones: a suite that
 * mostly stops on stability is one whose `minTime` could come down, and one
 * that mostly runs out of window is one where the stop rule is buying nothing.
 */
function reportMeasurement(stats: MeasurementStats): void {
  const spread =
    stats.spread === null || stats.worst === null
      ? 'no case completed its blocks'
      : `block spread ±${(stats.spread * 100).toFixed(2)}% median, ` +
        `±${(stats.worst * 100).toFixed(2)}% p95`
  out(
    dim(
      `  measurement: ${(stats.stable * 100).toFixed(0)}% of ${stats.cases} case-runs ` +
        `stopped on stability, ${spread}, ${stats.timed.toFixed(0)}s timed\n`,
    ),
  )
}

interface NoteRow {
  name: string
  note: string
}
interface MeasuredRow {
  name: string
  ratio: number
  moved: boolean
  noise: number
  median: number
}

/** @returns the process exit code */
function report(
  current: Record<string, CaseRecord>,
  baseline: Baseline,
  anchorNoise: { between: number; within: number },
  options: Options,
  files: readonly string[],
): number {
  const shared = Object.keys(current).filter((name) => baseline.cases[name] !== undefined)

  // A case whose fixture — or whose controls, which every stored number is in
  // units of — changed since the baseline cannot be compared against it.
  const changed = new Set(
    shared.filter(
      (name) =>
        baseline.cases[name].source !== current[name].source ||
        baseline.cases[name].controls !== current[name].controls,
    ),
  )

  // Both sides are milliseconds in units of their own session's machine, so
  // one of them has to be converted before they can be divided. `drift` is how
  // much slower this machine is than the one the baseline was recorded on, so
  // dividing today's normalised value by it puts it in the baseline's units;
  // the ratio is then baseline over current, above 1 for faster now.
  const ratioOf = (name: string): number => {
    const before = baseline.cases[name]
    const drift = machineRatio(current[name].machine, before.machine)
    return (before.normalised * drift) / current[name].normalised
  }
  const measured = shared.filter((name) => !isControl(name) && !changed.has(name))
  const broadMove =
    measured.length < MIN_BROAD_CASES ? null : geometricMean(measured.map(ratioOf))

  // Only the cases actually being compared, and not the controls: a control's
  // baseline entry belongs to whichever session recorded it last.
  checkEnvironment(baseline, measured, options.allowEnvironmentChange)

  // How much slower or faster the machine is than the one each case was
  // recorded on, from that case's *own* stored yardstick — the controls
  // measured in the session that produced its baseline number, which is the
  // only machine a comparison against it is really between.
  //
  // Grouped by that session rather than pooled, because a baseline may hold
  // several. One session recorded when the machine was 20% off would be a fifth
  // of the cases needing a correction past anything worth trusting, and a
  // median across all of them would report the other four fifths and say the
  // run was fine.
  const sessions = new Map<string, { drift: number; cases: number }>()
  for (const name of measured) {
    const before = baseline.cases[name]
    const session = sessions.get(before.recordedAt) ?? {
      drift: machineRatio(current[name].machine, before.machine),
      cases: 0,
    }
    session.cases++
    sessions.set(before.recordedAt, session)
  }

  const floor = options.quick ? QUICK_FLOOR : options.confirm ? CONFIRM_FLOOR : FLOOR
  const rows: (NoteRow | MeasuredRow)[] = []
  let regressions = 0
  let improvements = 0

  for (const name of Object.keys(current)) {
    const now = current[name]
    const before = baseline.cases[name]
    if (before === undefined) {
      rows.push({ name, note: 'no baseline entry' })
      continue
    }
    if (changed.has(name)) {
      rows.push({ name, note: 'definition changed' })
      continue
    }

    const control = isControl(name)
    // Controls are anchored to themselves, so their normalised ratio is ~1 by
    // construction and says nothing. Show what they actually did instead.
    const ratio = control ? before.median / now.median : ratioOf(name)
    const band = Math.max(now.noise + before.noise, floor)
    const moved = !control && Math.abs(Math.log(ratio)) > Math.log(1 + band)
    if (moved && ratio < 1) regressions++
    if (moved && ratio > 1) improvements++
    rows.push({ name, ratio, moved, noise: now.noise, median: now.median })
  }

  // A case's full name is `<file> > <group> > <case>`; only the last part varies
  // within a group, so the first two become a heading and the rows line up.
  const split = (name: string): [string, string] => {
    const at = name.lastIndexOf(' > ')
    return [name.slice(0, at), name.slice(at + 3)]
  }
  const width = Math.max(...rows.map((row) => split(row.name)[1].length))
  let heading: string | null = null

  out(
    `\n  ${'case'.padEnd(width)}  ${'median'.padStart(10)}  ${'vs base'.padStart(8)}  noise\n`,
  )

  for (const row of rows) {
    const [group, name] = split(row.name)
    if (group !== heading) {
      heading = group
      out(`\n  ${group}\n`)
    }

    const label = name.padEnd(width)
    if ('note' in row) {
      out(`  ${label}  ${dim(`${'—'.padStart(10)}  ${row.note}`)}\n`)
      continue
    }
    const timing = `${row.median.toFixed(4)}ms`.padStart(10)
    const ratio = `${row.ratio.toFixed(2)}x`.padStart(8)
    const noise = `±${(row.noise * 100).toFixed(1)}%`.padStart(6)
    const line = `  ${label}  ${timing}  ${ratio}  ${noise}`
    if (!row.moved) out(`${dim(line)}\n`)
    else out(`${row.ratio > 1 ? green(line) : red(line)}  !\n`)
  }

  // Only what this invocation asked to measure. Listing every baseline case a
  // filtered run left out reports the filter back as if it were a finding —
  // one file is already 122 lines of it. What is worth saying is that a case
  // this run *should* have produced did not: a rename, or a deletion.
  const missing = candidates(baseline, options, files).filter(
    (name) => current[name] === undefined,
  )
  if (missing.length > 0) {
    out(`\n  ${red('!')} ${missing.length} baseline case(s) not in this run:\n`)
    for (const name of missing) out(`    ${dim(name)}\n`)
    out(
      dim(`    a case the filter should have selected did not run — it was\n`) +
        dim(`    renamed or deleted. Re-record its file to settle the baseline.\n`),
    )
  }

  // A case that crossed the 0.1 ms batching threshold since the baseline was
  // recorded is being measured a different way than the number it is compared
  // against: one call per timed sample rather than several, or the reverse.
  // That is not a reason to distrust the ratio, but it is the first thing to
  // know about a suspicious one on a tiny case.
  //
  // Crossed, not merely moved. The batch is calibrated per run from a probe,
  // so a case sitting near a boundary reports x8 one day and x9 the next
  // without anything having happened. What is worth a line is a body that
  // stopped needing a batch, started needing one, or halved or doubled.
  const rebatched = measured.filter((name) => {
    const then = baseline.cases[name].batch
    const now = current[name].batch
    return (then === 1) !== (now === 1) || Math.max(then, now) >= 2 * Math.min(then, now)
  })
  if (rebatched.length > 0) {
    out(`\n  ${yellow('!')} ${rebatched.length} case(s) changed batching regime:\n`)
    for (const name of rebatched) {
      out(
        `    ${dim(`x${baseline.cases[name].batch} → x${current[name].batch}  ${name}`)}\n`,
      )
    }
  }

  // Quick mode measures nothing at the size the envelope describes, so the
  // sizes it reports would all be its own.
  if (!options.quick) reportSampleSizes(current)

  out('\n')
  let inconclusive = false

  if (anchorNoise.between > RUN_INSTABILITY_LIMIT) {
    out(
      `  ${red('!')} the machine moved ${percent(anchorNoise.between)} between repeats — it did not hold still\n`,
    )
    inconclusive = true
  }
  if (anchorNoise.within > RUN_INSTABILITY_LIMIT) {
    out(
      `  ${red('!')} the machine moved ${percent(anchorNoise.within)} between the start and end of a pass —\n` +
        `    cases measured early and late in it were not measured on the same machine\n`,
    )
    inconclusive = true
  }

  if (sessions.size === 0) {
    out(dim('  machine vs baseline: unknown — no case was comparable\n'))
  } else {
    const drifts = [...sessions.values()].map((session) => session.drift)
    const range =
      sessions.size === 1
        ? percent(drifts[0] - 1)
        : `${percent(Math.min(...drifts) - 1)} … ${percent(Math.max(...drifts) - 1)}`
    out(`  machine vs baseline, over ${sessions.size} recording session(s): ${range}`)
    out(dim(' (positive is slower now, and already divided out)\n'))

    for (const [recordedAt, session] of sessions) {
      if (Math.abs(Math.log(session.drift)) <= Math.log(1 + DRIFT_LIMIT)) continue
      out(
        `  ${red('!')} the ${session.cases} case(s) recorded ${recordedAt} needed a ` +
          `${percent(session.drift - 1)} correction — past ${percent(DRIFT_LIMIT)}, too far to trust\n`,
      )
      inconclusive = true
    }
  }

  // The number a whole-suite drift estimator would have absorbed: a real change
  // that moved everything at once, which no individual band would catch.
  const broadLimit = options.quick ? QUICK_BROAD_MOVE_LIMIT : BROAD_MOVE_LIMIT
  const broadRegression = broadMove !== null && broadMove < 1 / (1 + broadLimit)
  if (broadMove === null) {
    out(
      dim(
        `  suite move: n/a — ${measured.length} comparable case(s), fewer than the ${MIN_BROAD_CASES} this needs\n`,
      ),
    )
  } else if (Math.abs(Math.log(broadMove)) > Math.log(1 + broadLimit)) {
    const label = broadRegression ? red('broad regression') : yellow('broad improvement')
    out(`  the suite as a whole moved ${percent(broadMove - 1)} — ${label}\n`)
  } else {
    out(dim(`  the suite as a whole moved ${percent(broadMove - 1)}\n`))
  }

  const stale = shared.filter((name) => baseline.cases[name].repeats < 2).length
  if (stale > 0) {
    out(
      `  ${yellow('!')} ${stale} baseline case(s) were recorded with one repeat; their noise bands are zero\n`,
    )
  }
  if (changed.size > 0) {
    out(
      `  ${yellow('!')} ${changed.size} case(s) have a changed definition and were not compared\n`,
    )
  }

  out(
    `  ${improvements} improved, ${regressions} regressed beyond their noise band; ` +
      `${measured.length - improvements - regressions} unchanged\n`,
  )

  // A band is built from repeats inside one invocation, which share a machine
  // state that two invocations minutes apart do not. It therefore describes
  // less variation than actually separates a run from a baseline: recording
  // this suite and immediately comparing against it flagged eleven cases at
  // 0.94-0.97x, and the identical comparison run again flagged none of them.
  // So a flag is a place to look, not a finding — and the cheapest way to tell
  // the two apart is to ask again, with `--confirm`.
  if (regressions > 0 || improvements > 0) {
    out(
      dim('  a band covers the spread within one run, not between two — confirm\n') +
        dim('  anything flagged here with --confirm before believing it\n'),
    )
  }
  out('\n')

  // `missing` fails the gate, because a benchmark that stopped running is
  // indistinguishable from a benchmark that stopped being protected: a deleted
  // case reports nothing forever, and the run that deleted it would otherwise
  // be the quietest run in the log. Now that `--record <file>` replaces that
  // file's entries rather than merging into them, a deliberate deletion is
  // cleared by the same re-record the deletion needed anyway.
  //
  // A case with *no* baseline entry deliberately does not fail. That is a new
  // benchmark, and the run that adds one has nothing to compare it against yet.
  const failed =
    regressions > 0 ||
    broadRegression ||
    inconclusive ||
    changed.size > 0 ||
    missing.length > 0
  return options.failOnRegression && failed ? 1 : 0
}

/** @returns the process exit code */
function main(): number {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    out(USAGE)
    return 0
  }
  // Quick mode is one repeat by definition, and says so in its own banner.
  if (options.repeats < 2 && !options.quick) {
    out(
      `  ${yellow('!')} one repeat measures no spread, so every noise band falls back to the ±${(FLOOR * 100).toFixed(0)}% floor\n`,
    )
  }

  // Resolved first, because everything after it is keyed by the exact paths
  // this returns rather than by what was typed — and because a filter that
  // matches no file should say so now, not after the baseline has been read.
  const files = suiteFiles(options.files)

  // Read the baseline, and check what this run could compare against, before
  // spending minutes measuring.
  const baseline: Baseline | null = options.record
    ? null
    : JSON.parse(readFileSync(options.baseline, 'utf8'))
  if (baseline !== null) {
    checkEnvironment(
      baseline,
      candidates(baseline, options, files),
      options.allowEnvironmentChange,
      true,
    )
  }

  if (options.quick) {
    out(
      `  ${yellow('!')} quick mode: short windows, ±${(QUICK_FLOOR * 100).toFixed(0)}% threshold.\n` +
        `    It answers whether something broke, not whether something improved.\n`,
    )
  }

  const scratch = mkdtempSync(join(tmpdir(), 'rapidfuzz-bench-'))
  const bundleDir = join(scratch, 'bundles')
  const passes: Pass[] = []

  try {
    prebundle(bundleDir, files)
    for (let repeat = 1; repeat <= options.repeats; repeat++) {
      const started = Date.now()
      out(`  run ${repeat}/${options.repeats} …`)
      const pass = runPass(scratch, bundleDir, repeat, files, options)
      passes.push(pass)
      out(
        ` ${((Date.now() - started) / 1000).toFixed(0)}s` +
          dim(`, machine ${percent(pass.slope - 1)} across the pass\n`),
      )
    }
  } finally {
    rmSync(scratch, { recursive: true, force: true })
  }

  const { cases, measurement, anchorNoise } = aggregate(passes)

  // A pattern matching nothing is not an error to the runner — it skips every case
  // and exits 0 — so without this the run reports on a suite of no cases, and
  // `report` divides by a column width taken from an empty list. The likeliest
  // cause is the shape of the string being matched, so say what that is.
  if (Object.keys(cases).length === 0) {
    throw new Error(
      `-t ${JSON.stringify(options.name)} matched no case. The pattern is ` +
        `matched against "<group> <case>" — the file is not part of it, and ` +
        `neither are the " > " separators this report prints.`,
    )
  }

  reportMeasurement(measurement)

  // Null exactly when `--record` was asked for: nothing was read, because
  // there is nothing to compare against.
  if (baseline === null) {
    const previous: Record<string, CaseRecord> = existsSync(options.baseline)
      ? JSON.parse(readFileSync(options.baseline, 'utf8')).cases
      : {}
    // Recording a file replaces that file, rather than merging into it.
    //
    // Other files' entries must survive — a partial re-record measures a new
    // yardstick, but every case stores the yardstick from its own session, so
    // rewriting one file cannot renormalise an untouched one. Entries from the
    // files being recorded must *not* survive: merging would leave a case that
    // was renamed or deleted sitting in the baseline under its old name, with
    // nothing left to produce it and no way to tell it apart from a case that
    // simply was not measured today.
    //
    // This is the same invariant `--record` with `-t` is refused over: a file
    // is what a fingerprint covers, so a file is what gets recorded.
    //
    // With no file argument the replacement is the whole baseline, not each
    // measured file in turn. Those differ for exactly one case, and it is the
    // one that matters: `suiteFiles` builds its list from the files that exist
    // *now*, so a bench file that was deleted outright is in no file's scope
    // and a per-file replacement would preserve every one of its cases forever.
    // A full record is a statement that this is the whole suite.
    const rewritten = new Set(files)
    const keep = (name: string): boolean =>
      options.files.length > 0 && !rewritten.has(fileOf(name))
    const merged = {
      ...Object.fromEntries(Object.entries(previous).filter(([name]) => keep(name))),
      ...cases,
    }
    writeFileSync(options.baseline, `${JSON.stringify({ cases: merged }, null, 2)}\n`)
    out(`\n  recorded ${Object.keys(cases).length} cases to ${options.baseline}`)
    // Counted rather than subtracted: replacing three cases with three others
    // nets to zero, and would report a rename as nothing having happened.
    // Silence otherwise looks the same as there being nothing stale to drop.
    const stale = Object.keys(previous).filter(
      (name) => !keep(name) && cases[name] === undefined,
    )
    if (stale.length > 0) {
      out(dim(`, dropping ${stale.length} case(s) no longer measured:\n`))
      for (const name of stale) out(`    ${dim(name)}\n`)
    } else out('\n')
    out('\n')
    return 0
  }

  return report(cases, baseline, anchorNoise, options, files)
}

try {
  process.exitCode = main()
} catch (error) {
  // Every throw in here is a refusal to compare things that are not comparable,
  // which is a message to read rather than a stack to debug.
  process.stderr.write(
    `\n  ${red('!')} ${error instanceof Error ? error.message : error}\n\n`,
  )
  process.exitCode = 1
}
