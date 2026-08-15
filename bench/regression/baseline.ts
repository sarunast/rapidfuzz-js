/**
 * What `baseline.json` holds, and how a stored case is identified.
 *
 * Separate from `compare.ts` because this half is a library and that half is a
 * command: `compare.ts` runs `main()` at module load, so anything that imports
 * it measures the suite as a side effect. Everything here can be imported by a
 * tool that only wants to read or rewrite the stored file.
 */

import { createHash } from 'node:crypto'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { cpus } from 'node:os'
import { dirname, join } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

import { out, yellow } from './terminal.ts'

const REGRESSION_DIR = dirname(fileURLToPath(import.meta.url))
const BENCH_DIR = dirname(REGRESSION_DIR)
const PROJECT_DIR = dirname(BENCH_DIR)

/** Everything outside the code under test that changes what a number means. */
export interface Environment {
  node: string
  esbuild: string
  platform: string
  cpu: string
  nodeOptions: string
  /** The measurement method's version; see `MEASUREMENT_VERSION` in `compare.ts`. */
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
export interface CaseRecord extends Environment {
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

/** The stored file: every case ever recorded, by name. */
export interface Baseline {
  cases: Record<string, CaseRecord>
}

/**
 * Everything a case's timing depends on besides `src` and its own bench file.
 *
 * `bench/harness/runner.ts` is in here because the bundling options are decided
 * there, and changing any of them changes what a stored number means
 * without touching a benchmark. `MEASUREMENT_VERSION` covers a deliberate
 * change of method; this covers forgetting to bump it.
 */
export const SHARED_SOURCES = [
  'bench/harness/corpus.ts',
  'bench/harness/harness.ts',
  'bench/harness/runner.ts',
]

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
export function fingerprint(benchFile: string): string {
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

/** The bench file a stored case came from — the first segment of its name. */
export function fileOf(name: string): string {
  return name.slice(0, name.indexOf(' > '))
}

/** The file whose cases anchor every other case. See its header. */
export const CONTROL_FILE = 'bench/suites/control.bench.ts'

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
export const MEASUREMENT_VERSION = 3

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
export function environment(): Environment {
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

export const isControl = (name: string): boolean => name.startsWith(`${CONTROL_FILE} > `)

/**
 * The two filters that decide which stored cases a run is about: the files it
 * was pointed at, and the `-t` pattern over case names. Narrower than the CLI's
 * own options object on purpose — the parsed command line belongs to
 * `compare.ts`, and nothing down here should be able to reach the rest of it.
 */
export interface CandidateFilter {
  /** Canonical bench paths; empty means all. */
  files: readonly string[]
  /** `-t` regexp source, or null for no filter. */
  name: string | null
}

/**
 * The baseline entries this invocation could possibly compare against.
 *
 * Selected by the same two filters the run itself applies, so this picks
 * exactly the cases a filtered run will produce — which is what lets the
 * environment be checked before spending minutes measuring, without rejecting
 * a run over some unrelated file recorded on another Node.
 *
 * `files` is the resolved list of bench paths, not the raw arguments.
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
export function candidates(
  baseline: Baseline,
  options: CandidateFilter,
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
 * name with single spaces and never sees the file, so `bench/suites/fuzz.bench.ts >
 * ratio > 8 chars` is `ratio 8 chars` to a pattern. Predicting the filter with
 * the stored name instead would quietly disagree with the run — `-t '^ratio'`
 * matches every case here and none there.
 */
export function taskName(name: string): string {
  return name.slice(name.indexOf(' > ') + 3).replaceAll(' > ', ' ')
}

/**
 * Refuse to compare a baseline recorded somewhere else.
 *
 * A V8 release moves individual microbenchmarks by different amounts, which no
 * yardstick can repair, and a different CPU is not a comparison at all.
 */
export function checkEnvironment(
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

/** Read the stored baseline, or fail saying which path was missing. */
export function readBaseline(path: string): Baseline {
  const baseline: Baseline = JSON.parse(readFileSync(path, 'utf8'))
  return baseline
}

/**
 * The cases already on disk, or none.
 *
 * Recording is the one path that tolerates a missing file: the first baseline a
 * checkout ever writes has nothing to merge with.
 */
export function storedCases(path: string): Record<string, CaseRecord> {
  if (!existsSync(path)) return {}
  const baseline: Baseline = JSON.parse(readFileSync(path, 'utf8'))
  return baseline.cases
}

/** Write the baseline back, in the one shape {@link readBaseline} expects. */
export function writeBaseline(path: string, cases: Record<string, CaseRecord>): void {
  writeFileSync(path, `${JSON.stringify({ cases }, null, 2)}\n`)
}
