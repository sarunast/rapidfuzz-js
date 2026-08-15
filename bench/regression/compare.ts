#!/usr/bin/env node
/**
 * Run the benchmark suite several times and compare the result against
 * `bench/regression/baseline.json`.
 *
 * ## What a baseline stores
 *
 * Not a bare time. Every repeat brackets the suite with
 * `bench/suites/control.bench.ts` — four workloads this library cannot change — and
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
 * See {@link USAGE}, or `node bench/regression/compare.ts --help`.
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, relative, resolve, sep } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

import type { Baseline, CaseRecord } from './baseline.ts'
import {
  checkEnvironment,
  candidates,
  fileOf,
  readBaseline,
  storedCases,
  writeBaseline,
} from './baseline.ts'
import type { Pass } from './measurement.ts'
import { aggregate, prebundle, runPass, suiteFiles } from './measurement.ts'
import { FLOOR, report, reportMeasurement, QUICK_FLOOR } from './report.ts'
import { dim, out, percent, red, yellow } from './terminal.ts'

const REGRESSION_DIR = dirname(fileURLToPath(import.meta.url))
const BENCH_DIR = dirname(REGRESSION_DIR)
const PROJECT_DIR = dirname(BENCH_DIR)
const DEFAULT_BASELINE = join(REGRESSION_DIR, 'baseline.json')

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

const USAGE = `
  node bench/regression/compare.ts [options] [file …]

  Compare the benchmark suite against bench/regression/baseline.json. A file may
  be named by any substring of its path. With no file, the whole suite runs;
  bench/suites/control.bench.ts is the anchor and always runs either way, whatever is
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

    node bench/regression/compare.ts fuzz                 one file, by substring
    node bench/regression/compare.ts -t 'partialRatio'    one group, any file
    node bench/regression/compare.ts --quick -t '128 chars'  the fast look
    node bench/regression/compare.ts --record bench/suites/fuzz.bench.ts

  For the edit loop, \`pnpm bench:quick\` skips the comparison entirely.
`

/**
 * A path argument in the one spelling everything downstream compares against.
 *
 * Stored case names begin with the path the runner reports — `bench/suites/fuzz.bench.ts`,
 * project-relative and slash-separated — and that string is the identity behind
 * three separate things: which baseline entries a partial `--record` replaces,
 * which ones `candidates` considers, and which file gets fingerprinted. Left
 * raw, `./bench/suites/fuzz.bench.ts` and an absolute path each name the same file and
 * match none of them, so a re-record would quietly keep the stale entries it
 * was run to remove. Normalising here means there is only ever one namespace.
 */
function canonicalFile(arg: string): string {
  // Resolved against the project, not the process's cwd: `bench/suites/fuzz.bench.ts`
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
      // The one hazard of a two-token option: `-t bench/suites/fuzz.bench.ts` is a
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
  const baseline: Baseline | null = options.record ? null : readBaseline(options.baseline)
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
    const previous: Record<string, CaseRecord> = storedCases(options.baseline)
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
    writeBaseline(options.baseline, merged)
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
