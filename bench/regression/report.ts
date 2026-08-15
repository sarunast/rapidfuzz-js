/**
 * Reading a measured run against the stored baseline, and saying so.
 *
 * The verdict and its rendering are one function on purpose: every threshold
 * below decides both whether a case moved and how the line is painted, and
 * splitting them would need a verdict model neither side asks for yet.
 */

import type { Baseline, CaseRecord } from './baseline.ts'
import { candidates, checkEnvironment, isControl } from './baseline.ts'
import type { MeasurementStats } from './measurement.ts'
import { geometricMean, machineRatio } from './statistics.ts'
import { dim, green, out, percent, red, yellow } from './terminal.ts'

/**
 * What reporting needs from the parsed command line. Narrower than the CLI's
 * own options object: the command line belongs to `compare.ts`.
 */
export interface ReportOptions {
  quick: boolean
  confirm: boolean
  failOnRegression: boolean
  allowEnvironmentChange: boolean
  /** Canonical bench paths; empty means all. */
  files: readonly string[]
  /** `-t` regexp source, or null for no filter. */
  name: string | null
}

/**
 * Smallest move worth reporting when the measured noise is smaller than this.
 * Nothing this suite optimises for is a 2% change, and a floor keeps a
 * suspiciously quiet run from flagging every case.
 */
export const FLOOR = 0.03

/**
 * The same floor under `--quick`.
 *
 * Shorter windows make each median coarser and can measure a case before V8 has
 * finished tiering, which biases cases and controls by different amounts. A
 * threshold that pretended to the usual resolution would spend the run flagging
 * its own measurement error. At this width it still catches the thing quick
 * mode is for: something that got noticeably, obviously worse.
 */
export const QUICK_FLOOR = 0.15

/** Beyond this much machine drift the run is too unlike the baseline's run to
 *  mean anything, whatever the yardstick corrects for. */
export const DRIFT_LIMIT = 0.1

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
export const RUN_INSTABILITY_LIMIT = 0.05

/** How far the suite may move past the controls before that is a regression. */
export const BROAD_MOVE_LIMIT = 0.03

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
export const QUICK_BROAD_MOVE_LIMIT = QUICK_FLOOR

/**
 * The `--confirm` floor. Confirm mode exists to re-measure the one case a
 * normal comparison flagged, with windows past the point of diminishing
 * returns and more repeats, so its verdict is allowed to be finer than the
 * everyday ±3%.
 */
export const CONFIRM_FLOOR = 0.015

/**
 * Fewest cases before "the suite as a whole moved" means anything.
 *
 * The broad detector has no noise band — it exists to catch a change too even
 * to trip any single case's. Over one filtered case that is not a second
 * opinion, it is the same measurement judged by a stricter rule, and it would
 * fail a run that the case's own band correctly called noise.
 */
export const MIN_BROAD_CASES = 5

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
export const SAMPLE_TOO_SHORT = 0.02
export const SAMPLE_TOO_LONG = 5

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
export function reportSampleSizes(current: Record<string, CaseRecord>): void {
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
export function reportMeasurement(stats: MeasurementStats): void {
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
export function report(
  current: Record<string, CaseRecord>,
  baseline: Baseline,
  anchorNoise: { between: number; within: number },
  options: ReportOptions,
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
