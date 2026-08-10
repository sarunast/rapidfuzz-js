// @ts-check
/**
 * The timing loop both legs share, restated in Python in `rapidfuzz_bench.py`.
 *
 * Not `bench/_harness.ts`: that one is fingerprinted into every baseline entry
 * and answers a different question — how this library moves against its own
 * past. This one compares two implementations inside one process, where the
 * only defensible statistic is a median over repeats. A machine that spikes
 * mid-run (this one does, every few minutes) otherwise decides the winner.
 */

/**
 * @typedef {object} Timing
 * @property {number} median  seconds per pass, median over repeats
 * @property {number} min     the fastest pass
 * @property {number} spread  (max - min) / median, as a fraction
 * @property {number} passes
 * @property {number} inner   workload repeats inside one pass
 */

/**
 * @param {number[]} values
 * @returns {number}
 */
function median(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = sorted.length >> 1
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2
}

/** A pass shorter than this measures the clock and the scheduler, not the work. */
const TARGET_SECONDS = 0.05

/**
 * Time `run` over `passes` repeats, after warming it up and sizing a pass so
 * the number means something.
 *
 * Two things this does that a bare `hrtime` around a loop does not:
 *
 *   - **The warm-up is not a formality.** V8 needs the function tiered up
 *     before a measurement means anything, and the first pass of an unwarmed
 *     contender is routinely 5-20x its steady state. Every contender gets the
 *     same three passes, which is what makes the ratio fair.
 *   - **A pass is scaled to at least {@link TARGET_SECONDS}.** Timing 200
 *     eight-character comparisons directly measured spreads of 56% to 144% on
 *     this machine — it spikes every few minutes, and against a workload of
 *     tens of microseconds a spike is the whole measurement. Repeating the
 *     workload inside the pass until it is long enough is what makes the median
 *     stable; the reported number is still per single `run()`.
 *
 * @param {() => void} run  one pass over the whole workload
 * @param {number} [passes]
 * @returns {Timing}
 */
export function time(run, passes = 9) {
  for (let i = 0; i < 3; i++) run()

  // Calibrate: grow the inner count until a pass clears the target. The bound
  // stops a pathologically cheap workload from looping forever.
  let inner = 1
  for (;;) {
    const started = process.hrtime.bigint()
    for (let i = 0; i < inner; i++) run()
    const elapsed = Number(process.hrtime.bigint() - started) / 1e9
    if (elapsed >= TARGET_SECONDS || inner >= 1_000_000) break
    const factor = Math.max(2, Math.ceil(TARGET_SECONDS / Math.max(elapsed, 1e-9)))
    inner = Math.min(1_000_000, inner * factor)
  }

  /** @type {number[]} */
  const seconds = []
  for (let i = 0; i < passes; i++) {
    const started = process.hrtime.bigint()
    for (let j = 0; j < inner; j++) run()
    seconds.push(Number(process.hrtime.bigint() - started) / 1e9 / inner)
  }

  const centre = median(seconds)
  return {
    median: centre,
    min: Math.min(...seconds),
    spread: (Math.max(...seconds) - Math.min(...seconds)) / centre,
    passes,
    inner,
  }
}

/**
 * Render a ratio the way the report reads it: how many times faster the
 * baseline is than the contender, or the reverse.
 *
 * @param {number} ours
 * @param {number} theirs
 * @returns {string}
 */
export function speedup(ours, theirs) {
  const ratio = theirs / ours
  return ratio >= 1 ? `${ratio.toFixed(2)}x faster` : `${(1 / ratio).toFixed(2)}x slower`
}
