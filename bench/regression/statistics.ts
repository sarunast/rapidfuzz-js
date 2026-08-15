/**
 * The arithmetic the comparison is built from: robust middles, spread, and the
 * per-control ratio that turns two machines into one number.
 *
 * Everything here is pure and takes plain numbers, which is what lets both the
 * measurement side and the reporting side use it without reaching through each
 * other.
 */

export function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = sorted.length >> 1
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2
}

export function geometricMean(ratios: readonly number[]): number {
  if (ratios.length === 0) return 1
  return Math.exp(ratios.reduce((total, r) => total + Math.log(r), 0) / ratios.length)
}

/** The middle of a set of ratios, which one bad member cannot drag. */
export function geometricMedian(ratios: readonly number[]): number {
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
export function relativeSpread(values: readonly number[], centre: number): number {
  if (centre <= 0) return 0
  const deviations = values.map((value) => Math.abs(value - centre))
  return (MAD_TO_SIGMA * median(deviations)) / centre
}

/**
 * Fewest controls a yardstick may be built from.
 *
 * The whole case for taking the middle of the per-control ratios rather than
 * their mean is that one control having a bad run cannot drag it. With two
 * there is no middle, and with one there is nothing to be robust about.
 */
export const MIN_CONTROLS = 3

/**
 * The middle of the per-control ratios between two yardsticks: how much
 * slower (>1) or faster (<1) the machine described by `numerator` was than
 * the one described by `denominator`.
 */
export function machineRatio(
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
export function sameControls(
  left: readonly string[],
  right: readonly string[],
): string[] {
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
