/**
 * Anything this package can compare: a string, or an array-like of arbitrary
 * elements compared by identity. "String matching" here is really sequence
 * matching — arrays of numbers, typed arrays and arrays of objects all work.
 *
 * Both operands of a pair have to be the same kind of sequence. A string is
 * compared by its UTF-16 code units, so `'a'` and `[97]` are not equal.
 */
export type Sequence = string | ArrayLike<unknown>

/** A {@link Sequence} that may be absent — see {@link MissingPolicy}. */
export type MaybeSequence = Sequence | null | undefined

/**
 * Which way a metric's numbers run.
 *
 * `'similarity'` means higher is better and a threshold is a minimum;
 * `'distance'` means lower is better, `0` is identical, and a threshold is a
 * maximum. Everything built on top — thresholds, sorting, search — reads this
 * rather than being told which way to compare.
 */
export type Direction = 'similarity' | 'distance'
/**
 * What a similarity scorer does with a missing operand.
 *
 * `'compatible'` scores the pair `0`, including `null` against `null`: two
 * unknowns are not evidence of a match, and a perfect score there would put
 * every missing record at the top of a search or merge them in a dedup.
 * `'throw'` is for callers who would rather hear that the value never arrived.
 * Distance scorers have no such choice — there is no distance to report.
 */
export type MissingPolicy = 'compatible' | 'throw'
/**
 * Cleans a value before it is scored — lowercasing, stripping punctuation,
 * folding domain noise. `normalizeText` is the built-in one.
 *
 * Applied to both sides, so the comparison always happens between two values
 * cleaned the same way. Two rules follow from where it runs: it must be
 * deterministic, because a Matcher normalizes its choices once at construction,
 * and returning `null`/`undefined` marks the value as having nothing to search,
 * which doubles as a filter.
 */
export type Normalizer = (value: Sequence) => MaybeSequence

/** Mixed into the configuration of every metric that can be asked for a similarity. */
export interface SimilarityConfiguration {
  readonly missing?: MissingPolicy | undefined
}
