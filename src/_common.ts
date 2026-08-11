/**
 * Shared plumbing for every scorer: input conversion, the processor hook, and
 * the four `scoreCutoff` conventions used across the distance module.
 */

/** Anything a scorer can compare: a string, or a sequence of arbitrary values. */
export type Sequence = string | ArrayLike<unknown>

/**
 * Preprocessing hook applied to both inputs before scoring.
 *
 * A processor takes whatever the scorer was given, so one that only handles
 * strings — such as {@link import('./utils.js').defaultProcess} — is
 * responsible for rejecting other input itself.
 */
export type Processor = (s: Sequence) => Sequence

/**
 * Options every scorer accepts.
 *
 * Read-only throughout, as documentation of what a scorer does with the object
 * it is handed: it reads the options and never writes them back, so a caller
 * may share one options object across calls. {@link configure} is what turns
 * options into state, and it copies rather than retaining what it was given.
 */
export interface ScorerOptions {
  /** Applied to both inputs before scoring. */
  readonly processor?: Processor | undefined
  /**
   * Bound on the returned score. See the per-function docs — distances return
   * `scoreCutoff + 1` when exceeded, similarities return `0` when not met.
   */
  readonly scoreCutoff?: number | undefined
  /**
   * Expected score used to choose an initial bounded algorithm. This is only a
   * performance hint and never changes the returned score.
   */
  readonly scoreHint?: number | undefined
}

/**
 * Expand a string into code points so characters outside the BMP compare as a
 * single element, matching how Python iterates `str`.
 */
function toCodePoints(s: string): Uint32Array {
  const out = new Uint32Array(s.length)
  let n = 0

  for (let i = 0; i < s.length; i++) {
    const hi = s.charCodeAt(i)

    if (hi >= 0xd800 && hi <= 0xdbff && i + 1 < s.length) {
      const lo = s.charCodeAt(i + 1)
      if (lo >= 0xdc00 && lo <= 0xdfff) {
        out[n++] = (hi - 0xd800) * 0x400 + (lo - 0xdc00) + 0x10000
        i++
        continue
      }
    }

    out[n++] = hi
  }

  // Only a surrogate pair makes the code-point count differ from the code-unit
  // count. Taking the view unconditionally would cost several times more than
  // the loop above, and for BMP text it would be a view over the whole buffer
  // anyway.
  return n === s.length ? out : out.subarray(0, n)
}

/**
 * A high surrogate followed by a low one — the only way UTF-16 indexing can
 * split a code point.
 *
 * Deliberately a regular expression rather than the equivalent `charCodeAt`
 * loop. Every scorer asks this question about both of its inputs before it does
 * any work, so the scan is pure overhead on the common answer of "no", and a
 * loop pays for it one character at a time. A regexp hands the question to the
 * engine, which knows whether the subject is stored one byte or two per
 * character: a Latin-1 string cannot contain a surrogate at all, so the whole
 * match is refused without the string being read. That is the case nearly all
 * text takes, and it measured ~22x faster than the loop on 128-character
 * inputs; two-byte text, which does have to be scanned, still came out ~2x
 * ahead.
 *
 * `test` on a non-global regexp keeps no `lastIndex`, so sharing one instance
 * across calls is safe.
 */
const SURROGATE_PAIR = /[\uD800-\uDBFF][\uDC00-\uDFFF]/

/**
 * Whether UTF-16 indexing would split at least one Unicode code point.
 *
 * Exported so a scorer can ask the question {@link convPair} asks without
 * taking the tuple it allocates to answer it — see `levenshteinDistance_impl`.
 * A caller that skips `convPair` on the strength of this has to test both
 * inputs, exactly as `convPair` does: it keeps a pair as strings only when
 * neither side would split.
 */
export function hasSurrogatePair(s: string): boolean {
  return SURROGATE_PAIR.test(s)
}

/**
 * Port of `conv_sequence`: a single-character string becomes its code point so
 * that `['a', 'b']` compares equal to `'ab'`. Everything else is kept as-is and
 * compared with `===`, which stands in for Python's `hash(elem)`.
 */
function convElement(x: unknown): unknown {
  if (typeof x !== 'string') return x

  const cp = x.codePointAt(0)
  return cp !== undefined && String.fromCodePoint(cp).length === x.length ? cp : x
}

export function convSequence(s: Sequence): ArrayLike<unknown> {
  if (typeof s === 'string') return toCodePoints(s)
  if (ArrayBuffer.isView(s)) return s

  const out = new Array<unknown>(s.length)
  for (let i = 0; i < s.length; i++) out[i] = convElement(s[i])
  return out
}

/**
 * Runtime narrowing for values accepted by scorer conversion.
 *
 * `in` rather than `Reflect.get`: both narrow the type honestly, but the
 * builtin call costs several times a property read, and every prepared scoring
 * validates its choice through here.
 *
 * A `length` that is not a count is refused rather than passed on. Those cases
 * — `NaN`, a fraction, a negative — already fail, but they fail inside
 * {@link convSequence} as a `RangeError` from `new Array`, naming the array
 * constructor for an input problem. What the guard claims and what it proves
 * are the same thing this way, at the cost of two numeric tests on a value that
 * has already been read.
 */
export function isSequence(value: unknown): value is Sequence {
  if (typeof value === 'string') return true
  if (typeof value !== 'object' || value === null) return false
  if (!('length' in value)) return false

  const length = value.length
  return typeof length === 'number' && Number.isSafeInteger(length) && length >= 0
}

/**
 * Port of `conv_sequences`. Normalises a pair of inputs into indexable
 * sequences whose elements compare with `===`.
 *
 * Two strings take a fast path straight to code-point integers. A mixed pair
 * goes through {@link convSequence} on both sides, so a string compares equal
 * to an array of its characters *and* to a byte array of the same text.
 */
function convPair(s1: Sequence, s2: Sequence): [ArrayLike<unknown>, ArrayLike<unknown>] {
  if (typeof s1 === 'string' && typeof s2 === 'string') {
    // BMP-only strings are already indexable one character at a time, so the
    // hot path can avoid allocating two temporary Uint32Arrays. Astral text
    // retains code-point semantics through the existing conversion.
    if (!hasSurrogatePair(s1) && !hasSurrogatePair(s2)) return [s1, s2]
    return [toCodePoints(s1), toCodePoints(s2)]
  }

  return [convSequence(s1), convSequence(s2)]
}

/**
 * Direction and bounds of a scorer's output, so `process` knows whether a high
 * or a low score is better. Port of the `_RF_ScorerPy` flags upstream attaches
 * to every scorer.
 */
export interface ScorerFlags {
  /** Score of a maximally dissimilar pair. */
  readonly worstScore: number
  /** Score of a perfect match. */
  readonly optimalScore: number
  readonly symmetric: boolean
}

/** A scorer carrying the flags `process` reads off it. */
export interface Flagged {
  readonly rfScorerFlags: ScorerFlags
}

// Annotated `unique symbol` rather than left to inference: `isolatedDeclarations`
// requires an annotation here, and a plain `symbol` would not be usable as the
// computed key `Preparable` and `PreparedScore` declare below.
/** Internal hook used by process functions to cache the query side of a scorer. */
export const PREPARE_SCORER: unique symbol = Symbol('rapidfuzz.prepareScorer')
export const PREPARE_CHOICE: unique symbol = Symbol('rapidfuzz.prepareChoice')
/**
 * Brand on the choice index `search.prepareChoices` returns.
 *
 * Here rather than in `search.ts` for the reason above: the index type is
 * public, so its key has to be a `unique symbol` an external caller cannot
 * name — which is what makes the type unforgeable — and that has to be an
 * exported binding for the emitted declarations to refer to it.
 */
export const PREPARED_CHOICES: unique symbol = Symbol('rapidfuzz.preparedChoices')
/**
 * Brands on the two callable handles `search.prepareQuery` and
 * `search.prepareChoice` return, here for the same reason as the one above.
 */
export const PREPARED_QUERY_HANDLE: unique symbol = Symbol('rapidfuzz.preparedQuery')
export const PREPARED_CHOICE_HANDLE: unique symbol = Symbol('rapidfuzz.preparedChoice')

/**
 * Whether a value is one of the two prepared handles.
 *
 * A brand test rather than a shape test, for the reason the brands exist: a
 * handle is an ordinary function carrying two ordinary properties, and every
 * one of those is something a third-party scorer could also have.
 */
export function isPreparedHandle(value: unknown): boolean {
  return (
    typeof value === 'function' &&
    (PREPARED_QUERY_HANDLE in value || PREPARED_CHOICE_HANDLE in value)
  )
}

/**
 * Refuse a prepared handle where a scorer is expected.
 *
 * A handle is a function of two arguments returning a number, so it satisfies
 * {@link ErasedScorer} — parameters are contravariant and `never` is assignable
 * to everything — and type-checks at every seam that takes a scorer. What
 * happens then is not an error but a wrong number: {@link callScorer} hands the
 * handle the *choice* as its options bag, and {@link scorerFlagsOf} falls back
 * to the fuzz defaults, so an exact match scores `0` and a distance handle also
 * gets the direction backwards.
 *
 * In `_common.ts` rather than in `search.ts` so that `configure.ts` and
 * `match.ts` can ask without importing the search module.
 */
export function assertNotPreparedHandle(value: unknown): void {
  if (isPreparedHandle(value)) {
    throw new TypeError('a prepared query or choice cannot be used as a scorer')
  }
}

const PREPARED_SEQUENCE = Symbol('rapidfuzz.preparedSequence')

interface PreparedSequence {
  readonly [PREPARED_SEQUENCE]: true
  readonly value: ArrayLike<unknown>
}

/**
 * The per-choice preparation a scorer's factory offers.
 *
 * Named for what it is rather than for what it does, because `prepareChoice` is
 * a public entry point in `search.ts` now and a type sharing its name is one
 * import away from shadowing it.
 */
export type ChoicePreparer = (choice: unknown) => unknown

export interface PreparedScore {
  (choice: unknown, scoreCutoff: number | null, scoreHint: number | null): number
}

/**
 * A scorer's query-preparation factory, carrying the policy it applies to a
 * *choice*.
 *
 * The hook is on the factory rather than on each prepared score because that is
 * where the fact belongs: how a scorer wants a choice prepared is a property of
 * the scorer, not of the query it was last handed. It used to be assigned to
 * every prepared score, which meant a caller could only reach it by preparing a
 * query first — fine for `scoreMatrix`, which prepares one anyway, and a hack
 * for `prepareChoices`, which has no query at all and had to fabricate one from
 * the first choice to read a value that never varied with it.
 *
 * Absent is a real answer: a scorer may cache a query and still want its
 * choices untouched.
 *
 * Writable, and assigned rather than defined. `Object.defineProperty` was
 * measured at **109 ns** against **1.4 ns** for an assignment when this ran per
 * prepared query; it now runs once per scorer, so the difference no longer
 * shows — but the descriptor still buys nothing, since a symbol key is already
 * absent from `Object.keys`, `for...in` and `JSON.stringify`.
 */
export interface PrepareScorer {
  (query: Sequence, kwargs: Readonly<Record<string, unknown>>): PreparedScore
  [PREPARE_CHOICE]?: ChoicePreparer
}

/**
 * Attach a factory's choice-preparation policy, at the point the factory is
 * built. Every built-in goes through here, so the two cannot drift apart.
 */
export function withChoicePreparer(
  prepare: PrepareScorer,
  choicePreparer: ChoicePreparer,
): PrepareScorer {
  prepare[PREPARE_CHOICE] = choicePreparer
  return prepare
}

export type PreparedMetricKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

/**
 * The largest distance a score can have and still clear `rawCutoff`.
 *
 * Each of the four conventions bounds the score from a different direction, so
 * each turns a cutoff on the *reported* score into a bound on the underlying
 * distance. A kernel handed this may stop as soon as it exceeds the bound and
 * report `bound + 1`; every convention maps that back to the same rejection the
 * exact distance would have produced.
 *
 * `Infinity` — the result when no cutoff was given — is deliberate: it makes
 * "no bound" a value the kernels compare against rather than a case they test
 * for, so the bounded and unbounded paths stay one loop.
 *
 * The two raw conventions bound against {@link canonicalRawCutoff}, not against
 * the number the caller passed, so the kernel is held to the same integer the
 * reported score is compared against.
 */
export function distanceCutoffFor(
  kind: PreparedMetricKind,
  rawCutoff: number | null | undefined,
  maximum: number,
): number {
  if (rawCutoff == null) return Number.POSITIVE_INFINITY

  switch (kind) {
    case 'distance':
      return truncatedRawCutoff(rawCutoff)
    case 'similarity':
      return maximum - truncatedRawCutoff(rawCutoff)
    case 'normalizedDistance':
      return rawCutoff * maximum
    case 'normalizedSimilarity':
      return (1 - rawCutoff) * maximum
  }
}

/**
 * Build an allocation-free prepared adapter for a simple exact distance.
 *
 * `distance` receives the derived distance bound as its fourth argument. An
 * implementation may ignore it and return the exact distance — that is always
 * within the bound the caller would accept — or use it to bail out early and
 * return any value greater than the bound.
 */
export function prepareMetric(
  kind: PreparedMetricKind,
  distance: (
    query: ArrayLike<unknown>,
    choice: ArrayLike<unknown>,
    parsedKwargs: unknown,
    distanceCutoff: number,
  ) => number,
  maximum: (query: ArrayLike<unknown>, choice: ArrayLike<unknown>) => number,
  parseKwargs: (kwargs: Readonly<Record<string, unknown>>) => unknown = () => null,
): PrepareScorer {
  const prepare: PrepareScorer = (query, kwargs) => {
    const preparedQuery = preparedScorerSequence(prepareScorerChoice(query))
    if (preparedQuery === null) throw new TypeError('expected a sequence')
    const parsedKwargs = parseKwargs(kwargs)
    const score: PreparedScore = (rawChoice, rawCutoff) => {
      if (isNone(rawChoice)) {
        if (kind === 'normalizedDistance') return 1
        if (kind === 'normalizedSimilarity') return 0
      }
      let choice = preparedScorerSequence(rawChoice)
      if (choice === null) {
        if (!isSequence(rawChoice)) {
          throw new TypeError('expected a string or an array-like sequence')
        }
        choice = scorerSequence(rawChoice)
      }
      // `distance` compares elements with `===`, so the pair has to agree on how
      // a character is spelled before it is handed over.
      const query = alignRepresentation(preparedQuery, choice)
      const aligned = alignRepresentation(choice, preparedQuery)
      const max = maximum(query, aligned)
      const dist = distance(
        query,
        aligned,
        parsedKwargs,
        distanceCutoffFor(kind, rawCutoff, max),
      )
      switch (kind) {
        case 'distance':
          return distCutoff(dist, rawCutoff)
        case 'similarity':
          return simCutoff(max - dist, rawCutoff)
        case 'normalizedDistance':
          return normDistCutoff(normalize(dist, max), rawCutoff)
        case 'normalizedSimilarity':
          return normSimCutoff(1 - normalize(dist, max), rawCutoff)
      }
    }
    return score
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}

export interface Preparable {
  readonly [PREPARE_SCORER]: PrepareScorer
}

/**
 * Public shape of a distance or similarity scorer.
 *
 * Every exported scorer is annotated with this (or {@link NormalizedScorer})
 * rather than letting its type be inferred from the `withPreparedFlags` call.
 * Inference produced `typeof hammingDistance_impl & Flagged & Preparable` in
 * the emitted `.d.ts`, which named a private function in the public API;
 * annotating keeps the internal `*_impl` out of the declarations and satisfies
 * `isolatedDeclarations`.
 *
 * Deliberately *not* {@link Preparable}. Having a prepared-query factory is a
 * separate fact from being a scorer, and the run time already treats it as one:
 * {@link prepareScorerOf} answers `null` for a built-in registered without a
 * factory, which is exactly what {@link configure} produces from a scorer with
 * a baked-in processor. A `Scorer` that promised the factory would be a type
 * that a value this package itself hands out does not satisfy.
 */
export interface Scorer<O extends ScorerOptions = ScorerOptions> extends Flagged {
  (s1: Sequence, s2: Sequence, options?: O): number
}

/**
 * Public shape of a normalized scorer. These report maximum dissimilarity for a
 * "missing" input instead of throwing, so their inputs widen to
 * {@link MaybeSequence}.
 */
export interface NormalizedScorer<
  O extends ScorerOptions = ScorerOptions,
> extends Flagged {
  (s1: MaybeSequence, s2: MaybeSequence, options?: O): number
}

const builtInScorers = new WeakSet<object>()

/**
 * The prepared-query factory each built-in scorer was registered with.
 *
 * `process` could read the same function back off the `PREPARE_SCORER`
 * property, but only as an `unknown` it would then have to narrow — and no
 * runtime check can confirm a function's signature, so that narrowing would
 * assert a shape nothing had verified. Recording the factory here instead
 * keeps it typed from the point of registration, where it is known.
 */
const preparedFactories = new WeakMap<object, PrepareScorer>()

/**
 * How a scorer's flags change once options are baked into it.
 *
 * Recorded in a WeakMap for the same reason {@link preparedFactories} is.
 */
const flagsFactories = new WeakMap<object, ConfiguredFlags>()

/**
 * How a scorer turns options being baked into it into values it can keep.
 *
 * Recorded in a WeakMap for the same reason {@link preparedFactories} is.
 */
const optionCanonicalizers = new WeakMap<object, ConfigureOptions>()

/**
 * Derives a scorer's flags from the options {@link configure} baked into it.
 *
 * Only a scorer whose options can change its direction or its symmetry needs
 * one. Levenshtein is the case that exists: swapping the arguments swaps
 * insertion and deletion, so it is symmetric exactly when their costs are
 * equal, and a matrix over one input can only mirror its lower triangle when
 * they are. Asking the scorer that owns the option is what keeps `scoreMatrix`
 * from having to know that the option is spelled `weights`.
 */
export type ConfiguredFlags = (options: Readonly<Record<string, unknown>>) => ScorerFlags

/**
 * Replaces baked option values with ones the scorer can hold onto.
 *
 * {@link configure} copies the options object, but only one level deep — a
 * nested value stays shared with whoever passed it in. That is a correctness
 * problem, not just an aliasing wart, because {@link ConfiguredFlags} runs
 * *once*: bake `{ weights }`, then mutate `weights.deletion`, and the scorer
 * starts scoring asymmetrically while its recorded flags still say symmetric —
 * at which point `scoreMatrix` mirrors a triangle it never scored and reports
 * numbers that are simply wrong.
 *
 * Only the scorer that owns an option knows which of its values are structural,
 * so canonicalising is asked of it rather than attempted generically. A deep
 * copy here could not tell a weights array from a processor function.
 */
export type ConfigureOptions = (
  options: Readonly<Record<string, unknown>>,
) => Readonly<Record<string, unknown>>

/** Everything a scorer may register beyond its static flags. */
export interface ScorerRegistration {
  /**
   * Query-caching factory. Absent means this scorer has no prepared path and
   * {@link prepareScorerOf} reports `null` for it — which is the state a
   * configured scorer with a baked-in processor needs, since the processor has
   * to reach the scorer and the prepared path bypasses it.
   */
  readonly prepare?: PrepareScorer | undefined
  /** See {@link ConfiguredFlags}. Absent means options cannot change the flags. */
  readonly configuredFlags?: ConfiguredFlags | undefined
  /** See {@link ConfigureOptions}. Absent means no option value needs snapshotting. */
  readonly configureOptions?: ConfigureOptions | undefined
}

/**
 * Record a scorer this package built, with whatever metadata it has.
 *
 * Deliberately three independent facts rather than one bundle: being a built-in
 * (which lets `extract` tighten its cutoff against the running best), having a
 * prepared factory, and knowing how baked options change its flags. A
 * configured scorer needs the first without the second.
 */
export function registerScorer<F extends ErasedScorer>(
  fn: F,
  flags: ScorerFlags,
  registration: ScorerRegistration,
): F & Flagged {
  const scorer = Object.assign(fn, { rfScorerFlags: stableFlags(flags) })
  recordScorer(scorer, registration)
  return scorer
}

/**
 * Take a scorer's flags out of its supplier's hands.
 *
 * {@link ScorerFlags} is read once per scoring *run* and then trusted for every
 * pair in it, so a flag that changes underneath is not a stale display value —
 * `symmetric` turned on for an asymmetric scorer makes `scoreMatrix` mirror a
 * triangle it never scored. The same reasoning as {@link ConfigureOptions},
 * one level further out: what a scorer promises about its own output has to
 * stop being anyone else's to change once the scorer exists.
 *
 * The copy is what makes the freeze safe to apply to a caller's object, and it
 * costs one small object per scorer, inside the `@__PURE__` call a bundler can
 * still drop.
 */
export function stableFlags(flags: ScorerFlags): ScorerFlags {
  return Object.freeze({ ...flags })
}

/** Populate the registries, leaving the caller to attach the properties. */
function recordScorer(scorer: object, registration: ScorerRegistration): void {
  builtInScorers.add(scorer)
  if (registration.prepare !== undefined) {
    preparedFactories.set(scorer, registration.prepare)
  }
  if (registration.configuredFlags !== undefined) {
    flagsFactories.set(scorer, registration.configuredFlags)
  }
  if (registration.configureOptions !== undefined) {
    optionCanonicalizers.set(scorer, registration.configureOptions)
  }
}

/** Whether a scorer is one of this package's implementations. */
export function isBuiltInScorer(scorer: object): boolean {
  return builtInScorers.has(scorer)
}

/**
 * The prepared-query factory for a built-in scorer, or `null` for anything this
 * package did not build — third-party scorers take the generic call path.
 */
export function prepareScorerOf(scorer: object): PrepareScorer | null {
  return preparedFactories.get(scorer) ?? null
}

/** The flags-from-options resolver a scorer registered, or `null`. */
export function configuredFlagsOf(scorer: object): ConfiguredFlags | null {
  return flagsFactories.get(scorer) ?? null
}

/** The option canonicaliser a scorer registered, or `null`. */
export function configureOptionsOf(scorer: object): ConfigureOptions | null {
  return optionCanonicalizers.get(scorer) ?? null
}

function isPreparedSequence(value: unknown): value is PreparedSequence {
  if (typeof value !== 'object' || value === null) return false
  if (!(PREPARED_SEQUENCE in value) || value[PREPARED_SEQUENCE] !== true) return false
  return 'value' in value && isSequence(value.value)
}

/** Convert once and wrap so process matrix scorers can reuse the result. */
export function prepareScorerChoice(choice: unknown): unknown {
  if (!isSequence(choice)) return choice

  const value = scorerSequence(choice)
  const prepared: PreparedSequence = { [PREPARED_SEQUENCE]: true, value }
  return prepared
}

/**
 * Normalise a scorer sequence while retaining allocation-free BMP strings.
 *
 * The two forms this can return — a string, or code points — do not compare
 * elementwise with each other, so anything holding one of each has to put them
 * in a common representation first. See {@link alignRepresentation}.
 */
export function scorerSequence(choice: Sequence): ArrayLike<unknown> {
  return typeof choice === 'string' && !hasSurrogatePair(choice)
    ? choice
    : convSequence(choice)
}

/**
 * Expand `s` into code points when `other` is already in that form.
 *
 * {@link scorerSequence} keeps a BMP-only string as a string, so a prepared
 * query and a converted choice can meet as `'a'` and `97` — equal text that
 * `===` reports as different. The bit-parallel kernels read either form through
 * `charCodeAt` and so never notice, but every elementwise comparison does: a
 * common prefix, Hamming's mismatch count, Jaro's transposition pass. Calling
 * this on both sides settles it at the one place the pair is formed.
 *
 * Two inputs already in the same representation are returned untouched, which
 * is every pair that does not straddle the BMP.
 */
export function alignRepresentation(
  s: ArrayLike<unknown>,
  other: ArrayLike<unknown>,
): ArrayLike<unknown> {
  return typeof s === 'string' && typeof other !== 'string' ? convSequence(s) : s
}

/**
 * Ceiling on the probe below, whatever an eighth of the inputs comes to.
 *
 * The pair the probe is most expensive for is the one it sends to the trimming
 * kernel: an affix of 480 in 512 leaves that kernel 32 elements to score, so an
 * unbounded eighth would spend more choosing than scoring. Thirty-two costs
 * half of what an eighth of that pair would and reaches the same verdict, and
 * it is still wider than the nineteen-element prefixes a near-copy list throws
 * up at every width the suite measures — which is the confusion that matters,
 * since calling those an affix would give back the whole win.
 *
 * Past this width the probe over-reports: an affix of exactly this many in a
 * pair thousands long is called worth trimming when it removes almost nothing.
 * That is the safe direction — over-reporting keeps the behaviour the length
 * gate had on its own — and it is what bounds the cost.
 *
 * Halving it again to 16 was measured and is worse: it starts calling the
 * near-copy prefixes an affix, which took the wins on those from 0.56x and
 * 0.74x back to 0.65x and 0.84x, to save 2% on the affix-heavy shape.
 */
const AFFIX_PROBE_LIMIT = 32

/**
 * Whether the pair shares an affix long enough to be worth trimming.
 *
 * This is the content half of the dispatch each metric's length gate
 * makes on lengths alone. The unprepared kernel's advantage over the held
 * pattern is entirely that it removes a common prefix and suffix first, and a
 * tight cutoff produces a narrow band and a large affix alike, so lengths
 * cannot tell an affix-free pair from an affix-heavy one — which is why
 * relaxing that gate on lengths measured 1.6x to 7.7x faster on one population
 * and 3.2x slower on the other.
 *
 * Deliberately a lower bound rather than the affix itself: it answers "is there
 * at least an eighth", not "how much", because measuring the affix itself would
 * cost the whole scan the answer is meant to avoid.
 *
 * It does not make the affix-heavy case free, and is not meant to: scanning the
 * probe and then handing the pair to a kernel with 32 elements left to score
 * measures about 4% slower than not asking. What it buys for that 4% is 1.35x
 * to 7.0x on the pairs the length gate was refusing, where the previous attempt
 * — relaxing on lengths alone — cost 3.2x on this same shape.
 *
 * An eighth rather than a constant because what matters is the affix relative
 * to the work it removes — nineteen shared elements out of 1024 is noise, and
 * the near-copy corpus is full of exactly that. It also makes the probe pay for
 * itself: it runs to completion only when the affix really is an eighth or
 * more, and trimming then removes at least an eighth of the kernel's work,
 * while an affix-free pair leaves at the first mismatch.
 *
 * Representation is a property of the pair, not of either side — a BMP string
 * held as a string and a sequence converted to code points compare `'a'`
 * against `97` and agree nowhere. A mixed pair therefore reports `true`, which
 * is the conservative answer: it leaves the length gate deciding alone, exactly
 * as it did before this probe existed.
 */
export function sharesAffix(a: ArrayLike<unknown>, b: ArrayLike<unknown>): boolean {
  const probe = Math.min(Math.min(a.length, b.length) >>> 3, AFFIX_PROBE_LIMIT)
  const lastA = a.length - 1
  const lastB = b.length - 1
  if (typeof a === 'string') {
    if (typeof b !== 'string') return true
    let i = 0
    while (i < probe && a.charCodeAt(i) === b.charCodeAt(i)) i++
    if (i === probe) return true
    let j = 0
    while (j < probe && a.charCodeAt(lastA - j) === b.charCodeAt(lastB - j)) j++
    return j === probe
  }
  if (typeof b === 'string') return true
  let i = 0
  while (i < probe && a[i] === b[i]) i++
  if (i === probe) return true
  let j = 0
  while (j < probe && a[lastA - j] === b[lastB - j]) j++
  return j === probe
}

/** Read a process-prepared sequence after validating its opaque record. */
export function preparedScorerSequence(value: unknown): ArrayLike<unknown> | null {
  return isPreparedSequence(value) ? value.value : null
}

/**
 * A distance has no upper bound and neither does a raw similarity, so the score
 * at the unbounded end of each is `Infinity`.
 *
 * Upstream writes `2**63 - 1` here, which is `SIZE_T_MAX` in C and the largest
 * value its `uint64_t` scores can hold. As a `number` that expression does not
 * even evaluate to what it says — `2**63 - 1 === 2**63` — and it is not a score
 * any scorer here can return, so reading it as "the worst a distance can be"
 * meant reading a C type's ceiling as a fact about the metric.
 *
 * These two are compared against, never handed to a scorer: `search` admits a
 * score by comparing it with the bound below, and says "no cutoff" to the
 * scorer with `null`. That separation is what lets this be honest — a cutoff is
 * still a count of elements, and {@link canonicalRawCutoff} still refuses an
 * infinite one.
 */
export const DISTANCE_FLAGS: ScorerFlags = {
  optimalScore: 0,
  worstScore: Number.POSITIVE_INFINITY,
  symmetric: true,
}
export const SIMILARITY_FLAGS: ScorerFlags = {
  optimalScore: Number.POSITIVE_INFINITY,
  worstScore: 0,
  symmetric: true,
}
export const NORMALIZED_DISTANCE_FLAGS: ScorerFlags = {
  optimalScore: 0,
  worstScore: 1,
  symmetric: true,
}
export const NORMALIZED_SIMILARITY_FLAGS: ScorerFlags = {
  optimalScore: 1,
  worstScore: 0,
  symmetric: true,
}
/** Every `fuzz` scorer reports a percentage. */
export const FUZZ_FLAGS: ScorerFlags = {
  optimalScore: 100,
  worstScore: 0,
  symmetric: true,
}

/**
 * Attach scorer flags and the non-public prepared-query implementation.
 *
 * Call sites mark this `@__PURE__` so a bundler can still drop the scorer when
 * it is unused — that annotation is what keeps `"sideEffects": false` honest.
 */
export function withPreparedFlags<F extends ErasedScorer>(
  fn: F,
  flags: ScorerFlags,
  prepare: PrepareScorer,
  metadata: Omit<ScorerRegistration, 'prepare'> = {},
): F & Flagged & Preparable {
  // One `Object.assign`, not one per registration step: adding properties to a
  // function object in two passes transitions its shape twice, and every scorer
  // this package ships goes through here.
  const scorer = Object.assign(fn, {
    rfScorerFlags: stableFlags(flags),
    [PREPARE_SCORER]: prepare,
  })
  recordScorer(scorer, { ...metadata, prepare })
  return scorer
}

/**
 * Any scorer, seen from outside: nothing may be assumed about its inputs.
 *
 * `never` in the parameters is what makes every concrete scorer assignable
 * contravariantly, so a caller holding *some* scorer can name this type without
 * claiming to know which one it has. That is also what makes it usable as the
 * constraint on {@link registerScorer} and {@link withPreparedFlags}: they
 * attach a scorer's metadata, so what they take has to be a scorer, and `object`
 * let a plain record through.
 */
export type ErasedScorer = (s1: never, s2: never, options?: never) => number

/**
 * Call a scorer whose declared input types the caller cannot see.
 *
 * Scorers are declared with concrete inputs; `process` and {@link configure}
 * only know they hold *some* scorer. This is the one place the two meet, and it
 * stays honest by routing through `Reflect.apply` and checking the result,
 * rather than asserting a signature nothing has verified.
 *
 * `options` is `object` rather than a record because this function does not
 * read it — it hands it to the scorer untouched. Requiring a record would mean
 * a caller holding an options *interface* had to copy it key by key to satisfy
 * the parameter (an interface has no implicit index signature), and that copy
 * is observable: a scorer that reads `Object.keys`, a getter, or the identity
 * of what it was passed would see the copy rather than what the caller wrote.
 */
export function callScorer(
  scorer: ErasedScorer,
  s1: unknown,
  s2: unknown,
  options: object,
): number {
  const result: unknown = Reflect.apply(scorer, undefined, [s1, s2, options])

  if (typeof result !== 'number') {
    throw new TypeError('scorer did not return a number')
  }

  return result
}

/**
 * Call a scorer with no options argument at all.
 *
 * The one caller is a prepared handle invoked without call options, which is
 * standing in for `scorer(query, choice)` — so it makes that call rather than
 * one with two `undefined` fields. The difference is observable to any scorer
 * that tests `options === undefined` or reads `arguments.length`, which is
 * exactly the third-party scorer this path exists for.
 *
 * A sibling rather than an optional fourth parameter on {@link callScorer},
 * because that function is on every non-prepared scoring path in the package
 * and its call sites are measured where they stand.
 */
export function callScorerBare(scorer: ErasedScorer, s1: unknown, s2: unknown): number {
  const result: unknown = Reflect.apply(scorer, undefined, [s1, s2])

  if (typeof result !== 'number') {
    throw new TypeError('scorer did not return a number')
  }

  return result
}

/**
 * Copy an options object into a plain record.
 *
 * Options are typed per scorer, so they are not the `Record<string, unknown>`
 * the call plumbing spreads into a scorer's options. Copying key by key
 * produces one without claiming anything about the shape that the type system
 * has not already checked.
 */
export function toRecord(value: unknown): Readonly<Record<string, unknown>> {
  const out: Record<string, unknown> = {}
  if (typeof value !== 'object' || value === null) return out

  for (const key of Object.keys(value)) out[key] = Reflect.get(value, key)
  return out
}

/** The empty options record, shared so the common path allocates nothing. */
export const NO_OPTIONS: Readonly<Record<string, unknown>> = Object.freeze({})

/**
 * Read a scorer's flags, falling back to upstream's default for unknown scorers.
 *
 * The flags come off an arbitrary object through `Reflect.get`, so each field is
 * proved rather than spread: `{ ...FUZZ_FLAGS, ...flags }` produced a
 * {@link ScorerFlags} whose `symmetric` had never been checked, and carried any
 * other key the object happened to have along with it.
 *
 * `symmetric` still falls back rather than rejecting the whole object, because
 * the two bounds and the symmetry are independently useful: a third-party
 * scorer that declares only which direction is better is stating something
 * true, and dropping its bounds over a missing third field would score it as a
 * percentage.
 */
export function scorerFlagsOf(scorer: object): ScorerFlags {
  const flags = Reflect.get(scorer, 'rfScorerFlags')
  if (typeof flags !== 'object' || flags === null) return FUZZ_FLAGS

  const worstScore = Reflect.get(flags, 'worstScore')
  const optimalScore = Reflect.get(flags, 'optimalScore')
  if (typeof worstScore !== 'number' || typeof optimalScore !== 'number') {
    return FUZZ_FLAGS
  }

  const symmetric = Reflect.get(flags, 'symmetric')
  return {
    worstScore,
    optimalScore,
    symmetric: typeof symmetric === 'boolean' ? symmetric : FUZZ_FLAGS.symmetric,
  }
}

/**
 * A scorer input that may be "missing". Upstream's normalized scorers report
 * maximum dissimilarity for such an input rather than raising.
 *
 * Python spells "missing" three ways — `None`, and `float("nan")` — and
 * {@link isNone} still recognises the `NaN` at runtime, because the ported
 * suite pins that behaviour. It is deliberately not in this type: `number` here
 * would make `levenshteinNormalizedSimilarity(5, 'abc')` typecheck and then
 * throw, since every number that is not `NaN` is an error. JavaScript already
 * has two ways to say "missing" that carry no such trap.
 */
export type MaybeSequence = Sequence | null | undefined

/**
 * Port of `is_none`: `None` and `float("nan")` both mean "no result".
 *
 * Takes `unknown` because `process` tests arbitrary choices with it.
 */
export function isNone(s: unknown): boolean {
  return s == null || (typeof s === 'number' && Number.isNaN(s))
}

/**
 * Narrow a non-missing input to a {@link Sequence}. A bare number is not a
 * sequence, which is the case Python raises `TypeError` on.
 *
 * Takes `unknown` for the same reason {@link isNone} does: its job is the run
 * time boundary, and the values it has to refuse — a number, a plain object, a
 * `Symbol` — are ones no caller could pass if the parameter were typed to
 * exclude them. Narrowing through {@link isSequence} rather than by listing
 * what to reject also keeps one answer to "is this a sequence".
 */
export function asSequence(value: unknown): Sequence {
  if (!isSequence(value)) {
    throw new TypeError('expected a string or an array-like sequence')
  }

  return value
}

/** Options for the `editops` / `opcodes` entry points, which have no cutoff. */
export interface EditopsOptions {
  /** Applied to both inputs before the alignment is computed. */
  readonly processor?: Processor | undefined
}

/** Apply the processor (if any) and normalise both inputs into indexable sequences. */
export function conv(
  s1: Sequence,
  s2: Sequence,
  processor?: Processor | undefined,
): [ArrayLike<unknown>, ArrayLike<unknown>] {
  if (processor == null) return convPair(s1, s2)

  // A processor is a caller's own function, so its return is data rather than
  // something the type says anything about. `convSequence` reads a `length` off
  // whatever it is given, and `new Array(undefined)` is a one-element array —
  // so a processor returning a number turned both sides into `[undefined]` and
  // scored them as a perfect match. Upstream raises on the same returns, and
  // the check belongs here rather than in `convSequence`, which every prepared
  // comparison goes through.
  const a = processor(s1)
  const b = processor(s2)
  if (!isSequence(a) || !isSequence(b)) {
    throw new TypeError('processor must return a string or an array-like sequence')
  }

  return convPair(a, b)
}

/**
 * A cutoff on a raw score, reduced to the integer the score is compared against.
 *
 * Upstream's bindings convert a raw cutoff to a `uint64_t`, and both halves of
 * that conversion are observable. The truncation is: `score_cutoff=1.9` bounds
 * a distance at `1`, so `Prefix.distance('abc', 'xyz', score_cutoff=1.9)` is
 * `2` rather than `2.9`, and `Prefix.similarity('ab', 'ax', score_cutoff=1.9)`
 * is `1` rather than `0`. The refusal is: `NaN` and an infinity have no integer
 * to convert to, and one that truncates below zero does not fit an unsigned
 * type. `-0.5` therefore passes and behaves as `0`.
 *
 * What is *not* kept is upstream's ceiling. Its conversion also refuses
 * anything at or above `2 ** 64`, because that is where a `uint64_t` runs out —
 * a fact about the C type its bindings convert into, not about the metric. A
 * cutoff here is a count of elements held in a `number`, so the limit that
 * means something is finiteness, and `2 ** 64` is accepted. Nothing observable
 * hangs on the difference: a cutoff that large is unreachable by any distance
 * over a JavaScript string, so both answers are the exact score.
 *
 * Refusing a genuinely negative cutoff is worth it on its own account too. It
 * has no true answer to give: nothing satisfies `dist <= -1`, so every pair
 * collapses to `cutoff + 1`, and for `-1` that is `0` — a distance of zero
 * reported for inputs that differ.
 *
 * Truncating *here*, rather than at the point a score is finally clamped, is
 * what keeps one cutoff in play: the same integer bounds the kernel and decides
 * the reported score. A kernel handed the untruncated `max - 1.9` where the
 * cutoff really means `max - 1` stops one step early and reports a rejection
 * for a pair that clears the cutoff.
 *
 * These are the scorers in `distance/`. The ones in `fuzz.ts` score out of 100
 * and validate nothing, upstream included, so they do not come through here.
 */
function truncatedRawCutoff(cutoff: number): number {
  const truncated = Math.trunc(cutoff)

  // `Number.isFinite` refuses `NaN` and both infinities; `< 0` is left to say
  // only what it says. An infinite cutoff is refused rather than read as "no
  // bound": for a distance the two would agree, but for a similarity an
  // infinite cutoff is one nothing can meet, and a single spelling cannot mean
  // both. Absence is spelled by absence — see {@link canonicalRawCutoff}.
  if (!Number.isFinite(truncated) || truncated < 0) {
    throw new RangeError('scoreCutoff has to be a finite count of at least 0')
  }

  return truncated
}

/**
 * {@link truncatedRawCutoff} over an optional cutoff, keeping "no cutoff" as
 * `null`.
 *
 * A scorer that derives a bound for its kernel calls this once, at the top, and
 * uses the result for both the bound and the clamp on the way out.
 */
export function canonicalRawCutoff(cutoff: number | null | undefined): number | null {
  return cutoff == null ? null : truncatedRawCutoff(cutoff)
}

/**
 * A cutoff on a normalised score, which upstream requires to be in `[0, 1]`.
 *
 * `NaN` is deliberately let through, because upstream lets it through: its
 * check is a pair of comparisons and `NaN` fails both, so the cutoff survives
 * to be compared against the score, fails that comparison too, and yields the
 * worst score. Refusing it here would be stricter than the thing being ported.
 */
function checkNormalizedCutoff(cutoff: number): void {
  if (cutoff < 0 || cutoff > 1) {
    throw new RangeError('scoreCutoff has to be in the range of 0.0 - 1.0')
  }
}

/** Distances above `cutoff` collapse to `cutoff + 1`. */
export function distCutoff(dist: number, cutoff?: number | null | undefined): number {
  if (cutoff == null) return dist

  const bound = truncatedRawCutoff(cutoff)
  return dist <= bound ? dist : bound + 1
}

/** Similarities below `cutoff` collapse to `0`. */
export function simCutoff(sim: number, cutoff?: number | null | undefined): number {
  if (cutoff == null) return sim

  const bound = truncatedRawCutoff(cutoff)
  return sim >= bound ? sim : 0
}

/** Normalised distances above `cutoff` collapse to `1`. */
export function normDistCutoff(dist: number, cutoff?: number | null | undefined): number {
  if (cutoff == null) return dist
  checkNormalizedCutoff(cutoff)
  return dist <= cutoff ? dist : 1
}

/** Normalised similarities below `cutoff` collapse to `0`. */
export function normSimCutoff(sim: number, cutoff?: number | null | undefined): number {
  if (cutoff == null) return sim
  checkNormalizedCutoff(cutoff)
  return sim >= cutoff ? sim : 0
}

/** `dist / maximum`, with the degenerate "both inputs empty" case pinned to 0. */
export function normalize(dist: number, maximum: number): number {
  return maximum === 0 ? 0 : dist / maximum
}

/**
 * Number of leading elements `s1` and `s2` have in common.
 *
 * The string branch exists because {@link scorerSequence} deliberately keeps a
 * BMP-only string as a string: indexing one yields a fresh single-character
 * string per position, so `s1[i] === s2[i]` allocates twice and then compares
 * two heap values. `charCodeAt` compares two integers instead. Both inputs
 * share a representation by the time they reach here, so the question is asked
 * once for the whole scan rather than per position.
 */
export function commonPrefix(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  const limit = Math.min(s1.length, s2.length)
  let i = 0

  if (typeof s1 === 'string' && typeof s2 === 'string') {
    while (i < limit && s1.charCodeAt(i) === s2.charCodeAt(i)) i++
    return i
  }

  while (i < limit && s1[i] === s2[i]) i++
  return i
}

/** Number of trailing elements `s1` and `s2` have in common. See {@link commonPrefix}. */
export function commonSuffix(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  const limit = Math.min(s1.length, s2.length)
  const end1 = s1.length - 1
  const end2 = s2.length - 1
  let i = 0

  if (typeof s1 === 'string' && typeof s2 === 'string') {
    while (i < limit && s1.charCodeAt(end1 - i) === s2.charCodeAt(end2 - i)) i++
    return i
  }

  while (i < limit && s1[end1 - i] === s2[end2 - i]) i++
  return i
}
