/**
 * Port of `rapidfuzz.process` (see `src/rapidfuzz/process_py.py`).
 *
 * `cdist` and `cpdist` are `scoreMatrix` and `scorePairs` here, and return
 * typed arrays rather than NumPy ones — see `_scoreArray.ts` for what replaced
 * `dtype`.
 */
import {
  assertNotPreparedHandle,
  callScorer,
  callScorerBare,
  isNone,
  isBuiltInScorer,
  isPreparedHandle,
  isSequence,
  NO_OPTIONS,
  PREPARE_CHOICE,
  PREPARED_CHOICE_HANDLE,
  PREPARED_CHOICES,
  PREPARED_QUERY_HANDLE,
  prepareScorerOf,
  scorerFlagsOf,
  type ErasedScorer,
  type MaybeSequence,
  type ChoicePreparer,
  type PreparedScore,
  type Processor,
  type Sequence,
} from './_common.js'
import {
  allocateScores,
  buildScoreMatrix,
  roundHalfAwayFromZero,
  scoreArrayFactory,
  type ScoreArray,
  type ScoreArrayKind,
  type ScoreArrayOf,
  type ScoreMatrix,
} from './_scoreArray.js'
import { ratio, wRatio } from './fuzz.js'

/** What `into` defaults to: a double, which loses no score to its store. */
const DEFAULT_SCORE_KIND: ScoreArrayKind = 'f64'

/**
 * A scorer these functions can call: two inputs plus options, returning a
 * number.
 *
 * Deliberately erased, and the same type {@link callScorer} takes. Naming the
 * options this module supplies — `processor`, `scoreCutoff`, `scoreHint` —
 * would look more informative and be wrong in both directions: it does not
 * describe a scorer's own options, and because every one of those is optional
 * it refused any scorer whose options were unrelated, as a weak type sharing no
 * property with it. A third-party `(a, b, options?: { caseSensitive?: boolean })`
 * is a scorer these functions call correctly, so the type has to admit it.
 */
export type SearchScorer = ErasedScorer

/**
 * Choices may be a list, a `Map`, any other iterable (including a generator),
 * or a plain object — mirroring Python's list / dict / iterable support.
 *
 * A bare string is refused, where upstream enumerates its characters. That
 * follows from `for c in "abc"` rather than from anything `process` decided,
 * and here it is a mistake with no other reading: `extract(q, 'abc')` is
 * `['abc']` written wrong, and enumerating it scores three one-character
 * candidates instead of saying so. The type still admits one — `string`
 * satisfies `Iterable<string>` and TypeScript cannot subtract it from a union —
 * so the refusal is {@link entriesOf}'s, at run time.
 */
export type Choices<T> =
  | readonly T[]
  | ReadonlyMap<unknown, T>
  | Iterable<T>
  | Readonly<Record<string, T>>

/**
 * One scored choice.
 *
 * `key` is the index for a list or any other iterable, the map key for a `Map`,
 * and the property name for a plain object; the overloads below pick the right
 * one from the shape of `choices`.
 *
 * Upstream returns the positional tuple `(choice, score, key)`, which Python
 * unpacks by assignment. Named fields are the equivalent here: `result.score`
 * says what `result[1]` only implies, and reading one field does not mean
 * naming the two you did not want.
 */
export interface ExtractResult<T, K = unknown> {
  readonly choice: T
  readonly score: number
  readonly key: K
}

/**
 * The properties of a plain-object `choices` that `Object.keys` walks.
 *
 * Symbol-keyed ones are not among them, so they are excluded here rather than
 * only from the key type: taking the value type as `C[keyof C]` would put a
 * symbol-keyed property's type into `choice` for a value the loop never
 * reaches.
 */
type ObjectProperty<C> = Extract<keyof C, string | number>

/**
 * The keys of a plain-object `choices`, as `Object.keys` reports them at
 * runtime. Numeric keys are stringified — `{ 1: 'a' }` is walked as `'1'`, not
 * `1` — so the template literal is what keeps the type honest.
 */
type ObjectKey<C> = `${ObjectProperty<C>}`

/** The values of a plain-object `choices` that {@link ObjectKey} names. */
type ObjectValue<C> = C[ObjectProperty<C>]

export interface SearchOptions {
  /** Applied to the query and to every choice. Defaults to none. */
  readonly processor?: Processor | undefined
  /**
   * Defaults to {@link wRatio} for `extract*` and {@link ratio} for
   * `scoreMatrix`/`scorePairs`. To give the scorer its own options, bake them in with
   * `configure` — `scorer: configure(levenshteinDistance, { weights })`.
   */
  readonly scorer?: SearchScorer | undefined
  /** Results worse than this are dropped. Defaults to the scorer's worst score. */
  readonly scoreCutoff?: number | undefined
  /** Performance hint forwarded to built-in scorers; it never changes results. */
  readonly scoreHint?: number | undefined
}

export interface ExtractOptions extends SearchOptions {
  /** Maximum number of results. `null` returns every match. Defaults to 5. */
  readonly limit?: number | null | undefined
}

/**
 * Normalise the three accepted `choices` shapes into `[key, choice]` pairs.
 *
 * No array fast path, because there is nothing left to make fast: all four
 * callers test `Array.isArray` themselves and run a plain indexed loop, which
 * is worth 1.25x precisely because it keeps the choice out of a generator and
 * out of a tuple. An array reaching here anyway would still come out right —
 * {@link isIterable} admits one, and the position it counts is the index — so
 * this is a branch removed, not a case dropped.
 */
function* entriesOf<T>(choices: Choices<T>): Generator<[unknown, T]> {
  // See {@link Choices}: a string is admitted by the type and is never meant.
  if (typeof choices === 'string') {
    throw new TypeError('choices must be a collection, not a single string')
  }

  if (isMapLike<unknown, T>(choices)) {
    yield* choices.entries()
    return
  }

  // Any other iterable (a generator, a Set) is enumerated like a list.
  // Maps are already handled above, so this only sees value iterables.
  if (isIterable(choices)) {
    let i = 0
    for (const choice of choices) {
      yield [i, choice]
      i++
    }
    return
  }

  for (const key of Object.keys(choices)) {
    yield [key, Reflect.get(choices, key)]
  }
}

/**
 * Whether `choices` is a map: the check the public `ReadonlyMap` implies.
 *
 * `instanceof Map` would be a narrower question than the type asks — a `Map`
 * from another realm fails it while satisfying `ReadonlyMap` exactly — and
 * failing it is not a type error but a wrong score: the map falls through to
 * {@link isIterable}, where every `[key, value]` pair is scored *as* the choice
 * and the real keys are replaced by positions.
 *
 * `get` is what separates a map from a set, whose `entries()` yields
 * `[value, value]`; upstream asks the same question of a Python mapping as
 * `hasattr(choices, "items")`, which is likewise structural — its docs promise
 * a pandas `Series` works, and a `Series` is not a `dict`.
 */
function isMapLike<K, V>(value: unknown): value is ReadonlyMap<K, V> {
  if (typeof value !== 'object' || value === null) return false

  return (
    typeof Reflect.get(value, 'get') === 'function' &&
    typeof Reflect.get(value, 'has') === 'function' &&
    typeof Reflect.get(value, 'entries') === 'function' &&
    typeof Reflect.get(value, 'size') === 'number'
  )
}

function isIterable<T>(choices: Choices<T>): choices is Iterable<T> {
  // Only the object case is left: `entriesOf` refuses a string before this,
  // and `Reflect.get` throws on a primitive rather than boxing it.
  if (typeof choices !== 'object' || choices === null) return false
  return typeof Reflect.get(choices, Symbol.iterator) === 'function'
}

/**
 * The two options every `prepare*` entry point bakes in, because they decide
 * what a prepared operand may hold.
 *
 * Shared by {@link prepareChoices}, {@link prepareQuery} and
 * {@link prepareChoice}, which is why neither field says *which* half it is
 * applied to. The rest of a call — the cutoff, the hint, `limit` — stays where
 * it was, because none of them changes what preparing produces.
 */
export interface PrepareOptions {
  /**
   * Applied once, when the operand is prepared, instead of once per later call.
   * Whatever this is paired with has to have been prepared with the same one.
   */
  readonly processor?: Processor | undefined
  /**
   * The scorer this is prepared for. Defaults to {@link wRatio}, as `extract*`
   * does, and every later call has to name the same one.
   */
  readonly scorer?: SearchScorer | undefined
}

/**
 * Choices normalised, processed and scorer-prepared once, for reuse across
 * queries. Built by {@link prepareChoices} and passed to `extract*` in place of
 * the collection it was built from.
 *
 * What it exposes is what a caller can use: how many choices survived, which
 * ones, under which keys, and what it was prepared for. What it does not expose
 * is the prepared state itself — that is the scorer's, its shape is the
 * scorer's business, and it is positionally tied to `values` in a way nothing
 * outside this module can be expected to maintain. It lives in
 * {@link indexPayloads} instead, which is also what {@link isIndex} asks.
 *
 * The brand is the same claim at the type level: it is keyed by a symbol no
 * other module can name, so an object literal cannot be passed where an index
 * is expected and then silently scored as a plain-object collection.
 *
 * The value and both arrays are frozen, so the `readonly` above holds at run
 * time too. Read them freely; to change the choices, build another index.
 */
export interface PreparedChoiceIndex<T, K> {
  readonly [PREPARED_CHOICES]: true
  /** The scorer this index was built for. */
  readonly scorer: SearchScorer
  /** The processor already applied to every choice. */
  readonly processor: Processor | null
  /** The choices, in enumeration order, with missing values dropped. */
  readonly values: readonly T[]
  /** Their keys, positionally matching {@link values}. */
  readonly keys: readonly K[]
  /** How many choices the index holds, after dropping missing values. */
  readonly size: number
}

/**
 * Each index's prepared choices, positionally matching its `values`.
 *
 * Off the value for two reasons. It is internal scorer state — converted code
 * points, a tokenisation memo, the choice untouched — that no caller has a use
 * for, and `readonly` in the type is a promise to TypeScript that JavaScript
 * never hears. And membership is a stronger claim than shape: it says this
 * module built the thing, which is what the scoring loops are actually
 * trusting when they pair `prepared[i]` with `values[i]`.
 */
const indexPayloads = new WeakMap<object, readonly unknown[]>()

/**
 * The prepared choices of an index, or a `TypeError` naming what went wrong.
 *
 * The failure is not hypothetical, which is why it is a message rather than an
 * assertion: `{ ...index }` copies the brand — a spread copies symbol-keyed own
 * properties — and copies `values` and `keys`, but cannot copy an entry in a
 * table it cannot reach. Such a copy is branded, so it type-checks and reaches
 * the scoring loops, and without this it would be scored against prepared state
 * that is not there. Saying so beats scoring it wrong.
 */
function payloadOf(index: PreparedChoiceIndex<unknown, unknown>): readonly unknown[] {
  const prepared = indexPayloads.get(index)
  if (prepared === undefined) {
    throw new TypeError('a prepared choice index cannot be copied; pass the original')
  }
  return prepared
}

/**
 * Prepare `choices` once so a run of queries does not redo the per-choice work.
 *
 * `extract*` prepares the *query* and streams the choices, which is right for
 * one call and wasteful for a hundred over the same list: the processor runs on
 * every choice every time, every choice is converted every time, and for the
 * token scorers every choice is split, deduplicated, sorted and rejoined every
 * time. An index moves all of that to one pass.
 *
 * ```ts
 * const index = prepareChoices(titles, { scorer: tokenSortRatio })
 * for (const query of queries) extractOne(query, index)
 * ```
 *
 * Three things follow from what it holds, and all three are checked rather than
 * documented and hoped for:
 *
 * - **The scorer is baked in**, because what a prepared choice holds is the
 *   scorer's own business. `extract*` on this index either names the same
 *   scorer or names none.
 * - **The processor is baked in** and already applied. `extract*` still needs
 *   it for the query, and takes it from here rather than making the caller
 *   repeat it.
 * - **The cutoff, the hint and the limit are not**, because none of them
 *   changes a choice. They stay per-call, where they were.
 *
 * It is a snapshot, and the two things that follow from that are the caller's
 * to keep. **The choices must not change after it is built** — not the
 * collection, and not the contents of a choice that is itself mutable, since
 * rewriting an element of an array choice leaves the prepared state describing
 * what that choice used to be while the result still names the array. And **the
 * processor must be deterministic**: `extract` runs it once per choice per
 * query, an index runs it once per choice, so one that counts its calls is
 * entitled to disagree. Neither can be enforced without deep-freezing or
 * copying user values, and both would change what the results are.
 *
 * Missing choices are dropped as they are found rather than at every query,
 * which is why `values` can be shorter than the collection. Nothing else is
 * observable: `extract*` skips them anyway, and the position they held only
 * ever mattered against another position.
 *
 * The index holds the choices, so it keeps them alive; it also memoises derived
 * forms as queries ask for them, so it grows a little on first use and then
 * settles. Build one per list you query repeatedly, not one per call — a single
 * `extract` over a fresh index is slower than passing the list.
 */
export function prepareChoices<K, V>(
  choices: ReadonlyMap<K, V>,
  options?: PrepareOptions,
): PreparedChoiceIndex<V, K>
export function prepareChoices<T>(
  choices: readonly T[] | Iterable<T>,
  options?: PrepareOptions,
): PreparedChoiceIndex<T, number>
export function prepareChoices<C extends Readonly<Record<string, unknown>>>(
  choices: C,
  options?: PrepareOptions,
): PreparedChoiceIndex<ObjectValue<C>, ObjectKey<C>>
export function prepareChoices<T>(
  choices: Choices<T>,
  options?: PrepareOptions,
): PreparedChoiceIndex<T, unknown>
export function prepareChoices<T>(
  choices: Choices<T>,
  options: PrepareOptions = {},
): PreparedChoiceIndex<T, unknown> {
  assertNotPreparedHandle(options.scorer)
  const scorer = options.scorer ?? wRatio
  const processor = options.processor ?? null

  const values: T[] = []
  const keys: unknown[] = []
  const prepared: unknown[] = []

  // The same two shapes the scoring loops separate, for the same reason: a list
  // is what this is given in practice, and `entriesOf` puts a generator frame
  // and a tuple between every choice and this loop.
  if (Array.isArray(choices)) {
    for (let key = 0; key < choices.length; key++) {
      const choice = choices[key]
      if (isNone(choice)) continue
      values.push(choice)
      keys.push(key)
      prepared.push(processor != null ? processor(queryAsSequence(choice)) : choice)
    }
  } else {
    for (const [key, choice] of entriesOf(choices)) {
      if (isNone(choice)) continue
      values.push(choice)
      keys.push(key)
      prepared.push(processor != null ? processor(queryAsSequence(choice)) : choice)
    }
  }

  const choicePreparer = choicePreparerOf(scorer)
  if (choicePreparer !== null) {
    for (let i = 0; i < prepared.length; i++) prepared[i] = choicePreparer(prepared[i])
  }

  const index: PreparedChoiceIndex<T, unknown> = {
    [PREPARED_CHOICES]: true,
    scorer,
    processor,
    values,
    keys,
    size: values.length,
  }
  // `readonly` is a promise to the type checker and nothing else, and these four
  // are the last publicly reachable things that can drift out of step with the
  // hidden payload: `values[i]` names what `prepared[i]` was built from, and
  // `scorer` names what it was built for. A write to either is scored correctly
  // and reported wrongly, silently. Freezing is what the type already claims.
  //
  // The value is free to freeze. The two arrays are not: their elements move to
  // `PACKED_FROZEN_ELEMENTS`, whose loads measured 1.68x in a read loop, which
  // shows end to end wherever a scorer is cheap enough for two loads to matter
  // and every choice is admitted — 1.04-1.05x on a drained `extractIter` over
  // `ratio`, and nothing at all on the default `wRatio` at any limit. Paid.
  //
  // That figure is also the ceiling for buying it back. Leaving both unfrozen
  // and scoring off private copies re-measured at 1.05x on the same drained
  // `extractIter`, 1.03x on `levenshtein` and `tokenSortRatio`, and 1.00x on
  // every `extractOne` and every `extract(limit: 5)` — so a scheme that keeps
  // the public arrays frozen and duplicates them for the loops would pay two
  // arrays per index, permanently, for at most that.
  Object.freeze(values)
  Object.freeze(keys)
  Object.freeze(index)
  indexPayloads.set(index, prepared)
  return index
}

/**
 * The per-choice hook a scorer offers, or `null` for one that offers none.
 *
 * A scorer that registers no factory has none — a third-party scorer, or one
 * `configure` refused to prepare because a processor was baked into it. So does
 * a scorer that caches a query and still wants its choices as they came. Both
 * get an index; it holds the processed choices and stops there.
 */
function choicePreparerOf(scorer: SearchScorer): ChoicePreparer | null {
  return prepareScorerOf(scorer)?.[PREPARE_CHOICE] ?? null
}

/**
 * The two bounds a prepared handle still takes on each call.
 *
 * Deliberately just these two. The scorer and the processor are baked in — they
 * decide what a prepared operand *is*, so accepting either per call could only
 * mean a handle that had prepared for the wrong one — and the type is what says
 * so: `pq(choice, { scorer })` is an excess property, not a run time refusal.
 *
 * Absence is spelled `undefined` here rather than `null`, because these are the
 * caller's options and a scorer's own options spell it that way. Both of those
 * are what let a scorer with no prepared form be handed this object as it
 * stands: a handle called with options passes the caller's own object through,
 * and one called without passes no options argument at all. `pq(choice)` and
 * `pq(choice, {})` are therefore different calls, exactly as they would be
 * written by hand.
 */
export interface PreparedCallOptions {
  /** Bound on the returned score, in the scorer's own convention. */
  readonly scoreCutoff?: number | undefined
  /** Performance hint forwarded to built-in scorers; it never changes results. */
  readonly scoreHint?: number | undefined
}

/**
 * A query prepared for one scorer, callable against choices.
 *
 * Returned by {@link prepareQuery}. Frozen, reusable, and safe to hold for as
 * long as the query is worth scoring against.
 */
export interface PreparedQuery {
  (choice: Sequence | PreparedChoice, options?: PreparedCallOptions): number
  readonly [PREPARED_QUERY_HANDLE]: true
  /** The scorer this query was prepared for. */
  readonly scorer: SearchScorer
  /** The processor already applied to it, and owed to whatever it scores. */
  readonly processor: Processor | null
}

/**
 * A choice prepared for one scorer, callable against queries.
 *
 * Returned by {@link prepareChoice}. Note the operand order this implies and
 * does *not* change: `preparedChoice(query)` is `scorer(query, choice)`, the
 * same way round as everywhere else in this package. A handle holds the side it
 * holds; it does not swap the two.
 */
export interface PreparedChoice {
  (query: Sequence | PreparedQuery, options?: PreparedCallOptions): number
  readonly [PREPARED_CHOICE_HANDLE]: true
  /** The scorer this choice was prepared for. */
  readonly scorer: SearchScorer
  /** The processor already applied to it, and owed to whatever it scores. */
  readonly processor: Processor | null
}

/** What a prepared query holds: the processed sequence and the scorer's state. */
interface QueryPayload {
  readonly processed: Sequence
  readonly preparedScore: PreparedScore | null
}

/**
 * What a prepared choice holds.
 *
 * Both forms, because they go to different places. `scored` is what the
 * scorer's own hook made of the choice — a converted sequence, a tokenisation
 * memo — and only a `PreparedScore` from the same factory understands it;
 * `processed` is the plain sequence, which is the only thing that may be handed
 * to the scorer's public signature. Feeding the first where the second belongs
 * is silently wrong rather than an error: `levenshteinDistance` reads a record
 * with no `length` as a one-element sequence and answers `4` where the truth is
 * `0`.
 */
interface ChoicePayload {
  readonly processed: Sequence
  readonly scored: unknown
}

/**
 * Provenance for the handles, on the same footing as {@link indexPayloads}.
 *
 * The fast path never reads either table — a handle called on a raw operand
 * runs entirely out of its own closure. These exist for the composed calls,
 * where the operand is a handle someone else built and "someone else" has to
 * include a forger: `Object.assign(() => 123, realHandle)` copies the brand and
 * both identity fields, and cannot copy an entry in a table keyed by the
 * original.
 */
const queryPayloads = new WeakMap<object, QueryPayload>()
const choicePayloads = new WeakMap<object, ChoicePayload>()

function queryPayloadOf(handle: PreparedQuery): QueryPayload {
  const payload = queryPayloads.get(handle)
  if (payload === undefined) {
    throw new TypeError('a prepared query cannot be copied; pass the original')
  }
  return payload
}

function choicePayloadOf(handle: PreparedChoice): ChoicePayload {
  const payload = choicePayloads.get(handle)
  if (payload === undefined) {
    throw new TypeError('a prepared choice cannot be copied; pass the original')
  }
  return payload
}

function isPreparedQuery(value: unknown): value is PreparedQuery {
  return typeof value === 'function' && PREPARED_QUERY_HANDLE in value
}

function isPreparedChoice(value: unknown): value is PreparedChoice {
  return typeof value === 'function' && PREPARED_CHOICE_HANDLE in value
}

/**
 * Refuse two handles that were not prepared the same way.
 *
 * Both tests are identity, in both directions, `null` included — stricter than
 * {@link checkIndex}, which tolerates an absent processor because the index is
 * the one supplying it. Here neither side supplies anything to the other: a
 * query normalised by a processor, scored against a choice that was not, is a
 * plausible number with nothing to see. Two scorers that compute the same thing
 * still prepare an operand differently, so semantic equality would not do even
 * if it could be asked.
 */
function checkPairing(
  scorer: SearchScorer,
  processor: Processor | null,
  other: PreparedQuery | PreparedChoice,
  noun: string,
): void {
  if (other.scorer !== scorer) {
    throw new TypeError(`scorer differs from the one this prepared ${noun} was built for`)
  }
  if (other.processor !== processor) {
    throw new TypeError(
      `processor differs from the one this prepared ${noun} was built for`,
    )
  }
}

/**
 * The one place a prepared pair is scored, whichever half was prepared.
 *
 * Deliberately not {@link score}, which reads its hint off the state because an
 * `extract` run has one hint for every choice, where a handle takes one per
 * call. Threading a per-call hint through `score` would put an extra argument
 * in nine hot-loop call sites, to chase a difference the comment above it
 * already measured at 0.99-1.03x with no consistent sign.
 */
function scorePreparedPair(
  scorer: SearchScorer,
  query: Sequence,
  preparedScore: PreparedScore | null,
  choice: unknown,
  options: PreparedCallOptions | undefined,
): number {
  if (preparedScore !== null) {
    return preparedScore(choice, options?.scoreCutoff ?? null, options?.scoreHint ?? null)
  }

  // A handle called with no options is replacing `scorer(q, c)`, so it makes
  // that call — two arguments, not three with two `undefined` fields. See
  // {@link callScorerBare}.
  if (options === undefined) return callScorerBare(scorer, query, choice)

  // And when there are options, the scorer gets the caller's object itself, not
  // a rebuilt `{ scoreCutoff, scoreHint }` — for the same reason. Rebuilding
  // turns `pq(choice, {})` into a call carrying two `undefined` fields, which
  // `Object.keys`, a getter, or an identity test can all see, and this path
  // exists precisely for the third-party scorer that might look.
  return callScorer(scorer, query, choice, options)
}

/**
 * Prepare a query once, to score it against many choices.
 *
 * The half `extract*` already prepares, handed to a caller who is not going
 * through `extract*` — a custom ranker, a join, a loop over pairs. The scorer
 * and the processor are baked in; the cutoff and the hint stay per call:
 *
 * ```ts
 * const query = prepareQuery('new york mets', { scorer: tokenSortRatio })
 * const scores = titles.map((title) => query(title))
 * ```
 *
 * It composes with {@link prepareChoice}, and that is where the two together
 * are worth the most — `query(preparedChoice)` pays for neither half. Both
 * orders are the same call: the operand order a handle scores in is always
 * `scorer(query, choice)`, whichever of the two you happen to be holding.
 *
 * Two ways this is stricter than `extract*`, both deliberate. A missing
 * operand — `null`, `undefined`, `NaN` — is refused rather than dropped, here
 * and on every later call: dropping a choice is a decision `extract` can make
 * because it is producing a list, and a single score has no such answer to give.
 * And the scorer may not be a prepared handle, which would otherwise type-check
 * and score every pair wrongly in silence.
 *
 * **A handle is a snapshot, and the handle being frozen does not freeze what it
 * was built from.** `Object.freeze` here seals the three properties this
 * function returns; it does nothing to an array operand, or to a mutable
 * sequence a processor returned. So the caller keeps the same two rules an
 * index keeps: the operand must not change after preparation — rewriting an
 * element leaves the prepared state describing what the operand used to be,
 * while the handle still names the array — and the processor must be
 * deterministic, since it runs once here and once per raw operand later.
 * Rebuild the handle instead. Neither can be enforced without deep-freezing or
 * copying the caller's values, and both would change what the results are.
 */
export function prepareQuery(
  query: Sequence,
  options: PrepareOptions = {},
): PreparedQuery {
  assertNotPreparedHandle(options.scorer)
  const scorer = options.scorer ?? wRatio
  const processor = options.processor ?? null
  const processed = processOperand(query, processor)
  const factory = prepareScorerOf(scorer)
  const preparedScore = factory !== null ? factory(processed, NO_OPTIONS) : null

  const call = (
    choice: Sequence | PreparedChoice,
    callOptions?: PreparedCallOptions,
  ): number => {
    if (isPreparedChoice(choice)) {
      checkPairing(scorer, processor, choice, 'choice')
      const payload = choicePayloadOf(choice)
      // The ternary cannot take its second arm while `preparedScore` is
      // non-null: the identity test above means both sides agree about whether
      // this scorer has a factory, and only a factory produces a `scored` that
      // differs from `processed`. Written out anyway, because it makes "a
      // hooked payload never reaches a public scorer signature" a property of
      // these eight lines rather than a deduction across three modules — and
      // the second arm is reached in earnest by every scorer with no factory.
      return scorePreparedPair(
        scorer,
        processed,
        preparedScore,
        preparedScore !== null ? payload.scored : payload.processed,
        callOptions,
      )
    }

    return scorePreparedPair(
      scorer,
      processed,
      preparedScore,
      processOperand(choice, processor),
      callOptions,
    )
  }

  // A literal with no contextual type widens `true` to `boolean`, which is not
  // the brand's type — so the annotation is on a declaration of its own. An
  // annotation, not an assertion: it is checked against the initialiser.
  const brand: { readonly [PREPARED_QUERY_HANDLE]: true } = {
    [PREPARED_QUERY_HANDLE]: true,
  }
  const handle: PreparedQuery = Object.freeze(
    Object.assign(call, brand, { scorer, processor }),
  )
  queryPayloads.set(handle, { processed, preparedScore })
  return handle
}

/**
 * Prepare a choice once, to score many queries against it.
 *
 * The mirror of {@link prepareQuery}, and the half `extract*` never had: a
 * choice, holding whatever its scorer wants a choice to hold.
 *
 * ```ts
 * const choice = prepareChoice(title, { scorer: tokenSortRatio })
 * const scores = queries.map((query) => choice(query))
 * ```
 *
 * `choice(query)` is `scorer(query, choice)` — the operand order does not
 * follow which half is written first. That matters for the scorers where it is
 * observable: weighted Levenshtein with an insertion cost that differs from its
 * deletion cost is not symmetric, and neither is an arbitrary third-party
 * scorer.
 *
 * Where this pays is composition and the processor: `preparedChoice(query)`
 * against a raw query takes the scorer's ordinary path with the processor
 * already spent, while `preparedQuery(preparedChoice)` pays for neither half.
 * A prepared choice on its own, scored against raw queries with no processor,
 * is not much cheaper than calling the scorer — the query is the side these
 * scorers cache.
 *
 * The snapshot rule {@link prepareQuery} states holds here too, and one degree
 * more sharply: the choice-side state is built now, while the reversed one a
 * symmetric scorer uses is built on the first raw query. A choice mutated
 * between those two moments would leave the handle holding two states of the
 * same operand taken at different times.
 */
export function prepareChoice(
  choice: Sequence,
  options: PrepareOptions = {},
): PreparedChoice {
  assertNotPreparedHandle(options.scorer)
  const scorer = options.scorer ?? wRatio
  const processor = options.processor ?? null
  const processed = processOperand(choice, processor)
  // Eagerly, unlike the reversal below, and measured: building this on first
  // composed use instead cost 1.09-1.13x on composed `tokenSortRatio` over two
  // runs — the strongest number this API has — and 1.03-1.06x on composed
  // `wRatio`, against a ±3.5% noise floor read off the column neither variant
  // touches. What it bought was ~195 bytes a handle, and only for a handle
  // whose choice-side state is never used: over 2000 prepared choices it saved
  // 0.39 MB built-but-never-called and 0.38 MB of 6.1 MB scored against raw
  // queries, while *costing* 0.14-0.19 MB once anything composed. The
  // asymmetry is the point — a memo has to be paid for on the path that reads
  // it, and this one is read by the path worth protecting.
  const choicePreparer = choicePreparerOf(scorer)
  const scored = choicePreparer !== null ? choicePreparer(processed) : processed

  // Scored against a raw query, a symmetric scorer can be given the *choice* as
  // its prepared query and the query as its choice: the pair is the same pair,
  // and the expensive half is now the one held. It is what makes this handle
  // worth having on its own rather than only in composition — 0.54-0.81x
  // against the direct path on every scorer family measured.
  //
  // Built on first use, not at construction. A choice that is only ever
  // composed with a prepared query never scores through this, and building it
  // eagerly cost that path 2.5-8.3%.
  //
  // `null` means "no reversal available", `undefined` means "not asked yet".
  let reversed: PreparedScore | null | undefined = undefined
  const reversedScore = (): PreparedScore | null => {
    if (reversed === undefined) {
      const factory = prepareScorerOf(scorer)
      // No `isBuiltInScorer` test, which would be dead: a factory is only ever
      // recorded alongside the built-in registration, so having one already
      // proves this package built the scorer. That matters, because
      // `scorerFlagsOf` answers `symmetric: true` for anything it does not
      // know — a third-party scorer would otherwise be claiming a symmetry it
      // never promised, and reaching here at all is what that test guarded.
      reversed =
        factory !== null && scorerFlagsOf(scorer).symmetric
          ? factory(processed, NO_OPTIONS)
          : null
    }
    return reversed
  }

  const call = (
    query: Sequence | PreparedQuery,
    callOptions?: PreparedCallOptions,
  ): number => {
    if (isPreparedQuery(query)) {
      checkPairing(scorer, processor, query, 'query')
      // Before the handle is invoked, not after. A forged query carries the
      // brand and passes both identity tests, and calling it would run the
      // forger's body and hand back whatever it liked — so provenance is
      // established while refusing is still possible.
      queryPayloadOf(query)
      return query(handle, callOptions)
    }

    // Note what is *not* here: building the scorer's factory per call, around
    // the query, which is what an earlier sketch did. That measured 1.69x on
    // `ratio` and 1.31x on `levenshteinDistance` — a prepared query pays for
    // itself over many choices, and this call has exactly one. The reversal
    // above is the same idea done once per handle instead of once per call.
    const processedQuery = processOperand(query, processor)
    const prepared = reversedScore()
    return scorePreparedPair(
      scorer,
      processedQuery,
      prepared,
      prepared !== null ? processedQuery : processed,
      callOptions,
    )
  }

  const brand: { readonly [PREPARED_CHOICE_HANDLE]: true } = {
    [PREPARED_CHOICE_HANDLE]: true,
  }
  const handle: PreparedChoice = Object.freeze(
    Object.assign(call, brand, { scorer, processor }),
  )
  choicePayloads.set(handle, { processed, scored })
  return handle
}

/**
 * Validate a raw operand and apply the processor, if there is one.
 *
 * Validated even when there is no processor to run, which is what makes a
 * missing operand a refusal rather than a wrong answer several frames later:
 * `levenshteinDistance(someFunction, 'abcd')` returns `4` today, having read
 * the function's `length`.
 */
function processOperand(value: unknown, processor: Processor | null): Sequence {
  const sequence = queryAsSequence(value)
  return processor !== null ? processor(sequence) : sequence
}

/**
 * Whether `choices` is an index rather than a collection.
 *
 * The brand is checked, not the shape: every field an index exposes is one a
 * plain object could have, and a collection of choices is exactly where an
 * object with arbitrary fields turns up. Provenance is then established a
 * second time, by {@link payloadOf} — this says "meant as an index", that says
 * "built here".
 */
function isIndex<T>(
  choices: Choices<T> | PreparedChoiceIndex<T, unknown>,
): choices is PreparedChoiceIndex<T, unknown> {
  return typeof choices === 'object' && choices !== null && PREPARED_CHOICES in choices
}

/**
 * Refuse a call that disagrees with the index it names, or names an index this
 * module did not build. A collection is not an index and has nothing to check.
 *
 * Separate from {@link prepare} because one path has to run it without ever
 * preparing anything: `extract` returns `[]` for `limit <= 0` before it has a
 * query, a processor call or a scorer state, and a call that is wrong about its
 * index is wrong whether or not it asked for any results back. Two identity
 * tests and a `WeakMap` lookup is the whole cost, so that path pays nothing it
 * would not have paid at `limit: 1`.
 */
function checkIndex<T>(
  options: SearchOptions,
  choices: Choices<T> | PreparedChoiceIndex<T, unknown>,
): void {
  if (!isIndex(choices)) return

  // Both are identity tests on purpose: two scorers that do the same thing
  // still prepare a choice differently, and an index cannot say whether a
  // second processor would agree with the one it already applied.
  if (options.scorer !== undefined && options.scorer !== choices.scorer) {
    throw new TypeError('scorer differs from the one this index was prepared for')
  }
  if (options.processor != null && options.processor !== choices.processor) {
    throw new TypeError('processor differs from the one this index was prepared for')
  }
  // Provenance on the same footing as the two above: a copy is refused for what
  // it is, before anything asks it for prepared state it does not have.
  payloadOf(choices)
}

/** The scorer a call runs, which an index supplies when the options do not. */
function scorerFor<T>(
  options: SearchOptions,
  choices: Choices<T> | PreparedChoiceIndex<T, unknown>,
): SearchScorer {
  // Once per call, never per choice — and only on the options, since an index
  // was refused one at the point it was built.
  assertNotPreparedHandle(options.scorer)
  if (options.scorer !== undefined) return options.scorer
  return isIndex(choices) ? choices.scorer : wRatio
}

interface Prepared {
  scorer: SearchScorer
  optimalScore: number
  lowestScoreWorst: boolean
  /**
   * The bound handed to the scorer, and `null` when the caller named none.
   *
   * Absence is passed on as absence rather than as the scorer's worst score.
   * The two used to be the same field, which meant inventing a number that
   * admits everything and handing it over as though a caller had asked for it:
   * a distance scorer was told to bound at `2**63`. Every cutoff helper in
   * `_common.ts` already reads `null` as "no cutoff", and a third-party scorer
   * reading `options.scoreCutoff` now sees the same thing a direct call with no
   * cutoff shows it.
   */
  scoreCutoff: number | null
  /**
   * The score a result has to reach to be kept, which is the scorer's worst
   * score when there is no cutoff — `Infinity` for a distance, `0` for a
   * similarity. Only ever compared against, never passed on.
   */
  admits: number
  query: Sequence
  processor: Processor | null
  scoreHint: number | null
  preparedScore: PreparedScore | null
  builtIn: boolean
}

function prepare<T>(
  query: MaybeSequence,
  options: SearchOptions,
  choices: Choices<T> | PreparedChoiceIndex<T, unknown>,
): Prepared | null {
  const scorer = scorerFor(options, choices)
  const index = isIndex(choices) ? choices : null
  // Before the `isNone` return below, so a bad index is refused whatever the
  // query is.
  checkIndex(options, choices)
  // A handle here is a caller who expected `extract(preparedQuery, choices)` to
  // work. It falls through to `queryAsSequence` otherwise, where "expected a
  // string or an array-like sequence" is true and says nothing about what to do
  // instead. Once per call, not per choice — `queryAsSequence` runs per choice
  // whenever a processor is set.
  if (isPreparedHandle(query)) {
    throw new TypeError(
      'a prepared query or choice cannot be used as a query here; call the handle instead',
    )
  }
  const { worstScore, optimalScore } = scorerFlagsOf(scorer)
  const lowestScoreWorst = optimalScore > worstScore

  if (isNone(query)) return null

  // An index took its processor from `prepareChoices`, and applied it there.
  const processor = options.processor ?? index?.processor ?? null
  const raw =
    processor != null ? processor(queryAsSequence(query)) : queryAsSequence(query)
  const factory = prepareScorerOf(scorer)
  const builtIn = isBuiltInScorer(scorer)
  const preparedScore = factory !== null ? factory(raw, NO_OPTIONS) : null
  return {
    scorer,
    optimalScore,
    lowestScoreWorst,
    scoreCutoff: options.scoreCutoff ?? null,
    admits: options.scoreCutoff ?? worstScore,
    query: raw,
    // The choices in an index are processed already; only the query above was
    // still owed it. Holding `null` is what keeps {@link score} one function.
    processor: index === null ? processor : null,
    scoreHint: options.scoreHint ?? null,
    preparedScore,
    builtIn,
  }
}

function queryAsSequence(query: unknown): Sequence {
  if (isSequence(query)) return query

  throw new TypeError('expected a string or an array-like sequence')
}

/**
 * Call the scorer the way upstream does: query, processed choice, cutoff.
 *
 * One function for both shapes, including an index — where `state.processor` is
 * always `null`, so the first test is known false before the loop starts.
 * Resolving both tests once per call instead, and running the indexed loops off
 * a hoisted `preparedScore`, was measured over 2000 choices on `ratio`,
 * `levenshteinDistance`, `tokenSortRatio` and `wRatio` across `extractOne`,
 * `extract(limit: 5)` and a drained `extractIter`: 0.99-1.03x, no consistent
 * sign, both orders. Four duplicated loops for nothing measurable.
 */
function score<T>(state: Prepared, choice: T, scoreCutoff: number | null): number {
  const processed =
    state.processor != null ? state.processor(queryAsSequence(choice)) : choice
  if (state.preparedScore !== null) {
    return state.preparedScore(processed, scoreCutoff, state.scoreHint)
  }

  // `null` is this module's spelling of "not given", and `PreparedScore` above
  // takes it — but a scorer's own options are a public type, and there absence
  // is spelled `undefined`. A third-party scorer typed against `ScorerOptions`
  // would otherwise be handed a value its own signature says it cannot get.
  return callScorer(state.scorer, state.query, processed, {
    scoreCutoff: scoreCutoff ?? undefined,
    scoreHint: state.scoreHint ?? undefined,
  })
}

/**
 * Yield an {@link ExtractResult} for every choice that passes `scoreCutoff`.
 *
 * Results arrive in the order the choices do, unsorted.
 */
export function extractIter<T, K>(
  query: MaybeSequence,
  choices: PreparedChoiceIndex<T, K>,
  options?: SearchOptions,
): Generator<ExtractResult<T, K>>
export function extractIter<K, V>(
  query: MaybeSequence,
  choices: ReadonlyMap<K, V>,
  options?: SearchOptions,
): Generator<ExtractResult<V, K>>
export function extractIter<T>(
  query: MaybeSequence,
  choices: readonly T[] | Iterable<T>,
  options?: SearchOptions,
): Generator<ExtractResult<T, number>>
export function extractIter<C extends Readonly<Record<string, unknown>>>(
  query: MaybeSequence,
  choices: C,
  options?: SearchOptions,
): Generator<ExtractResult<ObjectValue<C>, ObjectKey<C>>>
export function extractIter<T>(
  query: MaybeSequence,
  choices: Choices<T> | PreparedChoiceIndex<T, unknown>,
  options?: SearchOptions,
): Generator<ExtractResult<T>>
export function* extractIter<T>(
  query: MaybeSequence,
  choices: Choices<T> | PreparedChoiceIndex<T, unknown>,
  options: SearchOptions = {},
): Generator<ExtractResult<T>> {
  if (isIndex(choices)) {
    const state = prepare(query, options, choices)
    if (state === null) return

    const { values, keys } = choices
    const prepared = payloadOf(choices)
    for (let i = 0; i < values.length; i++) {
      const value = score(state, prepared[i], state.scoreCutoff)
      const admitted = state.lowestScoreWorst
        ? value >= state.admits
        : value <= state.admits
      if (admitted) yield { choice: values[i], score: value, key: keys[i] }
    }
    return
  }

  const state = prepare(query, options, choices)
  if (state === null) return

  // A list skips `entriesOf`, which would otherwise put a second generator
  // frame and a `[key, choice]` tuple between every choice and this loop.
  if (Array.isArray(choices)) {
    for (let key = 0; key < choices.length; key++) {
      const choice = choices[key]
      if (isNone(choice)) continue

      const value = score(state, choice, state.scoreCutoff)
      const admitted = state.lowestScoreWorst
        ? value >= state.admits
        : value <= state.admits
      if (admitted) yield { choice, score: value, key }
    }
    return
  }

  for (const [key, choice] of entriesOf(choices)) {
    if (isNone(choice)) continue

    const value = score(state, choice, state.scoreCutoff)
    const admitted = state.lowestScoreWorst
      ? value >= state.admits
      : value <= state.admits
    if (admitted) yield { choice, score: value, key }
  }
}

/**
 * Best match among `choices`, or `undefined` when nothing passes `scoreCutoff`.
 *
 * Tightens the cutoff as it goes and stops early on a perfect score, so it is
 * cheaper than `extract(..., { limit: 1 })`.
 */
export function extractOne<T, K>(
  query: MaybeSequence,
  choices: PreparedChoiceIndex<T, K>,
  options?: SearchOptions,
): ExtractResult<T, K> | undefined
export function extractOne<K, V>(
  query: MaybeSequence,
  choices: ReadonlyMap<K, V>,
  options?: SearchOptions,
): ExtractResult<V, K> | undefined
export function extractOne<T>(
  query: MaybeSequence,
  choices: readonly T[] | Iterable<T>,
  options?: SearchOptions,
): ExtractResult<T, number> | undefined
export function extractOne<C extends Readonly<Record<string, unknown>>>(
  query: MaybeSequence,
  choices: C,
  options?: SearchOptions,
): ExtractResult<ObjectValue<C>, ObjectKey<C>> | undefined
export function extractOne<T>(
  query: MaybeSequence,
  choices: Choices<T> | PreparedChoiceIndex<T, unknown>,
  options?: SearchOptions,
): ExtractResult<T> | undefined
export function extractOne<T>(
  query: MaybeSequence,
  choices: Choices<T> | PreparedChoiceIndex<T, unknown>,
  options: SearchOptions = {},
): ExtractResult<T> | undefined {
  if (isIndex(choices)) {
    const state = prepare(query, options, choices)
    if (state === null) return undefined

    let cutoff = state.scoreCutoff
    let admits = state.admits
    let result: ExtractResult<T> | undefined = undefined
    const { values, keys } = choices
    const prepared = payloadOf(choices)

    for (let i = 0; i < values.length; i++) {
      const value = score(state, prepared[i], cutoff)
      if (state.lowestScoreWorst) {
        if (value >= admits && (result === undefined || value > result.score)) {
          cutoff = value
          admits = value
          result = { choice: values[i], score: value, key: keys[i] }
        }
      } else if (value <= admits && (result === undefined || value < result.score)) {
        cutoff = value
        admits = value
        result = { choice: values[i], score: value, key: keys[i] }
      }

      if (value === state.optimalScore) break
    }
    return result
  }

  const state = prepare(query, options, choices)
  if (state === null) return undefined

  // The running best tightens both at once: `cutoff` is what the scorer is
  // told, `admits` is what the score is compared against. They differ only
  // until the first result, when there is no cutoff to tell the scorer about
  // and every score still has to clear the scorer's worst.
  let cutoff = state.scoreCutoff
  let admits = state.admits
  let result: ExtractResult<T> | undefined = undefined

  if (Array.isArray(choices)) {
    for (let key = 0; key < choices.length; key++) {
      const choice = choices[key]
      if (isNone(choice)) continue

      const value = score(state, choice, cutoff)
      if (state.lowestScoreWorst) {
        if (value >= admits && (result === undefined || value > result.score)) {
          cutoff = value
          admits = value
          result = { choice, score: value, key }
        }
      } else if (value <= admits && (result === undefined || value < result.score)) {
        cutoff = value
        admits = value
        result = { choice, score: value, key }
      }

      if (value === state.optimalScore) break
    }
    return result
  }

  for (const [key, choice] of entriesOf(choices)) {
    if (isNone(choice)) continue

    const value = score(state, choice, cutoff)

    if (state.lowestScoreWorst) {
      if (value >= admits && (result === undefined || value > result.score)) {
        cutoff = value
        admits = value
        result = { choice, score: value, key }
      }
    } else if (value <= admits && (result === undefined || value < result.score)) {
      cutoff = value
      admits = value
      result = { choice, score: value, key }
    }

    if (value === state.optimalScore) break
  }

  return result
}

/** The best `limit` matches, ordered best first. */
export function extract<T, K>(
  query: MaybeSequence,
  choices: PreparedChoiceIndex<T, K>,
  options?: ExtractOptions,
): ExtractResult<T, K>[]
export function extract<K, V>(
  query: MaybeSequence,
  choices: ReadonlyMap<K, V>,
  options?: ExtractOptions,
): ExtractResult<V, K>[]
export function extract<T>(
  query: MaybeSequence,
  choices: readonly T[] | Iterable<T>,
  options?: ExtractOptions,
): ExtractResult<T, number>[]
export function extract<C extends Readonly<Record<string, unknown>>>(
  query: MaybeSequence,
  choices: C,
  options?: ExtractOptions,
): ExtractResult<ObjectValue<C>, ObjectKey<C>>[]
export function extract<T>(
  query: MaybeSequence,
  choices: Choices<T> | PreparedChoiceIndex<T, unknown>,
  options?: ExtractOptions,
): ExtractResult<T>[]
export function extract<T>(
  query: MaybeSequence,
  choices: Choices<T> | PreparedChoiceIndex<T, unknown>,
  options: ExtractOptions = {},
): ExtractResult<T>[] {
  const limit = options.limit === undefined ? 5 : options.limit

  // The heap below compares `heap.length` against this, so a fractional limit
  // silently returns one result too many and `NaN` reads an empty heap. Neither
  // is a limit anyone meant to ask for.
  if (limit != null && !Number.isSafeInteger(limit)) {
    throw new RangeError('limit must be an integer or null')
  }

  if (limit === 1) {
    const best = extractOne(query, choices, options)
    return best === undefined ? [] : [best]
  }

  // The scorer's direction is not read here. `prepare` resolves it from the same
  // flags on the way to building the state, and both branches below have that
  // state in hand before they need to know — so asking `scorerFlagsOf` twice per
  // call bought a binding that was already about to exist.
  if (limit == null) {
    // Not `[...extractIter(...)]`. Every choice would cross a generator
    // boundary twice — once out of `entriesOf`, once out of `extractIter` — and
    // carry a two-element tuple with it, to build a list this function then
    // sorts in place anyway. An array of choices is the shape `process` is
    // given in practice, so it gets the plain loop `extractOne` already has.
    const unlimited = prepare(query, options, choices)
    if (unlimited === null) return []

    const results: ExtractResult<T>[] = []
    const { scoreCutoff, admits, lowestScoreWorst: keepHigh } = unlimited

    // Written out three times rather than through a shared closure. One call per
    // choice is what this branch exists to remove, and routing the map case
    // through a closure to save six lines measured 1.03x — giving back more
    // than the generator it replaced had cost.
    if (isIndex(choices)) {
      const { values, keys } = choices
      const prepared = payloadOf(choices)
      for (let i = 0; i < values.length; i++) {
        const value = score(unlimited, prepared[i], scoreCutoff)
        if (keepHigh ? value >= admits : value <= admits) {
          results.push({ choice: values[i], score: value, key: keys[i] })
        }
      }
    } else if (Array.isArray(choices)) {
      for (let key = 0; key < choices.length; key++) {
        const choice = choices[key]
        if (isNone(choice)) continue
        const value = score(unlimited, choice, scoreCutoff)
        if (keepHigh ? value >= admits : value <= admits) {
          results.push({ choice, score: value, key })
        }
      }
    } else {
      for (const [key, choice] of entriesOf(choices)) {
        if (isNone(choice)) continue
        const value = score(unlimited, choice, scoreCutoff)
        if (keepHigh ? value >= admits : value <= admits) {
          results.push({ choice, score: value, key })
        }
      }
    }

    results.sort((a, b) => (keepHigh ? b.score - a.score : a.score - b.score))
    return results
  }

  // An empty result is still this index's answer, so the call still has to be
  // one the index accepts — a mismatched scorer is a mistake at `limit: 0` for
  // the same reason it is at `limit: 5`. Checked rather than prepared: no
  // processor runs and no query is built for a call that asked for nothing.
  if (limit <= 0) {
    assertNotPreparedHandle(options.scorer)
    checkIndex(options, choices)
    return []
  }

  const state = prepare(query, options, choices)
  if (state === null) return []

  interface HeapEntry {
    choice: T
    score: number
    key: unknown
    position: number
  }

  // Read once into a local: `worse` runs on every sift step of every insertion,
  // and a property load per comparison is what the old duplicate flags lookup
  // was really buying.
  const lowestScoreWorst = state.lowestScoreWorst

  // The root is the worst retained item. Equal scores prefer the earlier
  // iterable position, which preserves stable-sort behaviour.
  const worse = (a: HeapEntry, b: HeapEntry): boolean =>
    a.score === b.score
      ? a.position > b.position
      : lowestScoreWorst
        ? a.score < b.score
        : a.score > b.score
  const heap: HeapEntry[] = []

  const siftUp = (index: number): void => {
    while (index > 0) {
      const parent = (index - 1) >>> 1
      if (!worse(heap[index], heap[parent])) break
      const entry = heap[parent]
      heap[parent] = heap[index]
      heap[index] = entry
      index = parent
    }
  }
  const siftDown = (index: number): void => {
    for (;;) {
      const left = index * 2 + 1
      if (left >= heap.length) return
      const right = left + 1
      let child = left
      if (right < heap.length && worse(heap[right], heap[left])) child = right
      if (!worse(heap[child], heap[index])) return
      const entry = heap[index]
      heap[index] = heap[child]
      heap[child] = entry
      index = child
    }
  }

  // Scoring is split from placement, and neither half is given the choice or the
  // key. `extract` rejects almost everything — a heap of five over two thousand
  // — and on an index those two are loads from frozen arrays that only a winner
  // needs; fetching them per choice measured 1.15x on `ratio` over short
  // strings. So a driver reads them once `wins` has said the candidate earned
  // them. `winningScore` carries the one value across, rather than a second
  // return or a tuple to allocate.
  let winningScore = 0
  const wins = (scored: unknown, position: number): boolean => {
    // A full heap replaces both the bound and the comparison with the score at
    // its root; short of that they part company, because "no cutoff" is a
    // number to compare against but not one to hand a scorer.
    const tightened = state.builtIn && heap.length === limit
    const activeCutoff = tightened ? heap[0].score : state.scoreCutoff
    const admits = tightened ? heap[0].score : state.admits
    const value = score(state, scored, activeCutoff)
    const passes = state.lowestScoreWorst ? value >= admits : value <= admits
    if (!passes) return false
    winningScore = value
    if (heap.length < limit) return true

    // `worse(heap[0], entry)` with the entry not yet built. Clearing the cutoff
    // is not the same as beating the heap: a score equal to the root passes
    // `admits` and then loses this test, since a later position never displaces
    // an earlier one at the same score.
    const root = heap[0]
    return root.score === value
      ? root.position > position
      : lowestScoreWorst
        ? root.score < value
        : root.score > value
  }

  // Reached only for a candidate `wins` accepted, so the heap being full is the
  // whole of the push-or-replace question by this point.
  const place = (choice: T, key: unknown, position: number): void => {
    const entry = { choice, score: winningScore, key, position }
    if (heap.length < limit) {
      heap.push(entry)
      siftUp(heap.length - 1)
      return
    }
    heap[0] = entry
    siftDown(0)
  }

  // `scored` is what the scorer is handed and `choice` is what a result names.
  // They are the same value everywhere but an index, where the first is the
  // choice as the scorer's own hook left it and the second is what the caller
  // put in the collection. An index has no missing choices left to test for —
  // they were dropped when it was built.
  if (isIndex(choices)) {
    const { values, keys } = choices
    const prepared = payloadOf(choices)
    for (let i = 0; i < values.length; i++) {
      if (wins(prepared[i], i)) place(values[i], keys[i], i)
    }
  } else if (Array.isArray(choices)) {
    for (let i = 0; i < choices.length; i++) {
      const choice = choices[i]
      if (!isNone(choice) && wins(choice, i)) place(choice, i, i)
    }
  } else {
    let position = 0
    for (const [key, choice] of entriesOf(choices)) {
      if (!isNone(choice) && wins(choice, position)) place(choice, key, position)
      position++
    }
  }

  heap.sort((a, b) => {
    const delta = lowestScoreWorst ? b.score - a.score : a.score - b.score
    return delta || a.position - b.position
  })
  return heap.map(({ choice, score: value, key }) => ({ choice, score: value, key }))
}

export interface ScoreOptions<K extends ScoreArrayKind = 'f64'> extends SearchOptions {
  /**
   * Element type the scores are stored as. Defaults to `'f64'`.
   *
   * An integral kind rounds each score half away from zero, which is what an
   * integral NumPy dtype gets upstream. `'u8'` wraps above 255, as NumPy's
   * `astype` does; `'u8c'` saturates instead.
   *
   * The result type is read from this, so a variable holding it needs to say
   * which kind it holds: `ScoreOptions` alone means `ScoreOptions<'f64'>` and
   * refuses any other kind. Use `ScoreOptions<ScoreArrayKind>` when the kind is
   * only known at run time — the result is then the `ScoreArray` union, which
   * is what a matrix of unknown element type honestly is.
   */
  readonly into?: K | undefined
  /**
   * Multiplies every score before it is stored. Defaults to 1.
   *
   * This is what makes an integral `into` useful:
   * `{ into: 'u8', scoreMultiplier: 2.55 }` quantises a 0–100 `ratio` into a
   * byte.
   */
  readonly scoreMultiplier?: number | undefined
}

/**
 * Score every query against every choice, as a `queries × choices` matrix.
 *
 * Upstream's `cdist`. Renamed because it no longer returns what `cdist`
 * returns: the result is a {@link ScoreMatrix} over one typed array rather than
 * an array of arrays, and `dtype` is replaced by {@link ScoreOptions.into},
 * which selects storage rather than only rounding. `toArray()` gives back the
 * nested-array shape.
 */
export function scoreMatrix(
  queries: readonly MaybeSequence[],
  choices: readonly MaybeSequence[],
  options?: ScoreOptions<'f64'>,
): ScoreMatrix<Float64Array>
export function scoreMatrix<K extends ScoreArrayKind>(
  queries: readonly MaybeSequence[],
  choices: readonly MaybeSequence[],
  options: ScoreOptions<K> & { into: K },
): ScoreMatrix<ScoreArrayOf[K]>
/** For a caller whose `into` is only known at run time. */
export function scoreMatrix(
  queries: readonly MaybeSequence[],
  choices: readonly MaybeSequence[],
  options: ScoreOptions<ScoreArrayKind>,
): ScoreMatrix<ScoreArray>
export function scoreMatrix(
  queries: readonly MaybeSequence[],
  choices: readonly MaybeSequence[],
  options: ScoreOptions<ScoreArrayKind> = {},
): ScoreMatrix<ScoreArray> {
  assertNotPreparedHandle(options.scorer)
  const scorer = options.scorer ?? ratio
  const processor = options.processor ?? null
  const scoreCutoff = options.scoreCutoff ?? null
  const scoreHint = options.scoreHint ?? null

  const applyProcessor = (value: unknown): unknown =>
    processor != null && !isNone(value) ? processor(queryAsSequence(value)) : value

  // Both passes run even with no processor, where `applyProcessor` is identity
  // and the two arrays are copies. Skipping them then was measured over 200
  // 4x6 matrices — the shape where per-call setup is a share of the work rather
  // than a rounding error — at 1.03x on `hammingDistance`, 1.01x on `ratio` and
  // `prefixSimilarity`, and 1.00x on a 50x200. Inside the noise floor even at
  // the cheapest scorer there is, for a nested ternary on both bindings. So is
  // the cell loop below: hoisting `i * cols` out of it, and splitting it in two
  // so the invariant `integral` test does not run per cell, came to 0.99-1.02x
  // over the same matrices with no consistent sign.
  const processedChoices = choices.map(applyProcessor)
  const sameInput = Object.is(queries, choices)
  const processedQueries = sameInput ? processedChoices : queries.map(applyProcessor)
  const flags = scorerFlagsOf(scorer)
  const factory = prepareScorerOf(scorer)
  let cachedPrepared: PreparedScore | null = null
  let cachedPreparedIndex = -1
  // Hoisting the per-choice conversion out of the row loop is only possible if
  // the scorer's factory offers a hook. It always did while every prepared
  // scorer was a built-in; a configured one with a baked-in processor registers
  // no factory at all, and a third-party one may prepare a query and still want
  // its choices as they came — so the absence is real, and `preparedChoices`
  // has to fall back to the unconverted choices rather than to a default
  // converter that would disagree with what the scorer expects.
  let prepareMatrixChoice: ChoicePreparer | null = null
  if (factory !== null) {
    for (let i = 0; i < processedQueries.length; i++) {
      const query = processedQueries[i]
      if (isNone(query)) continue
      cachedPrepared = factory(queryAsSequence(query), NO_OPTIONS)
      cachedPreparedIndex = i
      // Read here rather than beside `factory`, so that a matrix whose every
      // query is missing prepares no choice: there would be nothing to read it.
      prepareMatrixChoice = factory[PREPARE_CHOICE] ?? null
      break
    }
  }
  const preparedChoices =
    prepareMatrixChoice !== null
      ? processedChoices.map(prepareMatrixChoice)
      : processedChoices
  // A scorer whose options make it asymmetric -- weighted Levenshtein -- says so
  // through its own flags, which `configure` resolved at the point the option
  // was baked in. Nothing here needs to know what that option is called.
  const symmetric = sameInput && factory !== null && flags.symmetric
  const multiplier = options.scoreMultiplier ?? 1

  const rows = processedQueries.length
  const cols = processedChoices.length

  const fill = (data: ScoreArray, integral: boolean): void => {
    for (let i = 0; i < rows; i++) {
      const query = processedQueries[i]
      const prepared =
        i === cachedPreparedIndex
          ? cachedPrepared
          : factory !== null && !isNone(query)
            ? factory(queryAsSequence(query), NO_OPTIONS)
            : null
      const start = symmetric ? i : 0

      for (let j = start; j < cols; j++) {
        const choice = processedChoices[j]
        const value =
          prepared !== null
            ? prepared(preparedChoices[j], scoreCutoff, scoreHint)
            : callScorer(scorer, query, choice, { scoreCutoff, scoreHint })
        const scaled = value * multiplier
        // Rounded once, before either store: writing the same already-integral
        // value into both triangles is what keeps them from drifting apart.
        const result = integral ? roundHalfAwayFromZero(scaled) : scaled
        data[i * cols + j] = result
        if (symmetric && i !== j) data[j * cols + i] = result
      }
    }
  }

  // The element type has to reach `buildScoreMatrix` as a literal, or the row
  // iterator can only promise the whole `ScoreArray` union. Dispatching here is
  // what keeps `scoreMatrix(a, b, { into: 'u8' })` iterating `Uint8Array`
  // without an assertion anywhere.
  switch (options.into ?? DEFAULT_SCORE_KIND) {
    case 'f32':
      return buildScoreMatrix('f32', rows, cols, 'scoreMatrix', fill)
    case 'i32':
      return buildScoreMatrix('i32', rows, cols, 'scoreMatrix', fill)
    case 'i16':
      return buildScoreMatrix('i16', rows, cols, 'scoreMatrix', fill)
    case 'i8':
      return buildScoreMatrix('i8', rows, cols, 'scoreMatrix', fill)
    case 'u32':
      return buildScoreMatrix('u32', rows, cols, 'scoreMatrix', fill)
    case 'u16':
      return buildScoreMatrix('u16', rows, cols, 'scoreMatrix', fill)
    case 'u8':
      return buildScoreMatrix('u8', rows, cols, 'scoreMatrix', fill)
    case 'u8c':
      return buildScoreMatrix('u8c', rows, cols, 'scoreMatrix', fill)
    default:
      return buildScoreMatrix('f64', rows, cols, 'scoreMatrix', fill)
  }
}

/**
 * Score `queries[i]` against `choices[i]`, pairwise.
 *
 * Upstream's `cpdist`. Returns the typed array bare: one dimension needs no
 * wrapper, since `length`, `at`, iteration and `Array.from` are already there.
 */
export function scorePairs(
  queries: readonly MaybeSequence[],
  choices: readonly MaybeSequence[],
  options?: ScoreOptions<'f64'>,
): Float64Array
export function scorePairs<K extends ScoreArrayKind>(
  queries: readonly MaybeSequence[],
  choices: readonly MaybeSequence[],
  options: ScoreOptions<K> & { into: K },
): ScoreArrayOf[K]
/** For a caller whose `into` is only known at run time. */
export function scorePairs(
  queries: readonly MaybeSequence[],
  choices: readonly MaybeSequence[],
  options: ScoreOptions<ScoreArrayKind>,
): ScoreArray
export function scorePairs(
  queries: readonly MaybeSequence[],
  choices: readonly MaybeSequence[],
  options: ScoreOptions<ScoreArrayKind> = {},
): ScoreArray {
  assertNotPreparedHandle(options.scorer)
  const scorer = options.scorer ?? ratio
  const processor = options.processor ?? null
  const scoreCutoff = options.scoreCutoff ?? null
  const scoreHint = options.scoreHint ?? null
  const multiplier = options.scoreMultiplier ?? 1

  const applyProcessor = (value: unknown): unknown =>
    processor != null && !isNone(value) ? processor(queryAsSequence(value)) : value

  if (queries.length !== choices.length) {
    throw new Error('Length of queries and choices must be the same!')
  }

  const length = queries.length
  const kind = options.into ?? DEFAULT_SCORE_KIND
  const { integral } = scoreArrayFactory(kind)
  const out = allocateScores(kind, length, 'scorePairs')

  // Deliberately not the prepared path `scoreMatrix` takes. Preparing a query
  // pays for itself by being scored against many choices, and pairwise scoring
  // gives it exactly one — so the masks, tokenisation and wrapper allocation are
  // built and then thrown away every iteration. Measured over 4000 title-like
  // pairs that made `ratio` 2.2x and `tokenSortRatio` 2.4x slower than calling
  // the scorer directly, with every scorer tested slower and none faster.
  // Specialising this loop was measured and dropped: a shared options object for
  // built-in scorers, plus separate no-processor and non-integral loops, came to
  // 0.985-0.992x over 4000 pairs on every scorer tried — under the ±3% noise
  // floor, for two duplicated loops. The scorer call is the cost here, not the
  // boundary around it.
  for (let i = 0; i < length; i++) {
    const value = callScorer(
      scorer,
      applyProcessor(queries[i]),
      applyProcessor(choices[i]),
      { scoreCutoff, scoreHint },
    )
    const scaled = value * multiplier
    out[i] = integral ? roundHalfAwayFromZero(scaled) : scaled
  }

  return out
}
