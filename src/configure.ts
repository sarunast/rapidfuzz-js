/**
 * Baking a scorer's options into it, so it can be passed as a scorer.
 *
 * Its own module rather than part of `_common.ts`: that file is the plumbing
 * every scorer's hot path runs through — `conv`, `commonPrefix`, the cutoff
 * conventions — and this is public API that runs once per configured scorer.
 * Keeping it out leaves the shared module the size it was.
 */
import {
  callScorer,
  configuredFlagsOf,
  configureOptionsOf,
  isBuiltInScorer,
  NO_OPTIONS,
  prepareScorerOf,
  registerScorer,
  scorerFlagsOf,
  stableFlags,
  toRecord,
  type Flagged,
  type ScorerOptions,
} from './_common.js'

/**
 * The options {@link configure} can bake in: a scorer's own, minus the two
 * bounds that only mean anything per call.
 */
export type ScorerConfig<O extends ScorerOptions> = Omit<O, 'scoreCutoff' | 'scoreHint'>

/** A scorer with options already applied. Callable, and carrying its flags. */
export interface ConfiguredScorer<I, O extends ScorerOptions> extends Flagged {
  (s1: I, s2: I, options?: O): number
}

/**
 * Bake options into a scorer, so it can be passed anywhere a scorer is expected.
 *
 * This replaces upstream's `scorer_kwargs`, which exists because Python has
 * `**kwargs`. Handing `process` a scorer *and* a bag of arguments to forward to
 * it means the bag cannot be checked against the scorer, and means every
 * consumer of a scorer has to thread the bag through. A configured scorer is
 * just a scorer.
 *
 * ```ts
 * scoreMatrix(a, b, { scorer: configure(levenshteinDistance, { weights: [1, 1, 2] }) })
 * ```
 *
 * Options given per call win over the baked ones. Both `I` and `O` are inferred
 * from the scorer's own call signature, which is what lets one signature serve
 * `Scorer` and `NormalizedScorer` alike — a configured
 * `levenshteinNormalizedDistance` still accepts `null`.
 *
 * A configured *built-in* keeps its prepared-query fast path, and stays a
 * built-in so `extract` can still tighten its cutoff against the running best.
 * A configured third-party scorer gets flags only: registering it would change
 * how many times `process` calls it, and that is observable.
 */
export function configure<I, O extends ScorerOptions>(
  scorer: (s1: I, s2: I, options?: O) => number,
  options: ScorerConfig<O>,
): ConfiguredScorer<I, O> {
  const given = toRecord(options)

  // `ScorerConfig` omits these, so a TypeScript caller cannot get here — but
  // this API is meant to be usable from JavaScript, and a baked cutoff is not
  // merely ignored, it makes the same scorer disagree with itself: a direct
  // call would honour it, while `scoreMatrix` and `extract*` supply their own
  // per-call cutoff that overrides it. Refusing is the only answer that leaves
  // one meaning.
  for (const perCallOnly of ['scoreCutoff', 'scoreHint']) {
    if (perCallOnly in given) {
      throw new TypeError(`${perCallOnly} is a per-call option and cannot be configured`)
    }
  }

  // Snapshot before anything reads it: `given` is one level deep, so a nested
  // value is still the caller's to mutate. See {@link ConfigureOptions}.
  const canonicalize = configureOptionsOf(scorer)
  const baked = canonicalize === null ? given : canonicalize(given)

  const resolveFlags = configuredFlagsOf(scorer)
  const flags = resolveFlags === null ? scorerFlagsOf(scorer) : resolveFlags(baked)

  // `{ ...baked, ...perCall }` cannot be typed as `O` for a generic `O` without
  // an assertion, so the merged record goes through the `Reflect.apply`
  // boundary instead — the same trade the comment on `callScorer` describes.
  const call = (s1: I, s2: I, perCall?: O): number =>
    callScorer(scorer, s1, s2, perCall === undefined ? baked : { ...baked, ...perCall })

  if (!isBuiltInScorer(scorer)) {
    return Object.assign(call, { rfScorerFlags: stableFlags(flags) })
  }

  // A baked-in processor has to reach the scorer, and the prepared path bypasses
  // it — so that case registers with no factory. It is still a built-in: the
  // processor runs once per choice however the cutoff moves, and never sees it.
  const inner = Reflect.get(baked, 'processor') == null ? prepareScorerOf(scorer) : null

  return registerScorer(call, flags, {
    // `configure` is the only supplier of a prepared factory's second argument
    // now. It still has to be honoured rather than replaced, because
    // configuring an already-configured scorer arrives here as exactly that: an
    // outer set of options handed to an inner factory that has its own. Merging
    // keeps the prepared path agreeing with the direct call, where the outer
    // options win the same way.
    prepare:
      inner === null
        ? undefined
        : (query, options) =>
            inner(query, options === NO_OPTIONS ? baked : { ...baked, ...options }),
    configuredFlags:
      resolveFlags === null
        ? undefined
        : (perCall) => resolveFlags({ ...baked, ...perCall }),
    // Carried through so configuring a configured scorer snapshots the outer
    // options the same way the inner ones were.
    configureOptions: canonicalize ?? undefined,
  })
}
