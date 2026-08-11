/**
 * The prepared-query hook every fuzz scorer shares.
 *
 * `process` scores one query against many choices, so anything derived from the
 * query alone is worth holding: its converted sequence, its LCS masks, its
 * tokenisation and the masks of its token-sorted form. {@link prepareFuzz} builds
 * that state once per query and returns a `PreparedScore` closure that consumes
 * it, which is what turns a `scoreMatrix` from `O(rows * cols)` tokenisations
 * into `O(rows + cols)`.
 *
 * ## The one-way rule
 *
 * This module sits above every scorer family and imports the reusable cores from
 * `basic`, `tokens` and `tokenScorers`. None of them may import this one, which
 * is what keeps the graph acyclic.
 *
 * `composite.ts` is deliberately **not** among them. Its `wRatio_impl` and
 * `qRatio_impl` validate and convert raw input, which is exactly the work
 * already done by the time a branch below runs — so the composite strategies are
 * reproduced here over prepared state instead of called. The `wRatio` branch is
 * the clearest case: it mirrors `wRatio_impl`'s ladder of length-ratio tests and
 * 0.95/0.9/0.6 scalings against held state rather than fresh input. The two must
 * be kept in step by hand; a change to one is a change to both.
 */
import {
  alignRepresentation,
  convSequence,
  isNone,
  isSequence,
  withChoicePreparer,
  prepareScorerChoice,
  preparedScorerSequence,
  scorerSequence,
  type ChoicePreparer,
  type PrepareScorer,
  type PreparedScore,
} from '../_common.js'
import type { PatternMask } from '../distance/_bitVector/index.js'
import {
  lcsSeqLengthPrepared,
  lcsSeqLengthPreparedBounded,
  prepareLcsPattern,
} from '../distance/lcsSeq.js'
import {
  type CharSet,
  charSetOf,
  partialAlignmentConverted,
  partialRatioConverted,
  partialRatioImpl,
} from './basic.js'
import {
  containsWhitespace,
  hasWhitespaceOf,
  preparedTokenChoice,
  sortedOf,
  stringContainsWhitespace,
  tokenChoicePreparer,
  tokenForm,
  tokenViewOf,
} from './tokens.js'
import {
  partialTokenRatioConverted,
  partialTokenSetRatioConverted,
  tokenRatioConverted,
  tokenSetRatioConverted,
  tokenSortRatioConverted,
} from './tokenScorers.js'
import type { PreparedFuzzKind } from './types.js'

/**
 * Ratio against immutable query-side LCS masks.
 *
 * Deliberately ignores `scoreHint`, which the caller receives and drops. The
 * only lever a hint could pull here is the bounded kernel's early exit, and a
 * hint is an estimate rather than a bound: budgeting the scan at an optimistic
 * hint means every candidate scoring between the cutoff and the hint is pruned
 * and then rescanned in full. Measured over an `extract` of 2000 choices where
 * the bounded path is reachable, that cost **1.49x** the kernel iterations at
 * `scoreHint: 90` and **1.71x** at `70`; with a cutoff already set it ran the
 * bounded kernel 1602 times where 806 sufficed. The same shape was already
 * found slower in Levenshtein, which sizes a *band* with its hint — a lever
 * the bit-parallel LCS does not have.
 */
function ratioPrepared(
  query: ArrayLike<unknown>,
  pattern: PatternMask,
  choice: ArrayLike<unknown>,
  scoreCutoff: number,
): number {
  const maximum = query.length + choice.length
  if (maximum === 0) return scoreCutoff <= 100 ? 100 : 0

  const ceiling =
    (1 - (maximum - 2 * Math.min(query.length, choice.length)) / maximum) * 100
  if (ceiling < scoreCutoff) return 0

  const required = Math.max(0, Math.floor((scoreCutoff * maximum) / 200))
  const lcs =
    scoreCutoff >= 70 && maximum >= 128
      ? lcsSeqLengthPreparedBounded(pattern, choice, 0, choice.length, required)
      : lcsSeqLengthPrepared(pattern, choice, 0, choice.length)
  if (lcs < 0) return 0
  const score = (1 - (maximum - 2 * lcs) / maximum) * 100
  return score >= scoreCutoff ? score : 0
}

/**
 * Whether this scorer ever splits an input into tokens.
 *
 * The ones that do take {@link tokenChoicePreparer}, so a choice is converted
 * once and its derived forms are memoised on the record as branches ask for
 * them; the ones that do not take {@link prepareScorerChoice} and never
 * tokenise at all.
 *
 * This used to be a table of *which* forms each scorer might read, so the
 * preparer could build exactly those up front. Per-scorer is one branch too
 * coarse — `wRatio` may read all three but on a given pair usually reads one —
 * so the choice is now made per access instead. See {@link PreparedTokenChoice}.
 */
function tokenisesInput(kind: PreparedFuzzKind): boolean {
  switch (kind) {
    case 'ratio':
    case 'partialRatio':
    case 'qRatio':
      return false
    case 'tokenSortRatio':
    case 'partialTokenSortRatio':
    case 'tokenSetRatio':
    case 'partialTokenSetRatio':
    case 'tokenRatio':
    case 'partialTokenRatio':
    case 'wRatio':
      return true
  }
}

/** Build the internal query-caching hook shared by all fuzz scorers. */
export function prepareFuzz(kind: PreparedFuzzKind): PrepareScorer {
  const usesTokens = tokenisesInput(kind)
  // Every token scorer but one splits whatever it is handed, so converting a
  // raw candidate up front costs nothing it would not spend anyway. `wRatio` is
  // the exception, and the reason this is a second predicate rather than a
  // reuse of `usesTokens`: its most common route answers with the base ratio
  // without ever splitting, and the LCS kernels read a BMP string through
  // `charCodeAt` exactly as they read code points. So it takes a candidate as
  // it comes and expands it only on the branches that tokenise — which, over an
  // `extract`, is the difference between one `Uint32Array` per candidate and
  // none at all.
  //
  // The query is untouched by this. It is prepared once per `extract`, so
  // converting it eagerly costs one allocation against the candidates'
  // thousands, and holding it converted is what keeps the token branches
  // comparable: a token of characters and a token of code points never compare
  // elementwise equal, so the two sides have to meet in one form.
  const convertsChoice = usesTokens && kind !== 'wRatio'
  // One preparer per scorer rather than per query: `prepareFuzz` runs once, at
  // the point each scorer below is built.
  const choicePreparer: ChoicePreparer = usesTokens
    ? tokenChoicePreparer()
    : prepareScorerChoice

  const prepare: PrepareScorer = (query) => {
    const queryTokenChoice = usesTokens
      ? preparedTokenChoice(choicePreparer(query))
      : null
    const heldQuery = usesTokens
      ? queryTokenChoice === null
        ? null
        : queryTokenChoice.sequence
      : preparedScorerSequence(prepareScorerChoice(query))
    if (heldQuery === null) throw new TypeError('fuzz scorers expect a sequence')
    const a = heldQuery
    // Built on first use rather than up front: only four of the ten kinds score
    // through LCS masks, and `wRatio` reaches them only on some inputs.
    let lcsPattern: PatternMask | null = null
    const patternOf = (): PatternMask =>
      (lcsPattern ??= prepareLcsPattern(a, 0, a.length))
    // The query's own view. Token branches are the only readers, and for them
    // `queryTokenChoice` is never null — a query that is not a sequence has
    // already thrown above. The stand-in empty split, token set and sorted form
    // that used to sit here are gone with the eager preparation that needed
    // them: a non-token kind now carries no tokenisation rather than an empty
    // one.
    const queryView = queryTokenChoice ?? undefined
    // The same view, named so that the sorted forms below can be typed without
    // re-testing what the throw above settled. A non-token kind never reads it
    // and takes the second arm, which is why that arm is a real one rather than
    // a fallback no input reaches.
    //
    // Not lazy, unlike everything around it. `tokenViewOf` is `{ sequence }`,
    // so the second arm costs one object literal per *prepared query* — not per
    // candidate — which an `extract` over ten thousand choices pays once. An
    // accessor to defer that would be more code than the allocation it saves.
    // The derived forms are the expensive part, and those are still on demand.
    const queryTokens = queryTokenChoice ?? tokenViewOf(a)
    // Masks for the token-sorted query. Token-sort scoring reaches the same
    // kernel as `ratio` but with a different left-hand sequence, so it needs
    // masks of its own; without them every choice rebuilt the sorted query's.
    // Lazy for the same reason as `patternOf`: `wRatio` reaches token-sort only
    // on some inputs, and `tokenSetRatio` never does.
    let sortedPattern: PatternMask | null = null
    const sortedPatternOf = (): PatternMask => {
      if (sortedPattern === null) {
        const sorted = sortedOf(queryTokens)
        sortedPattern = prepareLcsPattern(sorted, 0, sorted.length)
      }
      return sortedPattern
    }
    // The pruning set for the *sorted* query, which the partial token scorers
    // scan the same way `partialRatio` scans the query itself. One form suffices
    // where `partialRatio` needs two: a sorted form is always the array
    // `joinTokens` built, never a string, so `alignRepresentation` has nothing
    // to expand and both sides already agree.
    let sortedCharSet: CharSet | null = null
    const sortedCharSetOf = (): CharSet =>
      (sortedCharSet ??= charSetOf(sortedOf(queryTokens)))
    // The window scan's pruning set, which upstream's `CachedPartialRatio` also
    // holds beside its cached ratio. Rebuilt per candidate before this, which on
    // an `extract` of ten thousand is ten thousand walks of the same query.
    //
    // Two forms, because unlike `patternOf` this one is not representation-blind:
    // the scan prunes with `===`, where `'a' !== 97`. `alignRepresentation` hands
    // the scan either `a` itself or its code points depending on what the
    // *candidate* turned out to be, so both spellings can be asked for across one
    // query's candidates, and each is built at most once.
    let nativeCharSet: CharSet | null = null
    const nativeCharSetOf = (): CharSet => (nativeCharSet ??= charSetOf(a))
    let convertedCharSet: CharSet | null = null
    const convertedCharSetOf = (): CharSet =>
      (convertedCharSet ??= charSetOf(convSequence(a)))

    const score: PreparedScore = (rawChoice, rawCutoff) => {
      if (isNone(rawChoice)) return 0
      const tokenChoice = usesTokens ? preparedTokenChoice(rawChoice) : null
      // The choice's own view, when `process` prepared one. A choice reached
      // without preparation gets a fresh view inside whichever core needs it.
      const choiceView = tokenChoice ?? undefined
      let b = tokenChoice?.sequence ?? preparedScorerSequence(rawChoice)
      if (b === null) {
        if (!isSequence(rawChoice)) {
          throw new TypeError('fuzz scorers expect a string or an array-like sequence')
        }
        b = convertsChoice ? convSequence(rawChoice) : scorerSequence(rawChoice)
      }
      const cutoff = rawCutoff ?? 0

      switch (kind) {
        case 'ratio':
          return ratioPrepared(a, patternOf(), b, cutoff)
        case 'partialRatio': {
          // Unlike the mask kernels, the window scan prunes by comparing
          // elements with `===`, so a query held as a BMP string and a choice
          // expanded into code points have to be brought together first.
          const s1 = alignRepresentation(a, b)
          const s2 = alignRepresentation(b, a)
          // The held masks belong to `s1` whichever representation it ended up
          // in: a BMP string and its code points have the same element values,
          // so they build the same masks.
          //
          // The char set is the exception, and `s1 === a` is exactly the test
          // for which spelling this candidate produced: `alignRepresentation`
          // returns `a` untouched unless it had to expand it.
          return (
            partialAlignmentConverted(
              s1,
              s2,
              cutoff,
              true,
              patternOf(),
              s1 === a ? nativeCharSetOf() : convertedCharSetOf(),
            )?.score ?? 0
          )
        }
        case 'tokenSortRatio':
          return tokenSortRatioConverted(
            a,
            b,
            cutoff,
            queryView,
            choiceView,
            sortedPatternOf(),
          )
        case 'tokenSetRatio':
          return tokenSetRatioConverted(a, b, cutoff, queryView, choiceView)
        case 'tokenRatio':
          return tokenRatioConverted(a, b, cutoff, queryView, choiceView, sortedPatternOf)
        case 'partialTokenSortRatio': {
          // Through `partialAlignmentConverted` rather than
          // `partialRatioConverted`, so the sorted query's masks and pruning set
          // can be handed down. Without them this rebuilt the same mask for the
          // same sorted query once per candidate — the very thing the prepared
          // `partialRatio` path stopped doing.
          const sortedQuery = sortedOf(queryTokens)
          const sortedChoice = sortedOf(choiceView ?? tokenViewOf(b))
          // `partialAlignmentConverted` would ignore both when the candidate is
          // the shorter side, but these are arguments, so they would be built on
          // the way in regardless. Both memoise, so the waste is one mask and one
          // set per query rather than per candidate — still worth not paying for
          // a query that is never the needle.
          const preparedApplies = sortedQuery.length <= sortedChoice.length

          return (
            partialAlignmentConverted(
              sortedQuery,
              sortedChoice,
              cutoff,
              true,
              preparedApplies ? sortedPatternOf() : undefined,
              preparedApplies ? sortedCharSetOf() : undefined,
            )?.score ?? 0
          )
        }
        case 'partialTokenSetRatio':
          return partialTokenSetRatioConverted(a, b, cutoff, queryView, choiceView)
        case 'partialTokenRatio':
          return partialTokenRatioConverted(
            a,
            b,
            cutoff,
            queryView,
            choiceView,
            sortedPatternOf,
            sortedCharSetOf,
          )
        case 'qRatio':
          return a.length === 0 || b.length === 0
            ? 0
            : ratioPrepared(a, patternOf(), b, cutoff)
        case 'wRatio': {
          if (a.length === 0 || b.length === 0 || cutoff > 100) return 0
          const unbaseScale = 0.95
          const lenRatio = a.length > b.length ? a.length / b.length : b.length / a.length
          let dynamicCutoff = cutoff
          let result = ratioPrepared(a, patternOf(), b, dynamicCutoff)

          if (lenRatio < 1.5) {
            // Whether the query holds whitespace, not how many tokens it splits
            // into: a single token with a space around it still has its
            // token-sorted form differ from the input, so the shortcut past the
            // token scorers does not apply to it.
            //
            // This is the branch lazy preparation was for. Both sides are asked
            // only whether they hold whitespace, which reads the sequence and
            // never splits it — so a candidate that takes this shortcut is now
            // never tokenised at all, where before it had been split, deduped,
            // sorted and joined by the preparer before scoring even began.
            // `wRatio` tokenises, so `queryView` is always there — a query that
            // is not a sequence threw above. The fallback still reads the
            // sequence rather than answering `false`, because `false` is the
            // answer that *takes* the shortcut: if that invariant ever changed,
            // a wrong `false` would skip the token scorers outright, while a
            // wrong `true` only costs work.
            //
            // The candidate reaches here unconverted, so its own test goes
            // through whichever spelling it arrived in — a prepared view always
            // holds code points, an unprepared candidate may still be a string.
            if (
              !hasWhitespaceOf(queryTokens) &&
              !(choiceView === undefined
                ? typeof b === 'string'
                  ? stringContainsWhitespace(b)
                  : containsWhitespace(b)
                : hasWhitespaceOf(choiceView))
            )
              return result
            dynamicCutoff = Math.max(dynamicCutoff, result) / unbaseScale
            return Math.max(
              result,
              tokenRatioConverted(
                a,
                // Expanded here and not before: the shortcut above is the route
                // most candidates take, and it needs no tokens. `a` is already
                // code points, so this is what puts the two sides in the one
                // form their token sets have to share.
                tokenForm(b),
                dynamicCutoff,
                queryView,
                choiceView,
                sortedPatternOf,
              ) * unbaseScale,
            )
          }

          const partialScale = lenRatio <= 8 ? 0.9 : 0.6
          dynamicCutoff = Math.max(dynamicCutoff, result) / partialScale
          // `nativeCharSetOf` without the representation test the `partialRatio`
          // branch needs: `a` is the query, and `wRatio` holds a query as code
          // points, so the pruning set is only ever built over that spelling.
          // The candidate is the side that may still be a string, and the scan
          // prunes with `===` — so it is expanded here, which is the same job
          // `alignRepresentation` does for the `partialRatio` branch above.
          const bPartial = alignRepresentation(b, a)
          const partial =
            a.length <= bPartial.length
              ? partialRatioImpl(
                  a,
                  bPartial,
                  dynamicCutoff / 100,
                  patternOf(),
                  true,
                  nativeCharSetOf(),
                ).score
              : partialRatioConverted(a, bPartial, dynamicCutoff)
          result = Math.max(result, partial * partialScale)
          dynamicCutoff = Math.max(dynamicCutoff, result) / unbaseScale
          return Math.max(
            result,
            partialTokenRatioConverted(
              a,
              tokenForm(b),
              dynamicCutoff,
              queryView,
              choiceView,
              sortedPatternOf,
              sortedCharSetOf,
            ) *
              unbaseScale *
              partialScale,
          )
        }
      }
    }
    return score
  }
  return withChoicePreparer(prepare, choicePreparer)
}
