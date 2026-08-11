import { prepareLcsPattern } from '../../algorithms/lcs/implementation.js'
import type { PatternMask } from '../../algorithms/shared/bitmask/pattern.js'
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
 * `partialWindow.ts`, `tokens.ts` and `tokenSet.ts`. None of them may import this
 * one, which is what keeps the graph acyclic.
 *
 * `fuzzy.ts` is deliberately **not** among them. Its `wRatio_impl`, like every
 * other public scorer's implementation, validates and converts raw input — the
 * work already done by the time a branch below runs — so the composite strategy
 * is reproduced here over prepared state rather than called. The `wRatio` branch
 * mirrors that ladder of length-ratio tests and 0.95/0.9/0.6 scalings against
 * held state. The two must be kept in step by hand; a change to one is a change
 * to both, and `tests/fuzz/preparedParity.test.ts` is what says so out loud.
 */
import {
  alignRepresentation,
  convSequence,
  withChoicePreparer,
  prepareScorerChoice,
  preparedScorerSequence,
  type ChoicePreparer,
  type PrepareScorer,
  type PreparedScorerFactory,
  type PreparedScore,
} from '../../algorithms/shared/scorerSupport.js'
import type { PreparedFuzzKind } from '../types.js'
import {
  type CharSet,
  charSetOf,
  partialAlignmentConverted,
  partialRatioConverted,
  partialRatioImpl,
  ratioHeld,
} from './partialWindow.js'
import {
  hasWhitespaceOf,
  preparedTokenChoice,
  sortedOf,
  tokenChoicePreparer,
  tokenViewOf,
} from './tokens.js'
import {
  partialTokenRatioConverted,
  partialTokenSetRatioConverted,
  tokenRatioConverted,
  tokenSetRatioConverted,
} from './tokenSet.js'

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
    case 'partialRatio':
      return false
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
export function prepareFuzz(kind: PreparedFuzzKind): PreparedScorerFactory {
  const usesTokens = tokenisesInput(kind)
  // One preparer per scorer rather than per query: `prepareFuzz` runs once, at
  // the point each scorer below is built.
  const choicePreparer: ChoicePreparer = usesTokens
    ? tokenChoicePreparer()
    : prepareScorerChoice

  const prepare: PrepareScorer = (query) => {
    const queryTokenChoice = usesTokens
      ? preparedTokenChoice(choicePreparer(query))
      : null
    const heldQuery =
      queryTokenChoice === null
        ? preparedScorerSequence(prepareScorerChoice(query))
        : queryTokenChoice.sequence
    const a = heldQuery
    // Built on first use rather than up front: `partialRatio` and `wRatio` are
    // the only kinds that score through the query's own LCS masks, and `wRatio`
    // reaches them only on some inputs.
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

    // Each case reads the candidate for itself rather than a shared prelude
    // decoding it for all seven. The prelude had to be written for the union of
    // what the branches want — a `usesTokens` ternary, a `?.sequence`, a `??`
    // against the sequence preparer — and two branches then decoded the same
    // choice a second time to get a form the prelude had discarded the type of.
    // Six copies of one line buys every branch its own shape.
    const score: PreparedScore = (rawChoice, rawCutoff) => {
      const cutoff = rawCutoff ?? 0

      switch (kind) {
        case 'partialRatio': {
          const b = preparedScorerSequence(rawChoice)
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
        case 'tokenSetRatio': {
          const choice = preparedTokenChoice(rawChoice)
          return tokenSetRatioConverted(a, choice.sequence, cutoff, queryView, choice)
        }
        case 'tokenRatio': {
          const choice = preparedTokenChoice(rawChoice)
          return tokenRatioConverted(
            a,
            choice.sequence,
            cutoff,
            queryView,
            choice,
            sortedPatternOf,
          )
        }
        case 'partialTokenSortRatio': {
          const choice = preparedTokenChoice(rawChoice)
          // Through `partialAlignmentConverted` rather than
          // `partialRatioConverted`, so the sorted query's masks and pruning set
          // can be handed down. Without them this rebuilt the same mask for the
          // same sorted query once per candidate — the very thing the prepared
          // `partialRatio` path stopped doing.
          const sortedQuery = sortedOf(queryTokens)
          const sortedChoice = sortedOf(choice)
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
        case 'partialTokenSetRatio': {
          const choice = preparedTokenChoice(rawChoice)
          return partialTokenSetRatioConverted(
            a,
            choice.sequence,
            cutoff,
            queryView,
            choice,
          )
        }
        case 'partialTokenRatio': {
          const choice = preparedTokenChoice(rawChoice)
          return partialTokenRatioConverted(
            a,
            choice.sequence,
            cutoff,
            queryView,
            choice,
            sortedPatternOf,
            sortedCharSetOf,
          )
        }
        case 'wRatio': {
          const preparedTokens = preparedTokenChoice(rawChoice)
          const b = preparedTokens.sequence
          if (a.length === 0 || b.length === 0 || cutoff > 100) return 0
          const unbaseScale = 0.95
          const lenRatio = a.length > b.length ? a.length / b.length : b.length / a.length
          let dynamicCutoff = cutoff
          let result = ratioHeld(patternOf(), a.length, b, dynamicCutoff)

          if (lenRatio < 1.5) {
            // Raised ahead of the whitespace tests, exactly as `wRatio_impl`
            // does: every scorer below answers 0 to a cutoff above 100, so this
            // returns the same number without asking either side for its tokens.
            dynamicCutoff = Math.max(dynamicCutoff, result) / unbaseScale
            if (dynamicCutoff > 100) return result

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
            if (!hasWhitespaceOf(queryTokens) && !hasWhitespaceOf(preparedTokens))
              return result
            return Math.max(
              result,
              tokenRatioConverted(
                a,
                // Not through `tokenForm`, unlike `wRatio_impl`: a token kind
                // prepares its choices with `prepareTokenChoice`, which converts
                // unconditionally, so both sides are already the code points a
                // token set has to be compared in. The raw path needs the
                // expansion because it may still be holding two BMP strings.
                b,
                dynamicCutoff,
                queryView,
                preparedTokens,
                sortedPatternOf,
              ) * unbaseScale,
            )
          }

          const partialScale = lenRatio <= 8 ? 0.9 : 0.6
          dynamicCutoff = Math.max(dynamicCutoff, result) / partialScale
          // Rules out the window scan and the token component after it, whose
          // cutoff is this number divided again.
          if (dynamicCutoff > 100) return result

          // `nativeCharSetOf` without the representation test the `partialRatio`
          // branch needs, and the candidate without the `alignRepresentation`
          // that branch puts it through. The scan prunes with `===`, so the two
          // sides have to be spelled alike — and here they already are:
          // `prepareTokenChoice` converts every choice, and the query took the
          // same route, so neither can still be the BMP string that would meet
          // `97` as `'a'`. `partialRatio` prepares with `prepareScorerChoice`
          // instead, which does not convert, which is why the test stays there.
          const partial =
            a.length <= b.length
              ? partialRatioImpl(
                  a,
                  b,
                  dynamicCutoff / 100,
                  patternOf(),
                  true,
                  nativeCharSetOf(),
                ).score
              : partialRatioConverted(a, b, dynamicCutoff)
          result = Math.max(result, partial * partialScale)
          dynamicCutoff = Math.max(dynamicCutoff, result) / unbaseScale
          if (dynamicCutoff > 100) return result

          return Math.max(
            result,
            partialTokenRatioConverted(
              a,
              b,
              dynamicCutoff,
              queryView,
              preparedTokens,
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
