import type { ScorerOptions } from '#core/scoring/builtIn/options.js'

/** Cost of an insertion, a deletion, and a substitution, in that order. */
export type LevenshteinWeights = readonly [
  insert: number,
  delete_: number,
  replace: number,
]

/**
 * The same three costs, named.
 *
 * Upstream takes only the tuple, because that is what its C++ signature takes.
 * Nothing at a call site says which of `[1, 1, 2]` is the substitution, and
 * getting the order wrong produces a plausible wrong number rather than an
 * error — so the named form is the one to reach for, and the tuple stays for
 * parity with upstream's docs.
 */
export interface LevenshteinCosts {
  readonly insertion: number
  readonly deletion: number
  readonly substitution: number
}

/** Either spelling of the three costs. */
export type LevenshteinWeightsInput = LevenshteinWeights | LevenshteinCosts

export interface LevenshteinOptions extends ScorerOptions {
  /**
   * Defaults to uniform costs of `1`. Accepts
   * `{ insertion, deletion, substitution }` or the positional
   * `[insertion, deletion, substitution]`.
   */
  weights?: LevenshteinWeightsInput | undefined
}
