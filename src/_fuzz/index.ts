/**
 * Entry point for the fuzz scorer implementations.
 *
 * `fuzz.ts` is the public face of the module and holds no algorithm; everything
 * it exports is assembled from the modules below.
 *
 * ## What is where
 *
 * - `types.ts` — the shared data contracts. Imports nothing but types.
 * - `tokens.ts` — splitting, ordering, hashing, `UniqueTokenSet` and the
 *   prepared-choice branding. Holds the module-level ordinal counter and the
 *   brand `WeakSet`, which is why it cannot be split further.
 * - `basic.ts` — `ratio` and `partialRatio`, plus the processor plumbing. A
 *   *sibling* of `tokens.ts`, not a consumer: it must stay usable without
 *   tokenising anything.
 * - `tokenScorers.ts` — the six token scorers, which compose heavily enough that
 *   separating them would put a boundary through the middle of their control
 *   flow.
 * - `composite.ts` — `wRatio` and `qRatio`, which pick a strategy rather than
 *   define one, and so sit above every other family.
 * - `prepared.ts` — the per-query caching hook. It imports every core; no core
 *   imports it, which is what keeps the graph acyclic.
 *
 * The re-exports below are listed by name rather than with `export *`, for the
 * same reason the `distance` barrels are: `scripts/check-exports.mjs` fails the
 * build if a barrel widens and leaks an internal name.
 */
export { partialRatioAlignment, partialRatio_impl, ratio_impl } from './basic.js'
export { qRatio_impl, wRatio_impl } from './composite.js'
export { prepareFuzz } from './prepared.js'
export {
  partialTokenRatio_impl,
  partialTokenSetRatio_impl,
  partialTokenSortRatio_impl,
  tokenRatio_impl,
  tokenSetRatio_impl,
  tokenSortRatio_impl,
} from './tokenScorers.js'
export type { FuzzInput, FuzzOptions, PreparedFuzzKind, ScoreAlignment } from './types.js'
