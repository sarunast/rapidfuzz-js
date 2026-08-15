import type { Normalizer } from '../types.js'

/**
 * Assigned from inside the class body because its constructor is private.
 * These are runtime exports for `core/scoring/scorer` and `search/` alone: no
 * package entrypoint re-exports them, so no consumer can reach either door.
 */
export let createPreparedChoice: <TBrand>(
  owner: object,
  value: unknown,
  normalize: Normalizer | undefined,
) => PreparedChoice<TBrand>

export let resolvePreparedChoice: (
  owner: object,
  handle: unknown,
  normalize: Normalizer | undefined,
) => unknown

/**
 * The brand of a handle whose metric its type does not fix.
 *
 * `any` rather than `unknown` on purpose, and the one place this package uses
 * it: the brand is phantom and invariant, so `unknown` would be a brand like
 * any other — `Scorer<'similarity'>` would then accept no built-in scorer at
 * all. `any` is the wildcard the widening spellings need, and it gives up
 * nothing a caller could act on: the class has no member to reach.
 */
// oxlint-disable-next-line typescript/no-explicit-any -- the block above is the reason
export type AnyBrand = any

/**
 * A choice prepared by `scorer.prepareChoice`, opaque to the caller. The state
 * lives in `#` fields, so a spread or `Object.keys` sees nothing, and the
 * `Brand` parameter carries which built-in metric produced it — a Levenshtein
 * handle does not typecheck where a Jaro scorer's is expected.
 *
 * A built-in brand is the metric's own id literal, `'levenshtein.distance'`,
 * not a wrapper type: the brand is phantom and invariant, forgery is already
 * refused at runtime by `#owner`, and a bare literal is what lets a consumer's
 * own declaration emit name `Scorer<'similarity', 'fuzz.tokenSetRatio'>`
 * without importing anything of ours.
 */
export class PreparedChoice<TBrand = AnyBrand> {
  declare protected readonly brand: (value: TBrand) => TBrand

  readonly #owner: object
  readonly #value: unknown
  readonly #normalize: Normalizer | undefined

  private constructor(owner: object, value: unknown, normalize: Normalizer | undefined) {
    this.#owner = owner
    this.#value = value
    this.#normalize = normalize
  }

  static {
    createPreparedChoice = <TBrand>(
      owner: object,
      value: unknown,
      normalize: Normalizer | undefined,
    ) => new PreparedChoice<TBrand>(owner, value, normalize)
    resolvePreparedChoice = (owner, handle, normalize) => {
      if (typeof handle !== 'object' || handle === null || !(#owner in handle)) {
        throw new TypeError('getPrepared returned an invalid prepared choice')
      }
      if (handle.#owner !== owner) {
        throw new TypeError('prepared choice is incompatible with this scorer')
      }
      if (handle.#normalize !== normalize) {
        throw new TypeError(normalizerMismatch(handle.#normalize, normalize))
      }
      return handle.#value
    }
  }
}

function normalizerMismatch(
  prepared: Normalizer | undefined,
  search: Normalizer | undefined,
): string {
  if (search === undefined) {
    return 'prepared choice was normalized, this search is not'
  }
  if (prepared === undefined) {
    return 'this search normalizes, the prepared choice was not'
  }
  return 'prepared choice was normalized by a different function than this search'
}
