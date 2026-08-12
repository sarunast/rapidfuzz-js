/**
 * Assigned from inside the class body because the constructor is private —
 * anything reachable from an exported class is public API, and these are the
 * doors past it. Exported for `core/scorer` and `search/`, left out of every
 * public entrypoint.
 */
export let createPreparedChoice: <Brand>(
  owner: object,
  value: unknown,
) => PreparedChoice<Brand>

export let resolvePreparedChoice: (owner: object, handle: unknown) => unknown

/**
 * The brand of a handle whose metric its type does not fix.
 *
 * `any` rather than `unknown` on purpose, and the one place this package uses
 * it: the brand is phantom and invariant, so `unknown` would be a brand like
 * any other — `Scorer<'similarity'>` would then accept no built-in scorer at
 * all. `any` is the wildcard the widening spellings need, and it gives up
 * nothing a caller could act on: the class has no member to reach.
 */
export type AnyBrand = any

declare const METRIC_BRAND: unique symbol

/**
 * The brand of a built-in metric, told apart by the name the metric declares.
 *
 * Phantom, and keyed by a symbol so it reads as one: no value of this type is
 * ever made, and no ordinary object shape can be one by accident. The name is
 * a compile-time discriminator — nothing exposes it at runtime, and no API
 * takes one.
 */
export interface MetricBrand<Id extends string> {
  readonly [METRIC_BRAND]: Id
}

/**
 * A choice prepared by `scorer.prepareChoice`, opaque to the caller. The state
 * lives in `#` fields, so a spread or `Object.keys` sees nothing, and the
 * `Brand` parameter carries which built-in metric produced it — a Levenshtein
 * handle does not typecheck where a Jaro scorer's is expected.
 */
export class PreparedChoice<Brand = AnyBrand> {
  // Phantom, never assigned, never read. `(value: Brand) => Brand` keeps Brand
  // invariant: PreparedChoice<A> must not widen to PreparedChoice<A | B>.
  //
  // Protected rather than private because declaration emit erases the type of
  // a private member — `private brand;` — which would leave `Brand` unused in
  // the packed `.d.ts` and every consumer unprotected. Nothing can reach it:
  // the constructor is private, so the class cannot be extended.
  declare protected readonly brand: (value: Brand) => Brand

  readonly #owner: object
  readonly #value: unknown

  private constructor(owner: object, value: unknown) {
    this.#owner = owner
    this.#value = value
  }

  static {
    createPreparedChoice = (owner, value) => new PreparedChoice(owner, value)
    resolvePreparedChoice = (owner, handle) => {
      // `#owner in handle` refuses anything the private constructor did not
      // build, including `Object.create(PreparedChoice.prototype)` forgeries
      // and spread clones.
      if (typeof handle !== 'object' || handle === null || !(#owner in handle)) {
        throw new TypeError('getPrepared returned an invalid prepared choice')
      }
      if (handle.#owner !== owner) {
        throw new TypeError('prepared choice is incompatible with this scorer')
      }
      return handle.#value
    }
  }
}
