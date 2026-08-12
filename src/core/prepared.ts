/**
 * Assigned from inside the class body because its constructor is private.
 * These are runtime exports for `core/scorer` and `search/` alone: no package
 * entrypoint re-exports them, so no consumer can reach either door.
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

/**
 * A choice prepared by `scorer.prepareChoice`, opaque to the caller. The state
 * lives in `#` fields, so a spread or `Object.keys` sees nothing, and the
 * `Brand` parameter carries which built-in metric produced it — a Levenshtein
 * handle does not typecheck where a Jaro scorer's is expected.
 *
 * A built-in brand is the metric's own id literal, `'levenshtein.distance'`,
 * not a wrapper type: the brand is phantom and invariant, forgery is already
 * refused at runtime by `#owner`, and a bare literal is what lets a consumer's
 * own declaration emit name `Scorer<'similarity', 'fuzz.tokenSetSimilarity'>`
 * without importing anything of ours.
 */
export class PreparedChoice<Brand = AnyBrand> {
  // Phantom, never assigned, never read. `(value: Brand) => Brand` keeps Brand
  // invariant: PreparedChoice<A> must not widen to PreparedChoice<A | B>.
  //
  // Protected rather than private because declaration emit erases the type of
  // a private member — `private brand;` — which would leave `Brand` unused in
  // the packed `.d.ts` and every consumer unprotected. The private constructor
  // refuses `extends` through the TypeScript API rather than at runtime, which
  // is enough here: the class is exported as a type, never as a value.
  declare protected readonly brand: (value: Brand) => Brand

  readonly #owner: object
  readonly #value: unknown

  private constructor(owner: object, value: unknown) {
    this.#owner = owner
    this.#value = value
  }

  static {
    createPreparedChoice = <Brand>(owner: object, value: unknown) =>
      new PreparedChoice<Brand>(owner, value)
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
