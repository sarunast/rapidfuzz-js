/**
 * Call something with an argument its parameter types do not admit.
 *
 * Two ported behaviours need this. `NaN` is still "missing" at runtime, for
 * parity with Python's `float("nan")`, but it is deliberately not in
 * `MaybeSequence` — see the comment there. And a non-`NaN` number is the case
 * upstream raises `TypeError` on, which is now also a compile error, so the
 * test that pins the throw cannot spell the call directly.
 *
 * Routing through `Reflect.apply` rather than asserting a type is the same
 * trick `callScorer` in `src/process.ts` uses at the one place a scorer's
 * concrete inputs meet a caller that only knows it holds *some* scorer. The
 * rest parameter is `never[]` so that every function is assignable to it
 * contravariantly; `TResult` still comes back precisely, so a result can be indexed
 * or awaited as usual.
 */
export function callUntyped<TResult>(
  fn: (...args: never[]) => TResult,
  ...args: unknown[]
): TResult {
  return Reflect.apply(fn, undefined, args)
}
