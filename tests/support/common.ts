/**
 * Port of RapidFuzz's `tests/common.py` + `tests/distance/common.py`.
 *
 * Only representation helpers shared by the canonical algorithm tests live
 * here. Metric/scorer behavior is exercised through public subpath metrics.
 */
import type {
  EditopTag,
  Editops,
  MatchingBlock,
  OpcodeTag,
  Opcodes,
} from '../../src/algorithms/shared/editops/index.js'

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
 * contravariantly; `R` still comes back precisely, so a result can be indexed
 * or awaited as usual.
 */
export function callUntyped<R>(fn: (...args: never[]) => R, ...args: unknown[]): R {
  return Reflect.apply(fn, undefined, args)
}

/**
 * Upstream's `as_list()`, kept in the harness rather than on the collections.
 *
 * `Editops` and `Opcodes` hold records here — the Python list protocol and the
 * tuples that came with it are not part of this library's API. Every expected
 * value below is still transcribed from RapidFuzz's own suite, where an
 * operation *is* a tuple, so the conversion has to live somewhere; putting it
 * here keeps those tables comparable to upstream's line for line, and keeps
 * the API from carrying a method that exists for the tests.
 */
export function editopTuples(ops: Editops): Array<[EditopTag, number, number]> {
  return ops.operations.map((op) => [op.tag, op.srcPos, op.destPos])
}

export function opcodeTuples(
  ops: Opcodes,
): Array<[OpcodeTag, number, number, number, number]> {
  return ops.operations.map((op) => [
    op.tag,
    op.srcStart,
    op.srcEnd,
    op.destStart,
    op.destEnd,
  ])
}

export function blockTuples(
  blocks: readonly MatchingBlock[],
): Array<[number, number, number]> {
  return blocks.map((block) => [block.srcStart, block.destStart, block.length])
}
