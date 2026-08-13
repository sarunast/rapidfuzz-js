/**
 * Upstream's `as_list()`, kept in the harness rather than on the collections.
 *
 * `Editops` and `Opcodes` hold records here — the Python list protocol and the
 * tuples that came with it are not part of this library's API. Every expected
 * value in the ported suites is still transcribed from RapidFuzz's own tests,
 * where an operation *is* a tuple, so the conversion has to live somewhere;
 * putting it here keeps those tables comparable to upstream's line for line,
 * and keeps the API from carrying a method that exists for the tests.
 */
import type {
  EditopTag,
  Editops,
  OpcodeTag,
  Opcodes,
} from '../src/algorithms/shared/editops/index.js'

/** Flatten an `Editops` record list into upstream's `(tag, srcPos, destPos)`. */
export function editopTuples(ops: Editops): Array<[EditopTag, number, number]> {
  return ops.operations.map((op) => [op.tag, op.srcPos, op.destPos])
}

/** Flatten an `Opcodes` record list into upstream's five-field tuple. */
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
