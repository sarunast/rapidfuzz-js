/**
 * Shared port of `rapidfuzz.distance._initialize` (see `_initialize_py.py`) — the
 * alignments it describes, not the containers it describes them in.
 *
 * Upstream's `Editops` and `Opcodes` are mutable Python sequences: indexable
 * with negative integers, sliceable with a step, deletable in place, and
 * holding tuples that unpack. Reproducing that meant reimplementing `list`,
 * and JavaScript already has one. What is kept here is what carries meaning —
 * the tags, the positions, the two forms and the conversions between them —
 * over readonly records in a readonly array, where `operations.at(-1)`,
 * `.filter(…)` and `.slice(…)` are the language's own.
 *
 * `srcLen` and `destLen` keep their upstream spelling under the camelCase rule
 * (`src_len`, `dest_len`), so those field names in upstream's documentation
 * still map. See README's "Differences from Python RapidFuzz" for the rest.
 */

import { hasSurrogatePair } from '../sequence.js'
import type { Editop, MatchingBlock, Opcode, OpcodeTag } from './types.js'

export type { Editop, EditopTag, MatchingBlock, Opcode, OpcodeTag } from './types.js'

/**
 * What a tag and a position have to be before the checks below mean anything.
 *
 * Every bound is written as a comparison, and a comparison against `NaN` or
 * `undefined` is false — so an unchecked position passed each of them and
 * reached code that assumes it is a whole number. A tag outside its union is
 * worse than useless: several branches test for one tag and treat everything
 * else as its opposite, so an unknown one is silently read as a deletion.
 */
function checkEditopTag(tag: unknown): void {
  if (tag !== 'replace' && tag !== 'delete' && tag !== 'insert') {
    throw new TypeError(`invalid edit operation tag ${String(tag)}`)
  }
}

function checkOpcodeTag(tag: unknown): void {
  if (tag !== 'replace' && tag !== 'delete' && tag !== 'insert' && tag !== 'equal') {
    throw new TypeError(`invalid edit operation tag ${String(tag)}`)
  }
}

// Safe rather than merely whole: past 2^53 adjacent integers collide, so a
// position up there is no longer an exact coordinate, and the per-element
// loops in `opcodesToEditops` would be asked to run for years.
function checkPosition(position: unknown): void {
  if (typeof position !== 'number' || !Number.isSafeInteger(position) || position < 0) {
    throw new TypeError('edit operation positions must be whole and at least zero')
  }
}

/**
 * The types below are a promise to a type checker, and the checks here are for
 * everything that reaches this from JavaScript, or through an `any`, having
 * made no such promise. Reading `.tag` off `null` would otherwise report a
 * list of operations as a missing property.
 */
function checkOperation(op: unknown): void {
  if (typeof op !== 'object' || op === null) {
    throw new TypeError('an edit operation must be an object')
  }
}

/** The declared lengths a list of operations is validated against. */
function checkLengths(srcLen: number, destLen: number): void {
  checkPosition(srcLen)
  checkPosition(destLen)
}

/**
 * Copied as they are checked, so that what is stored is this library's, not a
 * record the caller still holds a reference to and can write through.
 */
function checkedEditop(op: Editop): Editop {
  checkOperation(op)
  checkEditopTag(op.tag)
  checkPosition(op.srcPos)
  checkPosition(op.destPos)

  return { tag: op.tag, srcPos: op.srcPos, destPos: op.destPos }
}

/** Mutable while being built: {@link listToOpcodes} merges adjacent blocks. */
interface DraftOpcode {
  tag: OpcodeTag
  srcStart: number
  srcEnd: number
  destStart: number
  destEnd: number
}

function checkedOpcode(op: Opcode): DraftOpcode {
  checkOperation(op)
  checkOpcodeTag(op.tag)
  checkPosition(op.srcStart)
  checkPosition(op.srcEnd)
  checkPosition(op.destStart)
  checkPosition(op.destEnd)

  return {
    tag: op.tag,
    srcStart: op.srcStart,
    srcEnd: op.srcEnd,
    destStart: op.destStart,
    destEnd: op.destEnd,
  }
}

/** Port of `_list_to_editops`. */
function listToEditops(
  ops: readonly Editop[],
  srcLen: number,
  destLen: number,
): Editop[] {
  checkLengths(srcLen, destLen)

  const blocks: Editop[] = []
  // Ordering is checked against the operation before it as the list is built,
  // rather than in a second walk over the finished array: every operation is
  // in hand once already. It does mean a list that breaks two rules now
  // reports whichever comes first in reading order, where the second walk
  // always reported every bound before any ordering. Both are refusals.
  let previous: Editop | undefined

  for (const raw of ops) {
    const op = checkedEditop(raw)

    if (op.srcPos > srcLen || op.destPos > destLen) {
      throw new Error('List of edit operations invalid')
    }
    if (op.srcPos === srcLen && op.tag !== 'insert') {
      throw new Error('List of edit operations invalid')
    }
    if (op.destPos === destLen && op.tag !== 'delete') {
      throw new Error('List of edit operations invalid')
    }
    if (previous !== undefined) {
      if (op.srcPos < previous.srcPos || op.destPos < previous.destPos) {
        throw new Error('List of edit operations out of order')
      }
      if (op.srcPos === previous.srcPos && op.destPos === previous.destPos) {
        throw new Error('Duplicated edit operation')
      }
    }

    blocks.push(op)
    previous = op
  }

  return blocks
}

/** Port of `_list_to_opcodes`. */
function listToOpcodes(
  ops: readonly Opcode[],
  srcLen: number,
  destLen: number,
): Opcode[] {
  checkLengths(srcLen, destLen)

  // No operations still describes both strings in full, as the single `equal`
  // block that covers them — which is what upstream answers here too, by way
  // of converting an empty editop list.
  if (ops.length === 0) return editopsToOpcodes([], srcLen, destLen)

  const blocks: DraftOpcode[] = []

  for (const raw of ops) {
    const op = checkedOpcode(raw)

    if (op.srcEnd > srcLen || op.destEnd > destLen) {
      throw new Error('List of edit operations invalid')
    }
    if (op.srcEnd < op.srcStart || op.destEnd < op.destStart) {
      throw new Error('List of edit operations invalid')
    }
    if (
      (op.tag === 'equal' || op.tag === 'replace') &&
      (op.srcEnd - op.srcStart !== op.destEnd - op.destStart || op.srcStart === op.srcEnd)
    ) {
      throw new Error('List of edit operations invalid')
    }
    if (
      op.tag === 'insert' &&
      (op.srcStart !== op.srcEnd || op.destStart === op.destEnd)
    ) {
      throw new Error('List of edit operations invalid')
    }
    if (
      op.tag === 'delete' &&
      (op.srcStart === op.srcEnd || op.destStart !== op.destEnd)
    ) {
      throw new Error('List of edit operations invalid')
    }

    const last = blocks[blocks.length - 1]

    if (last !== undefined) {
      // Merge adjacent blocks with the same tag.
      if (
        last.tag === op.tag &&
        last.srcEnd === op.srcStart &&
        last.destEnd === op.destStart
      ) {
        last.srcEnd = op.srcEnd
        last.destEnd = op.destEnd
        continue
      }

      // Continuity, against the block this one follows — the same comparison
      // the second walk over the finished array used to make, minus the walk.
      // A merged block cannot fail it: merging is that comparison, plus a tag.
      if (last.srcEnd !== op.srcStart || last.destEnd !== op.destStart) {
        throw new Error('List of edit operations is not continuous')
      }
    }

    blocks.push(op)
  }

  const head = blocks[0]
  const tail = blocks[blocks.length - 1]

  if (head.srcStart !== 0 || head.destStart !== 0) {
    throw new Error('List of edit operations does not start at position 0')
  }
  if (tail.srcEnd !== srcLen || tail.destEnd !== destLen) {
    throw new Error('List of edit operations does not end at the string ends')
  }

  return blocks
}

/**
 * The two conversions, as plain functions over plain arrays, so that the
 * factories below can reach them without building an instance to discard.
 */
function editopsToOpcodes(
  ops: readonly Editop[],
  srcLen: number,
  destLen: number,
): Opcode[] {
  const blocks: Opcode[] = []
  let srcPos = 0
  let destPos = 0
  let i = 0

  while (i < ops.length) {
    const op = ops[i]

    if (srcPos < op.srcPos || destPos < op.destPos) {
      blocks.push({
        tag: 'equal',
        srcStart: srcPos,
        srcEnd: op.srcPos,
        destStart: destPos,
        destEnd: op.destPos,
      })
      srcPos = op.srcPos
      destPos = op.destPos
    }

    const srcBegin = srcPos
    const destBegin = destPos
    const { tag } = op

    while (
      i < ops.length &&
      ops[i].tag === tag &&
      srcPos === ops[i].srcPos &&
      destPos === ops[i].destPos
    ) {
      if (tag === 'replace') {
        srcPos++
        destPos++
      } else if (tag === 'insert') {
        destPos++
      } else {
        srcPos++
      }

      i++
    }

    blocks.push({
      tag,
      srcStart: srcBegin,
      srcEnd: srcPos,
      destStart: destBegin,
      destEnd: destPos,
    })
  }

  if (srcPos < srcLen || destPos < destLen) {
    blocks.push({
      tag: 'equal',
      srcStart: srcPos,
      srcEnd: srcLen,
      destStart: destPos,
      destEnd: destLen,
    })
  }

  return blocks
}

function opcodesToEditops(blocks: readonly Opcode[]): Editop[] {
  const ops: Editop[] = []

  for (const op of blocks) {
    if (op.tag === 'replace') {
      for (let j = 0; j < op.srcEnd - op.srcStart; j++) {
        ops.push({ tag: 'replace', srcPos: op.srcStart + j, destPos: op.destStart + j })
      }
    } else if (op.tag === 'insert') {
      for (let j = 0; j < op.destEnd - op.destStart; j++) {
        ops.push({ tag: 'insert', srcPos: op.srcStart, destPos: op.destStart + j })
      }
    } else if (op.tag === 'delete') {
      for (let j = 0; j < op.srcEnd - op.srcStart; j++) {
        ops.push({ tag: 'delete', srcPos: op.srcStart + j, destPos: op.destStart })
      }
    }
  }

  return ops
}

/**
 * A view of `s` that can be indexed by code point.
 *
 * Every position in an edit script counts code points, so astral text has to be
 * split apart before it can be indexed at all. BMP text does not: UTF-16
 * indices and code point indices agree there, and the string can then be sliced
 * natively rather than rebuilt one character at a time. That is the rule
 * `convPair` applies to scorer inputs, asked here of each string separately —
 * `apply` reads its source and its destination independently, so one astral
 * argument need not slow the other down.
 */
function codePointView(s: string): string | string[] {
  return hasSurrogatePair(s) ? Array.from(s) : s
}

/**
 * What both `apply` methods demand of their strings, in code points.
 *
 * Upstream checks neither length: its Python `Editops.apply` raises IndexError
 * where its `Opcodes.apply` silently clamps, and both pass extra source text
 * through — three behaviours for one mistake, and the JavaScript spellings of
 * two of them would splice `"undefined"` into the answer. One refusal instead.
 */
function checkApplyLengths(
  srcLength: number,
  destLength: number,
  srcLen: number,
  destLen: number,
): void {
  if (srcLength !== srcLen || destLength !== destLen) {
    throw new RangeError(
      'apply expects strings whose lengths match the recorded srcLen and destLen',
    )
  }
}

/**
 * A run of unchanged text, as a string.
 *
 * The array branch is what {@link Opcodes.apply} has always done. Copying an
 * array run character by character instead measured 1.4-2.7x faster on short
 * runs and 1.6x slower on 64k-character ones, so it is not the clear win it
 * looks like; the win worth taking is the string branch, which allocates no
 * array at all.
 */
function textSlice(view: string | string[], start: number, end: number): string {
  return typeof view === 'string'
    ? view.slice(start, end)
    : view.slice(start, end).join('')
}

/**
 * Build a collection from operations this library produced itself, skipping
 * every check the public factories make.
 *
 * Deliberately not a static method. Anything reachable from an exported class
 * is public API, and this is the one door past validation — `fromValidated`
 * with a hand-written record would put a negative position into a collection
 * that promises it holds none. These are exported for the scorers in sibling
 * producers and left out of public entrypoints, so consumers cannot
 * name them.
 *
 * Assigned from inside the class bodies because a private constructor can be
 * called from nowhere else.
 */
export let editopsFromValidated: (
  operations: readonly Editop[],
  srcLen: number,
  destLen: number,
) => Editops

let opcodesFromValidated: (
  operations: readonly Opcode[],
  srcLen: number,
  destLen: number,
) => Opcodes

/**
 * The operations that turn `s1` into `s2`, one character at a time.
 *
 * Readonly rather than deeply frozen: the array itself is frozen, so the
 * collection cannot be lengthened or reordered behind `srcLen`'s back, and the
 * records in it are readonly by contract — freezing each one would mean a call
 * per operation on a path that produces hundreds of thousands of them, to
 * restate what their type already says. Every method here answers with a new
 * collection rather than editing this one.
 */
export class Editops {
  readonly operations: readonly Editop[]

  readonly srcLen: number
  readonly destLen: number

  private constructor(operations: readonly Editop[], srcLen: number, destLen: number) {
    // One call, whatever the length: what it buys is that `operations.push(…)`
    // cannot leave `srcLen` describing a different alignment from the one
    // stored. See the class comment for what it deliberately does not buy.
    this.operations = Object.freeze(operations)
    this.srcLen = srcLen
    this.destLen = destLen
  }

  static {
    editopsFromValidated = (operations, srcLen, destLen) =>
      new Editops(operations, srcLen, destLen)
  }

  /** Build from operations that have not been checked, as upstream's constructor does. */
  static fromOperations(
    operations: readonly Editop[],
    srcLen: number,
    destLen: number,
  ): Editops {
    return new Editops(listToEditops(operations, srcLen, destLen), srcLen, destLen)
  }

  static fromOpcodes(opcodes: Opcodes): Editops {
    return opcodes.toEditops()
  }

  toOpcodes(): Opcodes {
    return opcodesFromValidated(
      editopsToOpcodes(this.operations, this.srcLen, this.destLen),
      this.srcLen,
      this.destLen,
    )
  }

  toMatchingBlocks(): readonly MatchingBlock[] {
    const blocks: MatchingBlock[] = []
    let srcPos = 0
    let destPos = 0

    for (const op of this.operations) {
      if (srcPos < op.srcPos || destPos < op.destPos) {
        const length = Math.min(op.srcPos - srcPos, op.destPos - destPos)
        if (length > 0) blocks.push({ srcStart: srcPos, destStart: destPos, length })
        srcPos = op.srcPos
        destPos = op.destPos
      }

      if (op.tag === 'replace') {
        srcPos++
        destPos++
      } else if (op.tag === 'delete') {
        srcPos++
      } else {
        destPos++
      }
    }

    if (srcPos < this.srcLen || destPos < this.destLen) {
      const length = Math.min(this.srcLen - srcPos, this.destLen - destPos)
      if (length > 0) blocks.push({ srcStart: srcPos, destStart: destPos, length })
    }

    blocks.push({ srcStart: this.srcLen, destStart: this.destLen, length: 0 })
    return blocks
  }

  /** Describe how to turn the destination back into the source. */
  inverse(): Editops {
    const blocks = this.operations.map((op): Editop => ({
      tag: op.tag === 'delete' ? 'insert' : op.tag === 'insert' ? 'delete' : op.tag,
      srcPos: op.destPos,
      destPos: op.srcPos,
    }))

    return new Editops(blocks, this.destLen, this.srcLen)
  }

  removeSubsequence(subsequence: Editops): Editops {
    if (subsequence.operations.length > this.operations.length) {
      throw new Error('subsequence is not a subsequence')
    }

    const result: Editop[] = []
    let offset = 0
    let opPos = 0

    const sameOp = (a: Editop, b: Editop): boolean =>
      a.tag === b.tag && a.srcPos === b.srcPos && a.destPos === b.destPos

    for (const sop of subsequence.operations) {
      while (opPos !== this.operations.length && !sameOp(sop, this.operations[opPos])) {
        const op = this.operations[opPos]
        result.push({ tag: op.tag, srcPos: op.srcPos + offset, destPos: op.destPos })
        opPos++
      }

      if (opPos === this.operations.length) {
        throw new Error('subsequence is not a subsequence')
      }

      if (sop.tag === 'insert') offset += 1
      else if (sop.tag === 'delete') offset -= 1

      opPos++
    }

    while (opPos !== this.operations.length) {
      const op = this.operations[opPos]
      result.push({ tag: op.tag, srcPos: op.srcPos + offset, destPos: op.destPos })
      opPos++
    }

    return new Editops(result, this.srcLen, this.destLen)
  }

  /**
   * Apply these operations to `source`, drawing replacements from `destination`.
   *
   * The destination is only ever read one character at a time, so a view of it
   * serves both paths. The source is not: everything between two operations is
   * copied across unchanged, and how that run is copied is the whole cost of
   * this method. A BMP source hands the run to `slice`, which measured 2-3x
   * faster on edit-dense scripts and three orders of magnitude on sparse ones,
   * where the old loop copied a whole string one character at a time. The
   * astral loop below is left as it was: routing it through a slice instead
   * measured 1.3-2.8x *slower*, because its runs are short and each one then
   * pays for a call and an intermediate string.
   */
  apply(source: string, destination: string): string {
    const dest = codePointView(destination)
    let out = ''
    let srcPos = 0

    if (!hasSurrogatePair(source)) {
      checkApplyLengths(source.length, dest.length, this.srcLen, this.destLen)

      for (const op of this.operations) {
        if (srcPos < op.srcPos) {
          out += source.slice(srcPos, op.srcPos)
          srcPos = op.srcPos
        }

        if (op.tag === 'replace') {
          out += dest[op.destPos]
          srcPos++
        } else if (op.tag === 'insert') {
          out += dest[op.destPos]
        } else {
          srcPos++
        }
      }

      return srcPos < source.length ? out + source.slice(srcPos) : out
    }

    const src = Array.from(source)
    checkApplyLengths(src.length, dest.length, this.srcLen, this.destLen)

    for (const op of this.operations) {
      while (srcPos < op.srcPos) {
        out += src[srcPos]
        srcPos++
      }

      if (op.tag === 'replace') {
        out += dest[op.destPos]
        srcPos++
      } else if (op.tag === 'insert') {
        out += dest[op.destPos]
      } else {
        srcPos++
      }
    }

    while (srcPos < src.length) {
      out += src[srcPos]
      srcPos++
    }

    return out
  }

  equals(other: Editops): boolean {
    if (this.srcLen !== other.srcLen || this.destLen !== other.destLen) return false
    if (this.operations.length !== other.operations.length) return false

    // Indexed rather than `every`, which pays for a call per operation: 3.4x
    // over a thousand of them, and level with it once the two arrays are large
    // enough that fetching them is the cost.
    for (let i = 0; i < this.operations.length; i++) {
      const a = this.operations[i]
      const b = other.operations[i]

      if (a.tag !== b.tag || a.srcPos !== b.srcPos || a.destPos !== b.destPos)
        return false
    }

    return true
  }
}

/** The same alignment in runs, rather than one character at a time. */
export class Opcodes {
  readonly operations: readonly Opcode[]
  readonly srcLen: number
  readonly destLen: number

  private constructor(operations: readonly Opcode[], srcLen: number, destLen: number) {
    this.operations = Object.freeze(operations)
    this.srcLen = srcLen
    this.destLen = destLen
  }

  static {
    opcodesFromValidated = (operations, srcLen, destLen) =>
      new Opcodes(operations, srcLen, destLen)
  }

  static fromOperations(
    operations: readonly Opcode[],
    srcLen: number,
    destLen: number,
  ): Opcodes {
    return new Opcodes(listToOpcodes(operations, srcLen, destLen), srcLen, destLen)
  }

  static fromEditops(editops: Editops): Opcodes {
    return editops.toOpcodes()
  }

  toEditops(): Editops {
    return editopsFromValidated(
      opcodesToEditops(this.operations),
      this.srcLen,
      this.destLen,
    )
  }

  toMatchingBlocks(): readonly MatchingBlock[] {
    const blocks: MatchingBlock[] = []
    for (const op of this.operations) {
      if (op.tag !== 'equal') continue
      const length = Math.min(op.srcEnd - op.srcStart, op.destEnd - op.destStart)
      if (length > 0) {
        blocks.push({ srcStart: op.srcStart, destStart: op.destStart, length })
      }
    }
    blocks.push({ srcStart: this.srcLen, destStart: this.destLen, length: 0 })
    return blocks
  }

  inverse(): Opcodes {
    const blocks = this.operations.map((op): Opcode => ({
      tag: op.tag === 'delete' ? 'insert' : op.tag === 'insert' ? 'delete' : op.tag,
      srcStart: op.destStart,
      srcEnd: op.destEnd,
      destStart: op.srcStart,
      destEnd: op.srcEnd,
    }))
    return new Opcodes(blocks, this.destLen, this.srcLen)
  }

  apply(source: string, destination: string): string {
    const src = codePointView(source)
    const dest = codePointView(destination)
    checkApplyLengths(src.length, dest.length, this.srcLen, this.destLen)
    let out = ''
    for (const op of this.operations) {
      if (op.tag === 'equal') {
        out += textSlice(src, op.srcStart, op.srcEnd)
      } else if (op.tag === 'replace' || op.tag === 'insert') {
        out += textSlice(dest, op.destStart, op.destEnd)
      }
    }
    return out
  }

  equals(other: Opcodes): boolean {
    if (this.srcLen !== other.srcLen || this.destLen !== other.destLen) return false
    if (this.operations.length !== other.operations.length) return false
    for (let i = 0; i < this.operations.length; i++) {
      const a = this.operations[i]
      const b = other.operations[i]
      if (
        a.tag !== b.tag ||
        a.srcStart !== b.srcStart ||
        a.srcEnd !== b.srcEnd ||
        a.destStart !== b.destStart ||
        a.destEnd !== b.destEnd
      ) {
        return false
      }
    }
    return true
  }
}
