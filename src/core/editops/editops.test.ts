// Ported from RapidFuzz tests/distance/test_init.py, minus the half of that
// file that exercises Python's list protocol — indexing, slicing, `del`, tuple
// unpacking. Those tests do not survive the port because what they test is
// `list`, and `operations` is a JavaScript array: `at(-1)`, `.slice(…)` and
// `.filter(…)` are the language's, tested by the language.
//
// Upstream's tuples are transcribed as the records this library holds. The
// values are the same values.
import { describe, expect, it } from 'vitest'

import * as indel from '#algorithms/indel/index.js'
import {
  levenshteinEditops,
  levenshteinOpcodes,
} from '#algorithms/levenshtein/editops.js'

import { callUntyped } from '../../../testing/untyped.js'
import { Editops, Opcodes, type Editop, type EditopTag, type Opcode } from './index.js'

it('exposes canonical edit-operation producers in both forms', () => {
  const blocks = indel.opcodes('abc', 'axc')
  expect(blocks.toEditops().operations).toHaveLength(2)
  expect(blocks.toMatchingBlocks()).toEqual([
    { srcStart: 0, destStart: 0, length: 1 },
    { srcStart: 2, destStart: 2, length: 1 },
    { srcStart: 3, destStart: 3, length: 0 },
  ])
})

const EDITOP_LIST: readonly Editop[] = [
  { tag: 'delete', srcPos: 1, destPos: 1 },
  { tag: 'replace', srcPos: 2, destPos: 1 },
  { tag: 'insert', srcPos: 6, destPos: 5 },
  { tag: 'insert', srcPos: 6, destPos: 6 },
  { tag: 'insert', srcPos: 6, destPos: 7 },
]

const OPCODE_LIST: readonly Opcode[] = [
  { tag: 'equal', srcStart: 0, srcEnd: 1, destStart: 0, destEnd: 1 },
  { tag: 'delete', srcStart: 1, srcEnd: 2, destStart: 1, destEnd: 1 },
  { tag: 'replace', srcStart: 2, srcEnd: 3, destStart: 1, destEnd: 2 },
  { tag: 'equal', srcStart: 3, srcEnd: 6, destStart: 2, destEnd: 5 },
  { tag: 'insert', srcStart: 6, srcEnd: 6, destStart: 5, destEnd: 8 },
  { tag: 'equal', srcStart: 6, srcEnd: 7, destStart: 8, destEnd: 9 },
]

function editops(): Editops {
  return Editops.fromOperations(EDITOP_LIST, 7, 9)
}

function opcodes(): Opcodes {
  return Opcodes.fromOperations(OPCODE_LIST, 7, 9)
}

it('holds the operations it was given', () => {
  expect(editops().operations).toEqual(EDITOP_LIST)
  expect(opcodes().operations).toEqual(OPCODE_LIST)
  expect(editops().srcLen).toBe(7)
  expect(editops().destLen).toBe(9)
})

it('iterates its operations, and counts them without handing the array out', () => {
  expect([...editops()]).toEqual(EDITOP_LIST)
  expect([...opcodes()]).toEqual(OPCODE_LIST)
  expect(editops().length).toBe(EDITOP_LIST.length)
  expect(opcodes().length).toBe(OPCODE_LIST.length)

  const seen: EditopTag[] = []
  for (const op of editops()) seen.push(op.tag)
  expect(seen).toEqual(['delete', 'replace', 'insert', 'insert', 'insert'])

  const empty = Editops.fromOperations([], 0, 0)
  expect([...empty]).toEqual([])
  expect(empty.length).toBe(0)
})

it('compares Editops for equality', () => {
  const ops = levenshteinEditops('aaabaaa', 'abbaaabba')
  const rebuilt = Editops.fromOperations(ops.operations, ops.srcLen, ops.destLen)

  expect(ops.equals(ops)).toBe(true)
  expect(ops.equals(rebuilt)).toBe(true)
  expect(ops.equals(Editops.fromOperations([], ops.srcLen, ops.destLen))).toBe(false)
})

it('compares Opcodes for equality', () => {
  const ops = levenshteinOpcodes('aaabaaa', 'abbaaabba')
  const rebuilt = Opcodes.fromOperations(ops.operations, ops.srcLen, ops.destLen)

  expect(ops.equals(ops)).toBe(true)
  expect(ops.equals(rebuilt)).toBe(true)
})

it('inverts Editops', () => {
  expect(editops().inverse().operations).toEqual([
    { tag: 'insert', srcPos: 1, destPos: 1 },
    { tag: 'replace', srcPos: 1, destPos: 2 },
    { tag: 'delete', srcPos: 5, destPos: 6 },
    { tag: 'delete', srcPos: 6, destPos: 6 },
    { tag: 'delete', srcPos: 7, destPos: 6 },
  ])
})

it('inverts Opcodes', () => {
  expect(opcodes().inverse().operations).toEqual([
    { tag: 'equal', srcStart: 0, srcEnd: 1, destStart: 0, destEnd: 1 },
    { tag: 'insert', srcStart: 1, srcEnd: 1, destStart: 1, destEnd: 2 },
    { tag: 'replace', srcStart: 1, srcEnd: 2, destStart: 2, destEnd: 3 },
    { tag: 'equal', srcStart: 2, srcEnd: 5, destStart: 3, destEnd: 6 },
    { tag: 'delete', srcStart: 5, srcEnd: 8, destStart: 6, destEnd: 6 },
    { tag: 'equal', srcStart: 8, srcEnd: 9, destStart: 6, destEnd: 7 },
  ])
})

it('converts an empty list to Opcodes', () => {
  let ops = Opcodes.fromOperations([], 0, 0)
  expect(ops.operations).toEqual([])
  expect(ops.srcLen).toBe(0)
  expect(ops.destLen).toBe(0)

  ops = Opcodes.fromOperations([], 0, 3)
  expect(ops.operations).toEqual([
    { tag: 'equal', srcStart: 0, srcEnd: 0, destStart: 0, destEnd: 3 },
  ])
  expect(ops.srcLen).toBe(0)
  expect(ops.destLen).toBe(3)
})

it('converts an empty list to Editops', () => {
  let ops = Editops.fromOperations([], 0, 0)
  expect(ops.operations).toEqual([])
  expect(ops.srcLen).toBe(0)
  expect(ops.destLen).toBe(0)

  ops = Editops.fromOperations([], 0, 3)
  expect(ops.operations).toEqual([])
  expect(ops.srcLen).toBe(0)
  expect(ops.destLen).toBe(3)
})

// Upstream builds either collection from either shape by looking at the length
// of the first tuple. Here the two shapes are two types, and each conversion is
// named — but the round trips those tests pin down still have to hold.
it('round-trips between the two forms', () => {
  for (const [s1, s2] of [
    ['aaabaaa', 'abbaaabba'],
    ['skdsakldsakdlasda', 'djkajkdfkdgkhdfjrmecsidjf'],
  ]) {
    const eops = levenshteinEditops(s1, s2)
    const ops = levenshteinOpcodes(s1, s2)

    expect(eops.toOpcodes().equals(ops)).toBe(true)
    expect(ops.toEditops().equals(eops)).toBe(true)
    expect(Editops.fromOpcodes(ops).equals(eops)).toBe(true)
    expect(Opcodes.fromEditops(eops).equals(ops)).toBe(true)
    expect(eops.toOpcodes().toEditops().equals(eops)).toBe(true)
  }
})

it('merges adjacent blocks', () => {
  const merged: Opcode[] = [
    { tag: 'equal', srcStart: 0, srcEnd: 3, destStart: 0, destEnd: 3 },
  ]
  const split: Opcode[] = [
    { tag: 'equal', srcStart: 0, srcEnd: 1, destStart: 0, destEnd: 1 },
    { tag: 'equal', srcStart: 1, srcEnd: 3, destStart: 1, destEnd: 3 },
  ]

  expect(
    Opcodes.fromOperations(merged, 3, 3).equals(Opcodes.fromOperations(split, 3, 3)),
  ).toBe(true)
  expect(
    Opcodes.fromOperations(split, 3, 3).equals(
      Opcodes.fromOperations(split, 3, 3).toEditops().toOpcodes(),
    ),
  ).toBe(true)
})

it('handles empty matching blocks', () => {
  const empty = { srcStart: 0, destStart: 0, length: 0 }

  expect(Editops.fromOperations([], 0, 0).toMatchingBlocks()).toEqual([empty])
  expect(Editops.fromOperations([], 0, 3).toMatchingBlocks()).toEqual([
    { srcStart: 0, destStart: 3, length: 0 },
  ])
  expect(Editops.fromOperations([], 3, 0).toMatchingBlocks()).toEqual([
    { srcStart: 3, destStart: 0, length: 0 },
  ])

  expect(Opcodes.fromOperations([], 0, 0).toMatchingBlocks()).toEqual([empty])
  expect(Opcodes.fromOperations([], 0, 3).toMatchingBlocks()).toEqual([
    { srcStart: 0, destStart: 3, length: 0 },
  ])
  expect(Opcodes.fromOperations([], 3, 0).toMatchingBlocks()).toEqual([
    { srcStart: 3, destStart: 0, length: 0 },
  ])
})

// Not ported — upstream raises on all of these, but its tests do not cover
// them, and in JavaScript the same inputs did not merely give a wrong answer.
describe('malformed input is refused rather than acted on', () => {
  it('rejects a tag no branch knows how to read', () => {
    // A tag outside the set is read as its opposite by branches that test for
    // one tag and treat everything else as the other, so an unknown one had
    // been silently applied as a deletion.
    const badEditop = { tag: 'equal', srcPos: 0, destPos: 0 }
    const badOpcode = { tag: 'foo', srcStart: 0, srcEnd: 3, destStart: 0, destEnd: 3 }

    expect(() => callUntyped(Editops.fromOperations, [badEditop], 3, 3)).toThrow(
      TypeError,
    )
    expect(() => callUntyped(Opcodes.fromOperations, [badOpcode], 3, 3)).toThrow(
      TypeError,
    )
  })

  it('rejects positions and lengths that are no position at all', () => {
    expect(() =>
      Editops.fromOperations([{ tag: 'delete', srcPos: -1, destPos: 0 }], 3, 3),
    ).toThrow(TypeError)
    expect(() =>
      Editops.fromOperations([{ tag: 'delete', srcPos: 0, destPos: Number.NaN }], 3, 3),
    ).toThrow(TypeError)
    expect(() =>
      Editops.fromOperations([{ tag: 'delete', srcPos: 0.5, destPos: 0 }], 3, 3),
    ).toThrow(TypeError)
    expect(() => Editops.fromOperations([], -1, 3)).toThrow(TypeError)
    expect(() => Editops.fromOperations([], 3, Number.NaN)).toThrow(TypeError)
    expect(() => Opcodes.fromOperations([], 3, -1)).toThrow(TypeError)
    // Past 2^53 adjacent integers collide, so a coordinate up there is not one.
    expect(() => Editops.fromOperations([], 2 ** 53, 3)).toThrow(TypeError)
    expect(() =>
      Opcodes.fromOperations(
        [{ tag: 'equal', srcStart: 0, srcEnd: 2 ** 53, destStart: 0, destEnd: 3 }],
        2 ** 53,
        3,
      ),
    ).toThrow(TypeError)
  })

  // Upstream checks neither string against the recorded lengths, and its three
  // paths answer the same mistake three ways — IndexError, silent clamping,
  // and extra source text passed through. Here both forms refuse alike.
  it('rejects apply() strings whose lengths do not match the recorded ones', () => {
    const ops = Editops.fromOperations([{ tag: 'replace', srcPos: 1, destPos: 1 }], 3, 3)
    const blocks = ops.toOpcodes()

    expect(() => ops.apply('abc', 'x')).toThrow(RangeError)
    expect(() => ops.apply('abcd', 'xyz')).toThrow(RangeError)
    expect(() => blocks.apply('abc', 'x')).toThrow(RangeError)
    expect(() => blocks.apply('abcd', 'xyz')).toThrow(RangeError)

    // Lengths count code points, not UTF-16 units, on the astral path too.
    expect(() => ops.apply('a💩c', 'x💩zw')).toThrow(RangeError)
    expect(ops.apply('a💩c', 'x💩z')).toBe('a💩c')
    expect(ops.apply('abc', 'x💩z')).toBe('a💩c')
    expect(blocks.apply('abc', 'x💩z')).toBe('a💩c')
  })

  // Reading `.tag` off one of these reported a list of operations as a missing
  // property, several frames away from the list that was wrong.
  it('rejects an operation that is not an object', () => {
    for (const malformed of [null, undefined, 3, 'delete']) {
      expect(() => callUntyped(Editops.fromOperations, [malformed], 1, 1)).toThrow(
        TypeError,
      )
      expect(() => callUntyped(Opcodes.fromOperations, [malformed], 1, 1)).toThrow(
        TypeError,
      )
    }
  })

  it('rejects operations that do not describe one alignment', () => {
    expect(() =>
      Editops.fromOperations(
        [
          { tag: 'delete', srcPos: 2, destPos: 1 },
          { tag: 'delete', srcPos: 1, destPos: 1 },
        ],
        7,
        9,
      ),
    ).toThrow('List of edit operations out of order')

    expect(() =>
      Editops.fromOperations(
        [
          { tag: 'delete', srcPos: 1, destPos: 1 },
          { tag: 'delete', srcPos: 1, destPos: 1 },
        ],
        7,
        9,
      ),
    ).toThrow('Duplicated edit operation')

    expect(() =>
      Opcodes.fromOperations(
        [{ tag: 'equal', srcStart: 1, srcEnd: 3, destStart: 1, destEnd: 3 }],
        3,
        3,
      ),
    ).toThrow('List of edit operations does not start at position 0')

    expect(() =>
      Opcodes.fromOperations(
        [{ tag: 'equal', srcStart: 0, srcEnd: 2, destStart: 0, destEnd: 2 }],
        3,
        3,
      ),
    ).toThrow('List of edit operations does not end at the string ends')

    expect(() =>
      Opcodes.fromOperations(
        [
          { tag: 'equal', srcStart: 0, srcEnd: 1, destStart: 0, destEnd: 1 },
          { tag: 'replace', srcStart: 2, srcEnd: 3, destStart: 2, destEnd: 3 },
        ],
        3,
        3,
      ),
    ).toThrow('List of edit operations is not continuous')
  })
})

// Also not ported, and the other half of the same story: upstream's
// `_list_to_editops` and `_list_to_opcodes` raise on each of these, and its
// suite exercises none of them. Every rule gets both of its sub-conditions,
// because each is a separate way for an alignment to be impossible.
describe('operations that describe no alignment at all', () => {
  it('refuses editops outside the lengths they claim', () => {
    expect(() =>
      Editops.fromOperations([{ tag: 'delete', srcPos: 8, destPos: 0 }], 7, 9),
    ).toThrow('List of edit operations invalid')
    expect(() =>
      Editops.fromOperations([{ tag: 'delete', srcPos: 0, destPos: 10 }], 7, 9),
    ).toThrow('List of edit operations invalid')
  })

  // Past the end of the source there is nothing left to delete or replace, and
  // past the end of the destination nothing left to insert or replace.
  it('refuses an editop at a position with nothing to edit', () => {
    for (const tag of ['delete', 'replace'] as const) {
      expect(() =>
        Editops.fromOperations([{ tag, srcPos: 7, destPos: 0 }], 7, 9),
      ).toThrow('List of edit operations invalid')
    }
    for (const tag of ['insert', 'replace'] as const) {
      expect(() =>
        Editops.fromOperations([{ tag, srcPos: 0, destPos: 9 }], 7, 9),
      ).toThrow('List of edit operations invalid')
    }
  })

  it('refuses opcodes outside the lengths they claim', () => {
    expect(() =>
      Opcodes.fromOperations(
        [{ tag: 'equal', srcStart: 0, srcEnd: 4, destStart: 0, destEnd: 3 }],
        3,
        3,
      ),
    ).toThrow('List of edit operations invalid')
    expect(() =>
      Opcodes.fromOperations(
        [{ tag: 'equal', srcStart: 0, srcEnd: 3, destStart: 0, destEnd: 4 }],
        3,
        3,
      ),
    ).toThrow('List of edit operations invalid')
  })

  it('refuses a block that ends before it starts', () => {
    expect(() =>
      Opcodes.fromOperations(
        [{ tag: 'equal', srcStart: 2, srcEnd: 1, destStart: 0, destEnd: 1 }],
        3,
        3,
      ),
    ).toThrow('List of edit operations invalid')
    expect(() =>
      Opcodes.fromOperations(
        [{ tag: 'equal', srcStart: 0, srcEnd: 1, destStart: 2, destEnd: 1 }],
        3,
        3,
      ),
    ).toThrow('List of edit operations invalid')
  })

  // Each tag constrains the two spans: `equal` and `replace` cover the same
  // non-zero number of elements on both sides, `insert` covers none of the
  // source and some of the destination, `delete` the other way round.
  it('refuses a block whose spans contradict its tag', () => {
    const malformed: readonly Opcode[] = [
      { tag: 'equal', srcStart: 0, srcEnd: 2, destStart: 0, destEnd: 1 },
      { tag: 'replace', srcStart: 0, srcEnd: 0, destStart: 0, destEnd: 0 },
      { tag: 'insert', srcStart: 0, srcEnd: 1, destStart: 0, destEnd: 1 },
      { tag: 'insert', srcStart: 0, srcEnd: 0, destStart: 0, destEnd: 0 },
      { tag: 'delete', srcStart: 0, srcEnd: 0, destStart: 0, destEnd: 0 },
      { tag: 'delete', srcStart: 0, srcEnd: 1, destStart: 0, destEnd: 1 },
    ]

    for (const op of malformed) {
      expect(() => Opcodes.fromOperations([op], 3, 3), JSON.stringify(op)).toThrow(
        'List of edit operations invalid',
      )
    }
  })
})

// Upstream's `Editops.remove_subsequence`: given the operations that turn `a`
// into `c` and the subset of them that turns `a` into `b`, the ones left over
// turn `b` into `c`. The positions of what is left shift by what was removed —
// an insertion that is gone leaves the source one element shorter than the
// remaining operations expect, and a deletion one longer.
describe('removing a subsequence of operations', () => {
  const OPS: readonly Editop[] = [
    { tag: 'insert', srcPos: 1, destPos: 1 },
    { tag: 'delete', srcPos: 1, destPos: 2 },
    { tag: 'replace', srcPos: 2, destPos: 3 },
  ]
  const all = (): Editops => Editops.fromOperations(OPS, 3, 4)

  it('leaves everything when nothing is removed', () => {
    expect(all().removeSubsequence(Editops.fromOperations([], 3, 4)).operations).toEqual(
      OPS,
    )
  })

  it('shifts what follows a removed insertion forwards', () => {
    const removed = all().removeSubsequence(Editops.fromOperations([OPS[0]], 3, 4))
    expect(removed.operations).toEqual([
      { tag: 'delete', srcPos: 2, destPos: 2 },
      { tag: 'replace', srcPos: 3, destPos: 3 },
    ])
  })

  it('shifts what follows a removed deletion back', () => {
    const removed = all().removeSubsequence(Editops.fromOperations([OPS[1]], 3, 4))
    expect(removed.operations).toEqual([
      { tag: 'insert', srcPos: 1, destPos: 1 },
      { tag: 'replace', srcPos: 1, destPos: 3 },
    ])
  })

  it('leaves the operations before and after a removed replacement alone', () => {
    const removed = all().removeSubsequence(Editops.fromOperations([OPS[2]], 3, 4))
    expect(removed.operations).toEqual([OPS[0], OPS[1]])
  })

  it('refuses a list that is not a subsequence', () => {
    expect(() =>
      all().removeSubsequence(
        Editops.fromOperations(
          [
            { tag: 'insert', srcPos: 0, destPos: 0 },
            { tag: 'insert', srcPos: 0, destPos: 1 },
            { tag: 'insert', srcPos: 0, destPos: 2 },
            { tag: 'insert', srcPos: 0, destPos: 3 },
          ],
          3,
          4,
        ),
      ),
    ).toThrow('subsequence is not a subsequence')
    expect(() =>
      all().removeSubsequence(
        Editops.fromOperations([{ tag: 'delete', srcPos: 0, destPos: 0 }], 3, 4),
      ),
    ).toThrow('subsequence is not a subsequence')
  })
})

describe('what equality and matching blocks answer at the edges', () => {
  it('reports collections of different lengths as unequal', () => {
    const ops = Editops.fromOperations([], 1, 1)
    expect(ops.equals(Editops.fromOperations([], 2, 1))).toBe(false)
    expect(ops.equals(Editops.fromOperations([], 1, 2))).toBe(false)

    const blocks = Opcodes.fromOperations([], 1, 1)
    expect(blocks.equals(Opcodes.fromOperations([], 2, 1))).toBe(false)
    expect(blocks.equals(Opcodes.fromOperations([], 1, 2))).toBe(false)
    expect(
      blocks.equals(
        Opcodes.fromOperations(
          [
            { tag: 'delete', srcStart: 0, srcEnd: 1, destStart: 0, destEnd: 0 },
            { tag: 'insert', srcStart: 1, srcEnd: 1, destStart: 0, destEnd: 1 },
          ],
          1,
          1,
        ),
      ),
    ).toBe(false)
  })

  it('reports collections that differ in one operation as unequal', () => {
    const a = Editops.fromOperations([{ tag: 'delete', srcPos: 0, destPos: 0 }], 2, 1)
    const b = Editops.fromOperations([{ tag: 'delete', srcPos: 1, destPos: 0 }], 2, 1)
    expect(a.equals(b)).toBe(false)

    const c = levenshteinOpcodes('aaabaaa', 'abbaaabba')
    const d = levenshteinOpcodes('aaabaaa', 'abbaaabbb')
    expect(c.equals(d)).toBe(false)

    // Same lengths and the same number of blocks, so the comparison has to
    // reach the blocks themselves rather than stopping at the shape.
    const deleteFirst = Opcodes.fromOperations(
      [
        { tag: 'delete', srcStart: 0, srcEnd: 1, destStart: 0, destEnd: 0 },
        { tag: 'equal', srcStart: 1, srcEnd: 2, destStart: 0, destEnd: 1 },
      ],
      2,
      1,
    )
    const deleteLast = Opcodes.fromOperations(
      [
        { tag: 'equal', srcStart: 0, srcEnd: 1, destStart: 0, destEnd: 1 },
        { tag: 'delete', srcStart: 1, srcEnd: 2, destStart: 1, destEnd: 1 },
      ],
      2,
      1,
    )
    expect(deleteFirst.equals(deleteLast)).toBe(false)
  })

  // The runs between two operations are only matching blocks when both sides
  // advanced; an insertion followed by a deletion leaves one of them at zero.
  it('drops a run that is empty on one side', () => {
    const ops = Editops.fromOperations(
      [
        { tag: 'insert', srcPos: 0, destPos: 0 },
        { tag: 'delete', srcPos: 0, destPos: 2 },
      ],
      1,
      3,
    )
    expect(ops.toMatchingBlocks()).toEqual([{ srcStart: 1, destStart: 3, length: 0 }])
  })
})

// Not ported — upstream's collections are mutable and its operations are
// tuples, so neither of these is a question there. Here the collection is the
// result a scorer hands back, and it says it does not change.
describe('a collection cannot be edited through what it hands out', () => {
  it('freezes the array, so it cannot be lengthened past its declared lengths', () => {
    const ops = editops()

    expect(Object.isFrozen(ops.operations)).toBe(true)
    // Through `Reflect` because the type forbids it, which is the other half
    // of the contract: the freeze is what answers a caller who has no types.
    expect(() =>
      Reflect.apply(Array.prototype.push, ops.operations, [EDITOP_LIST[0]]),
    ).toThrow(TypeError)
    expect(ops.operations.length).toBe(EDITOP_LIST.length)
  })

  it('copies what it was built from, so the caller keeps no way in', () => {
    const source: Array<{ tag: EditopTag; srcPos: number; destPos: number }> = [
      { tag: 'delete', srcPos: 1, destPos: 1 },
    ]
    const ops = Editops.fromOperations(source, 7, 9)

    source[0].srcPos = 99
    source.push({ tag: 'delete', srcPos: 2, destPos: 1 })

    expect(ops.operations).toEqual([{ tag: 'delete', srcPos: 1, destPos: 1 }])
  })

  it('does not share an array between a collection and its conversions', () => {
    const ops = editops()

    expect(ops.toOpcodes().operations).not.toBe(ops.operations)
    expect(ops.toMatchingBlocks()).not.toBe(ops.toMatchingBlocks())
    expect(ops.inverse().operations).not.toBe(ops.operations)
  })
})
