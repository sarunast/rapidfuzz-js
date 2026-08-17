import type { ChoiceIndex, ChoiceIndexBuilder } from '#core/scoring/choiceIndex.js'
import { convSequence } from '#core/sequence.js'
import type { Sequence } from '#core/types.js'

import { feasibleRadices } from '../key.js'
import {
  decodeGramKey,
  encodeGramKey,
  extractGrams,
  extractOrdinalGrams,
  NEEDS_ORDINALS,
  NEEDS_WIDER_RADIX,
  radixFor,
  type RadixWidening,
  repackKey,
} from './keys.js'
import { ordinalizeChoice, resolveOrdinals } from './ordinals.js'

export interface Postings {
  readonly ordinals: Map<string | number, number>
  readonly offsets: Uint32Array
  readonly ids: Uint16Array | Uint32Array
  readonly counts: Uint8Array | Uint16Array | Uint32Array | null
  readonly dense: Uint8Array | null
}

interface PostingBuilder {
  readonly ids: number[]
  readonly counts: number[]
}

interface GramlessChoice {
  readonly id: number
  readonly elements: readonly unknown[]
}

export interface SealedIndex<TNorm extends Float64Array | null> {
  readonly gramSize: number
  readonly radix: number | null
  /**
   * The element-to-ordinal table when the index keys by ordinal, and `null`
   * when its posting keys hold elements directly. It says which representation
   * the keys are in, not whether every element in the corpus is an integer — a
   * gramless choice never reaches a key at all.
   */
  readonly elementOrdinals: ReadonlyMap<unknown, number> | null
  readonly choiceCount: number
  readonly postings: Postings
  readonly gramCount: Uint32Array
  readonly maxGramCount: number
  readonly maxSquaredNorm: number
  readonly squaredNorm: TNorm
  readonly gramless: readonly GramlessChoice[]
}

const DENSE_CUTOFF = 2 / 3

const MAX_UINT32 = 0xffff_ffff

export function assertAddressable(
  choiceCount: number,
  postingEntries: number,
  gramCount: number,
): void {
  if (choiceCount > MAX_UINT32) {
    throw new RangeError('an indexed collection cannot exceed 4294967295 choices')
  }
  if (postingEntries > MAX_UINT32) {
    throw new RangeError('an index cannot exceed 4294967295 posting entries')
  }
  if (gramCount > MAX_UINT32) {
    throw new RangeError('an indexed choice cannot exceed 4294967295 grams')
  }
}

function copyElements(elements: ArrayLike<unknown>): unknown[] {
  const copy = new Array<unknown>(elements.length)
  for (let index = 0; index < elements.length; index++) copy[index] = elements[index]
  return copy
}

function qualifiesAsDense(
  length: number,
  exceptions: number,
  choiceCount: number,
): boolean {
  return choiceCount > 0 && length >= DENSE_CUTOFF * choiceCount && exceptions < length
}

function compact(
  builder: Map<string | number, PostingBuilder>,
  choiceCount: number,
): Postings {
  let widest = 0
  for (const posting of builder.values()) {
    for (const count of posting.counts) if (count > widest) widest = count
  }
  const inverted = new Set<string | number>()
  let total = 0
  for (const [key, posting] of builder) {
    const length = posting.ids.length
    let exceptions = choiceCount - length
    for (const count of posting.counts) if (count !== 1) exceptions++
    if (qualifiesAsDense(length, exceptions, choiceCount)) {
      inverted.add(key)
      total += exceptions
    } else {
      total += length
    }
  }
  assertAddressable(choiceCount, total, 0)
  const ordinals = new Map<string | number, number>()
  const offsets = new Uint32Array(builder.size + 1)
  const ids = choiceCount <= 0x1_0000 ? new Uint16Array(total) : new Uint32Array(total)
  const counts =
    widest <= 1
      ? null
      : widest < 0x100
        ? new Uint8Array(total)
        : widest < 0x1_0000
          ? new Uint16Array(total)
          : new Uint32Array(total)
  const dense = inverted.size === 0 ? null : new Uint8Array(builder.size)
  let ordinal = 0
  let at = 0
  for (const [key, posting] of builder) {
    ordinals.set(key, ordinal)
    offsets[ordinal] = at
    const sourceIds = posting.ids
    if (dense !== null && inverted.has(key)) {
      dense[ordinal] = 1
      let cursor = 0
      if (counts === null) {
        for (let id = 0; id < choiceCount; id++) {
          if (cursor < sourceIds.length && sourceIds[cursor] === id) {
            cursor++
            continue
          }
          ids[at] = id
          at++
        }
      } else {
        for (let id = 0; id < choiceCount; id++) {
          if (cursor < sourceIds.length && sourceIds[cursor] === id) {
            const count = posting.counts[cursor]
            cursor++
            if (count === 1) continue
            ids[at] = id
            counts[at] = count
            at++
            continue
          }
          ids[at] = id
          counts[at] = 0
          at++
        }
      }
    } else {
      for (let index = 0; index < sourceIds.length; index++) {
        ids[at] = sourceIds[index]
        if (counts !== null) counts[at] = posting.counts[index]
        at++
      }
    }
    ordinal++
  }
  offsets[ordinal] = at
  return { ordinals, offsets, ids, counts, dense }
}

export class NGramIndexBuilder<
  TNorm extends Float64Array | null,
> implements ChoiceIndexBuilder {
  private postings: Map<string | number, PostingBuilder> | null = new Map()
  private radix: number | null
  private readonly initialRadix: number | null
  private elementOrdinals: Map<unknown, number> | null = null
  private readonly gramCount: number[] = []
  private readonly squaredNorm: number[] = []
  private readonly gramless: GramlessChoice[] = []
  private readonly keys: (string | number)[] = []
  private readonly counts: number[] = []
  private readonly ordinals: number[] = []
  private readonly widening: RadixWidening = { from: 0, to: null }
  private entries = 0
  private maxGramCount = 0
  private maxSquaredNorm = 0

  constructor(
    private readonly gramSize: number,
    private readonly norms: (values: readonly number[]) => TNorm,
    private readonly build: (sealed: SealedIndex<TNorm>) => ChoiceIndex,
  ) {
    this.initialRadix = feasibleRadices(gramSize)[0] ?? null
    this.radix = this.initialRadix
  }

  add(choice: Sequence): void {
    const postings = this.postings
    if (postings === null) throw new TypeError('this index is already sealed')
    const id = this.gramCount.length
    const elements = convSequence(choice)
    const total = elements.length - this.gramSize + 1
    assertAddressable(id + 1, this.entries, total < 0 ? 0 : total)
    if (total <= 0) {
      this.gramCount.push(0)
      this.squaredNorm.push(0)
      this.gramless.push({ id, elements: copyElements(elements) })
      return
    }
    for (;;) {
      const squaredNorm = this.extract(elements)
      if (squaredNorm === NEEDS_ORDINALS) {
        this.ordinalize(postings)
        continue
      }
      if (squaredNorm === NEEDS_WIDER_RADIX) {
        this.rekey(postings, this.widening.from, this.widening.to)
        continue
      }
      this.gramCount.push(total)
      if (total > this.maxGramCount) this.maxGramCount = total
      if (squaredNorm > this.maxSquaredNorm) this.maxSquaredNorm = squaredNorm
      this.squaredNorm.push(squaredNorm)
      this.record(postings, id)
      return
    }
  }

  private extract(elements: ArrayLike<unknown>): number {
    const table = this.elementOrdinals
    if (table === null) {
      return extractGrams(
        elements,
        this.gramSize,
        this.radix,
        this.widening,
        this.keys,
        this.counts,
      )
    }
    ordinalizeChoice(elements, this.gramSize, table, this.ordinals)
    return extractOrdinalGrams(
      this.ordinals,
      this.gramSize,
      this.radix,
      this.widening,
      this.keys,
      this.counts,
    )
  }

  /**
   * Moves an index that keys elements directly onto ordinal keys, once, without
   * re-reading a choice: the direct spelling is reversible, so each posting's
   * key is decoded, its elements take ordinals, and the posting moves to the
   * re-encoded key. Ordinals are dense from zero, so this usually narrows the
   * radix a text corpus had been forced to widen.
   *
   * Two passes, because the radix the second one encodes with is not known
   * until every ordinal has been handed out. The alternative — holding each
   * gram's ordinals from the first pass — allocates one array per distinct
   * gram, which is the corpus's gram variety; a second decode is arithmetic
   * over the keys already in hand, and this runs once per index.
   *
   * The first pass defines the vocabulary and the second encodes against it,
   * frozen: `resolveOrdinals` cannot add one, so no ordinal can appear that the
   * chosen radix was not sized for.
   */
  private ordinalize(postings: Map<string | number, PostingBuilder>): void {
    const gramSize = this.gramSize
    const table = new Map<unknown, number>()
    const elements = new Array<number>(gramSize)
    const ordinals: number[] = []
    for (const key of postings.keys()) {
      decodeGramKey(key, gramSize, this.radix, elements)
      ordinalizeChoice(elements, gramSize, table, ordinals)
    }
    // An empty table means no gram survived to carry an element — every window
    // of the choice that forced this held an unmatchable one — so there is no
    // largest ordinal to size the radix from, and the index starts over.
    const radix =
      table.size === 0 ? this.initialRadix : radixFor(gramSize, table.size - 1)
    const rekeyed = new Map<string | number, PostingBuilder>()
    for (const [key, posting] of postings) {
      decodeGramKey(key, gramSize, this.radix, elements)
      resolveOrdinals(elements, table, ordinals)
      rekeyed.set(encodeGramKey(ordinals, gramSize, radix), posting)
    }
    postings.clear()
    for (const [key, posting] of rekeyed) postings.set(key, posting)
    this.elementOrdinals = table
    this.radix = radix
  }

  private record(postings: Map<string | number, PostingBuilder>, id: number): void {
    const keys = this.keys
    const counts = this.counts
    for (let index = 0; index < keys.length; index++) {
      const posting = postings.get(keys[index])
      if (posting === undefined) {
        postings.set(keys[index], { ids: [id], counts: [counts[index]] })
      } else {
        posting.ids.push(id)
        posting.counts.push(counts[index])
      }
      this.entries++
    }
  }

  private rekey(
    postings: Map<string | number, PostingBuilder>,
    from: number,
    to: number | null,
  ): void {
    const rekeyed = new Map<string | number, PostingBuilder>()
    for (const [key, posting] of postings) {
      rekeyed.set(repackKey(key, from, to, this.gramSize), posting)
    }
    postings.clear()
    for (const [key, posting] of rekeyed) postings.set(key, posting)
    this.radix = to
  }

  seal(): ChoiceIndex {
    const postings = this.postings
    if (postings === null) throw new TypeError('this index is already sealed')
    this.postings = null
    const choiceCount = this.gramCount.length
    return this.build({
      gramSize: this.gramSize,
      radix: this.radix,
      elementOrdinals: this.elementOrdinals,
      choiceCount,
      postings: compact(postings, choiceCount),
      gramCount: Uint32Array.from(this.gramCount),
      maxGramCount: this.maxGramCount,
      maxSquaredNorm: this.maxSquaredNorm,
      squaredNorm: this.norms(this.squaredNorm),
      gramless: this.gramless,
    })
  }
}
