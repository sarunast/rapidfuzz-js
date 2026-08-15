import type {
  ChoiceIndex,
  ChoiceIndexBuilder,
} from '../../../../core/scoring/choiceIndex.js'
import type { Sequence } from '../../../../core/types.js'
import { convSequence } from '../../sequence.js'
import { feasibleRadices } from '../key.js'
import { extractGrams, OutOfRadix, radixFor, repackKey } from './keys.js'

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

export interface GramlessChoice {
  readonly id: number
  readonly elements: readonly unknown[]
}

export interface SealedIndex<TNorm extends Float64Array | null> {
  readonly gramSize: number
  readonly radix: number | null
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
  private readonly gramCount: number[] = []
  private readonly squaredNorm: number[] = []
  private readonly gramless: GramlessChoice[] = []
  private readonly keys: (string | number)[] = []
  private readonly counts: number[] = []
  private entries = 0
  private maxGramCount = 0
  private maxSquaredNorm = 0

  constructor(
    private readonly gramSize: number,
    private readonly norms: (values: readonly number[]) => TNorm,
    private readonly build: (sealed: SealedIndex<TNorm>) => ChoiceIndex,
  ) {
    this.radix = feasibleRadices(gramSize)[0] ?? null
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
      try {
        const squaredNorm = extractGrams(
          elements,
          this.gramSize,
          this.radix,
          true,
          this.keys,
          this.counts,
        )
        this.gramCount.push(total)
        if (total > this.maxGramCount) this.maxGramCount = total
        if (squaredNorm > this.maxSquaredNorm) this.maxSquaredNorm = squaredNorm
        this.squaredNorm.push(squaredNorm)
        this.record(postings, id)
        return
      } catch (error) {
        if (!(error instanceof OutOfRadix)) throw error
        this.rekey(postings, error.radix, radixFor(this.gramSize, error.element))
      }
    }
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
