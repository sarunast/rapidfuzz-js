import type {
  ChoiceIndex,
  ChoiceIndexBuilder,
} from '../../../../core/scoring/choiceIndex.js'
import type { Sequence } from '../../../../core/types.js'
import { convSequence } from '../../sequence.js'
import { feasibleRadices } from '../key.js'
import { extractGrams, OutOfRadix, radixFor, repackKey } from './keys.js'

/**
 * Posting lists in compressed-sparse-row form: one `ids` array for the whole
 * index, with `offsets[ordinal] .. offsets[ordinal + 1]` marking each gram's
 * slice of it, and a map from gram key to ordinal.
 *
 * The shape it replaces was two typed arrays per distinct gram, so a corpus of
 * seventeen thousand distinct trigrams carried as many posting objects, twice
 * as many typed arrays and as many buffers — object headers and collector work
 * proportional to gram variety rather than to the data, and a walk scattered
 * across that many allocations where this streams one array.
 */
export interface Postings {
  readonly ordinals: Map<string | number, number>
  readonly offsets: Uint32Array
  /**
   * `Uint16` when the corpus fits in one, which halves the largest stream the
   * accumulation loop reads. It buys no time — the stream is sequential and
   * prefetched, so what that loop costs is the loop rather than the bytes — but
   * it took retained memory down by nearly a third on a real corpus.
   */
  readonly ids: Uint16Array | Uint32Array
  /**
   * The narrowest word holding the largest frequency anywhere in the index, and
   * `null` only when no gram repeats within any single choice.
   *
   * That last case sounds common and is not: 99.9% of entries are `1` on
   * 26-letter trigrams, but one gram repeated three times in one choice gives
   * the whole corpus a counts array.
   */
  readonly counts: Uint8Array | Uint16Array | Uint32Array | null
  /**
   * `1` for an ordinal whose slice holds the choices that *lack* the gram
   * rather than the ones that have it, and `null` when no list qualified.
   *
   * Both spellings share the slice layout, so only the meaning of an entry
   * changes: a sparse entry is a choice holding that gram `counts[at]` times, a
   * dense entry is an exception to a default frequency of `1` — an absence at
   * count `0`, or a repeat at `2` or more.
   */
  readonly dense: Uint8Array | null
}

interface PostingBuilder {
  readonly ids: number[]
  readonly counts: number[]
}

/** A choice shorter than one gram, which only an equal one can match. */
export interface GramlessChoice {
  readonly id: number
  readonly elements: readonly unknown[]
}

/**
 * Everything an index needs once the builder is done with it.
 *
 * Generic in the norms so Cosine's array is *typed* present rather than checked
 * for: Dice does not allocate one, and a shared nullable field would put a
 * runtime guard in front of every Cosine score for a state that cannot happen.
 */
export interface SealedIndex<TNorm extends Float64Array | null> {
  readonly gramSize: number
  readonly radix: number | null
  readonly choiceCount: number
  readonly postings: Postings
  readonly gramCount: Uint32Array
  /** The largest of `gramCount`, which is what bounds a Cosine dot product. */
  readonly maxGramCount: number
  /**
   * The largest of `squaredNorm`, which is the other thing a Cosine score has
   * to keep inside the exact integers — see `assertCosineNormsExact`.
   */
  readonly maxSquaredNorm: number
  /** Cosine's denominator; `null` on a Dice index, which has no use for it. */
  readonly squaredNorm: TNorm
  /** Ascending by id, because ids are the order the choices arrived in. */
  readonly gramless: readonly GramlessChoice[]
}

/**
 * The share of the corpus a posting list has to cover before it is cheaper
 * stored inverted, and `2/3` rather than the obvious `1/2` because inverting
 * costs a second thing: any query touching a dense list has to score every
 * choice, since a default frequency then applies to all of them. Writing that
 * out, a dense gram changes the work by `(N − 2·length + exceptions)` in
 * accumulation and at most `(N − length)` in selection, and the sum only turns
 * negative above `2N/3`. At exactly one half the storage saving is zero and the
 * scan is pure loss.
 *
 * Derived rather than swept: real corpora have discontinuous gram frequencies,
 * so the ones measured had no list anywhere near this band and every cutoff
 * from 0.5 to 0.9 performed identically.
 */
const DENSE_CUTOFF = 2 / 3

/** What each fixed-width array in the representation can address. */
const MAX_UINT32 = 0xffff_ffff

/**
 * Refuse what the representation cannot address, rather than truncating into a
 * typed array and answering the wrong choice.
 *
 * Three bounds and not one: a corpus can exceed any of them without exceeding
 * the others, since a million choices can still overflow the posting-entry
 * total and one long sequence can overflow its own gram count whatever the
 * corpus size. Separate from where the numbers come from so it can be exercised
 * directly — the inputs that would trip it are four billion choices and a
 * four-billion-element sequence, neither of which a test can build.
 */
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

/**
 * The elements of a gramless choice, copied rather than referenced. A converted
 * string is a `Uint32Array` the conversion owns — or a view into one, where a
 * surrogate pair shortened it — and a typed-array input is handed back to us
 * untouched, so retaining either would keep a caller's buffer alive for the
 * handful of elements this needs. They are fewer than `gramSize` values, so the
 * copy is bounded by the gramless case itself.
 */
function copyElements(elements: ArrayLike<unknown>): unknown[] {
  const copy = new Array<unknown>(elements.length)
  for (let index = 0; index < elements.length; index++) copy[index] = elements[index]
  return copy
}

/** Whether a list covering this much of the corpus is cheaper stored inverted. */
function qualifiesAsDense(
  length: number,
  exceptions: number,
  choiceCount: number,
): boolean {
  return choiceCount > 0 && length >= DENSE_CUTOFF * choiceCount && exceptions < length
}

/** Growable per-gram lists into the flat arrays a query reads. */
function compact(
  builder: Map<string | number, PostingBuilder>,
  choiceCount: number,
): Postings {
  let widest = 0
  for (const posting of builder.values()) {
    for (const count of posting.counts) if (count > widest) widest = count
  }
  // Which lists to invert, and how much room that takes, before anything is
  // allocated: a dense list's slice is a different size from its sparse one, so
  // the decision has to be made in a pass of its own.
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
  // A `Uint16` id holds 0…65,535, so a corpus of exactly 65,536 choices is the
  // largest that fits.
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
      // One merge of the sorted list against every id: what is missing becomes
      // an absence at count 0, what is present with a frequency other than 1
      // becomes that frequency, and the overwhelmingly common present-once entry
      // is stored nowhere at all.
      let cursor = 0
      if (counts === null) {
        // No frequency anywhere exceeds 1, so an exception can only be an
        // absence and there is no count to write for it.
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

/**
 * The build path, shared by both metrics: which grams a choice holds is not a
 * question the metric changes.
 *
 * Ids are the order `add` is called in, so nothing has to be told how many
 * choices are coming — which matters because a caller cannot know how many it
 * will keep until it has read them all. The cost is that per-choice metadata
 * grows as plain arrays and becomes typed at `seal`, when the count is finally
 * known; the sealed representation is the same either way.
 */
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
    // A loop, not one attempt and a fallback: a single choice can need more than
    // one rung. `'\ud800😀'` is a lone surrogate followed by an astral
    // character, so its first element pushes a byte radix up to BMP and its
    // second pushes that one up again. Each rung is strictly wider than the
    // element that forced it, so this cannot cycle.
    //
    // Nothing has to be rolled back on the way round: the extraction fills the
    // scratch arrays and only `record` writes to a posting list, so a throw
    // leaves the index exactly as it was.
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

  /**
   * Widen the corpus-wide key representation one rung — to the narrowest radix
   * holding the element that did not fit, or to joined strings when no packed
   * radix can. Everything already ingested is re-keyed rather than re-read,
   * which makes a late widening cost the gram variety rather than the corpus.
   */
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
