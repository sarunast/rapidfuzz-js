import type {
  ChoiceIndex,
  ChoiceIndexBuilder,
  SelectedChoices,
} from '../../core/protocol.js'
import type { Sequence } from '../../core/types.js'
import { elementsEqual } from './ngram.js'
import { convSequence } from './sequence.js'

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
interface Postings {
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
interface GramlessChoice {
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
interface SealedIndex<TNorm extends Float64Array | null> {
  readonly gramSize: number
  readonly radix: number | null
  readonly choiceCount: number
  readonly postings: Postings
  readonly gramCount: Uint32Array
  /** The largest of `gramCount`, which is what bounds a Cosine dot product. */
  readonly maxGramCount: number
  /** Cosine's denominator; `null` on a Dice index, which has no use for it. */
  readonly squaredNorm: TNorm
  /** Ascending by id, because ids are the order the choices arrived in. */
  readonly gramless: readonly GramlessChoice[]
}

/**
 * Carries both the element that did not fit and the radix it did not fit — the
 * second so the re-key needs no null check for a state it cannot be in. Only a
 * packed radix can raise this, and the error itself is the proof.
 */
class OutOfRadix extends Error {
  constructor(
    readonly element: number,
    readonly radix: number,
  ) {
    super('gram element does not fit the packed key radix')
  }
}

/**
 * The rungs a packed gram key can sit on, narrowest first: a byte for Latin-1,
 * a BMP word, and the full code-point range.
 */
const RADIX_LADDER: readonly number[] = [0x100, 0x1_0000, 0x11_0000]

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
 * Dice accumulates `Σ min(a, b)`, which cannot exceed the query's own gram
 * count, so this is the whole of its `Int32Array` accumulator's exactness
 * condition. Unreachable for any real text — it is 2.1 billion grams — and
 * checked rather than assumed because its failure mode is a wrong score rather
 * than a thrown error.
 */
export function assertQueryIndexable(gramCount: number): void {
  if (gramCount > 0x7fff_ffff) {
    throw new RangeError('a query of more than 2147483647 grams cannot be indexed')
  }
}

/**
 * Cosine's dot product is `Σ qᵢ·cᵢ`, which is bounded by
 * `gramCount(query) · gramCount(choice)` — so while that product is a safe
 * integer, every term and every partial sum is exact whatever order they are
 * added in, and the index matches the exhaustive scorer to the bit.
 *
 * Above it they can disagree, because a dense list decomposes a repeated gram's
 * contribution as `q·(c-1) + q` where a sparse one computes `q·c`: at
 * `q = 116,982,125` and `c = 105,643,526` those are 12358404163972748 and
 * 12358404163972750. Checked rather than assumed, for the reason the Dice bound
 * above is: the failure mode is a wrong score rather than a thrown error. It
 * takes ~100-million-gram sequences on both sides to reach.
 */
export function assertCosineExact(queryGrams: number, maxChoiceGrams: number): void {
  if (queryGrams * maxChoiceGrams >= Number.MAX_SAFE_INTEGER) {
    throw new RangeError(
      'a cosine query of this many grams cannot be scored exactly against a choice this long',
    )
  }
}

function integerElement(element: unknown): number {
  if (typeof element !== 'number' || !Number.isInteger(element)) {
    throw new TypeError(
      `an indexed choice holds integer elements only, and one of them is ${String(element)}`,
    )
  }
  return element
}

/**
 * The radices that hold a gram of this depth inside one safe integer, narrowest
 * first. Latin-1 text needs 8 bits an element, so `'abc'` packs into 24 —
 * `0x616263` — where a BMP radix spends 48 on the same three letters, and small
 * integer keys are the ones a `Map` handles best.
 *
 * Depth decides how far the ladder reaches: a byte radix holds six elements, a
 * BMP radix three, the full code-point radix two. A trigram over astral text
 * therefore has no packed rung at all and falls back to joined strings.
 */
export function feasibleRadices(gramSize: number): readonly number[] {
  return RADIX_LADDER.filter(
    (radix) => Math.pow(radix, gramSize) <= Number.MAX_SAFE_INTEGER,
  )
}

/**
 * The narrowest feasible radix holding `element`, or `null` for joined strings.
 *
 * A negative element goes straight to strings: positional packing has no room
 * below zero, so answering with a rung the element is merely *less than* would
 * hand the re-key a target no wider than the one that just failed, and the
 * ladder would report it could not widen on an element strings represent
 * exactly.
 */
function radixFor(gramSize: number, element: number): number | null {
  if (element < 0) return null
  for (const radix of feasibleRadices(gramSize)) if (element < radix) return radix
  return null
}

/**
 * The same gram, re-spelled for a wider radix or for joined strings. Packing is
 * positional and so reversible, which is what lets an index that has already
 * ingested a million choices change key scheme without re-reading one of them.
 *
 * A key that is already joined comes back unchanged. No build reaches that arm —
 * only a packed radix can raise the error that starts a re-key, and once the
 * scheme is joined nothing raises it again — so the case is pinned by a direct
 * test rather than by a corpus.
 */
export function repackKey(
  key: string | number,
  from: number,
  to: number | null,
  gramSize: number,
): string | number {
  if (typeof key === 'string') return key
  const elements: number[] = new Array<number>(gramSize)
  let rest = key
  for (let position = gramSize - 1; position >= 0; position--) {
    elements[position] = rest % from
    rest = Math.floor(rest / from)
  }
  if (to === null) return elements.join(',')
  let packed = 0
  for (const element of elements) packed = packed * to + element
  return packed
}

function joinGram(elements: ArrayLike<unknown>, start: number, gramSize: number): string {
  let joined = String(integerElement(elements[start]))
  for (let offset = 1; offset < gramSize; offset++) {
    joined += `,${integerElement(elements[start + offset])}`
  }
  return joined
}

/**
 * Every distinct gram of a sequence, with its frequency, written into
 * caller-owned arrays.
 *
 * One walk shared by indexing and by querying, so the two cannot drift apart on
 * how a gram becomes a key — the only way an index could disagree with the
 * metric it reproduces. `widening` is the difference between them: during a
 * build an element too wide for the current radix has to widen the whole index,
 * while on a query it is spelled as a joined string instead, which no packed
 * index holds and which therefore matches nothing.
 *
 * That fallback is not the same as dropping the gram, and the difference is
 * Cosine's denominator: an unmatchable gram still counts toward the query's own
 * norm, so it has to reach `counts` even though no posting list will name it.
 *
 * Returns the squared norm, since counting is where it comes from for free —
 * `Σ c²` accumulates as `2·previous + 1` per occurrence.
 */
function extractGrams(
  elements: ArrayLike<unknown>,
  gramSize: number,
  radix: number | null,
  widening: boolean,
  keys: (string | number)[],
  counts: number[],
): number {
  keys.length = 0
  counts.length = 0
  const total = elements.length - gramSize + 1
  const seen = new Map<string | number, number>()
  let squaredNorm = 0
  for (let start = 0; start < total; start++) {
    let key: string | number
    if (radix === null) {
      key = joinGram(elements, start, gramSize)
    } else {
      let packed = 0
      let fits = true
      for (let offset = 0; offset < gramSize; offset++) {
        const value = integerElement(elements[start + offset])
        if (value < 0 || value >= radix) {
          if (widening) throw new OutOfRadix(value, radix)
          fits = false
          break
        }
        packed = packed * radix + value
      }
      key = fits ? packed : joinGram(elements, start, gramSize)
    }
    const previous = seen.get(key)
    if (previous === undefined) {
      seen.set(key, counts.length)
      keys.push(key)
      counts.push(1)
      squaredNorm += 1
      continue
    }
    squaredNorm += 2 * counts[previous] + 1
    counts[previous]++
  }
  return squaredNorm
}

/**
 * The exhaustive drivers' ranking rule as a predicate: a higher score wins, and
 * a tie goes to the earlier stored position.
 *
 * Load-bearing, and not derivable from the scan order: a dense scan does count
 * upward and each posting list is sorted, but the touched set is filled across
 * several lists, so a gram matching id 9 before another matches id 2 leaves it
 * out of order.
 */
function outranks(
  score: number,
  id: number,
  otherScore: number,
  otherId: number,
): boolean {
  return score > otherScore || (score === otherScore && id < otherId)
}

/**
 * Whether a score of exactly `0` belongs in the result. When it does, every
 * choice the postings never reached has to be accounted for; when it does not,
 * they can simply vanish.
 */
function zeroesQualify(threshold: number | null): boolean {
  return threshold === null || threshold <= 0
}

/**
 * The elements of a gramless choice, copied rather than referenced: a string's
 * are a `Uint32Array` view, and holding one would retain a buffer the index has
 * no reason to keep alive. They are fewer than `gramSize` values.
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
class NGramIndexBuilder<TNorm extends Float64Array | null> implements ChoiceIndexBuilder {
  private postings: Map<string | number, PostingBuilder> | null = new Map()
  private radix: number | null
  private readonly gramCount: number[] = []
  private readonly squaredNorm: number[] = []
  private readonly gramless: GramlessChoice[] = []
  private readonly keys: (string | number)[] = []
  private readonly counts: number[] = []
  private entries = 0
  private maxGramCount = 0

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
      squaredNorm: this.norms(this.squaredNorm),
      gramless: this.gramless,
    })
  }
}

/**
 * What every query of either metric needs: the flattened query, the set of
 * choices accumulation reached, and the arrays a result is written into.
 *
 * Held for the index's lifetime and reused, so none of it is allocated per
 * query. What still is: the `seen` map inside `extractGrams`, a gramless
 * query's matches, and an unlimited call's sorted result — none of them on the
 * path this exists to keep cheap, and query preparation measured a few percent
 * of a query.
 */
class QueryState {
  readonly keys: (string | number)[] = []
  readonly counts: number[] = []
  readonly touched: number[] = []
  /** What every choice scores before its own accumulator entry is added. */
  base = 0
  /** Set when a dense list has put every choice into the scan. */
  scannedAll = false
  ids: Uint32Array = new Uint32Array(0)
  scores: Float64Array = new Float64Array(0)

  /** Grown before anything is written, never after: this does not preserve. */
  reserve(needed: number): void {
    if (this.ids.length >= needed) return
    this.ids = new Uint32Array(needed)
    this.scores = new Float64Array(needed)
  }
}

/**
 * How many results a call can produce, which is what the result arrays are
 * grown to once, before selection writes anything into them.
 */
function roomFor(limit: number | null, choiceCount: number): number {
  return limit === null ? choiceCount : limit < choiceCount ? limit : choiceCount
}

/**
 * A gramless query scores `1` against a choice that is gramless and equal, and
 * `0` against everything else — a zero-gram similarity is `1` only when both
 * sides have no grams. So this needs the short choices' elements and nothing
 * else, which is why they are the one thing an index retains besides postings.
 */
function gramlessResult(
  sealed: SealedIndex<Float64Array | null>,
  state: QueryState,
  elements: ArrayLike<unknown>,
  threshold: number | null,
  limit: number | null,
  ascending: boolean,
): SelectedChoices {
  const matched: number[] = []
  if (threshold === null || threshold <= 1) {
    for (const entry of sealed.gramless) {
      if (elementsEqual(elements, entry.elements)) matched.push(entry.id)
    }
  }
  const zeroes = zeroesQualify(threshold)
  const room = roomFor(limit, sealed.choiceCount)
  state.reserve(zeroes ? room : matched.length < room ? matched.length : room)
  const ids = state.ids
  const scores = state.scores
  let length = 0
  // With no zeroes to place, every result scores `1` and the two orders agree:
  // ascending id is also `(score desc, id asc)`, and `matched` is already in it.
  if (!zeroes) {
    for (const id of matched) {
      if (length === room) break
      ids[length] = id
      scores[length] = 1
      length++
    }
    return { ids, scores, length }
  }
  // Otherwise they diverge: ranked order puts every `1` ahead of every `0`,
  // while ascending order interleaves them, so a match at id 7 and a zero at
  // id 3 come back 3 then 7.
  if (!ascending) {
    for (const id of matched) {
      if (length === room) break
      ids[length] = id
      scores[length] = 1
      length++
    }
  }
  let next = 0
  for (let id = 0; id < sealed.choiceCount && length < room; id++) {
    const isMatch = next < matched.length && matched[next] === id
    if (isMatch) next++
    if (isMatch && !ascending) continue
    ids[length] = id
    scores[length] = isMatch ? 1 : 0
    length++
  }
  return { ids, scores, length }
}

/**
 * Ids no posting list named, at the score they all share, appended after the
 * ranked results — where they belong, because under a sparse scan every touched
 * choice shares at least one gram and so scores above zero.
 */
function fillZeroes(
  sealed: SealedIndex<Float64Array | null>,
  state: QueryState,
  accumulator: Int32Array | Float64Array,
  length: number,
  threshold: number | null,
  room: number,
): number {
  if (!zeroesQualify(threshold)) return length
  // Every choice was already scored and offered, so there is nothing to fill —
  // walking for an untouched accumulator entry here would re-add choices the
  // dense base had already put in the result.
  if (state.scannedAll) return length
  const ids = state.ids
  const scores = state.scores
  let filled = length
  for (let id = 0; id < sealed.choiceCount && filled < room; id++) {
    if (accumulator[id] !== 0) continue
    ids[filled] = id
    scores[filled] = 0
    filled++
  }
  return filled
}

/**
 * Does this query reach a dense list? If it does, every choice is in play — a
 * default frequency applies to choices no posting entry names — so selection
 * runs over the whole corpus rather than over the touched set.
 */
function reachesDenseList(
  postings: Postings,
  dense: Uint8Array,
  keys: (string | number)[],
): boolean {
  for (let index = 0; index < keys.length; index++) {
    const ordinal = postings.ordinals.get(keys[index])
    if (ordinal !== undefined && dense[ordinal] === 1) return true
  }
  return false
}

/**
 * An unlimited call's results, ranked by sorting the collected set once.
 *
 * `top` places each qualifying choice by walking the results it already holds,
 * which is what makes a small limit cheap and what makes an unlimited one
 * quadratic: with room for the whole corpus every qualifying choice can shift
 * every earlier one. A scan collects the same set in id order, so the ranking
 * left to do is `O(k log k)` over it.
 *
 * The arrays come back freshly allocated rather than borrowed from the query
 * scratch, which is what the collected set already fills.
 */
function rankSelected(found: SelectedChoices): SelectedChoices {
  const length = found.length
  const collectedIds = found.ids
  const collectedScores = found.scores
  const order = new Array<number>(length)
  for (let at = 0; at < length; at++) order[at] = at
  order.sort(
    (left, right) =>
      collectedScores[right] - collectedScores[left] ||
      collectedIds[left] - collectedIds[right],
  )
  const ids = new Uint32Array(length)
  const scores = new Float64Array(length)
  for (let at = 0; at < length; at++) {
    const from = order[at]
    ids[at] = collectedIds[from]
    scores[at] = collectedScores[from]
  }
  return { ids, scores, length }
}

/**
 * The ids a sparse query may still qualify, ascending.
 *
 * Only under a positive threshold: nothing untouched can clear one, so the walk
 * is confined to what accumulation reached — and that set arrives unordered,
 * because it is filled across several posting lists.
 */
function sortedTouched(state: QueryState): number[] {
  const touched = state.touched
  touched.sort((left, right) => left - right)
  return touched
}

/**
 * A Sørensen-Dice index: `2 · Σ min(a, b) / (gramsA + gramsB)`, with the shared
 * count coming straight out of accumulation. No kernel call is left — the index
 * *is* the scorer.
 *
 * Its accumulator is an `Int32Array` where Cosine's is a `Float64Array`, which
 * is why the two are separate classes rather than one carrying a mode: Dice's
 * overlap is a sum of `min(queryCount, choiceCount)` terms and so bounded by the
 * query's own gram count, and narrowing it measured 1.05–1.68x. The
 * read-modify-write did not shrink so much as vanish — an integer add on four
 * bytes where a `Float64Array` cost a conversion, a double add and eight.
 */
class DiceIndex implements ChoiceIndex {
  private readonly state = new QueryState()
  private readonly accumulator: Int32Array

  constructor(private readonly sealed: SealedIndex<null>) {
    this.accumulator = new Int32Array(sealed.choiceCount)
  }

  /**
   * `Σ min(a, b) ≤ gramCount(query)`, so a query of more than 2.1 billion grams
   * is the whole of the narrow accumulator's exactness condition. Unreachable
   * for any real text, and checked rather than assumed because its failure mode
   * is a wrong score rather than a thrown error.
   */
  private begin(query: Sequence): ArrayLike<unknown> {
    const elements = convSequence(query)
    assertQueryIndexable(elements.length - this.sealed.gramSize + 1)
    return elements
  }

  select(
    query: Sequence,
    threshold: number | null,
    limit: number | null,
  ): SelectedChoices {
    // Collect and sort rather than insert into place: with no limit there is no
    // room bound to make the insertion walk in `top` cheap.
    if (limit === null) return rankSelected(this.collect(query, threshold, false))
    const sealed = this.sealed
    const state = this.state
    const elements = this.begin(query)
    if (elements.length < sealed.gramSize) {
      return gramlessResult(sealed, state, elements, threshold, limit, false)
    }
    const queryGrams = elements.length - sealed.gramSize + 1
    extractGrams(elements, sealed.gramSize, sealed.radix, false, state.keys, state.counts)
    this.accumulate()
    const room = roomFor(limit, sealed.choiceCount)
    state.reserve(room)
    const length = fillZeroes(
      sealed,
      state,
      this.accumulator,
      this.top(queryGrams, threshold, room),
      threshold,
      room,
    )
    this.reset()
    return { ids: state.ids, scores: state.scores, length }
  }

  scan(query: Sequence, threshold: number | null): SelectedChoices {
    return this.collect(query, threshold, true)
  }

  /**
   * Every qualifying choice, in the cheapest order the caller can use: `scan`
   * needs ascending ids and pays for them, while a ranked call sorts by score
   * afterwards and would throw that order away. Ordering the touched set is not
   * a rounding error — it measured 84% of a `threshold: 0.5` query over 10,000
   * choices, where accumulation itself was 5%.
   */
  private collect(
    query: Sequence,
    threshold: number | null,
    ascending: boolean,
  ): SelectedChoices {
    const sealed = this.sealed
    const state = this.state
    const elements = this.begin(query)
    if (elements.length < sealed.gramSize) {
      return gramlessResult(sealed, state, elements, threshold, null, ascending)
    }
    const queryGrams = elements.length - sealed.gramSize + 1
    extractGrams(elements, sealed.gramSize, sealed.radix, false, state.keys, state.counts)
    this.accumulate()
    const everyChoice = state.scannedAll || zeroesQualify(threshold)
    const source = everyChoice ? null : ascending ? sortedTouched(state) : state.touched
    const total = source === null ? sealed.choiceCount : source.length
    state.reserve(total)
    const ids = state.ids
    const scores = state.scores
    const accumulator = this.accumulator
    const gramCount = sealed.gramCount
    const base = state.base
    let length = 0
    for (let index = 0; index < total; index++) {
      const id = source === null ? index : source[index]
      const grams = gramCount[id]
      const score =
        grams === 0 ? 0 : (2 * (base + accumulator[id])) / (queryGrams + grams)
      if (threshold !== null && score < threshold) continue
      ids[length] = id
      scores[length] = score
      length++
    }
    this.reset()
    return { ids, scores, length }
  }

  /**
   * Two literal loops per posting-list shape rather than one with a branch in
   * it: this is the innermost frame of the whole representation, and the shape
   * is known before it starts.
   *
   * No membership marks. Where the touched set is read no dense list was
   * reached, so every contribution below is strictly positive and an untouched
   * accumulator entry is still exactly zero; where a dense list was reached the
   * set is never read at all. A generation-mark array cost 26% of this loop for
   * a set that is either unread or already implied.
   */
  private accumulate(): void {
    const sealed = this.sealed
    const state = this.state
    const postings = sealed.postings
    const accumulator = this.accumulator
    const touched = state.touched
    const keys = state.keys
    const queryCounts = state.counts
    const ids = postings.ids
    const postingCounts = postings.counts
    const offsets = postings.offsets
    const dense = postings.dense
    state.base = 0
    if (dense !== null && reachesDenseList(postings, dense, keys)) state.scannedAll = true
    const tracking = !state.scannedAll
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      if (ordinal === undefined) continue
      const queryCount = queryCounts[index]
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      if (dense !== null && dense[ordinal] === 1) {
        // Every choice holds this gram once unless the slice says otherwise, so
        // the whole corpus takes `min(queryCount, 1)` in one addition and the
        // loop walks only the exceptions.
        // `min(queryCount, 1)` is `1`: `extractGrams` starts every frequency at
        // one and only increments, so a query gram is never seen zero times.
        state.base += 1
        if (postingCounts === null) {
          for (let at = from; at < upto; at++) accumulator[ids[at]] -= 1
          continue
        }
        for (let at = from; at < upto; at++) {
          const count = postingCounts[at]
          accumulator[ids[at]] += (queryCount < count ? queryCount : count) - 1
        }
        continue
      }
      if (!tracking) {
        if (postingCounts === null) {
          for (let at = from; at < upto; at++) accumulator[ids[at]] += 1
          continue
        }
        for (let at = from; at < upto; at++) {
          const count = postingCounts[at]
          accumulator[ids[at]] += queryCount < count ? queryCount : count
        }
        continue
      }
      if (postingCounts === null) {
        for (let at = from; at < upto; at++) {
          const id = ids[at]
          if (accumulator[id] === 0) touched.push(id)
          accumulator[id] += 1
        }
        continue
      }
      for (let at = from; at < upto; at++) {
        const id = ids[at]
        if (accumulator[id] === 0) touched.push(id)
        const count = postingCounts[at]
        accumulator[id] += queryCount < count ? queryCount : count
      }
    }
  }

  /**
   * Top-k with Dice's arithmetic in the loop rather than behind a callback.
   *
   * The callback was a good trade while selection ran once per *touched* choice.
   * A dense list makes it run once per choice in the corpus, and inlining the
   * arithmetic measured 1.41–1.92x end to end where closing the callback over
   * locals recovered only 1.04–1.18x. The rest is the call boundary itself,
   * which only duplication removes.
   */
  private top(queryGrams: number, threshold: number | null, room: number): number {
    const sealed = this.sealed
    const state = this.state
    const touched = state.touched
    const accumulator = this.accumulator
    const gramCount = sealed.gramCount
    const base = state.base
    const everyChoice = state.scannedAll
    const total = everyChoice ? sealed.choiceCount : touched.length
    const ids = state.ids
    const scores = state.scores
    let length = 0
    for (let index = 0; index < total; index++) {
      const id = everyChoice ? index : touched[index]
      const grams = gramCount[id]
      // Scored rather than skipped: a dense list puts every choice into this
      // walk, gramless ones included, and they are then not zero-filled either.
      // A gramless choice against a query that has grams shares nothing.
      const score =
        grams === 0 ? 0 : (2 * (base + accumulator[id])) / (queryGrams + grams)
      if (threshold !== null && score < threshold) continue
      let at = length
      if (at === room) {
        if (!outranks(score, id, scores[room - 1], ids[room - 1])) continue
        at = room - 1
      } else {
        length++
      }
      while (at > 0 && outranks(score, id, scores[at - 1], ids[at - 1])) {
        ids[at] = ids[at - 1]
        scores[at] = scores[at - 1]
        at--
      }
      ids[at] = id
      scores[at] = score
    }
    return length
  }

  /**
   * Clears only what the query touched. Walking the whole accumulator would put
   * a cost proportional to the corpus back into every query, which is the one
   * thing this representation exists to avoid — except where a dense list has
   * already made the walk the whole corpus, and then `fill` beats it.
   */
  private reset(): void {
    const state = this.state
    const accumulator = this.accumulator
    const touched = state.touched
    if (state.scannedAll) accumulator.fill(0)
    else
      for (let index = 0; index < touched.length; index++) accumulator[touched[index]] = 0
    touched.length = 0
    state.scannedAll = false
    state.base = 0
  }
}

/**
 * A Cosine index: `Σ a·b / √(‖a‖² ‖b‖²)`, clamped.
 *
 * Its accumulator stays `Float64Array`. The dot product is bounded by
 * `queryGrams × choiceGrams` rather than by the query alone, so a long query
 * against a long choice can carry it past what an `Int32Array` holds, and the
 * failure mode would be a wrong score rather than a thrown error.
 */
class CosineIndex implements ChoiceIndex {
  private readonly state = new QueryState()
  private readonly accumulator: Float64Array

  constructor(private readonly sealed: SealedIndex<Float64Array>) {
    this.accumulator = new Float64Array(sealed.choiceCount)
  }

  /**
   * `Σ qᵢ·cᵢ ≤ gramCount(query) · gramCount(choice)`, and the longest choice in
   * the index is the one that can carry it past a double's exact integers —
   * where a dense list and a sparse one stop agreeing to the bit.
   */
  private begin(query: Sequence): ArrayLike<unknown> {
    const elements = convSequence(query)
    assertCosineExact(
      elements.length - this.sealed.gramSize + 1,
      this.sealed.maxGramCount,
    )
    return elements
  }

  select(
    query: Sequence,
    threshold: number | null,
    limit: number | null,
  ): SelectedChoices {
    // Collect and sort rather than insert into place, for the reason
    // `DiceIndex.select` gives.
    if (limit === null) return rankSelected(this.collect(query, threshold, false))
    const sealed = this.sealed
    const state = this.state
    const elements = this.begin(query)
    if (elements.length < sealed.gramSize) {
      return gramlessResult(sealed, state, elements, threshold, limit, false)
    }
    const queryNorm = extractGrams(
      elements,
      sealed.gramSize,
      sealed.radix,
      false,
      state.keys,
      state.counts,
    )
    this.accumulate()
    const room = roomFor(limit, sealed.choiceCount)
    state.reserve(room)
    const length = fillZeroes(
      sealed,
      state,
      this.accumulator,
      this.top(queryNorm, threshold, room),
      threshold,
      room,
    )
    this.reset()
    return { ids: state.ids, scores: state.scores, length }
  }

  scan(query: Sequence, threshold: number | null): SelectedChoices {
    return this.collect(query, threshold, true)
  }

  /** Every qualifying choice, as `DiceIndex.collect` explains. */
  private collect(
    query: Sequence,
    threshold: number | null,
    ascending: boolean,
  ): SelectedChoices {
    const sealed = this.sealed
    const state = this.state
    const elements = this.begin(query)
    if (elements.length < sealed.gramSize) {
      return gramlessResult(sealed, state, elements, threshold, null, ascending)
    }
    const queryNorm = extractGrams(
      elements,
      sealed.gramSize,
      sealed.radix,
      false,
      state.keys,
      state.counts,
    )
    this.accumulate()
    const everyChoice = state.scannedAll || zeroesQualify(threshold)
    const source = everyChoice ? null : ascending ? sortedTouched(state) : state.touched
    const total = source === null ? sealed.choiceCount : source.length
    state.reserve(total)
    const ids = state.ids
    const scores = state.scores
    const accumulator = this.accumulator
    const squaredNorm = sealed.squaredNorm
    const base = state.base
    let length = 0
    for (let index = 0; index < total; index++) {
      const id = source === null ? index : source[index]
      const norm = squaredNorm[id]
      const score =
        norm === 0 ? 0 : clamp((base + accumulator[id]) / Math.sqrt(queryNorm * norm))
      if (threshold !== null && score < threshold) continue
      ids[length] = id
      scores[length] = score
      length++
    }
    this.reset()
    return { ids, scores, length }
  }

  private accumulate(): void {
    const sealed = this.sealed
    const state = this.state
    const postings = sealed.postings
    const accumulator = this.accumulator
    const touched = state.touched
    const keys = state.keys
    const queryCounts = state.counts
    const ids = postings.ids
    const postingCounts = postings.counts
    const offsets = postings.offsets
    const dense = postings.dense
    state.base = 0
    if (dense !== null && reachesDenseList(postings, dense, keys)) state.scannedAll = true
    const tracking = !state.scannedAll
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      if (ordinal === undefined) continue
      const queryCount = queryCounts[index]
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      if (dense !== null && dense[ordinal] === 1) {
        // The dot product's default term is `queryCount × 1`, and an exception
        // replaces it: an absent choice gives back the whole term, a repeated
        // gram adds the extra `count − 1` copies.
        state.base += queryCount
        if (postingCounts === null) {
          for (let at = from; at < upto; at++) accumulator[ids[at]] -= queryCount
          continue
        }
        for (let at = from; at < upto; at++) {
          accumulator[ids[at]] += queryCount * (postingCounts[at] - 1)
        }
        continue
      }
      if (!tracking) {
        if (postingCounts === null) {
          for (let at = from; at < upto; at++) accumulator[ids[at]] += queryCount
          continue
        }
        for (let at = from; at < upto; at++) {
          accumulator[ids[at]] += queryCount * postingCounts[at]
        }
        continue
      }
      if (postingCounts === null) {
        for (let at = from; at < upto; at++) {
          const id = ids[at]
          if (accumulator[id] === 0) touched.push(id)
          accumulator[id] += queryCount
        }
        continue
      }
      for (let at = from; at < upto; at++) {
        const id = ids[at]
        if (accumulator[id] === 0) touched.push(id)
        accumulator[id] += queryCount * postingCounts[at]
      }
    }
  }

  private top(queryNorm: number, threshold: number | null, room: number): number {
    const sealed = this.sealed
    const state = this.state
    const touched = state.touched
    const accumulator = this.accumulator
    const squaredNorm = sealed.squaredNorm
    const base = state.base
    const everyChoice = state.scannedAll
    const total = everyChoice ? sealed.choiceCount : touched.length
    const ids = state.ids
    const scores = state.scores
    let length = 0
    for (let index = 0; index < total; index++) {
      const id = everyChoice ? index : touched[index]
      const norm = squaredNorm[id]
      // See `DiceIndex.top`. A zero norm would divide to the infinity the clamp
      // turns into a perfect score, which is what a gramless choice used to get.
      const score =
        norm === 0 ? 0 : clamp((base + accumulator[id]) / Math.sqrt(queryNorm * norm))
      if (threshold !== null && score < threshold) continue
      let at = length
      if (at === room) {
        if (!outranks(score, id, scores[room - 1], ids[room - 1])) continue
        at = room - 1
      } else {
        length++
      }
      while (at > 0 && outranks(score, id, scores[at - 1], ids[at - 1])) {
        ids[at] = ids[at - 1]
        scores[at] = scores[at - 1]
        at--
      }
      ids[at] = id
      scores[at] = score
    }
    return length
  }

  private reset(): void {
    const state = this.state
    const accumulator = this.accumulator
    const touched = state.touched
    if (state.scannedAll) accumulator.fill(0)
    else
      for (let index = 0; index < touched.length; index++) accumulator[touched[index]] = 0
    touched.length = 0
    state.scannedAll = false
    state.base = 0
  }
}

/**
 * One square root of the product and then a clamp, which is the arithmetic the
 * exhaustive kernel uses and for its reason: `Math.sqrt(3) * Math.sqrt(3)` is
 * `3.0000000000000004`, which would leave a profile scored against itself just
 * short of `1`.
 */
function clamp(similarity: number): number {
  return similarity < 1 ? similarity : 1
}

/** A builder for a Sørensen-Dice index over grams of `gramSize` elements. */
export function createDiceIndexBuilder(gramSize: number): ChoiceIndexBuilder {
  return new NGramIndexBuilder(
    gramSize,
    () => null,
    (sealed) => new DiceIndex(sealed),
  )
}

/** A builder for a Cosine index over grams of `gramSize` elements. */
export function createCosineIndexBuilder(gramSize: number): ChoiceIndexBuilder {
  return new NGramIndexBuilder(
    gramSize,
    (values) => Float64Array.from(values),
    (sealed) => new CosineIndex(sealed),
  )
}
