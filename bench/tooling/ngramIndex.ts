/**
 * A corpus-wide inverted n-gram index, as an experiment: can one of these
 * replace the N prepared `NGramProfile` tries a Dice/Cosine Matcher retains,
 * and still reproduce its results exactly?
 *
 * Bench-only on purpose. `src/` requires every module to be reachable from a
 * public entrypoint and covered to 100%, so an unwired prototype cannot live
 * there — and until the numbers say this is worth wiring, unwired is what it is.
 *
 * The index answers queries from index-only state and never falls back to a
 * scorer. A fallback would mean retaining profiles beside the index, and the
 * memory question would then be "what does an index cost on top of profiles"
 * rather than "can it replace them".
 */

import {
  elementsEqual,
  type GramNode,
  type NGramProfile,
} from '../../src/algorithms/shared/ngram.js'
import { convSequence } from '../../src/algorithms/shared/sequence.js'

/** One choice id and its score, in the index's own id space. */
export interface Scored {
  readonly id: number
  readonly score: number
}

/**
 * What one query cost, structurally. Reset at the start of every query, so a
 * caller reads them straight after the call that produced them.
 *
 * `postingEntriesTouched` is the number this experiment turns on: it is
 * `Σ |postings(g)|` over the query's distinct grams, and it scales with how
 * *common* those grams are rather than with how many candidates match.
 */
export interface IndexCounters {
  postingEntriesTouched: number
  distinctQueryGrams: number
  candidatesTouched: number
  candidatesQualified: number
  zeroFillCandidates: number
  /** Grams the prefix scan walked, and the ones it left for verification. */
  prefixGrams: number
  suffixGrams: number
  /** Candidates whose exact score the suffix had to be probed for. */
  verifiedCandidates: number
  /** Binary searches those verifications cost. */
  verifyProbes: number
  /** Posting entries the suffix walk read, when probing looked more expensive. */
  suffixWalked: number
}

/**
 * Posting lists in compressed-sparse-row form: one `ids` array for the whole
 * index, and `offsets[ordinal] .. offsets[ordinal + 1]` marking each gram's
 * slice of it. The map holds an ordinal rather than an object.
 *
 * The shape it replaces was two typed arrays per distinct gram, so a corpus
 * with seventeen thousand distinct trigrams carried seventeen thousand posting
 * objects, thirty-four thousand typed arrays and as many array buffers — object
 * headers and collector work proportional to gram variety rather than to the
 * data. It also scatters the walk across that many allocations, where this
 * streams one array.
 *
 * `counts` is the narrowest word that holds the largest frequency in the index,
 * which measured one byte on every corpus here — a 32-bit word for a number
 * that is almost always 1 was most of the payload.
 *
 * It is `null` only when *no* frequency anywhere exceeds 1, which sounds like
 * the common case and is not: 99.9% of entries are 1 on 26-letter trigrams and
 * 95.0% on Zipf text, but a maximum of 3 and 4 respectively disables the
 * shortcut for the whole corpus. What is true per list — 93.5% and 59.8% of
 * lists are all-ones — is where a sparse representation would go, if the byte
 * word ever stops being enough.
 */
interface Postings {
  readonly ordinals: Map<string | number, number>
  readonly offsets: Uint32Array
  readonly ids: Uint32Array
  readonly counts: Uint8Array | Uint16Array | Uint32Array | null
}

interface PostingBuilder {
  readonly ids: number[]
  readonly counts: number[]
}

interface GramlessChoice {
  readonly id: number
  readonly elements: readonly unknown[]
}

/** Carries the offending element, so the index knows which rung it needs. */
class OutOfRadix extends Error {
  constructor(readonly element: number) {
    super('gram element does not fit the packed key radix')
  }
}

function integerElement(element: unknown): number {
  if (typeof element !== 'number' || !Number.isInteger(element)) {
    throw new TypeError('the ngram index prototype accepts integer gram elements only')
  }
  return element
}

/**
 * The rungs a packed gram key can sit on, narrowest first: a byte for Latin-1,
 * a BMP word, and the full code-point range. An index starts on the narrowest
 * its depth allows and widens when an element does not fit.
 */
const RADIX_LADDER: readonly number[] = [0x100, 0x1_0000, 0x11_0000]

/**
 * The radices that hold a gram of this depth inside one safe integer, smallest
 * first. Latin-1 text needs 8 bits per element, so `'abc'` packs into 24 —
 * `0x616263` — where a BMP radix spends 48 on the same three letters. Small
 * integer keys are the ones V8 handles best, and the ladder is what lets a
 * corpus use the smallest one its content allows.
 *
 * Depth decides how far the ladder reaches: a byte radix holds six elements,
 * a BMP radix three, a full code-point radix two.
 */
export function feasibleRadices(gramSize: number): readonly number[] {
  return RADIX_LADDER.filter(
    (radix) => Math.pow(radix, gramSize) <= Number.MAX_SAFE_INTEGER,
  )
}

/**
 * The smallest feasible radix that can hold `element`, or `null` for strings.
 *
 * A negative element goes straight to strings. Positional packing has no room
 * below zero, so answering with a rung the element is merely *less than* would
 * hand `rekey` a target no wider than the one that just failed — the ladder would
 * report that it could not widen, on an element the joined-string scheme
 * represents exactly.
 */
function radixFor(gramSize: number, element: number): number | null {
  if (element < 0) return null
  for (const radix of feasibleRadices(gramSize)) if (element < radix) return radix
  return null
}

/**
 * Every distinct gram of a built profile, with its frequency.
 *
 * One walk shared by indexing and by query flattening, so the two cannot drift
 * apart on how a gram becomes a key — which is the only way this index could
 * disagree with the metric it is reproducing.
 *
 * Iterative over an explicit stack, as `sharedFrequency` is: `gramSize` equals
 * the trie depth and is caller-supplied, so recursion would put a stack overflow
 * inside the range of valid inputs.
 *
 * The `NaN`-is-unmatchable rule needs no reimplementing here: such a gram is
 * never inserted into the trie, so this walk cannot see one, while `gramCount`
 * and `squaredNorm` still count it — and those two are copied off the profile.
 *
 * `radix` picks how a gram becomes a key: a positional integer when the depth
 * and the element range allow one, and a joined string otherwise. The string
 * form allocates once per gram *per choice*, which at 100k choices is millions
 * of short-lived strings; the packed form allocates nothing.
 */
function eachGram(
  profile: NGramProfile,
  radix: number | null,
  lenient: boolean,
  visit: (key: string | number, count: number) => void,
): void {
  const last = profile.gramSize - 1
  const nodes: GramNode[] = [profile.root]
  const depths: number[] = [0]
  let top = 1
  if (radix === null) {
    const prefixes: string[] = ['']
    while (top > 0) {
      top--
      const node = nodes[top]
      const prefix = prefixes[top]
      const depth = depths[top]
      if (depth === last) {
        const counts = node.counts
        if (counts !== null) {
          for (const [element, count] of counts) {
            visit(prefix + integerElement(element), count)
          }
        }
        continue
      }
      const children = node.children
      if (children === null) continue
      for (const [element, child] of children) {
        nodes[top] = child
        prefixes[top] = `${prefix}${integerElement(element)},`
        depths[top] = depth + 1
        top++
      }
    }
    return
  }
  const partials: number[] = [0]
  while (top > 0) {
    top--
    const node = nodes[top]
    const partial = partials[top]
    const depth = depths[top]
    if (depth === last) {
      const counts = node.counts
      if (counts !== null) {
        for (const [element, count] of counts) {
          const value = integerElement(element)
          if (value < 0 || value >= radix) {
            // On a query this gram simply cannot be in a packed index, so it
            // matches nothing and skipping it is the answer. On a build it means
            // the whole index has to change key scheme.
            if (lenient) continue
            throw new OutOfRadix(value)
          }
          visit(partial * radix + value, count)
        }
      }
      continue
    }
    const children = node.children
    if (children === null) continue
    for (const [element, child] of children) {
      const value = integerElement(element)
      if (value < 0 || value >= radix) {
        if (lenient) continue
        throw new OutOfRadix(value)
      }
      nodes[top] = child
      partials[top] = partial * radix + value
      depths[top] = depth + 1
      top++
    }
  }
}

/**
 * The same gram, re-spelled for a wider radix or for the string scheme. Packing
 * is positional and therefore reversible, which is what lets an index that has
 * already ingested a million choices change key scheme without revisiting one
 * of them.
 */
function repackKey(
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

/**
 * Flatten into arrays the index owns rather than fresh ones per query.
 *
 * The prefix path needs a collection because it sorts; full accumulation does
 * not, and paid two arrays and an object per call for the privilege of sharing
 * one code path. Reused arrays keep the single path and drop the allocation.
 */
function flattenQueryInto(
  query: NGramProfile,
  radix: number | null,
  keys: (string | number)[],
  counts: number[],
): void {
  keys.length = 0
  counts.length = 0
  eachGram(query, radix, true, (key, count) => {
    keys.push(key)
    counts.push(count)
  })
}

/**
 * A profile with no grams is the one shape that retains its elements, which is
 * what `zeroGramSimilarity` compares. Anything else here is a broken invariant
 * rather than a bad input.
 */
function gramlessElements(profile: NGramProfile): ArrayLike<unknown> {
  const elements = profile.elements
  if (elements === null) {
    throw new TypeError('a profile with no grams must retain its elements')
  }
  return elements
}

/**
 * Copied rather than referenced: the elements of a string profile are a
 * `Uint32Array` view, and holding one would retain a buffer this index has no
 * reason to keep alive. They are fewer than `gramSize` values.
 *
 * Not passed through {@link integerKey}: no key is built from them, so a
 * gramless choice may hold whatever `convElement` left it holding, and
 * `elementsEqual` compares any values with `!==`.
 */
function copyElements(elements: ArrayLike<unknown>): unknown[] {
  const copy = new Array<unknown>(elements.length)
  for (let index = 0; index < elements.length; index++) copy[index] = elements[index]
  return copy
}

/**
 * The exhaustive drivers' ranking rule, as a predicate: a higher score wins, and
 * a tie goes to the earlier stored position.
 */
function outranks(score: number, id: number, other: Scored): boolean {
  return score > other.score || (score === other.score && id < other.id)
}

/**
 * `resultLimit` in `search/snapshot.ts` accepts null or a non-negative safe
 * integer and refuses everything else, so this does too: `0.5`, `NaN` and
 * `Infinity` would otherwise reach the insertion-sorted top-k, where a limit
 * that is never reached by `top.length` silently becomes "unlimited" and `NaN`
 * compares false against everything.
 */
function validLimit(limit: number | null): number | null {
  if (limit === null) return null
  if (!Number.isSafeInteger(limit) || limit < 0) {
    throw new RangeError('limit must be null or a non-negative safe integer')
  }
  return limit
}

/** One choice's frequency for a gram, or 0. The posting list is sorted by id. */
function frequencyOf(postings: Postings, ordinal: number, id: number): number {
  const ids = postings.ids
  let low = postings.offsets[ordinal]
  let high = postings.offsets[ordinal + 1] - 1
  while (low <= high) {
    const middle = (low + high) >>> 1
    const found = ids[middle]
    if (found === id) return postings.counts === null ? 1 : postings.counts[middle]
    if (found < id) low = middle + 1
    else high = middle - 1
  }
  return 0
}

/**
 * How much query frequency the prefix has to cover before the rest can be
 * skipped: `A - t + 1`, where `t` is the fewest shared grams any candidate could
 * qualify on.
 *
 * `t` rises with the candidate's gram count, so the binding case is the shortest
 * candidate that could still reach the threshold — `2B/(A+B) >= threshold`. Both
 * steps round *away* from the bound they need, because a `t` one too large
 * shortens the prefix and a short prefix is the one error that loses a result.
 * Costing a gram too many only costs a gram too many.
 */
function prefixTarget(gramCount: number, threshold: number): number {
  const shortest = Math.max(1, Math.ceil((threshold * gramCount) / (2 - threshold)) - 1)
  const needed = Math.max(1, Math.floor((threshold * (gramCount + shortest)) / 2))
  return gramCount - needed + 1
}

interface PrefixPlan {
  readonly suffixKeys: (string | number)[]
  readonly suffixCounts: number[]
  /** Query frequency still unaccounted for — the most any suffix can add. */
  readonly remaining: number
  /** Posting entries a suffix walk would read, against which probing is judged. */
  readonly walkCost: number
  readonly probeSteps: number
}

const EMPTY_SUFFIX: PrefixPlan = {
  suffixKeys: [],
  suffixCounts: [],
  remaining: 0,
  walkCost: 0,
  probeSteps: 0,
}

/**
 * A binary search is several times the cost of one sequential posting step —
 * branchy, and it misses cache where the walk streams. Four is a guess with the
 * right sign; what makes it safe is that both completions are exact, so the
 * constant only ever picks the slower of two correct answers.
 */
const PROBE_WEIGHT = 4

export class NGramIndex {
  private builder: Map<string | number, PostingBuilder> | null = new Map()
  private postings: Postings | null = null
  private radix: number | null
  /** How many times an out-of-range element forced the key scheme to widen. */
  rekeyed = 0
  /**
   * Choices must arrive in id order, because that is what leaves every posting
   * list sorted and lets `frequencyOf` binary-search it. Checked on the way in
   * rather than at `compact`, where a duplicate id would already have written
   * itself into every list it touched.
   */
  private nextChoiceId = 0
  private readonly gramCount: Uint32Array
  private readonly squaredNorm: Float64Array
  private readonly gramless: GramlessChoice[] = []
  /**
   * Per-query scratch. `Float64Array` rather than an integer width: a single
   * gram can repeat far more than a `Uint16Array` holds — 100k identical
   * characters is a legal choice — and at 1M choices this is 8 MB with no
   * accumulation-overflow question at all.
   */
  private readonly accumulator: Float64Array
  private readonly touched: number[] = []
  /** Candidates that outlived the cheap prunes, reused across queries. */
  private readonly survivors: number[] = []
  /** The flattened query, reused across queries for the same reason. */
  private readonly queryKeys: (string | number)[] = []
  private readonly queryCounts: number[] = []

  readonly counters: IndexCounters = {
    postingEntriesTouched: 0,
    distinctQueryGrams: 0,
    candidatesTouched: 0,
    candidatesQualified: 0,
    zeroFillCandidates: 0,
    prefixGrams: 0,
    suffixGrams: 0,
    verifiedCandidates: 0,
    verifyProbes: 0,
    suffixWalked: 0,
  }

  constructor(
    readonly gramSize: number,
    readonly choiceCount: number,
    /**
     * Packed integer keys where the depth allows them, which is the default.
     * `false` keeps the joined-string keys, and exists so the two can be
     * measured against each other in separate processes — one `Map.get` site
     * seeing both key types in one process would measure the mixture.
     */
    packedKeys = true,
    /** Pin the starting rung, so the ladder's rungs can be compared. */
    startRadix: number | null = null,
  ) {
    // A rung too wide for this depth overflows the safe-integer range in
    // `partial * radix + value`, and the loss of precision shows up as two grams
    // sharing a key — a wrong score, not a thrown error. Refuse it here rather
    // than let a pinned rung answer quietly.
    if (startRadix !== null && !feasibleRadices(gramSize).includes(startRadix)) {
      throw new RangeError(`radix ${startRadix} cannot hold ${gramSize} elements`)
    }
    this.radix = packedKeys ? (startRadix ?? feasibleRadices(gramSize)[0] ?? null) : null
    this.gramCount = new Uint32Array(choiceCount)
    this.squaredNorm = new Float64Array(choiceCount)
    this.accumulator = new Float64Array(choiceCount)
  }

  /**
   * The caller keeps no reference to `profile` after this returns — that is the
   * point of the experiment, so nothing here stores one either.
   */
  add(choiceId: number, profile: NGramProfile): void {
    const builder = this.builder
    if (builder === null) throw new TypeError('the index is already compacted')
    if (profile.gramSize !== this.gramSize) {
      throw new TypeError('profile gram size does not match the index')
    }
    this.acceptChoiceId(choiceId)
    this.gramCount[choiceId] = profile.gramCount
    this.squaredNorm[choiceId] = profile.squaredNorm
    if (profile.gramCount === 0) {
      this.gramless.push({
        id: choiceId,
        elements: copyElements(gramlessElements(profile)),
      })
      return
    }
    // A loop, not one attempt and a fallback: a single choice can need more than
    // one rung. `'\ud800😀'` is a lone surrogate followed by an astral
    // character, so the first element pushes a byte radix up to BMP and the
    // second pushes that one up again. Each rung is strictly wider than the
    // element that forced it, so this cannot cycle.
    while (this.radix !== null) {
      const before = this.radix
      try {
        this.insert(builder, choiceId, profile)
        return
      } catch (error) {
        if (!(error instanceof OutOfRadix)) throw error
        this.rekey(builder, choiceId, radixFor(this.gramSize, error.element))
      }
      if (this.radix === before) throw new Error('key scheme failed to widen')
    }
    this.insert(builder, choiceId, profile)
  }

  /**
   * Ingest a choice without building a profile for it.
   *
   * `add` goes through `NGramProfile`, which is a trie of nested `Map`s built
   * per choice and thrown away immediately — the index needs each gram once,
   * not a structure that can be walked. This extracts the grams straight from
   * the converted elements into one flat count map, which is the same
   * information with none of the nodes.
   *
   * Integer elements only — but so is {@link add}, which reaches
   * `integerElement` just the same. That is the honest scope of this whole
   * experiment: *can an integer/code-point n-gram index replace prepared
   * profiles for ordinary text*. The metric itself is more general — its trie is
   * keyed by `unknown` and treats `NaN` as unmatchable — and an index for that
   * would intern arbitrary elements to integer symbols first. Not Stage B.
   *
   * What this skips is the profile: `add` builds a trie per choice and throws it
   * away, where the index needs each gram once.
   */
  addSequence(choiceId: number, sequence: string): void {
    const builder = this.builder
    if (builder === null) throw new TypeError('the index is already compacted')
    this.acceptChoiceId(choiceId)
    const elements = convSequence(sequence)
    const gramSize = this.gramSize
    const gramCount = elements.length - gramSize + 1
    if (gramCount <= 0) {
      this.gramCount[choiceId] = 0
      this.squaredNorm[choiceId] = 0
      this.gramless.push({ id: choiceId, elements: copyElements(elements) })
      return
    }
    const radix = this.radix
    const counts = new Map<string | number, number>()
    let squaredNorm = 0
    for (let start = 0; start < gramCount; start++) {
      let key: string | number
      if (radix === null) {
        let joined = String(integerElement(elements[start]))
        for (let offset = 1; offset < gramSize; offset++) {
          joined += `,${integerElement(elements[start + offset])}`
        }
        key = joined
      } else {
        let packed = 0
        for (let offset = 0; offset < gramSize; offset++) {
          const value = integerElement(elements[start + offset])
          if (value < 0 || value >= radix) {
            // Rare enough to pay for: widen the whole index one rung and start
            // this choice again.
            this.rekey(builder, choiceId, radixFor(this.gramSize, value))
            this.nextChoiceId--
            this.addSequence(choiceId, sequence)
            return
          }
          packed = packed * radix + value
        }
        key = packed
      }
      const previous = counts.get(key) ?? 0
      squaredNorm += 2 * previous + 1
      counts.set(key, previous + 1)
    }
    this.gramCount[choiceId] = gramCount
    this.squaredNorm[choiceId] = squaredNorm
    for (const [key, count] of counts) {
      const posting = builder.get(key)
      if (posting === undefined) {
        builder.set(key, { ids: [choiceId], counts: [count] })
        continue
      }
      posting.ids.push(choiceId)
      posting.counts.push(count)
    }
  }

  private acceptChoiceId(choiceId: number): void {
    if (choiceId !== this.nextChoiceId) {
      throw new RangeError(
        `choices must arrive in id order: expected ${this.nextChoiceId}, got ${choiceId}`,
      )
    }
    if (choiceId >= this.choiceCount) {
      throw new RangeError('choice id is outside the index')
    }
    this.nextChoiceId++
  }

  private insert(
    builder: Map<string | number, PostingBuilder>,
    choiceId: number,
    profile: NGramProfile,
  ): void {
    eachGram(profile, this.radix, false, (key, count) => {
      const posting = builder.get(key)
      if (posting === undefined) {
        builder.set(key, { ids: [choiceId], counts: [count] })
        return
      }
      posting.ids.push(choiceId)
      posting.counts.push(count)
    })
  }

  /**
   * Widen the corpus-wide key representation one rung — to the narrowest radix
   * that holds the element that did not fit, or to joined strings when no packed
   * radix can. Everything already ingested is re-keyed rather than re-read, and
   * the choice that triggered it is rolled back first: its entries are the last
   * in whichever lists it reached, because choices arrive in id order.
   *
   * Called from a loop, because one choice can force more than one rung:
   * `'\ud800😀'` pushes a byte radix to BMP on its first element and BMP to the
   * full code-point range on its second.
   *
   * A real implementation would rather decide up front, and could: `convSequence`
   * already knows whether a string held a surrogate pair. This is the fallback
   * for when it turns out to be wrong.
   */
  private rekey(
    builder: Map<string | number, PostingBuilder>,
    choiceId: number,
    to: number | null,
  ): void {
    const radix = this.radix
    if (radix === null || radix === to) return
    const rekeyed = new Map<string | number, PostingBuilder>()
    for (const [key, posting] of builder) {
      const ids = posting.ids
      while (ids.length > 0 && ids[ids.length - 1] === choiceId) {
        ids.pop()
        posting.counts.pop()
      }
      if (ids.length === 0) continue
      rekeyed.set(repackKey(key, radix, to, this.gramSize), posting)
    }
    builder.clear()
    for (const [key, posting] of rekeyed) builder.set(key, posting)
    this.radix = to
    this.rekeyed++
  }

  compact(): void {
    const builder = this.builder
    if (builder === null) throw new TypeError('the index is already compacted')
    // Ids are contiguous by construction, so a short build would leave the tail
    // of the corpus indistinguishable from choices that score zero — and
    // `selectBest` would answer `{ id: 0, score: 0 }` for an index that never
    // saw choice 0. Completing the invariant is the point of having it.
    if (this.nextChoiceId !== this.choiceCount) {
      throw new Error(
        `expected ${this.choiceCount} choices, received ${this.nextChoiceId}`,
      )
    }
    let total = 0
    let widest = 0
    for (const posting of builder.values()) {
      total += posting.ids.length
      for (const count of posting.counts) if (count > widest) widest = count
    }
    const ordinals = new Map<string | number, number>()
    const offsets = new Uint32Array(builder.size + 1)
    const ids = new Uint32Array(total)
    const counts =
      widest <= 1
        ? null
        : widest < 0x100
          ? new Uint8Array(total)
          : widest < 0x1_0000
            ? new Uint16Array(total)
            : new Uint32Array(total)
    let ordinal = 0
    let at = 0
    for (const [key, posting] of builder) {
      ordinals.set(key, ordinal)
      offsets[ordinal] = at
      const sourceIds = posting.ids
      for (let index = 0; index < sourceIds.length; index++) {
        // Ascending by construction, because choices arrive in id order — and
        // `frequencyOf` binary-searches these, so it is worth saying out loud
        // rather than leaving as a property someone could quietly break.
        if (index > 0 && sourceIds[index - 1] >= sourceIds[index]) {
          throw new Error('posting list is not sorted by id')
        }
        ids[at] = sourceIds[index]
        if (counts !== null) counts[at] = posting.counts[index]
        at++
      }
      ordinal++
    }
    offsets[ordinal] = at
    this.postings = { ordinals, offsets, ids, counts }
    this.builder = null
  }

  /** Distinct grams in the compacted index. */
  gramVariety(): number {
    return this.requirePostings().ordinals.size
  }

  /**
   * What the index knows about itself at build time, and the reason it is worth
   * knowing: these two numbers predict whether querying it will beat scoring
   * every choice, without running a single query.
   *
   * `meanShare` is the fraction of the corpus an average gram's posting list
   * covers — grams-per-choice over distinct-grams, near enough. It is the right
   * predictor only when every gram is equally likely, which is true of random
   * text and false of every real corpus.
   *
   * `weightedShare` is the fraction covered by the gram a *query* is likely to
   * ask for, which is a different average: a gram appearing in half the corpus
   * is drawn far more often than one appearing twice. `Σ len² / Σ len` is that
   * expectation, and on skewed text it runs an order of magnitude above the
   * mean. Below roughly 0.1 the index reads a tenth of the corpus per gram and
   * wins; approaching 1 it reads everything and cannot.
   */
  postingStatistics(): {
    distinctGrams: number
    totalEntries: number
    meanShare: number
    weightedShare: number
    termWeightedShare: number
    countsWidthBytes: number
    maxCount: number
    singletonEntryShare: number
    singletonListShare: number
  } {
    const postings = this.requirePostings()
    const offsets = postings.offsets
    const counts = postings.counts
    const distinctGrams = postings.ordinals.size
    let totalEntries = 0
    let squared = 0
    let termTotal = 0
    let termWeighted = 0
    for (let ordinal = 0; ordinal < distinctGrams; ordinal++) {
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      const documentFrequency = upto - from
      totalEntries += documentFrequency
      squared += documentFrequency * documentFrequency
      let termFrequency = documentFrequency
      if (counts !== null) {
        termFrequency = 0
        for (let at = from; at < upto; at++) termFrequency += counts[at]
      }
      termTotal += termFrequency
      termWeighted += termFrequency * documentFrequency
    }
    // What the corpus-wide `counts === null` shortcut would have needed, against
    // what it actually gets: one repeated gram anywhere disables it, so the
    // share of entries that *are* 1 is the number worth reporting.
    let singletonEntries = 0
    let singletonLists = 0
    // An empty index has no frequency at all, so it reports none: `counts ===
    // null` means every entry is 1, which is vacuous when there are no entries.
    let maxCount = totalEntries === 0 ? 0 : counts === null ? 1 : 0
    for (let ordinal = 0; ordinal < distinctGrams; ordinal++) {
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      if (counts === null) {
        singletonEntries += upto - from
        singletonLists++
        continue
      }
      let allOne = true
      for (let at = from; at < upto; at++) {
        const count = counts[at]
        if (count === 1) singletonEntries++
        else allOne = false
        if (count > maxCount) maxCount = count
      }
      if (allOne) singletonLists++
    }
    return {
      distinctGrams,
      totalEntries,
      meanShare:
        distinctGrams === 0 ? 0 : totalEntries / distinctGrams / this.choiceCount,
      weightedShare: totalEntries === 0 ? 0 : squared / totalEntries / this.choiceCount,
      termWeightedShare:
        termTotal === 0 ? 0 : termWeighted / termTotal / this.choiceCount,
      countsWidthBytes: counts === null ? 0 : counts.BYTES_PER_ELEMENT,
      maxCount,
      singletonEntryShare: totalEntries === 0 ? 0 : singletonEntries / totalEntries,
      singletonListShare: distinctGrams === 0 ? 0 : singletonLists / distinctGrams,
    }
  }

  diceBest(query: NGramProfile, threshold: number | null): Scored | undefined {
    this.beginQuery(query)
    if (query.gramCount === 0) return this.gramlessBest(query, threshold)
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    this.diceAccumulate()
    const found = this.selectBest((id) => this.diceScore(query, id), threshold)
    this.reset()
    return found
  }

  diceSearch(
    query: NGramProfile,
    threshold: number | null,
    limit: number | null,
  ): Scored[] {
    this.beginQuery(query)
    if (validLimit(limit) === 0) return []
    if (query.gramCount === 0) return this.gramlessSearch(query, threshold, limit)
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    this.diceAccumulate()
    const found = this.select((id) => this.diceScore(query, id), threshold, limit)
    this.reset()
    return found
  }

  /**
   * Dice search that walks only a prefix of the query's grams.
   *
   * If a candidate needs `t` shared grams to reach the threshold and the query
   * holds `A` gram occurrences, a candidate sharing nothing with the query's
   * first `A - t + 1` occurrences can reach at most `t - 1` and cannot qualify.
   * Ordering the query's grams by posting length puts the common grams — the
   * long lists — outside that prefix, where they are never walked. Survivors are
   * then completed exactly against the skipped lists, so the result is the same
   * one {@link diceSearch} produces.
   *
   * Only for a positive threshold: `t` is what buys the prefix, and without one
   * there is nothing to be short of. Dice only, too — the argument needs a
   * threshold on the shared *count*, and Cosine's does not translate into one
   * without the norms, which is the same asymmetry that gives Dice a length
   * bound and Cosine none.
   */
  dicePrefixSearch(
    query: NGramProfile,
    threshold: number | null,
    limit: number | null,
  ): Scored[] {
    this.beginQuery(query)
    if (validLimit(limit) === 0) return []
    if (threshold === null || threshold <= 0)
      return this.diceSearch(query, threshold, limit)
    if (query.gramCount === 0) return this.gramlessSearch(query, threshold, limit)
    const plan = this.prefixScan(query, threshold)
    const found = this.verifyTop(query, plan, threshold, limit)
    this.reset()
    return found
  }

  /** {@link dicePrefixSearch} with a limit of one. */
  dicePrefixBest(query: NGramProfile, threshold: number | null): Scored | undefined {
    const found = this.dicePrefixSearch(query, threshold, 1)
    return found.length === 0 ? undefined : found[0]
  }

  cosineBest(query: NGramProfile, threshold: number | null): Scored | undefined {
    this.beginQuery(query)
    if (query.gramCount === 0) return this.gramlessBest(query, threshold)
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    this.cosineAccumulate()
    const found = this.selectBest((id) => this.cosineScore(query, id), threshold)
    this.reset()
    return found
  }

  cosineSearch(
    query: NGramProfile,
    threshold: number | null,
    limit: number | null,
  ): Scored[] {
    this.beginQuery(query)
    if (validLimit(limit) === 0) return []
    if (query.gramCount === 0) return this.gramlessSearch(query, threshold, limit)
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    this.cosineAccumulate()
    const found = this.select((id) => this.cosineScore(query, id), threshold, limit)
    this.reset()
    return found
  }

  /**
   * Walks the query's grams cheapest-first until the prefix covers enough query
   * frequency, and hands back what it skipped.
   *
   * Grams absent from the index sort first and cover their frequency for free:
   * no candidate holds them, so including them in the prefix only strengthens
   * the bound while costing no traversal at all.
   */
  private prefixScan(query: NGramProfile, threshold: number): PrefixPlan {
    const postings = this.requirePostings()
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    const keys = this.queryKeys
    const counts = this.queryCounts
    const lengths: number[] = new Array<number>(keys.length)
    const order: number[] = new Array<number>(keys.length)
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      lengths[index] =
        ordinal === undefined
          ? 0
          : postings.offsets[ordinal + 1] - postings.offsets[ordinal]
      order[index] = index
    }
    // Cost per unit of prefix coverage, not raw posting length. The prefix
    // target is measured in query gram *occurrences*, so a gram the query holds
    // twenty times covers twenty of them for one list walk. Ordering by
    // `length / queryCount` picks the cheaper list per occurrence covered;
    // where every query count is 1, which is most n-gram text, it is the same
    // order as before. Any prefix satisfying the target is exact, so this only
    // changes which valid one gets chosen.
    order.sort(
      (left, right) => lengths[left] / counts[left] - lengths[right] / counts[right],
    )

    const target = prefixTarget(query.gramCount, threshold)
    const accumulator = this.accumulator
    const touched = this.touched
    let covered = 0
    let entries = 0
    let index = 0
    for (; index < order.length && covered < target; index++) {
      const at = order[index]
      const queryCount = counts[at]
      covered += queryCount
      const ordinal = postings.ordinals.get(keys[at])
      if (ordinal === undefined) continue
      const ids = postings.ids
      const postingCounts = postings.counts
      const from = postings.offsets[ordinal]
      const upto = postings.offsets[ordinal + 1]
      entries += upto - from
      if (postingCounts === null) {
        for (let scan = from; scan < upto; scan++) {
          const id = ids[scan]
          if (accumulator[id] === 0) touched.push(id)
          accumulator[id] += queryCount < 1 ? queryCount : 1
        }
        continue
      }
      for (let scan = from; scan < upto; scan++) {
        const id = ids[scan]
        if (accumulator[id] === 0) touched.push(id)
        const count = postingCounts[scan]
        accumulator[id] += queryCount < count ? queryCount : count
      }
    }
    const suffixKeys: (string | number)[] = []
    const suffixCounts: number[] = []
    let remaining = 0
    let walkCost = 0
    let probeSteps = 0
    for (; index < order.length; index++) {
      const at = order[index]
      suffixKeys.push(keys[at])
      suffixCounts.push(counts[at])
      remaining += counts[at]
      const length = lengths[at]
      walkCost += length
      probeSteps += Math.log2(length + 1)
    }
    const counters = this.counters
    counters.distinctQueryGrams = keys.length
    counters.postingEntriesTouched = entries
    counters.candidatesTouched = touched.length
    counters.prefixGrams = keys.length - suffixKeys.length
    counters.suffixGrams = suffixKeys.length
    return { suffixKeys, suffixCounts, remaining, walkCost, probeSteps }
  }

  /**
   * Finishes the suffix by walking its posting lists instead of probing them,
   * for the candidates the prefix already found.
   *
   * Chosen when many candidates survive: probing costs one binary search per
   * survivor per suffix gram, and past a few thousand survivors that overtakes
   * reading the lists straight through. Candidates the prefix never touched are
   * skipped rather than accumulated — the prefix bound has already proved they
   * cannot qualify, and admitting them here would only add work.
   */
  private completeSuffix(plan: PrefixPlan): void {
    const postings = this.requirePostings()
    const accumulator = this.accumulator
    let entries = 0
    for (let at = 0; at < plan.suffixKeys.length; at++) {
      const ordinal = postings.ordinals.get(plan.suffixKeys[at])
      if (ordinal === undefined) continue
      const ids = postings.ids
      const counts = postings.counts
      const queryCount = plan.suffixCounts[at]
      const from = postings.offsets[ordinal]
      const upto = postings.offsets[ordinal + 1]
      entries += upto - from
      if (counts === null) {
        for (let scan = from; scan < upto; scan++) {
          const id = ids[scan]
          if (accumulator[id] === 0) continue
          accumulator[id] += queryCount < 1 ? queryCount : 1
        }
        continue
      }
      for (let scan = from; scan < upto; scan++) {
        const id = ids[scan]
        if (accumulator[id] === 0) continue
        const count = counts[scan]
        accumulator[id] += queryCount < count ? queryCount : count
      }
    }
    this.counters.postingEntriesTouched += entries
    this.counters.suffixWalked = entries
  }

  /**
   * Completes each surviving candidate against the skipped grams and keeps the
   * best `limit`.
   *
   * Three prunes before any probe, cheapest first: Dice's own length bound,
   * which the index can apply because it kept every candidate's gram count; the
   * partial overlap plus everything the suffix could still add; and, once the
   * result set is full, the score of the one at the bottom of it. That last is
   * the rising cutoff the exhaustive drivers get from their heap, and the thing
   * full accumulation has no way to use.
   */
  private verifyTop(
    query: NGramProfile,
    plan: PrefixPlan,
    threshold: number,
    limit: number | null,
  ): Scored[] {
    // Survivors, not touched candidates, decide how the suffix is finished.
    // Nearly every touched candidate dies here — to Dice's length bound, or to
    // the partial overlap plus everything the suffix could still add — and both
    // tests are arithmetic on numbers the index already holds. Counting them
    // first is what makes the choice below reflect the work that remains rather
    // than the work already done.
    const survivors = this.survivors
    survivors.length = 0
    const gramCounts = this.gramCount
    const accumulator = this.accumulator
    const queryGrams = query.gramCount
    for (let index = 0; index < this.touched.length; index++) {
      const id = this.touched[index]
      const denominator = queryGrams + gramCounts[id]
      const smaller = queryGrams < gramCounts[id] ? queryGrams : gramCounts[id]
      if ((2 * smaller) / denominator < threshold) continue
      if ((2 * (accumulator[id] + plan.remaining)) / denominator < threshold) continue
      survivors.push(id)
    }
    this.counters.verifiedCandidates = survivors.length

    // Both completions produce the same overlap; this only picks the cheaper.
    if (
      plan.walkCost > 0 &&
      plan.walkCost < survivors.length * plan.probeSteps * PROBE_WEIGHT
    ) {
      this.completeSuffix(plan)
      return this.verifySuffix(query, EMPTY_SUFFIX, threshold, limit)
    }
    return this.verifySuffix(query, plan, threshold, limit)
  }

  private verifySuffix(
    query: NGramProfile,
    plan: PrefixPlan,
    threshold: number,
    limit: number | null,
  ): Scored[] {
    const postings = this.requirePostings()
    const accumulator = this.accumulator
    const gramCounts = this.gramCount
    const survivors = this.survivors
    const suffixKeys = plan.suffixKeys
    const suffixCounts = plan.suffixCounts
    const queryGrams = query.gramCount
    const found: Scored[] = []
    let cutoff = threshold
    let qualified = 0
    let probes = 0
    for (let index = 0; index < survivors.length; index++) {
      const id = survivors[index]
      const choiceGrams = gramCounts[id]
      const denominator = queryGrams + choiceGrams
      const smaller = queryGrams < choiceGrams ? queryGrams : choiceGrams
      if ((2 * smaller) / denominator < cutoff) continue
      let shared = accumulator[id]
      let remaining = plan.remaining
      if ((2 * (shared + remaining)) / denominator < cutoff) continue
      let alive = true
      for (let at = 0; at < suffixKeys.length; at++) {
        const ordinal = postings.ordinals.get(suffixKeys[at])
        const queryCount = suffixCounts[at]
        remaining -= queryCount
        if (ordinal !== undefined) {
          probes++
          const count = frequencyOf(postings, ordinal, id)
          if (count > 0) shared += queryCount < count ? queryCount : count
        }
        if ((2 * (shared + remaining)) / denominator < cutoff) {
          alive = false
          break
        }
      }
      if (!alive) continue
      const score = (2 * shared) / denominator
      if (score < threshold) continue
      qualified++
      if (limit === null) {
        found.push({ id, score })
        continue
      }
      let at = found.length
      if (at === limit) {
        if (!outranks(score, id, found[limit - 1])) continue
        at = limit - 1
      }
      while (at > 0 && outranks(score, id, found[at - 1])) {
        found[at] = found[at - 1]
        at--
      }
      found[at] = { id, score }
      // Only ever upward, and only once the set is full: below that every
      // candidate still has a place waiting for it.
      if (found.length === limit) {
        const last = found[limit - 1].score
        if (last > cutoff) cutoff = last
      }
    }
    const counters = this.counters
    counters.candidatesQualified = qualified
    counters.verifyProbes = probes
    if (limit === null) {
      found.sort((left, right) => right.score - left.score || left.id - right.id)
    }
    return found
  }

  /**
   * Clears only what the query touched. Walking the whole accumulator would put
   * a cost proportional to the corpus back into every query, which is the one
   * thing this representation exists to avoid.
   */
  private reset(): void {
    const accumulator = this.accumulator
    const touched = this.touched
    for (let index = 0; index < touched.length; index++) accumulator[touched[index]] = 0
    touched.length = 0
  }

  private requirePostings(): Postings {
    const postings = this.postings
    if (postings === null) throw new TypeError('the index has not been compacted')
    return postings
  }

  private beginQuery(query: NGramProfile): void {
    if (query.gramSize !== this.gramSize) {
      throw new TypeError('query gram size does not match the index')
    }
    this.requirePostings()
    const counters = this.counters
    counters.postingEntriesTouched = 0
    counters.distinctQueryGrams = 0
    counters.candidatesTouched = 0
    counters.candidatesQualified = 0
    counters.zeroFillCandidates = 0
    counters.prefixGrams = 0
    counters.suffixGrams = 0
    counters.verifiedCandidates = 0
    counters.verifyProbes = 0
    counters.suffixWalked = 0
  }

  /**
   * The two accumulation loops are duplicated deliberately, for the reason
   * `sharedFrequency` and `dotProduct` are two literal traversals: this is the
   * innermost frame of the whole experiment, and a mode flag or a combiner
   * callback in it is the one thing that would make every measurement here a
   * measurement of megamorphic dispatch.
   */
  private diceAccumulate(): void {
    const postings = this.requirePostings()
    const accumulator = this.accumulator
    const touched = this.touched
    const keys = this.queryKeys
    const queryCounts = this.queryCounts
    let entries = 0
    const ids = postings.ids
    const postingCounts = postings.counts
    const offsets = postings.offsets
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      if (ordinal === undefined) continue
      const queryCount = queryCounts[index]
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      entries += upto - from
      // Split once per posting list rather than branching per entry: where
      // every frequency is 1 the whole counts array is absent, and the shared
      // minimum collapses to a constant.
      if (postingCounts === null) {
        const capped = queryCount < 1 ? queryCount : 1
        for (let at = from; at < upto; at++) {
          const id = ids[at]
          if (accumulator[id] === 0) touched.push(id)
          accumulator[id] += capped
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
    this.counters.distinctQueryGrams = keys.length
    this.counters.postingEntriesTouched = entries
    this.counters.candidatesTouched = touched.length
  }

  private cosineAccumulate(): void {
    const postings = this.requirePostings()
    const accumulator = this.accumulator
    const touched = this.touched
    const keys = this.queryKeys
    const queryCounts = this.queryCounts
    let entries = 0
    const ids = postings.ids
    const postingCounts = postings.counts
    const offsets = postings.offsets
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      if (ordinal === undefined) continue
      const queryCount = queryCounts[index]
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      entries += upto - from
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
    this.counters.distinctQueryGrams = keys.length
    this.counters.postingEntriesTouched = entries
    this.counters.candidatesTouched = touched.length
  }

  private diceScore(query: NGramProfile, id: number): number {
    return (2 * this.accumulator[id]) / (query.gramCount + this.gramCount[id])
  }

  /**
   * One square root of the product, then a clamp — the arithmetic
   * `profileSimilarity` uses and for its reason: `Math.sqrt(3) * Math.sqrt(3)`
   * is 3.0000000000000004, which would leave a profile scored against itself
   * just short of 1.
   */
  private cosineScore(query: NGramProfile, id: number): number {
    const similarity =
      this.accumulator[id] / Math.sqrt(query.squaredNorm * this.squaredNorm[id])
    return similarity < 1 ? similarity : 1
  }

  /**
   * `scoreOf` is a callback where the accumulators are literal loops, and the
   * asymmetry is the point: this runs once per *touched* candidate, which is the
   * quantity the index exists to keep small, while an accumulator runs once per
   * posting entry.
   */
  private selectBest(
    scoreOf: (id: number) => number,
    threshold: number | null,
  ): Scored | undefined {
    const touched = this.touched
    let bestId = -1
    let bestScore = 0
    let qualified = 0
    for (let index = 0; index < touched.length; index++) {
      const id = touched[index]
      const score = scoreOf(id)
      if (threshold !== null && score < threshold) continue
      qualified++
      if (bestId === -1 || score > bestScore || (score === bestScore && id < bestId)) {
        bestId = id
        bestScore = score
      }
    }
    this.counters.candidatesQualified = qualified
    if (bestId !== -1) return { id: bestId, score: bestScore }
    // Nothing was touched — a touched candidate shares a gram, so it scores
    // above zero and cannot have failed a threshold this branch is reached
    // under. So every choice scores 0, and `bestSimilarity` takes the first
    // item unconditionally rather than answering `undefined`.
    if (this.zeroesQualify(threshold) && this.choiceCount > 0) {
      this.counters.zeroFillCandidates = 1
      return { id: 0, score: 0 }
    }
    return undefined
  }

  private select(
    scoreOf: (id: number) => number,
    threshold: number | null,
    limit: number | null,
  ): Scored[] {
    const found =
      limit === null
        ? this.selectAll(scoreOf, threshold)
        : this.selectTop(scoreOf, threshold, limit)
    if (!this.zeroesQualify(threshold)) return found
    const accumulator = this.accumulator
    let filled = 0
    for (let id = 0; id < this.choiceCount; id++) {
      if (limit !== null && found.length >= limit) break
      if (accumulator[id] !== 0) continue
      found.push({ id, score: 0 })
      filled++
    }
    this.counters.zeroFillCandidates = filled
    return found
  }

  private selectAll(scoreOf: (id: number) => number, threshold: number | null): Scored[] {
    const touched = this.touched
    const found: Scored[] = []
    for (let index = 0; index < touched.length; index++) {
      const id = touched[index]
      const score = scoreOf(id)
      if (threshold !== null && score < threshold) continue
      found.push({ id, score })
    }
    // Ties lose on id, which is the stored order the exhaustive drivers rank by.
    // A total comparator, so posting-list arrival order cannot reach the result —
    // and no sort of `touched` is needed to get there.
    found.sort((left, right) => right.score - left.score || left.id - right.id)
    this.counters.candidatesQualified = found.length
    return found
  }

  /**
   * A bounded, insertion-sorted top-k rather than "collect every qualifier and
   * sort". The exhaustive driver keeps a heap of `limit`, so collecting all of
   * them would have measured 98,000 allocations and an `O(C log C)` sort against
   * its five-element heap — a comparison of this file's bookkeeping rather than
   * of the two representations. Insertion sort, not a heap, because `limit` is
   * small and the array stays in result order throughout.
   */
  private selectTop(
    scoreOf: (id: number) => number,
    threshold: number | null,
    limit: number,
  ): Scored[] {
    const touched = this.touched
    const top: Scored[] = []
    let qualified = 0
    for (let index = 0; index < touched.length; index++) {
      const id = touched[index]
      const score = scoreOf(id)
      if (threshold !== null && score < threshold) continue
      qualified++
      let at = top.length
      if (at === limit) {
        if (!outranks(score, id, top[limit - 1])) continue
        at = limit - 1
      }
      while (at > 0 && outranks(score, id, top[at - 1])) {
        top[at] = top[at - 1]
        at--
      }
      top[at] = { id, score }
    }
    this.counters.candidatesQualified = qualified
    return top
  }

  /**
   * Whether a score of exactly 0 belongs in the result. When it does, every
   * choice the postings never reached has to be accounted for; when it does
   * not, they can simply vanish.
   */
  private zeroesQualify(threshold: number | null): boolean {
    return threshold === null || threshold <= 0
  }

  /**
   * A gramless query scores 1 against a choice that is gramless and equal, and
   * 0 against everything else — `zeroGramSimilarity` is 1 only when both sides
   * have no grams. So this needs the short choices' elements and nothing else,
   * which is why they are the one thing the index retains besides postings.
   */
  private gramlessMatches(query: NGramProfile): number[] {
    const elements = gramlessElements(query)
    const equal: number[] = []
    for (const entry of this.gramless) {
      if (elementsEqual(elements, entry.elements)) equal.push(entry.id)
    }
    return equal
  }

  private gramlessBest(
    query: NGramProfile,
    threshold: number | null,
  ): Scored | undefined {
    const equal = this.gramlessMatches(query)
    if (equal.length > 0 && (threshold === null || threshold <= 1)) {
      this.counters.candidatesQualified = equal.length
      let best = equal[0]
      for (const id of equal) if (id < best) best = id
      return { id: best, score: 1 }
    }
    if (this.zeroesQualify(threshold) && this.choiceCount > 0) {
      this.counters.zeroFillCandidates = 1
      return { id: 0, score: 0 }
    }
    return undefined
  }

  private gramlessSearch(
    query: NGramProfile,
    threshold: number | null,
    limit: number | null,
  ): Scored[] {
    const equal = this.gramlessMatches(query)
    const scoring = new Set<number>(equal)
    const found: Scored[] = []
    if (threshold === null || threshold <= 1) {
      for (const id of [...equal].sort((left, right) => left - right)) {
        if (limit !== null && found.length >= limit) break
        found.push({ id, score: 1 })
      }
    }
    this.counters.candidatesQualified = found.length
    if (!this.zeroesQualify(threshold)) return found
    let filled = 0
    for (let id = 0; id < this.choiceCount; id++) {
      if (limit !== null && found.length >= limit) break
      if (scoring.has(id)) continue
      found.push({ id, score: 0 })
      filled++
    }
    this.counters.zeroFillCandidates = filled
    return found
  }
}
