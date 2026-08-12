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

import process from 'node:process'

import {
  buildProfile,
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
  /**
   * Candidates an accumulation actually wrote to, and `null` under a dense scan,
   * which no longer counts them: the marks that answered this were 26% of the
   * loop and their only other reader — `touched` — is unused there.
   *
   * It was measured while it existed, and the answer was 93% of the corpus on
   * the classes that matter, which is what closed skip-the-defaults selection.
   */
  modifiedCandidates: number | null
  /**
   * Whether a dense list put every candidate in play. It breaks the rule the
   * sparse representation runs on — a positive score and a posting-list hit are
   * the same event — because a default frequency applies to candidates no
   * posting entry ever named.
   */
  scannedAllCandidates: boolean
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
  /**
   * Narrowed to `Uint16` when the corpus fits, which is the largest single rung
   * of the accumulation loop: the ids are read sequentially, one per posting
   * entry, so halving them halves the loop's stream.
   *
   * One process must build at one width or the load site sees both element
   * kinds and the measurement is of the mixture — the same reason `packedKeys`
   * exists as a flag rather than a heuristic.
   */
  readonly ids: Uint16Array | Uint32Array
  readonly counts: Uint8Array | Uint16Array | Uint32Array | null
  /**
   * Which ordinals are stored inverted — `1` for a list whose slice holds the
   * choices that *lack* the gram rather than the ones that have it. `null` when
   * no list qualified, which is most corpora, and then nothing reads it.
   *
   * The two spellings share the slice layout, so only the meaning of an entry
   * changes: a sparse entry is a choice with that gram at `counts[at]`, a dense
   * entry is an exception to a default frequency of 1 — an absence at count `0`,
   * or a repeat at `2` or more. With `counts === null` the corpus has no repeats
   * at all, so a dense entry can only be an absence.
   */
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

/** See `qualifiesAsDense` for why this is two thirds and not one half. */
export const DENSE_CUTOFF = 2 / 3

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
/**
 * A dense list inverts both answers, so the two spellings need saying out loud:
 * in a sparse list a hit is the stored frequency and a miss is `0`, and in a
 * dense one a hit is the stored *exception* — `0` for an absence — while a miss
 * is the default frequency of `1`.
 *
 * Nothing reaches this with a dense ordinal today, because prefix filtering
 * falls back to full accumulation the moment a query gram is dense. Written to
 * be right anyway: a helper whose contract holds only because of where it
 * happens to be called from is a trap set for the next change.
 */
function frequencyOf(postings: Postings, ordinal: number, id: number): number {
  const ids = postings.ids
  const dense = postings.dense !== null && postings.dense[ordinal] === 1
  let low = postings.offsets[ordinal]
  let high = postings.offsets[ordinal + 1] - 1
  while (low <= high) {
    const middle = (low + high) >>> 1
    const found = ids[middle]
    if (found === id) {
      if (postings.counts !== null) return postings.counts[middle]
      return dense ? 0 : 1
    }
    if (found < id) low = middle + 1
    else high = middle - 1
  }
  return dense ? 1 : 0
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
   * Per-query scratch, 8 MB at a million choices. `Float64Array` by default
   * because Cosine's dot product has no width that is obviously enough — a
   * single gram can repeat far more than a `Uint16Array` holds, 100k identical
   * characters being a legal choice. `narrowAccumulator` halves it for Dice,
   * whose overlap is bounded by the query's own gram count.
   */
  private readonly accumulator: Int32Array | Float64Array
  private readonly touched: number[] = []
  /** Candidates that outlived the cheap prunes, reused across queries. */
  private readonly survivors: number[] = []
  /** The flattened query, reused across queries for the same reason. */
  private readonly queryKeys: (string | number)[] = []
  private readonly queryCounts: number[] = []

  /**
   * What every candidate scores before its own accumulator entry is added — the
   * sum of the dense lists' default contributions, and `0` whenever the query
   * reached none.
   */
  private base = 0

  /** Set when a dense list has put every candidate into `touched`. */
  private scannedAll = false

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
    scannedAllCandidates: false,
    modifiedCandidates: 0,
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
    /**
     * The share of the corpus a posting list has to cover before it is stored
     * inverted. `null` keeps every list sparse, which is what the representation
     * did before dense lists existed and what the A/B measures against.
     */
    private readonly denseCutoff: number | null = DENSE_CUTOFF,
    /**
     * Store posting ids in the narrowest word the corpus fits in. `false` keeps
     * `Uint32Array` at every size, which is what the A/B measures against — and
     * it has to be a flag rather than a size threshold alone, because a process
     * that builds one index of each width measures neither.
     */
    private readonly narrowIds = true,
    /**
     * Hold Dice's shared counts in an `Int32Array` rather than a `Float64Array`.
     * They are integral — a sum of `min(queryCount, choiceCount)` terms, less one
     * per dense list — and bounded by the query's gram count, so the narrow word
     * is exact up to a query of 2.1 billion grams.
     *
     * Dice only: Cosine's dot product is integral too but bounded by
     * `queryGrams × choiceGrams`, which a long query against a long choice can
     * carry past 32 bits. An index built this way refuses Cosine rather than
     * quietly wrapping.
     */
    private readonly narrowAccumulator = false,
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
    this.accumulator = narrowAccumulator
      ? new Int32Array(choiceCount)
      : new Float64Array(choiceCount)
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
    // Which lists to invert, and how much room that takes, before anything is
    // allocated — a dense list's slice is a different size from its sparse one,
    // so the decision has to be made in a pass of its own.
    const inverted = new Set<string | number>()
    let hybridTotal = 0
    for (const [key, posting] of builder) {
      const length = posting.ids.length
      let exceptions = this.choiceCount - length
      for (const count of posting.counts) if (count !== 1) exceptions++
      if (this.qualifiesAsDense(length, exceptions)) {
        inverted.add(key)
        hybridTotal += exceptions
      } else {
        hybridTotal += length
      }
    }
    const ordinals = new Map<string | number, number>()
    const offsets = new Uint32Array(builder.size + 1)
    // A `Uint16` id holds 0…65,535, so a corpus of exactly 65,536 choices is the
    // largest that fits.
    const ids =
      this.narrowIds && this.choiceCount <= 0x1_0000
        ? new Uint16Array(hybridTotal)
        : new Uint32Array(hybridTotal)
    const counts =
      widest <= 1
        ? null
        : widest < 0x100
          ? new Uint8Array(hybridTotal)
          : widest < 0x1_0000
            ? new Uint16Array(hybridTotal)
            : new Uint32Array(hybridTotal)
    const dense = inverted.size === 0 ? null : new Uint8Array(builder.size)
    let ordinal = 0
    let at = 0
    for (const [key, posting] of builder) {
      ordinals.set(key, ordinal)
      offsets[ordinal] = at
      const sourceIds = posting.ids
      for (let index = 1; index < sourceIds.length; index++) {
        // Ascending by construction, because choices arrive in id order — and
        // `frequencyOf` binary-searches these, so it is worth saying out loud
        // rather than leaving as a property someone could quietly break. The
        // inverted walk below depends on it too.
        if (sourceIds[index - 1] >= sourceIds[index]) {
          throw new Error('posting list is not sorted by id')
        }
      }
      if (dense !== null && inverted.has(key)) {
        dense[ordinal] = 1
        // One merge of the sorted list against every id: what is missing becomes
        // an absence at count 0, what is present with a frequency other than 1
        // becomes that frequency, and the overwhelmingly common present-once
        // entry is stored nowhere at all.
        let cursor = 0
        for (let id = 0; id < this.choiceCount; id++) {
          if (cursor < sourceIds.length && sourceIds[cursor] === id) {
            const count = posting.counts[cursor]
            cursor++
            if (count === 1) continue
            ids[at] = id
            if (counts !== null) counts[at] = count
            at++
            continue
          }
          ids[at] = id
          if (counts !== null) counts[at] = 0
          at++
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
    this.postings = { ordinals, offsets, ids, counts, dense }
    this.builder = null
  }

  /**
   * A list covering enough of the corpus is cheaper stored inverted, and `2/3`
   * rather than the obvious `1/2` is the cutoff because inverting costs a second
   * thing: any query touching a dense list has to score every candidate, since a
   * default frequency applies to all of them. Writing that out, a dense gram
   * changes the work by `(N − 2·length + exceptions)` in accumulation and at most
   * `(N − length)` in selection, and the sum only turns negative above `2N/3`.
   * At exactly one half the storage saving is zero and the scan is pure loss.
   */
  private qualifiesAsDense(length: number, exceptions: number): boolean {
    const cutoff = this.denseCutoff
    if (cutoff === null || this.choiceCount === 0) return false
    return length >= cutoff * this.choiceCount && exceptions < length
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
    /** Entries actually held, which a dense list makes smaller than its corpus share. */
    storedEntries: number
    /** `Σ` document frequency — the logical size, and what the shares divide by. */
    documentEntries: number
    meanShare: number
    weightedShare: number
    termWeightedShare: number
    countsWidthBytes: number
    idsWidthBytes: number
    maxCount: number
    singletonEntryShare: number
    singletonListShare: number
    denseLists: number
  } {
    const postings = this.requirePostings()
    const offsets = postings.offsets
    const counts = postings.counts
    const denseFlags = postings.dense
    const distinctGrams = postings.ordinals.size
    let storedEntries = 0
    let documentEntries = 0
    let squared = 0
    let termTotal = 0
    let termWeighted = 0
    // What the corpus-wide `counts === null` shortcut would have needed, against
    // what it actually gets: one repeated gram anywhere disables it, so the
    // share of entries that *are* 1 is the number worth reporting.
    let singletonEntries = 0
    let singletonLists = 0
    let denseLists = 0
    let maxCount = 0
    for (let ordinal = 0; ordinal < distinctGrams; ordinal++) {
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      const length = upto - from
      storedEntries += length
      let documentFrequency: number
      let repeated = 0
      let extra = 0
      if (denseFlags !== null && denseFlags[ordinal] === 1) {
        denseLists++
        // A slice entry is an exception: absent at `0`, repeated above `1`. With
        // no counts array at all, the corpus has no repeats, so every one of
        // them is an absence.
        let absent = length
        if (counts !== null) {
          absent = 0
          for (let at = from; at < upto; at++) {
            const count = counts[at]
            if (count === 0) absent++
            else {
              repeated++
              extra += count - 1
              if (count > maxCount) maxCount = count
            }
          }
        }
        documentFrequency = this.choiceCount - absent
      } else {
        documentFrequency = length
        if (counts !== null) {
          for (let at = from; at < upto; at++) {
            const count = counts[at]
            if (count !== 1) {
              repeated++
              extra += count - 1
            }
            if (count > maxCount) maxCount = count
          }
        }
      }
      documentEntries += documentFrequency
      singletonEntries += documentFrequency - repeated
      if (repeated === 0) singletonLists++
      const termFrequency = documentFrequency + extra
      squared += documentFrequency * documentFrequency
      termTotal += termFrequency
      termWeighted += termFrequency * documentFrequency
    }
    // An empty index has no frequency at all, so it reports none.
    if (maxCount === 0 && documentEntries > 0) maxCount = 1
    return {
      distinctGrams,
      storedEntries,
      documentEntries,
      meanShare:
        distinctGrams === 0 ? 0 : documentEntries / distinctGrams / this.choiceCount,
      weightedShare:
        documentEntries === 0 ? 0 : squared / documentEntries / this.choiceCount,
      termWeightedShare:
        termTotal === 0 ? 0 : termWeighted / termTotal / this.choiceCount,
      countsWidthBytes: counts === null ? 0 : counts.BYTES_PER_ELEMENT,
      idsWidthBytes: postings.ids.BYTES_PER_ELEMENT,
      maxCount,
      singletonEntryShare: documentEntries === 0 ? 0 : singletonEntries / documentEntries,
      singletonListShare: distinctGrams === 0 ? 0 : singletonLists / distinctGrams,
      denseLists,
    }
  }

  /**
   * What a dense "default frequency 1, store the exceptions" posting would cost
   * against what the sparse one costs, for one query. A probe, not an
   * implementation: it walks the same lists the accumulator would and counts,
   * and it touches nothing in the hot path, so building the real thing can wait
   * on a number rather than on an argument.
   *
   * A list covering more than half the corpus is cheaper stored inverted — its
   * absences plus the entries whose frequency is not 1. Per list the better of
   * the two is what a hybrid would pick, so `hybridWork` is a lower bound on any
   * such representation and `sparseWork` is what runs today.
   *
   * Both sides carry their selection scan, and they are not the same scan: the
   * sparse path visits the candidates it touched, a dense list forces all of
   * them, because once a base frequency applies to everyone nothing is untouched.
   * Charging that to the hybrid alone made a corpus where `touched` is already
   * `N` look like a regression when it is a small win.
   */
  /**
   * What a dense build would store, measured on an index built without one.
   * The corpus-level companion to {@link denseProbe}.
   */
  denseOutlook(cutoff: number): { denseLists: number; hybridEntries: number } {
    const postings = this.requireSparseIndex()
    const offsets = postings.offsets
    const counts = postings.counts
    let denseLists = 0
    let hybridEntries = 0
    for (let ordinal = 0; ordinal < postings.ordinals.size; ordinal++) {
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      const length = upto - from
      let notOne = 0
      if (counts !== null) {
        for (let at = from; at < upto; at++) if (counts[at] !== 1) notOne++
      }
      const inverted = this.choiceCount - length + notOne
      if (length >= cutoff * this.choiceCount && inverted < length) {
        denseLists++
        hybridEntries += inverted
      } else {
        hybridEntries += length
      }
    }
    return { denseLists, hybridEntries }
  }

  /**
   * Of the posting entries one query reads, how many sit in a list that needs no
   * count at all — every sparse frequency `1`, or every dense exception an
   * absence. Those are the entries a per-list implicit mode would relieve of the
   * count load, and the ladder's count step times this share is what the whole
   * idea can be worth.
   *
   * Asked before building it, because the flag array and the per-list dispatch
   * *are* the implementation, and a share small enough is the cheaper answer.
   */
  implicitOutlook(query: NGramProfile): {
    lists: number
    implicitLists: number
    entries: number
    implicitEntries: number
  } {
    const postings = this.requirePostings()
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    const keys = this.queryKeys
    const offsets = postings.offsets
    const counts = postings.counts
    const dense = postings.dense
    let lists = 0
    let implicitLists = 0
    let entries = 0
    let implicitEntries = 0
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      if (ordinal === undefined) continue
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      lists++
      entries += upto - from
      // A dense entry is an exception to a default of 1, so its implicit
      // spelling is an absence — count `0` — where a sparse entry's is `1`.
      const implied = dense !== null && dense[ordinal] === 1 ? 0 : 1
      let differs = 0
      if (counts !== null) {
        for (let at = from; at < upto; at++) if (counts[at] !== implied) differs++
      }
      if (differs === 0) {
        implicitLists++
        implicitEntries += upto - from
      }
    }
    return { lists, implicitLists, entries, implicitEntries }
  }

  denseProbe(
    query: NGramProfile,
    cutoff: number,
  ): {
    queryGrams: number
    denseGrams: number
    sparseWork: number
    hybridWork: number
    touched: number
    choiceCount: number
  } {
    const postings = this.requireSparseIndex()
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    const keys = this.queryKeys
    const offsets = postings.offsets
    const counts = postings.counts
    let sparseWork = 0
    let hybridWork = 0
    let denseGrams = 0
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      if (ordinal === undefined) continue
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      const length = upto - from
      let notOne = 0
      if (counts !== null) {
        for (let at = from; at < upto; at++) if (counts[at] !== 1) notOne++
      }
      const inverted = this.choiceCount - length + notOne
      sparseWork += length
      if (length >= cutoff * this.choiceCount && inverted < length) {
        denseGrams++
        hybridWork += inverted
      } else {
        hybridWork += length
      }
    }
    this.diceAccumulate()
    const touched = this.counters.candidatesTouched
    this.reset()
    sparseWork += touched
    hybridWork += denseGrams > 0 ? this.choiceCount : touched
    return {
      queryGrams: keys.length,
      denseGrams,
      sparseWork,
      hybridWork,
      touched,
      choiceCount: this.choiceCount,
    }
  }

  diceBest(query: NGramProfile, threshold: number | null): Scored | undefined {
    this.beginQuery(query)
    if (query.gramCount === 0) return this.gramlessBest(query, threshold)
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    this.diceAccumulate()
    const found = this.diceBestOf(query, threshold)
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
    const found =
      limit === null
        ? this.select(this.diceScorer(query), threshold, limit)
        : this.fillZeroes(this.diceTop(query, threshold, limit), threshold, limit)
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
    // Prefix filtering skips lists a candidate cannot qualify through, and its
    // bound is stated over lists that name who *has* a gram. A dense list names
    // who does not, so it inverts the meaning of every step — and it is also the
    // cheapest list there is, one addition plus its exceptions, so there was
    // never anything to gain by skipping one. Full accumulation instead, which
    // is exact and is what the dense representation is for.
    if (this.usesDenseList(query)) return this.diceSearch(query, threshold, limit)
    const plan = this.prefixScan(query, threshold)
    const found = this.verifyTop(query, plan, threshold, limit)
    this.reset()
    return found
  }

  private usesDenseList(query: NGramProfile): boolean {
    const postings = this.requirePostings()
    const dense = postings.dense
    if (dense === null) return false
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    const keys = this.queryKeys
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      if (ordinal !== undefined && dense[ordinal] === 1) return true
    }
    return false
  }

  /** {@link dicePrefixSearch} with a limit of one. */
  dicePrefixBest(query: NGramProfile, threshold: number | null): Scored | undefined {
    const found = this.dicePrefixSearch(query, threshold, 1)
    return found.length === 0 ? undefined : found[0]
  }

  cosineBest(query: NGramProfile, threshold: number | null): Scored | undefined {
    this.requireWideAccumulator()
    this.beginQuery(query)
    if (query.gramCount === 0) return this.gramlessBest(query, threshold)
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    this.cosineAccumulate()
    const found = this.cosineBestOf(query, threshold)
    this.reset()
    return found
  }

  cosineSearch(
    query: NGramProfile,
    threshold: number | null,
    limit: number | null,
  ): Scored[] {
    this.requireWideAccumulator()
    this.beginQuery(query)
    if (validLimit(limit) === 0) return []
    if (query.gramCount === 0) return this.gramlessSearch(query, threshold, limit)
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    this.cosineAccumulate()
    const found =
      limit === null
        ? this.select(this.cosineScorer(query), threshold, limit)
        : this.fillZeroes(this.cosineTop(query, threshold, limit), threshold, limit)
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
    // `fill` beats the walk once the walk is the whole corpus, and a dense list
    // makes it exactly that.
    if (this.scannedAll) accumulator.fill(0)
    else
      for (let index = 0; index < touched.length; index++) accumulator[touched[index]] = 0
    touched.length = 0
    this.scannedAll = false
    this.base = 0
  }

  /**
   * Both dense diagnostics read a slice length as a document frequency, which is
   * exactly what a dense list is not. They answer "what would inverting buy",
   * so they only make sense before anything has been inverted — and since dense
   * is now the default, asking on the wrong index is one flag away.
   */
  private requireSparseIndex(): Postings {
    const postings = this.requirePostings()
    if (postings.dense !== null) {
      throw new TypeError('the dense diagnostics need an index built with no dense lists')
    }
    return postings
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
    // Dice's overlap cannot exceed the query's own gram count, so this is the
    // whole of the narrow accumulator's exactness condition. Unreachable for any
    // real text — it is 2.1 billion grams — and stated rather than assumed,
    // because the failure mode is a wrong score rather than a thrown error.
    if (this.narrowAccumulator && query.gramCount > 0x7fff_ffff) {
      throw new RangeError('query is too large for a narrow accumulator')
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
    counters.scannedAllCandidates = false
    counters.modifiedCandidates = 0
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
    const dense = postings.dense
    this.base = 0
    if (dense !== null && this.reachesDenseList(dense)) {
      this.scannedAll = true
      this.counters.scannedAllCandidates = true
    }
    // A dense list already put every candidate into the scan, so `touched` is
    // never read again and nothing has to be recorded at all. Where it is read,
    // no dense list was reached, every contribution below is strictly positive,
    // and `accumulator[id] === 0` is a first-touch test again — which is why the
    // generation marks are gone from both kernels. The ladder priced them at 26%
    // of this loop: a random read and a random write per posting entry, into a
    // fourth array, for a set that is either unread or already implied.
    const tracking = !this.scannedAll
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      if (ordinal === undefined) continue
      const queryCount = queryCounts[index]
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      entries += upto - from
      if (dense !== null && dense[ordinal] === 1) {
        // Every candidate holds this gram once unless the slice says otherwise,
        // so the whole corpus takes `min(queryCount, 1)` in one addition and the
        // loop only walks the exceptions. A dense list can only be reached with
        // `tracking` already false, so no marks here under any query.
        this.base += queryCount < 1 ? queryCount : 1
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
      // Split once per posting list rather than branching per entry: where
      // every frequency is 1 the whole counts array is absent, and the shared
      // minimum collapses to a constant.
      if (!tracking) {
        if (postingCounts === null) {
          const capped = queryCount < 1 ? queryCount : 1
          for (let at = from; at < upto; at++) accumulator[ids[at]] += capped
          continue
        }
        for (let at = from; at < upto; at++) {
          const count = postingCounts[at]
          accumulator[ids[at]] += queryCount < count ? queryCount : count
        }
        continue
      }
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
    this.counters.modifiedCandidates = tracking ? touched.length : null
    this.counters.distinctQueryGrams = keys.length
    this.counters.postingEntriesTouched = entries
    this.counters.candidatesTouched = this.scannedAll ? this.choiceCount : touched.length
  }

  /**
   * Does this query reach a dense list? If it does, every candidate is in play —
   * a default frequency applies to choices no posting entry names — so selection
   * runs over the whole corpus instead of over `touched`.
   *
   * `touched` is deliberately *not* filled in with every id, which was the first
   * shape of this and was wrong twice over: the sparse loops then pushed ids that
   * were already there and the result carried duplicates, and a million pushes
   * cost more than the branch the selection loops pay instead.
   */
  private reachesDenseList(dense: Uint8Array): boolean {
    const postings = this.requirePostings()
    const keys = this.queryKeys
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      if (ordinal !== undefined && dense[ordinal] === 1) return true
    }
    return false
  }

  /**
   * Cosine's dot product is bounded by `queryGrams × choiceGrams`, not by the
   * query alone, so the narrow accumulator is not safe for it.
   *
   * On the entry points rather than in `cosineAccumulate`, because a gramless
   * query is answered before accumulation is reached: guarding the loop alone
   * let `cosineSearch('')` succeed on an index that refused every other query,
   * which is a contract that holds for all inputs or for none.
   */
  private requireWideAccumulator(): void {
    if (this.narrowAccumulator) {
      throw new TypeError('a narrow accumulator holds Dice only')
    }
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
    const dense = postings.dense
    this.base = 0
    if (dense !== null && this.reachesDenseList(dense)) {
      this.scannedAll = true
      this.counters.scannedAllCandidates = true
    }
    // See `diceAccumulate`: `touched` is unread under a dense scan, and a
    // product of two positive frequencies cannot land on zero, so the sparse
    // path can test the accumulator itself.
    const tracking = !this.scannedAll
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      if (ordinal === undefined) continue
      const queryCount = queryCounts[index]
      const from = offsets[ordinal]
      const upto = offsets[ordinal + 1]
      entries += upto - from
      if (dense !== null && dense[ordinal] === 1) {
        // The dot product's default term is `queryCount × 1`, and an exception
        // replaces it: an absent choice gives back the whole term, a repeated
        // gram adds the extra `count − 1` copies.
        this.base += queryCount
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
    this.counters.modifiedCandidates = tracking ? touched.length : null
    this.counters.distinctQueryGrams = keys.length
    this.counters.postingEntriesTouched = entries
    this.counters.candidatesTouched = this.scannedAll ? this.choiceCount : touched.length
  }

  /**
   * The same arithmetic over locals rather than over `this`, built once per
   * query and handed to selection.
   *
   * Four property loads per candidate is nothing while selection runs once per
   * *touched* candidate. A dense list made it run once per candidate in the
   * corpus, and then it measured 0.0447 ms against 0.0259 ms for the same loop
   * with the score inlined — 1.7x of a stage that is 40% of the query. Closing
   * over the fields recovers nearly all of that and keeps one scoring loop
   * rather than one per metric.
   *
   * **Built after accumulation, never before**: `base` is a value here, not a
   * field read, so a scorer made too early would carry a zero base and score
   * every candidate as though no dense list had contributed.
   */
  private diceScorer(query: NGramProfile): (id: number) => number {
    const base = this.base
    const accumulator = this.accumulator
    const gramCount = this.gramCount
    const queryGrams = query.gramCount
    return (id) => {
      const grams = gramCount[id]
      if (grams === 0) return 0
      return (2 * (base + accumulator[id])) / (queryGrams + grams)
    }
  }

  /**
   * {@link diceScorer} for Cosine, built after accumulation for the same reason.
   *
   * One square root of the product, then a clamp — the arithmetic
   * `profileSimilarity` uses and for its reason: `Math.sqrt(3) * Math.sqrt(3)` is
   * 3.0000000000000004, which would leave a profile scored against itself just
   * short of 1. A zero norm would divide to the infinity that clamp turns into a
   * perfect score, which is exactly what a gramless choice used to get.
   */
  private cosineScorer(query: NGramProfile): (id: number) => number {
    const base = this.base
    const accumulator = this.accumulator
    const squaredNorm = this.squaredNorm
    const queryNorm = query.squaredNorm
    return (id) => {
      const norm = squaredNorm[id]
      if (norm === 0) return 0
      const similarity = (base + accumulator[id]) / Math.sqrt(queryNorm * norm)
      return similarity < 1 ? similarity : 1
    }
  }

  /**
   * Where accumulation's time goes, one operation at a time.
   *
   * Selection turned out to cost a call boundary rather than arithmetic, so the
   * same question is worth putting to the loop that now owns three quarters of a
   * query. Each rung adds roughly one operation to the one below it — the
   * `Map.get` per distinct gram, the offsets, the id load, the accumulator
   * read-modify-write, the count load, the generation marks, the `touched` push.
   *
   * Read the steps as a **decomposition, not an isolation**. A rung changes what
   * the loop accumulates as well as what it reads, and the dense test goes
   * through a closure here where the real loop has it inline, so a step is the
   * ceiling on what removing that operation could buy rather than its exact
   * price. That is what they are used for: resolving ordinals once is capped by
   * rung 1's step, per-list implicit counts by rung 5's, and dropping the marks
   * by rung 6's.
   *
   * The ordering matters more than it should — a variant profiled after another
   * measured 1.8x its own time, reproducibly, so `control:` re-runs an earlier
   * rung last and any gap between the two is the ladder's own noise floor.
   *
   * Dice only — Cosine differs by one multiply where this differs by whole
   * operations.
   */
  profileAccumulation(
    query: NGramProfile,
    runs: number,
  ): { name: string; ms: number; check: number }[] {
    this.beginQuery(query)
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    const postings = this.requirePostings()
    const accumulator = this.accumulator
    const touched = this.touched
    const keys = this.queryKeys
    const queryCounts = this.queryCounts
    const ids = postings.ids
    const postingCounts = postings.counts
    const offsets = postings.offsets
    const dense = postings.dense
    // The kernels no longer carry these; the ladder still has to be able to
    // price what removing them bought, so it keeps its own copy.
    const marks = new Uint32Array(this.choiceCount)
    let generation = 0
    const everyCandidate = dense !== null && this.reachesDenseList(dense)
    const tracking = !everyCandidate
    // What "resolve the query's ordinals once" would hand the loop, built
    // outside the timing because that is exactly the claim: the resolution moves
    // into preparation, it does not disappear.
    const resolved = new Int32Array(keys.length)
    for (let index = 0; index < keys.length; index++) {
      const ordinal = postings.ordinals.get(keys[index])
      resolved[index] = ordinal === undefined ? -1 : ordinal
    }
    const isDense = (ordinal: number): boolean => dense !== null && dense[ordinal] === 1
    const variants: { name: string; body: () => number }[] = [
      {
        name: 'query gram loop',
        body: () => {
          let sum = 0
          for (let index = 0; index < keys.length; index++) sum += queryCounts[index]
          return sum
        },
      },
      {
        name: '+ ordinal Map.get',
        body: () => {
          let sum = 0
          for (let index = 0; index < keys.length; index++) {
            const ordinal = postings.ordinals.get(keys[index])
            if (ordinal === undefined) continue
            sum += ordinal
          }
          return sum
        },
      },
      {
        name: '+ offsets',
        body: () => {
          let sum = 0
          for (let index = 0; index < keys.length; index++) {
            const ordinal = postings.ordinals.get(keys[index])
            if (ordinal === undefined) continue
            sum += offsets[ordinal + 1] - offsets[ordinal]
          }
          return sum
        },
      },
      {
        name: '+ posting scan, ids only',
        body: () => {
          let sum = 0
          for (let index = 0; index < keys.length; index++) {
            const ordinal = postings.ordinals.get(keys[index])
            if (ordinal === undefined) continue
            const upto = offsets[ordinal + 1]
            for (let at = offsets[ordinal]; at < upto; at++) sum += ids[at]
          }
          return sum
        },
      },
      {
        name: '+ accumulator update',
        body: () => {
          let base = 0
          for (let index = 0; index < keys.length; index++) {
            const ordinal = postings.ordinals.get(keys[index])
            if (ordinal === undefined) continue
            const queryCount = queryCounts[index]
            const upto = offsets[ordinal + 1]
            if (isDense(ordinal)) {
              base += queryCount < 1 ? queryCount : 1
              for (let at = offsets[ordinal]; at < upto; at++) accumulator[ids[at]] -= 1
              continue
            }
            const capped = queryCount < 1 ? queryCount : 1
            for (let at = offsets[ordinal]; at < upto; at++)
              accumulator[ids[at]] += capped
          }
          return base
        },
      },
      {
        name: '+ count load and min',
        body: () => {
          let base = 0
          for (let index = 0; index < keys.length; index++) {
            const ordinal = postings.ordinals.get(keys[index])
            if (ordinal === undefined) continue
            const queryCount = queryCounts[index]
            const from = offsets[ordinal]
            const upto = offsets[ordinal + 1]
            if (isDense(ordinal)) {
              base += queryCount < 1 ? queryCount : 1
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
            if (postingCounts === null) {
              const capped = queryCount < 1 ? queryCount : 1
              for (let at = from; at < upto; at++) accumulator[ids[at]] += capped
              continue
            }
            for (let at = from; at < upto; at++) {
              const count = postingCounts[at]
              accumulator[ids[at]] += queryCount < count ? queryCount : count
            }
          }
          return base
        },
      },
      {
        name: '+ generation marks',
        body: () => {
          generation++
          let modified = 0
          for (let index = 0; index < keys.length; index++) {
            const ordinal = postings.ordinals.get(keys[index])
            if (ordinal === undefined) continue
            const queryCount = queryCounts[index]
            const from = offsets[ordinal]
            const upto = offsets[ordinal + 1]
            const denseList = isDense(ordinal)
            if (postingCounts === null) {
              const step = denseList ? -1 : queryCount < 1 ? queryCount : 1
              for (let at = from; at < upto; at++) {
                const id = ids[at]
                if (marks[id] !== generation) {
                  marks[id] = generation
                  modified++
                }
                accumulator[id] += step
              }
              continue
            }
            const offset = denseList ? -1 : 0
            for (let at = from; at < upto; at++) {
              const id = ids[at]
              if (marks[id] !== generation) {
                marks[id] = generation
                modified++
              }
              const count = postingCounts[at]
              accumulator[id] += (queryCount < count ? queryCount : count) + offset
            }
          }
          return modified
        },
      },
      {
        name: '+ touched push',
        body: () => {
          generation++
          touched.length = 0
          let modified = 0
          for (let index = 0; index < keys.length; index++) {
            const ordinal = postings.ordinals.get(keys[index])
            if (ordinal === undefined) continue
            const queryCount = queryCounts[index]
            const from = offsets[ordinal]
            const upto = offsets[ordinal + 1]
            const denseList = isDense(ordinal)
            if (postingCounts === null) {
              const step = denseList ? -1 : queryCount < 1 ? queryCount : 1
              for (let at = from; at < upto; at++) {
                const id = ids[at]
                if (marks[id] !== generation) {
                  marks[id] = generation
                  modified++
                  if (tracking) touched.push(id)
                }
                accumulator[id] += step
              }
              continue
            }
            const offset = denseList ? -1 : 0
            for (let at = from; at < upto; at++) {
              const id = ids[at]
              if (marks[id] !== generation) {
                marks[id] = generation
                modified++
                if (tracking) touched.push(id)
              }
              const count = postingCounts[at]
              accumulator[id] += (queryCount < count ? queryCount : count) + offset
            }
          }
          touched.length = 0
          return modified
        },
      },
      {
        name: 'diceAccumulate',
        body: () => {
          this.diceAccumulate()
          const entries = this.counters.postingEntriesTouched
          touched.length = 0
          this.scannedAll = false
          this.base = 0
          return entries
        },
      },
      {
        name: 'ordinals resolved once',
        body: () => {
          generation++
          touched.length = 0
          let modified = 0
          for (let index = 0; index < resolved.length; index++) {
            const ordinal = resolved[index]
            if (ordinal < 0) continue
            const queryCount = queryCounts[index]
            const from = offsets[ordinal]
            const upto = offsets[ordinal + 1]
            const denseList = isDense(ordinal)
            if (postingCounts === null) {
              const step = denseList ? -1 : queryCount < 1 ? queryCount : 1
              for (let at = from; at < upto; at++) {
                const id = ids[at]
                if (marks[id] !== generation) {
                  marks[id] = generation
                  modified++
                  if (tracking) touched.push(id)
                }
                accumulator[id] += step
              }
              continue
            }
            const offset = denseList ? -1 : 0
            for (let at = from; at < upto; at++) {
              const id = ids[at]
              if (marks[id] !== generation) {
                marks[id] = generation
                modified++
                if (tracking) touched.push(id)
              }
              const count = postingCounts[at]
              accumulator[id] += (queryCount < count ? queryCount : count) + offset
            }
          }
          touched.length = 0
          return modified
        },
      },
      {
        // A verbatim copy of the rung above but for the Map lookup, run last so
        // that anything the ordering does to a variant shows up as a difference
        // between two identical loops rather than as a finding about ordinals.
        name: 'control: Map.get again, last',
        body: () => {
          generation++
          touched.length = 0
          let modified = 0
          for (let index = 0; index < keys.length; index++) {
            const ordinal = postings.ordinals.get(keys[index])
            if (ordinal === undefined) continue
            const queryCount = queryCounts[index]
            const from = offsets[ordinal]
            const upto = offsets[ordinal + 1]
            const denseList = isDense(ordinal)
            if (postingCounts === null) {
              const step = denseList ? -1 : queryCount < 1 ? queryCount : 1
              for (let at = from; at < upto; at++) {
                const id = ids[at]
                if (marks[id] !== generation) {
                  marks[id] = generation
                  modified++
                  if (tracking) touched.push(id)
                }
                accumulator[id] += step
              }
              continue
            }
            const offset = denseList ? -1 : 0
            for (let at = from; at < upto; at++) {
              const id = ids[at]
              if (marks[id] !== generation) {
                marks[id] = generation
                modified++
                if (tracking) touched.push(id)
              }
              const count = postingCounts[at]
              accumulator[id] += (queryCount < count ? queryCount : count) + offset
            }
          }
          touched.length = 0
          return modified
        },
      },
      {
        name: 'reset (what the query pays after)',
        body: () => {
          if (everyCandidate) accumulator.fill(0)
          return everyCandidate ? this.choiceCount : 0
        },
      },
    ]
    const results: { name: string; ms: number; check: number }[] = []
    try {
      for (const variant of variants) {
        let last = 0
        for (let run = 0; run < runs; run++) last = variant.body()
        let best = Number.POSITIVE_INFINITY
        for (let run = 0; run < runs; run++) {
          // Between rungs, never inside one: a rung that inherits the previous
          // rung's numbers would be timing a different accumulator.
          accumulator.fill(0)
          const started = process.hrtime.bigint()
          last = variant.body()
          const elapsed = Number(process.hrtime.bigint() - started) / 1e6
          if (elapsed < best) best = elapsed
        }
        results.push({ name: variant.name, ms: best, check: last })
      }
    } finally {
      // `reset()` is not enough here and silently was not: it walks `touched`,
      // which these variants empty themselves, so a sparse query would have left
      // every accumulated value behind for the next real query to read.
      accumulator.fill(0)
      touched.length = 0
      this.scannedAll = false
      this.base = 0
    }
    return results
  }

  /**
   * Where one query's time actually goes, stage by stage.
   *
   * Each stage is timed as a prefix of the whole call and the differences are
   * read off, because timing a stage in isolation would measure it on state the
   * previous stage never built. The minimum of many runs rather than the median:
   * these are tens of microseconds on a machine that spikes, and the fastest run
   * is the one least contaminated by something else.
   */
  profilePhases(
    text: string,
    threshold: number | null,
    limit: number | null,
    runs: number,
  ): { name: string; ms: number }[] {
    const gramSize = this.gramSize
    const profile = buildProfile(text, gramSize)
    const stages: { name: string; body: () => unknown }[] = [
      { name: 'buildProfile', body: () => buildProfile(text, gramSize) },
      {
        name: '+ flatten',
        body: () => {
          const built = buildProfile(text, gramSize)
          flattenQueryInto(built, this.radix, this.queryKeys, this.queryCounts)
          return this.queryKeys.length
        },
      },
      {
        name: '+ accumulate',
        body: () => {
          const built = buildProfile(text, gramSize)
          this.beginQuery(built)
          flattenQueryInto(built, this.radix, this.queryKeys, this.queryCounts)
          this.diceAccumulate()
          const reached = this.counters.candidatesTouched
          this.reset()
          return reached
        },
      },
      {
        name: '+ select (whole query)',
        body: () => this.diceSearch(buildProfile(text, gramSize), threshold, limit),
      },
      {
        name: 'accumulate alone, prepared query',
        body: () => {
          this.beginQuery(profile)
          flattenQueryInto(profile, this.radix, this.queryKeys, this.queryCounts)
          this.diceAccumulate()
          const reached = this.counters.candidatesTouched
          this.reset()
          return reached
        },
      },
    ]
    const sink: { value: unknown } = { value: undefined }
    const results: { name: string; ms: number }[] = []
    for (const stage of stages) {
      for (let run = 0; run < runs; run++) sink.value = stage.body()
      let best = Number.POSITIVE_INFINITY
      for (let run = 0; run < runs; run++) {
        const started = process.hrtime.bigint()
        sink.value = stage.body()
        const elapsed = Number(process.hrtime.bigint() - started) / 1e6
        if (elapsed < best) best = elapsed
      }
      results.push({ name: stage.name, ms: best })
    }
    return results
  }

  /**
   * Where the time goes once accumulation stops being the cost.
   *
   * Dense postings cut the traffic of the query made of common grams by 30x and
   * moved its latency by 3%, which says the budget that binds it is this loop.
   * Accumulation runs once here and every variant then scans the same state, so
   * the differences are the loop's own and nothing else's.
   *
   * The variants climb from a floor: an empty pass, then the reads, then the
   * arithmetic, then the two ways of not doing the arithmetic.
   */
  profileSelection(
    query: NGramProfile,
    threshold: number,
    runs: number,
  ): { name: string; ms: number; qualified: number }[] {
    this.beginQuery(query)
    flattenQueryInto(query, this.radix, this.queryKeys, this.queryCounts)
    this.diceAccumulate()
    const accumulator = this.accumulator
    const gramCount = this.gramCount
    const touched = this.touched
    const everyCandidate = this.scannedAll
    const length = everyCandidate ? this.choiceCount : touched.length
    const base = this.base
    const queryGrams = query.gramCount
    // The two gram counts alone cap Dice at `2·min / (q + g)`, so the candidates
    // that can reach the threshold at all sit in one contiguous band of lengths.
    // Outside it no accumulator read is needed and no division is either.
    const lowest = Math.ceil((threshold * queryGrams) / (2 - threshold))
    const highest = Math.floor((queryGrams * (2 - threshold)) / threshold)
    // Reaching through `this` on purpose: the point of this variant is to
    // measure the property loads the hoisted one below does not make.
    const scoreOf = (id: number): number => {
      const grams = this.gramCount[id]
      if (grams === 0) return 0
      return (2 * (this.base + this.accumulator[id])) / (query.gramCount + grams)
    }
    const hoisted = (id: number): number => {
      const grams = gramCount[id]
      if (grams === 0) return 0
      return (2 * (base + accumulator[id])) / (queryGrams + grams)
    }
    const variants: { name: string; body: () => number }[] = [
      {
        name: 'loop only',
        body: () => {
          let sum = 0
          for (let index = 0; index < length; index++) {
            sum += everyCandidate ? index : touched[index]
          }
          return sum
        },
      },
      {
        name: '+ accumulator read',
        body: () => {
          let sum = 0
          for (let index = 0; index < length; index++) {
            sum += accumulator[everyCandidate ? index : touched[index]]
          }
          return sum
        },
      },
      {
        name: '+ gramCount read',
        body: () => {
          let sum = 0
          for (let index = 0; index < length; index++) {
            const id = everyCandidate ? index : touched[index]
            sum += accumulator[id] + gramCount[id]
          }
          return sum
        },
      },
      {
        name: '+ divide, inline',
        body: () => {
          let qualified = 0
          for (let index = 0; index < length; index++) {
            const id = everyCandidate ? index : touched[index]
            const grams = gramCount[id]
            if (grams === 0) continue
            const score = (2 * (base + accumulator[id])) / (queryGrams + grams)
            if (score >= threshold) qualified++
          }
          return qualified
        },
      },
      {
        name: '+ divide, through the callback',
        body: () => {
          let qualified = 0
          for (let index = 0; index < length; index++) {
            const id = everyCandidate ? index : touched[index]
            if (scoreOf(id) >= threshold) qualified++
          }
          return qualified
        },
      },
      {
        name: 'callback over hoisted locals',
        body: () => {
          const scorer = (id: number): number => {
            const grams = gramCount[id]
            if (grams === 0) return 0
            return (2 * (base + accumulator[id])) / (queryGrams + grams)
          }
          let qualified = 0
          for (let index = 0; index < length; index++) {
            const id = everyCandidate ? index : touched[index]
            if (scorer(id) >= threshold) qualified++
          }
          return qualified
        },
      },
      {
        name: 'callback + top-5 insertion',
        body: () => {
          const top: Scored[] = []
          let qualified = 0
          for (let index = 0; index < length; index++) {
            const id = everyCandidate ? index : touched[index]
            const score = scoreOf(id)
            if (score < threshold) continue
            qualified++
            let at = top.length
            if (at === 5) {
              if (!outranks(score, id, top[4])) continue
              at = 4
            }
            while (at > 0 && outranks(score, id, top[at - 1])) {
              top[at] = top[at - 1]
              at--
            }
            top[at] = { id, score }
          }
          return qualified
        },
      },
      {
        name: 'hoisted callback + top-5 insertion',
        body: () => {
          const top: Scored[] = []
          let qualified = 0
          for (let index = 0; index < length; index++) {
            const id = everyCandidate ? index : touched[index]
            const score = hoisted(id)
            if (score < threshold) continue
            qualified++
            let at = top.length
            if (at === 5) {
              if (!outranks(score, id, top[4])) continue
              at = 4
            }
            while (at > 0 && outranks(score, id, top[at - 1])) {
              top[at] = top[at - 1]
              at--
            }
            top[at] = { id, score }
          }
          return qualified
        },
      },
      {
        name: 'inline + top-5 insertion',
        body: () => {
          const top: Scored[] = []
          let qualified = 0
          for (let index = 0; index < length; index++) {
            const id = everyCandidate ? index : touched[index]
            const grams = gramCount[id]
            if (grams === 0) continue
            const score = (2 * (base + accumulator[id])) / (queryGrams + grams)
            if (score < threshold) continue
            qualified++
            let at = top.length
            if (at === 5) {
              if (!outranks(score, id, top[4])) continue
              at = 4
            }
            while (at > 0 && outranks(score, id, top[at - 1])) {
              top[at] = top[at - 1]
              at--
            }
            top[at] = { id, score }
          }
          return qualified
        },
      },
      {
        name: 'length band, then divide',
        body: () => {
          let qualified = 0
          for (let index = 0; index < length; index++) {
            const id = everyCandidate ? index : touched[index]
            const grams = gramCount[id]
            if (grams < lowest || grams > highest) continue
            const score = (2 * (base + accumulator[id])) / (queryGrams + grams)
            if (score >= threshold) qualified++
          }
          return qualified
        },
      },
      {
        name: 'band + integer test, then divide',
        body: () => {
          let qualified = 0
          for (let index = 0; index < length; index++) {
            const id = everyCandidate ? index : touched[index]
            const grams = gramCount[id]
            if (grams < lowest || grams > highest) continue
            // `shared ≥ threshold·(q + g)/2` is the same test as the score
            // against the threshold, with the division moved to the survivors.
            const shared = base + accumulator[id]
            if (2 * shared < threshold * (queryGrams + grams)) continue
            qualified++
          }
          return qualified
        },
      },
    ]
    const results: { name: string; ms: number; qualified: number }[] = []
    for (const variant of variants) {
      let last = 0
      for (let run = 0; run < runs; run++) last = variant.body()
      const samples: number[] = []
      for (let run = 0; run < runs; run++) {
        const started = process.hrtime.bigint()
        last = variant.body()
        samples.push(Number(process.hrtime.bigint() - started) / 1e6)
      }
      samples.sort((left, right) => left - right)
      results.push({ name: variant.name, ms: samples[0], qualified: last })
    }
    this.reset()
    return results
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
    return this.fillZeroes(found, threshold, limit)
  }

  /**
   * The candidates no posting list named, at the score they all share. Shared by
   * the generic path and the specialised kernels rather than copied into both:
   * it runs once per query, not once per candidate, so it is the one part of
   * selection an abstraction costs nothing.
   */
  private fillZeroes(
    found: Scored[],
    threshold: number | null,
    limit: number | null,
  ): Scored[] {
    if (!this.zeroesQualify(threshold)) return found
    // Every candidate was already scored and offered, so there is nothing left
    // to fill in — walking for `accumulator[id] === 0` here would re-add
    // candidates that the dense base had already put in the result.
    if (this.scannedAll) return found
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

  /**
   * Top-k with Dice's arithmetic in the loop rather than behind a callback.
   *
   * The callback was a good trade while selection ran once per *touched*
   * candidate. A dense list makes it run once per candidate in the corpus, and
   * measured against the same loop inlined it cost 0.0447 ms to 0.0259 ms — 1.7x
   * of a stage that is 40% of the query. Closing the scorer over locals
   * recovered 1.10–1.18x of that; the rest is the call boundary itself, which
   * only duplication removes.
   *
   * Two literal kernels rather than one parameterised loop, for the reason
   * `bestDistance` and `bestSimilarity` are two loops in `search/`: this is the
   * innermost frame, and the metric is known before it starts.
   */
  private diceTop(
    query: NGramProfile,
    threshold: number | null,
    limit: number,
  ): Scored[] {
    const touched = this.touched
    const accumulator = this.accumulator
    const gramCount = this.gramCount
    const base = this.base
    const queryGrams = query.gramCount
    const everyCandidate = this.scannedAll
    const length = everyCandidate ? this.choiceCount : touched.length
    const top: Scored[] = []
    let qualified = 0
    for (let index = 0; index < length; index++) {
      const id = everyCandidate ? index : touched[index]
      const grams = gramCount[id]
      if (grams === 0) continue
      const score = (2 * (base + accumulator[id])) / (queryGrams + grams)
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

  /** {@link diceTop} for Cosine. */
  private cosineTop(
    query: NGramProfile,
    threshold: number | null,
    limit: number,
  ): Scored[] {
    const touched = this.touched
    const accumulator = this.accumulator
    const squaredNorm = this.squaredNorm
    const base = this.base
    const queryNorm = query.squaredNorm
    const everyCandidate = this.scannedAll
    const length = everyCandidate ? this.choiceCount : touched.length
    const top: Scored[] = []
    let qualified = 0
    for (let index = 0; index < length; index++) {
      const id = everyCandidate ? index : touched[index]
      const norm = squaredNorm[id]
      if (norm === 0) continue
      const raw = (base + accumulator[id]) / Math.sqrt(queryNorm * norm)
      const score = raw < 1 ? raw : 1
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

  /** {@link diceTop}'s single-winner kernel: no insertion, no result array. */
  private diceBestOf(query: NGramProfile, threshold: number | null): Scored | undefined {
    const touched = this.touched
    const accumulator = this.accumulator
    const gramCount = this.gramCount
    const base = this.base
    const queryGrams = query.gramCount
    const everyCandidate = this.scannedAll
    const length = everyCandidate ? this.choiceCount : touched.length
    let bestId = -1
    let bestScore = 0
    let qualified = 0
    for (let index = 0; index < length; index++) {
      const id = everyCandidate ? index : touched[index]
      const grams = gramCount[id]
      if (grams === 0) continue
      const score = (2 * (base + accumulator[id])) / (queryGrams + grams)
      if (threshold !== null && score < threshold) continue
      qualified++
      if (bestId === -1 || score > bestScore || (score === bestScore && id < bestId)) {
        bestId = id
        bestScore = score
      }
    }
    return this.bestOrZero(bestId, bestScore, qualified, threshold)
  }

  /** {@link diceBestOf} for Cosine. */
  private cosineBestOf(
    query: NGramProfile,
    threshold: number | null,
  ): Scored | undefined {
    const touched = this.touched
    const accumulator = this.accumulator
    const squaredNorm = this.squaredNorm
    const base = this.base
    const queryNorm = query.squaredNorm
    const everyCandidate = this.scannedAll
    const length = everyCandidate ? this.choiceCount : touched.length
    let bestId = -1
    let bestScore = 0
    let qualified = 0
    for (let index = 0; index < length; index++) {
      const id = everyCandidate ? index : touched[index]
      const norm = squaredNorm[id]
      if (norm === 0) continue
      const raw = (base + accumulator[id]) / Math.sqrt(queryNorm * norm)
      const score = raw < 1 ? raw : 1
      if (threshold !== null && score < threshold) continue
      qualified++
      if (bestId === -1 || score > bestScore || (score === bestScore && id < bestId)) {
        bestId = id
        bestScore = score
      }
    }
    return this.bestOrZero(bestId, bestScore, qualified, threshold)
  }

  /**
   * A tie goes to the lower id, and the test for it is load-bearing rather than
   * decoration. Dropping it looked safe — the dense scan counts upward, and each
   * posting list is sorted — but `touched` is filled across *several* lists, so
   * a gram matching id 9 before another matches id 2 leaves it out of order.
   * Parity caught it on a two-choice corpus.
   */
  private bestOrZero(
    bestId: number,
    bestScore: number,
    qualified: number,
    threshold: number | null,
  ): Scored | undefined {
    this.counters.candidatesQualified = qualified
    if (bestId !== -1) return { id: bestId, score: bestScore }
    // Nothing qualified, and nothing touched could have: a touched candidate
    // shares a gram, so it scores above zero. So every choice scores 0, and
    // `bestSimilarity` takes the first item rather than answering `undefined`.
    if (this.zeroesQualify(threshold) && this.choiceCount > 0) {
      this.counters.zeroFillCandidates = 1
      return { id: 0, score: 0 }
    }
    return undefined
  }

  private selectAll(scoreOf: (id: number) => number, threshold: number | null): Scored[] {
    const touched = this.touched
    const everyCandidate = this.scannedAll
    const length = everyCandidate ? this.choiceCount : touched.length
    const found: Scored[] = []
    for (let index = 0; index < length; index++) {
      const id = everyCandidate ? index : touched[index]
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
    const everyCandidate = this.scannedAll
    const length = everyCandidate ? this.choiceCount : touched.length
    const top: Scored[] = []
    let qualified = 0
    for (let index = 0; index < length; index++) {
      const id = everyCandidate ? index : touched[index]
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
