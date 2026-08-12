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

interface Posting {
  readonly ids: Uint32Array
  readonly counts: Uint32Array
}

interface PostingBuilder {
  readonly ids: number[]
  readonly counts: number[]
}

interface GramlessChoice {
  readonly id: number
  readonly elements: readonly unknown[]
}

interface FlatQuery {
  readonly keys: string[]
  readonly counts: number[]
}

/**
 * Grams are keyed by their elements joined with a separator no component can
 * contain, rather than by `String.fromCodePoint(...gram)`.
 *
 * That encoding collides between grams of the same length, because it accepts
 * lone surrogates: `[0x1f600, 0xd83d, 0xde01]` and `[0xd83d, 0xde00, 0x1f601]`
 * both serialize to the units `D83D DE00 D83D DE01`. Decimal integers cannot
 * collide that way. `-0` stringifies as `0`, which matches the `Map` the profile
 * trie keys by SameValueZero.
 */
function integerKey(element: unknown): string {
  if (typeof element !== 'number' || !Number.isInteger(element)) {
    throw new TypeError('the ngram index prototype accepts integer gram elements only')
  }
  return String(element)
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
 */
function eachGram(
  profile: NGramProfile,
  visit: (key: string, count: number) => void,
): void {
  const last = profile.gramSize - 1
  const nodes: GramNode[] = [profile.root]
  const prefixes: string[] = ['']
  const depths: number[] = [0]
  let top = 1
  while (top > 0) {
    top--
    const node = nodes[top]
    const prefix = prefixes[top]
    const depth = depths[top]
    if (depth === last) {
      const counts = node.counts
      if (counts !== null) {
        for (const [element, count] of counts) visit(prefix + integerKey(element), count)
      }
      continue
    }
    const children = node.children
    if (children === null) continue
    for (const [element, child] of children) {
      nodes[top] = child
      prefixes[top] = `${prefix}${integerKey(element)},`
      depths[top] = depth + 1
      top++
    }
  }
}

function flattenQuery(query: NGramProfile): FlatQuery {
  const keys: string[] = []
  const counts: number[] = []
  eachGram(query, (key, count) => {
    keys.push(key)
    counts.push(count)
  })
  return { keys, counts }
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

/** One choice's frequency for a gram, or 0. The posting list is sorted by id. */
function frequencyOf(posting: Posting, id: number): number {
  const ids = posting.ids
  let low = 0
  let high = ids.length - 1
  while (low <= high) {
    const middle = (low + high) >>> 1
    const found = ids[middle]
    if (found === id) return posting.counts[middle]
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
  readonly suffixKeys: string[]
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
  private builder: Map<string, PostingBuilder> | null = new Map()
  private postings: Map<string, Posting> | null = null
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
  ) {
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
    if (choiceId < 0 || choiceId >= this.choiceCount) {
      throw new RangeError('choice id is outside the index')
    }
    this.gramCount[choiceId] = profile.gramCount
    this.squaredNorm[choiceId] = profile.squaredNorm
    if (profile.gramCount === 0) {
      this.gramless.push({
        id: choiceId,
        elements: copyElements(gramlessElements(profile)),
      })
      return
    }
    eachGram(profile, (key, count) => {
      const posting = builder.get(key)
      if (posting === undefined) {
        builder.set(key, { ids: [choiceId], counts: [count] })
        return
      }
      posting.ids.push(choiceId)
      posting.counts.push(count)
    })
  }

  compact(): void {
    const builder = this.builder
    if (builder === null) throw new TypeError('the index is already compacted')
    const postings = new Map<string, Posting>()
    for (const [key, posting] of builder) {
      const ids = Uint32Array.from(posting.ids)
      // Ascending by construction, because choices arrive in id order — and
      // `frequencyOf` binary-searches these, so it is worth saying out loud
      // rather than leaving as a property someone could quietly break.
      for (let at = 1; at < ids.length; at++) {
        if (ids[at - 1] >= ids[at]) throw new Error('posting list is not sorted by id')
      }
      postings.set(key, { ids, counts: Uint32Array.from(posting.counts) })
    }
    this.postings = postings
    this.builder = null
  }

  /** Distinct grams in the compacted index — the posting map's size. */
  gramVariety(): number {
    return this.requirePostings().size
  }

  diceBest(query: NGramProfile, threshold: number | null): Scored | undefined {
    this.beginQuery(query)
    if (query.gramCount === 0) return this.gramlessBest(query, threshold)
    this.diceAccumulate(flattenQuery(query))
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
    if (query.gramCount === 0) return this.gramlessSearch(query, threshold, limit)
    this.diceAccumulate(flattenQuery(query))
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
    if (threshold === null || threshold <= 0)
      return this.diceSearch(query, threshold, limit)
    this.beginQuery(query)
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
    this.cosineAccumulate(flattenQuery(query))
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
    if (query.gramCount === 0) return this.gramlessSearch(query, threshold, limit)
    this.cosineAccumulate(flattenQuery(query))
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
    const { keys, counts } = flattenQuery(query)
    const lengths: number[] = new Array<number>(keys.length)
    const order: number[] = new Array<number>(keys.length)
    for (let index = 0; index < keys.length; index++) {
      lengths[index] = postings.get(keys[index])?.ids.length ?? 0
      order[index] = index
    }
    order.sort((left, right) => lengths[left] - lengths[right])

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
      const posting = postings.get(keys[at])
      if (posting === undefined) continue
      const ids = posting.ids
      const postingCounts = posting.counts
      entries += ids.length
      for (let scan = 0; scan < ids.length; scan++) {
        const id = ids[scan]
        if (accumulator[id] === 0) touched.push(id)
        const count = postingCounts[scan]
        accumulator[id] += queryCount < count ? queryCount : count
      }
    }
    const suffixKeys: string[] = []
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
      const posting = postings.get(plan.suffixKeys[at])
      if (posting === undefined) continue
      const ids = posting.ids
      const counts = posting.counts
      const queryCount = plan.suffixCounts[at]
      entries += ids.length
      for (let scan = 0; scan < ids.length; scan++) {
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
        const posting = postings.get(suffixKeys[at])
        const queryCount = suffixCounts[at]
        remaining -= queryCount
        if (posting !== undefined) {
          probes++
          const count = frequencyOf(posting, id)
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

  private requirePostings(): Map<string, Posting> {
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
  private diceAccumulate(query: FlatQuery): void {
    const postings = this.requirePostings()
    const accumulator = this.accumulator
    const touched = this.touched
    const keys = query.keys
    const queryCounts = query.counts
    let entries = 0
    for (let index = 0; index < keys.length; index++) {
      const posting = postings.get(keys[index])
      if (posting === undefined) continue
      const ids = posting.ids
      const counts = posting.counts
      const queryCount = queryCounts[index]
      entries += ids.length
      for (let at = 0; at < ids.length; at++) {
        const id = ids[at]
        if (accumulator[id] === 0) touched.push(id)
        const count = counts[at]
        accumulator[id] += queryCount < count ? queryCount : count
      }
    }
    this.counters.distinctQueryGrams = keys.length
    this.counters.postingEntriesTouched = entries
    this.counters.candidatesTouched = touched.length
  }

  private cosineAccumulate(query: FlatQuery): void {
    const postings = this.requirePostings()
    const accumulator = this.accumulator
    const touched = this.touched
    const keys = query.keys
    const queryCounts = query.counts
    let entries = 0
    for (let index = 0; index < keys.length; index++) {
      const posting = postings.get(keys[index])
      if (posting === undefined) continue
      const ids = posting.ids
      const counts = posting.counts
      const queryCount = queryCounts[index]
      entries += ids.length
      for (let at = 0; at < ids.length; at++) {
        const id = ids[at]
        if (accumulator[id] === 0) touched.push(id)
        accumulator[id] += queryCount * counts[at]
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
