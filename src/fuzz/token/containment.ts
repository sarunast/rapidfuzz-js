/**
 * Token-set containment as an {@link OptimumProof}.
 *
 * `tokenSetRatioConverted` answers 100 exactly when one non-empty token set
 * contains the other: `normDistance` reaches 100 only at distance 0, its two
 * differences are disjoint token sets, and disjoint sets join to equal
 * sequences only when both are empty. So the perfect matches are describable
 * structurally, and an index over tokens finds every one of them without
 * scoring a pair.
 *
 * Containment runs both ways, and the two directions want opposite structures:
 *
 * - `query ⊆ choice` — every candidate holds *every* query token, so the
 *   query's rarest token names a complete candidate list. Ids ascend inside a
 *   posting list, so the first verified hit is already the earliest.
 * - `choice ⊆ query` — the choice's whole set is a subset of the query's, which
 *   no single posting list narrows. Enumerating the query's subsets and looking
 *   each up by exact set key finds them without touching a posting list at all.
 */
import type { OptimumProof } from '../../core/scoring/optimumProof.js'
import type { Sequence } from '../../core/types.js'
import { prepareTokenChoice, preparedTokenChoice, uniqueOf } from './tokens.js'

/**
 * Longest query the subset channel will enumerate, as `2 ** SUBSET_CAP` masks.
 *
 * Past it the proof declines rather than truncating: a truncated enumeration
 * misses `choice ⊆ query` matches silently, and the id it then reports is not
 * the earliest — a wrong answer rather than a slow one.
 */
const SUBSET_CAP = 12

/**
 * Canonical key for a set of token keys.
 *
 * Length-prefixed because token keys are arbitrary strings and are *not*
 * self-delimiting: a packed key opens with a fixed prefix but may spell that
 * same prefix again inside its body, so joining on any single separator lets
 * `['ab', 'c']` and `['a', 'bc']` collapse onto one key — and two different
 * token sets would then be treated as equal.
 */
function setKeyOf(sorted: readonly string[]): string {
  let key = ''
  for (let i = 0; i < sorted.length; i++) key += `${sorted[i].length}:${sorted[i]}`
  return key
}

interface ContainmentIndex {
  /** Token key to the ascending ids holding it. */
  readonly postings: Map<string, number[]>
  /** Exact set key to every ascending id with that set. */
  readonly bySet: Map<string, number[]>
  /** Each choice's packed keys, or null where it has none to offer. */
  readonly keysById: readonly (ReadonlySet<string> | null)[]
}

/**
 * A choice's mixed tokens do not disqualify it from the posting side.
 *
 * `query ⊆ choice` asks only whether the choice holds every query token, and a
 * packed-only query's tokens are all packed — whatever else the choice carries
 * is irrelevant. The reverse is not true, so a choice with mixed tokens stays
 * out of `bySet`: it cannot be contained by a query that has none.
 */
function build(prepared: readonly unknown[]): ContainmentIndex {
  const postings = new Map<string, number[]>()
  const bySet = new Map<string, number[]>()
  const keysById: (ReadonlySet<string> | null)[] = new Array(prepared.length)

  for (let id = 0; id < prepared.length; id++) {
    const unique = uniqueOf(preparedTokenChoice(prepared[id]))
    const packed = [...unique.packed.keys()]
    if (packed.length === 0) {
      // No packed tokens: nothing can be contained in it by a packed-only
      // query, and an empty token set scores 0 rather than 100 either way.
      keysById[id] = null
      continue
    }
    keysById[id] = new Set(packed)
    for (const key of packed) {
      let list = postings.get(key)
      if (list === undefined) postings.set(key, (list = []))
      list.push(id)
    }
    if (unique.mixed.size === 0) {
      const setKey = setKeyOf(packed.slice().sort())
      let ids = bySet.get(setKey)
      if (ids === undefined) bySet.set(setKey, (ids = []))
      ids.push(id)
    }
  }

  return { postings, bySet, keysById }
}

/**
 * The query's token keys, or `undefined` where no proof is possible.
 *
 * A mixed key is a collision bucket rather than an identity, so a query holding
 * one cannot be compared by key alone. An empty set never scores 100. And past
 * {@link SUBSET_CAP} the subset channel would have to be truncated, which would
 * make the answer wrong rather than absent.
 */
function queryKeys(query: Sequence): string[] | undefined {
  const unique = uniqueOf(prepareTokenChoice(query))
  if (unique.mixed.size !== 0) return undefined

  const keys = [...unique.packed.keys()]
  if (keys.length === 0 || keys.length > SUBSET_CAP) return undefined
  return keys
}

/** The shortest posting list among the query's tokens, or none if any is absent. */
function rarestPosting(
  index: ContainmentIndex,
  keys: readonly string[],
): number[] | undefined {
  let rarest: number[] | undefined
  for (let i = 0; i < keys.length; i++) {
    const list = index.postings.get(keys[i])
    // A query token no choice holds rules out `query ⊆ choice` entirely.
    if (list === undefined) return undefined
    if (rarest === undefined || list.length < rarest.length) rarest = list
  }
  return rarest
}

function holdsEvery(choiceKeys: ReadonlySet<string>, keys: readonly string[]): boolean {
  for (let i = 0; i < keys.length; i++) {
    if (!choiceKeys.has(keys[i])) return false
  }
  return true
}

/**
 * The `limit` smallest distinct ids offered, kept sorted.
 *
 * Both channels can name the same choice — an equal set is contained each way —
 * so the ids have to be merged rather than concatenated, and the bound keeps a
 * corpus with many matching subsets from collecting far more than the caller
 * asked for.
 */
class SmallestIds {
  private readonly ids: number[] = []
  private readonly seen = new Set<number>()

  constructor(private readonly limit: number) {}

  add(id: number): void {
    if (this.seen.has(id)) return
    if (this.ids.length === this.limit && id > this.ids[this.ids.length - 1]) return

    let at = this.ids.length
    while (at > 0 && this.ids[at - 1] > id) at--
    this.ids.splice(at, 0, id)
    this.seen.add(id)

    if (this.ids.length > this.limit) {
      const dropped = this.ids[this.ids.length - 1]
      this.ids.length = this.limit
      this.seen.delete(dropped)
    }
  }

  get full(): boolean {
    return this.ids.length === this.limit
  }

  get collected(): readonly number[] {
    return this.ids
  }
}

/**
 * Walk every non-empty subset of the query's tokens, offering each matching
 * set's ids. `wanted` bounds how many of one set's ids are worth reading.
 */
function offerSubsets(
  index: ContainmentIndex,
  sorted: readonly string[],
  wanted: number,
  into: (id: number) => void,
): void {
  const total = 1 << sorted.length
  for (let mask = 1; mask < total; mask++) {
    const subset: string[] = []
    for (let bit = 0; bit < sorted.length; bit++) {
      if ((mask & (1 << bit)) !== 0) subset.push(sorted[bit])
    }
    const ids = index.bySet.get(setKeyOf(subset))
    if (ids === undefined) continue
    for (let i = 0; i < ids.length && i < wanted; i++) into(ids[i])
  }
}

/**
 * Build the proof over `prepared`, indexing nothing until a query arrives that
 * could actually be settled.
 *
 * The order matters: a query that is empty, mixed or over the cap must cost its
 * own tokenisation and nothing else. Indexing first and declining afterwards
 * would charge a whole collection walk for an answer of "I cannot help", on top
 * of the scan that then has to happen anyway.
 */
export function tokenContainmentProof(prepared: readonly unknown[]): OptimumProof {
  let index: ContainmentIndex | null = null

  return {
    best(query: Sequence): number | undefined {
      const keys = queryKeys(query)
      if (keys === undefined) return undefined
      const built = (index ??= build(prepared))

      let earliest = -1
      const rarest = rarestPosting(built, keys)
      if (rarest !== undefined) {
        for (let i = 0; i < rarest.length; i++) {
          const id = rarest[i]
          const choiceKeys = built.keysById[id]
          // Ids ascend, so the first choice holding every query token is the
          // earliest one that can.
          if (choiceKeys !== null && holdsEvery(choiceKeys, keys)) {
            earliest = id
            break
          }
        }
      }

      const sorted = keys.slice().sort()
      offerSubsets(built, sorted, 1, (id) => {
        if (earliest === -1 || id < earliest) earliest = id
      })

      return earliest === -1 ? undefined : earliest
    },

    top(query: Sequence, limit: number): readonly number[] | undefined {
      const keys = queryKeys(query)
      if (keys === undefined) return undefined
      const built = (index ??= build(prepared))

      const smallest = new SmallestIds(limit)
      const rarest = rarestPosting(built, keys)
      if (rarest !== undefined) {
        for (let i = 0; i < rarest.length && !smallest.full; i++) {
          const id = rarest[i]
          const choiceKeys = built.keysById[id]
          if (choiceKeys !== null && holdsEvery(choiceKeys, keys)) smallest.add(id)
        }
      }

      const sorted = keys.slice().sort()
      offerSubsets(built, sorted, limit, (id) => {
        smallest.add(id)
      })

      // Short of the limit settles nothing: the rest of the result still has to
      // be scored, and scoring it visits every choice anyway.
      return smallest.full ? smallest.collected : undefined
    },
  }
}
