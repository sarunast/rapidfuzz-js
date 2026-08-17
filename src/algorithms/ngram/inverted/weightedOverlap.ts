import type { Postings } from './builder.js'

/** No share of any query group yet: a head or a link of `0`, so that an
 * untouched candidate reads as zero the way an accumulator does. Every stored
 * reference is therefore one past its index. */
export const NO_SHARE = 0

const RETAINED_SHARE_ENTRIES = 1 << 16

/** The entry a one-based `Uint32` reference can no longer name. */
const MAX_SHARE_ENTRIES = 0xffff_ffff

/**
 * The next capacity for the share entries, doubling up to the last entry a
 * one-based reference can address.
 *
 * The refusal cannot be reached through a real index — `assertAddressable`
 * refuses a corpus with more posting entries than this, and a query prepends at
 * most one entry per posting entry it reads — but the doubling is what has to
 * hold to the boundary for that to stay true, so it is stated rather than
 * implied, and exported so it can be checked without allocating 32 GB.
 */
export function grownShareCapacity(length: number): number {
  if (length >= MAX_SHARE_ENTRIES) {
    throw new RangeError('a query cannot read more than 4294967295 shared postings')
  }
  return length === 0 ? 64 : Math.min(MAX_SHARE_ENTRIES, length * 2)
}

/**
 * Per-candidate shares of each query weight group, as one ascending list per
 * candidate.
 *
 * A list rather than a row per group: exact weighted scoring needs each group's
 * shared count separately — one rounded total per candidate is provably not
 * enough — and a row per group would cost the corpus size times the query's
 * weight variety, which a caller passing a large IDF map would feel. Entries are
 * only what the postings actually held.
 *
 * Groups are traversed **descending** and entries prepended, so each candidate's
 * list comes out ascending, which is the order the score has to fold in.
 *
 * Every reference is unsigned, and sufficient rather than merely large: the
 * traversal prepends at most one entry per stored posting entry, since distinct
 * query elements spell distinct keys and each names one posting, and
 * `assertAddressable` has already refused an index holding more than
 * `0xffff_ffff` of those. Growth holds to the same boundary rather than trusting
 * that argument — see `grownShareCapacity`. `entryShare` is the one signed array,
 * because a dense list stores absences as corrections.
 */
export class WeightedShares {
  readonly head: Uint32Array
  readonly touched: number[] = []
  entryNext: Uint32Array = new Uint32Array(0)
  entryGroup: Uint32Array = new Uint32Array(0)
  entryShare: Float64Array = new Float64Array(0)
  entryCount = 0
  scannedAll = false

  constructor(choiceCount: number) {
    this.head = new Uint32Array(choiceCount)
  }

  prepend(id: number, group: number, share: number): void {
    const at = this.entryCount
    if (at === this.entryNext.length) this.grow()
    const previous = this.head[id]
    if (previous === NO_SHARE && !this.scannedAll) this.touched.push(id)
    this.entryNext[at] = previous
    this.entryGroup[at] = group
    this.entryShare[at] = share
    this.head[id] = at + 1
    this.entryCount = at + 1
  }

  reset(): void {
    const head = this.head
    if (this.scannedAll) head.fill(NO_SHARE)
    else for (const id of this.touched) head[id] = NO_SHARE
    this.touched.length = 0
    this.entryCount = 0
    this.scannedAll = false
    if (this.entryNext.length > RETAINED_SHARE_ENTRIES) {
      this.entryNext = new Uint32Array(0)
      this.entryGroup = new Uint32Array(0)
      this.entryShare = new Float64Array(0)
    }
  }

  private grow(): void {
    const capacity = grownShareCapacity(this.entryNext.length)
    const next = new Uint32Array(capacity)
    const group = new Uint32Array(capacity)
    const share = new Float64Array(capacity)
    next.set(this.entryNext)
    group.set(this.entryGroup)
    share.set(this.entryShare)
    this.entryNext = next
    this.entryGroup = group
    this.entryShare = share
  }
}

/**
 * Accumulates each candidate's `Σ min(query frequency, choice frequency)` per
 * weight group, in integers.
 *
 * Integral on purpose: a group's count is exact whatever order the postings
 * arrive in, which is what lets one canonical ascending fold at scoring time
 * agree with the exhaustive path to the bit — and it keeps a dense list's `base`
 * plus stored corrections exact, the way the unweighted walk relies on. A
 * weighted contribution added posting by posting would reorder the additions
 * instead, and a correction is negative, which no unsigned row could hold.
 *
 * Keys arrive ordered by weight group and are read back to front, so the shares
 * each candidate collects end up ascending by group.
 */
export function accumulateWeightedShares(
  postings: Postings,
  keys: readonly (string | number)[],
  keyCounts: readonly number[],
  keyGroups: readonly number[],
  groupBase: Float64Array,
  shares: WeightedShares,
): void {
  const ids = postings.ids
  const postingCounts = postings.counts
  const offsets = postings.offsets
  const dense = postings.dense
  for (let index = keys.length - 1; index >= 0; index--) {
    const ordinal = postings.ordinals.get(keys[index])
    if (ordinal === undefined) continue
    const group = keyGroups[index]
    const queryCount = keyCounts[index]
    const from = offsets[ordinal]
    const upto = offsets[ordinal + 1]
    if (dense !== null && dense[ordinal] === 1) {
      groupBase[group] += 1
      if (postingCounts === null) {
        for (let at = from; at < upto; at++) shares.prepend(ids[at], group, -1)
        continue
      }
      for (let at = from; at < upto; at++) {
        const count = postingCounts[at]
        shares.prepend(ids[at], group, (queryCount < count ? queryCount : count) - 1)
      }
      continue
    }
    if (postingCounts === null) {
      for (let at = from; at < upto; at++) shares.prepend(ids[at], group, 1)
      continue
    }
    for (let at = from; at < upto; at++) {
      const count = postingCounts[at]
      shares.prepend(ids[at], group, queryCount < count ? queryCount : count)
    }
  }
}
