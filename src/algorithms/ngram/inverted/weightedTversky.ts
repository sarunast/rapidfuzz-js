import type {
  ChoiceIndex,
  ChoiceIndexBuilder,
  SelectedChoices,
} from '#core/scoring/choiceIndex.js'
import { convSequence, isUnmatchableElement } from '#core/sequence.js'
import type { Sequence } from '#core/types.js'

import { weightedTverskyScore } from '../weightedTverskyScore.js'
import { NGramIndexBuilder, type SealedIndex } from './builder.js'
import { elementKey } from './keys.js'
import {
  gramlessResult,
  outranks,
  QueryState,
  rankSelected,
  reachesDenseList,
  roomFor,
  zeroesQualify,
} from './query.js'
import { accumulateWeightedShares, NO_SHARE, WeightedShares } from './weightedOverlap.js'

const MAX_UINT32 = 0xffff_ffff

export function assertWeightGroupsAddressable(entries: number): void {
  if (entries > MAX_UINT32) {
    throw new RangeError('an index cannot exceed 4294967295 weight group entries')
  }
}

/**
 * What each choice carries per weight group, choice-major and ascending: the
 * occurrences the group holds, including the ones no query can match.
 *
 * Enough to price what a candidate alone carries exactly, which one weighted
 * total per candidate cannot: a penalty has to be a sum of real unmatched
 * occurrences, group by group, never the residue of two rounded masses. The
 * ignored group is left out, since it prices nothing.
 */
interface ChoiceGroups {
  readonly offsets: Uint32Array
  readonly groups: Uint32Array
  readonly counts: Uint32Array
}

class WeightedTverskyIndex implements ChoiceIndex {
  private readonly state = new QueryState()
  private readonly shares: WeightedShares
  private readonly elementCounts = new Map<unknown, number>()
  private readonly groupTotals = new Map<number, number>()
  private readonly groupIndexOf = new Map<number, number>()
  private readonly queryGroups: number[] = []
  private readonly queryTotals: number[] = []
  private readonly queryWeights: number[] = []
  private readonly keys: (string | number)[] = []
  private readonly keyCounts: number[] = []
  private readonly keyGroups: number[] = []
  private readonly pendingKeys: (string | number)[] = []
  private readonly pendingCounts: number[] = []
  private readonly pendingGroups: number[] = []
  private groupBase: Float64Array = new Float64Array(0)
  private queryLength = 0
  private zeroMass = false

  constructor(
    private readonly sealed: SealedIndex<null>,
    private readonly alpha: number,
    private readonly beta: number,
    private readonly groupWeights: Float64Array,
    private readonly groupOf: ReadonlyMap<unknown, number>,
    private readonly defaultGroup: number,
    private readonly choiceGroups: ChoiceGroups,
  ) {
    this.shares = new WeightedShares(sealed.choiceCount)
  }

  private groupFor(element: unknown): number {
    const group = this.groupOf.get(element)
    return group === undefined ? this.defaultGroup : group
  }

  /**
   * Counts the query's elements, totals them per weight group, and spells one
   * posting key per distinct element — ordered by group, which is what lets the
   * traversal collect each candidate's shares ascending.
   *
   * The rule is the exhaustive weighted profile's, and has to stay so: the same
   * first-occurrence counts, the same per-group totals including unmatchable
   * elements, and the ignored group left out of the fold.
   */
  private prepareQuery(elements: ArrayLike<unknown>): void {
    const counts = this.elementCounts
    const totals = this.groupTotals
    if (counts.size !== 0) counts.clear()
    if (totals.size !== 0) totals.clear()
    if (this.groupIndexOf.size !== 0) this.groupIndexOf.clear()
    this.queryLength = elements.length
    for (let index = 0; index < elements.length; index++) {
      const element = elements[index]
      const group = this.groupFor(element)
      if (group !== 0) totals.set(group, (totals.get(group) ?? 0) + 1)
      if (isUnmatchableElement(element)) continue
      counts.set(element, (counts.get(element) ?? 0) + 1)
    }
    this.zeroMass = totals.size === 0
    const groups = this.queryGroups
    const groupTotals = this.queryTotals
    const weights = this.queryWeights
    groups.length = 0
    groupTotals.length = 0
    weights.length = 0
    if (this.zeroMass) {
      // One pseudo-group over every element: an all-ignored query is answered by
      // proving multiset equality from plain counts, not by weighted mass.
      groups.push(0)
      groupTotals.push(this.queryLength)
      weights.push(0)
    } else {
      const ascending = [...totals].sort((left, right) => left[0] - right[0])
      for (const [group, total] of ascending) {
        this.groupIndexOf.set(group, groups.length)
        groups.push(group)
        groupTotals.push(total)
        weights.push(this.groupWeights[group])
      }
    }
    if (this.groupBase.length < groups.length) {
      this.groupBase = new Float64Array(groups.length)
    } else {
      this.groupBase.fill(0, 0, groups.length)
    }
    this.spellKeys()
  }

  private spellKeys(): void {
    const pendingKeys = this.pendingKeys
    const pendingCounts = this.pendingCounts
    const pendingGroups = this.pendingGroups
    pendingKeys.length = 0
    pendingCounts.length = 0
    pendingGroups.length = 0
    const radix = this.sealed.radix
    const ordinals = this.sealed.elementOrdinals
    for (const [element, count] of this.elementCounts) {
      const group = this.zeroMass ? 0 : this.groupIndexOf.get(this.groupFor(element))
      // An ignored element in a query that has mass elsewhere: it can share
      // nothing and cost nothing, so it never reaches a posting.
      if (group === undefined) continue
      const key = elementKey(element, radix, ordinals)
      if (key === null) continue
      pendingKeys.push(key)
      pendingCounts.push(count)
      pendingGroups.push(group)
    }
    const order = pendingGroups.map((_, index) => index)
    order.sort((left, right) => pendingGroups[left] - pendingGroups[right])
    const keys = this.keys
    const keyCounts = this.keyCounts
    const keyGroups = this.keyGroups
    keys.length = 0
    keyCounts.length = 0
    keyGroups.length = 0
    for (const at of order) {
      keys.push(pendingKeys[at])
      keyCounts.push(pendingCounts[at])
      keyGroups.push(pendingGroups[at])
    }
  }

  private accumulate(): void {
    const shares = this.shares
    const postings = this.sealed.postings
    const dense = postings.dense
    if (dense !== null && reachesDenseList(postings, dense, this.keys)) {
      shares.scannedAll = true
    }
    accumulateWeightedShares(
      postings,
      this.keys,
      this.keyCounts,
      this.keyGroups,
      this.groupBase,
      shares,
    )
  }

  /**
   * Folds the ascending merge of the query's weight groups and the candidate's,
   * exactly as the exhaustive comparison does: each component collects one
   * addition per group, in one canonical order, so the two agree to the bit.
   */
  private score(id: number): number {
    if (this.zeroMass) return this.zeroMassScore(id)
    const groups = this.choiceGroups
    let cursor = groups.offsets[id]
    const upto = groups.offsets[id + 1]
    // No positive group at all: a candidate with no weighted mass shares none,
    // and the zero-mass rule has already refused it against a query that has.
    if (cursor === upto) return 0
    const shares = this.shares
    const queryGroups = this.queryGroups
    let entry = shares.head[id]
    let shared = 0
    let firstOnly = 0
    let secondOnly = 0
    for (let at = 0; at < queryGroups.length; at++) {
      const group = queryGroups[at]
      while (cursor < upto && groups.groups[cursor] < group) {
        secondOnly += this.groupWeights[groups.groups[cursor]] * groups.counts[cursor]
        cursor++
      }
      let count = this.groupBase[at]
      while (entry !== NO_SHARE && shares.entryGroup[entry - 1] === at) {
        count += shares.entryShare[entry - 1]
        entry = shares.entryNext[entry - 1]
      }
      const weight = this.queryWeights[at]
      shared += weight * count
      firstOnly += weight * (this.queryTotals[at] - count)
      if (cursor < upto && groups.groups[cursor] === group) {
        secondOnly += weight * (groups.counts[cursor] - count)
        cursor++
      }
    }
    while (cursor < upto) {
      secondOnly += this.groupWeights[groups.groups[cursor]] * groups.counts[cursor]
      cursor++
    }
    return weightedTverskyScore(shared, firstOnly, secondOnly, this.alpha, this.beta)
  }

  /**
   * An all-ignored query shares no mass with anything, so the only score above
   * zero is the exact-equality one, and the postings already prove it: every
   * query element matched once and the candidate holds nothing else. `NaN`
   * cannot satisfy it — it counts toward a length and reaches no posting — which
   * is what the exhaustive rule says too.
   */
  private zeroMassScore(id: number): number {
    const shares = this.shares
    let count = this.groupBase[0]
    let entry = shares.head[id]
    while (entry !== NO_SHARE) {
      count += shares.entryShare[entry - 1]
      entry = shares.entryNext[entry - 1]
    }
    return count === this.queryLength && this.sealed.gramCount[id] === this.queryLength
      ? 1
      : 0
  }

  select(
    query: Sequence,
    threshold: number | null,
    limit: number | null,
  ): SelectedChoices {
    if (limit === null) return rankSelected(this.collect(query, threshold, false))
    const sealed = this.sealed
    const state = this.state
    const elements = convSequence(query)
    if (elements.length < sealed.gramSize) {
      return gramlessResult(sealed, state, elements, threshold, limit, false)
    }
    this.prepareQuery(elements)
    this.accumulate()
    const room = roomFor(limit, sealed.choiceCount)
    state.reserve(room)
    const length = this.fillMissing(this.top(threshold, room), threshold, room)
    this.shares.reset()
    return { ids: state.ids, scores: state.scores, length }
  }

  scan(query: Sequence, threshold: number | null): SelectedChoices {
    return this.collect(query, threshold, true)
  }

  private collect(
    query: Sequence,
    threshold: number | null,
    ascending: boolean,
  ): SelectedChoices {
    const sealed = this.sealed
    const state = this.state
    const elements = convSequence(query)
    if (elements.length < sealed.gramSize) {
      return gramlessResult(sealed, state, elements, threshold, null, ascending)
    }
    this.prepareQuery(elements)
    this.accumulate()
    const shares = this.shares
    const everyChoice = shares.scannedAll || zeroesQualify(threshold)
    const touched = shares.touched
    if (!everyChoice && ascending) touched.sort((left, right) => left - right)
    const source = everyChoice ? null : touched
    const total = source === null ? sealed.choiceCount : source.length
    state.reserve(total)
    const ids = state.ids
    const scores = state.scores
    let length = 0
    for (let index = 0; index < total; index++) {
      const id = source === null ? index : source[index]
      const score = this.score(id)
      if (threshold !== null && score < threshold) continue
      ids[length] = id
      scores[length] = score
      length++
    }
    shares.reset()
    return { ids, scores, length }
  }

  private top(threshold: number | null, room: number): number {
    if (room === 0) return 0
    const sealed = this.sealed
    const shares = this.shares
    const touched = shares.touched
    const everyChoice = shares.scannedAll
    const total = everyChoice ? sealed.choiceCount : touched.length
    let length = 0
    for (let index = 0; index < total; index++) {
      const id = everyChoice ? index : touched[index]
      const score = this.score(id)
      if (threshold !== null && score < threshold) continue
      length = this.insert(score, id, room, length)
    }
    return length
  }

  /**
   * The choices the query never reached, which score `0` — inserted rather than
   * appended, because a weighted candidate the query *did* reach can score `0`
   * too: a dense list credits every member and corrects the absences, and a huge
   * `alpha` can round a real share away. Appending would then put a zero of id 6
   * ahead of a zero of id 0.
   */
  private fillMissing(length: number, threshold: number | null, room: number): number {
    if (!zeroesQualify(threshold)) return length
    if (this.shares.scannedAll) return length
    const head = this.shares.head
    const ids = this.state.ids
    const scores = this.state.scores
    let filled = length
    for (let id = 0; id < head.length; id++) {
      if (head[id] !== NO_SHARE) continue
      // Ids only grow from here, so once a zero cannot displace the last entry,
      // no later one can either.
      if (filled === room && !outranks(0, id, scores[room - 1], ids[room - 1])) break
      filled = this.insert(0, id, room, filled)
    }
    return filled
  }

  private insert(score: number, id: number, room: number, length: number): number {
    const ids = this.state.ids
    const scores = this.state.scores
    let at = length
    let filled = length
    if (at === room) {
      if (!outranks(score, id, scores[room - 1], ids[room - 1])) return length
      at = room - 1
    } else {
      filled++
    }
    while (at > 0 && outranks(score, id, scores[at - 1], ids[at - 1])) {
      ids[at] = ids[at - 1]
      scores[at] = scores[at - 1]
      at--
    }
    ids[at] = id
    scores[at] = score
    return filled
  }
}

class WeightedTverskyIndexBuilder implements ChoiceIndexBuilder {
  private readonly inner: NGramIndexBuilder<null>
  private readonly offsets: number[] = [0]
  private readonly groups: number[] = []
  private readonly counts: number[] = []
  private readonly scratchGroups: number[] = []
  private readonly scratchCounts: number[] = []

  constructor(
    alpha: number,
    beta: number,
    groupWeights: Float64Array,
    private readonly groupOf: ReadonlyMap<unknown, number>,
    private readonly defaultGroup: number,
  ) {
    // Weights are only defined over single elements, so the gram size is not a
    // parameter here: `gramSize: 1` is what the configuration already proved.
    this.inner = new NGramIndexBuilder(
      1,
      () => null,
      (sealed) =>
        new WeightedTverskyIndex(
          sealed,
          alpha,
          beta,
          groupWeights,
          this.groupOf,
          this.defaultGroup,
          {
            offsets: Uint32Array.from(this.offsets),
            groups: Uint32Array.from(this.groups),
            counts: Uint32Array.from(this.counts),
          },
        ),
    )
  }

  add(choice: Sequence): void {
    const elements = convSequence(choice)
    this.inner.addElements(elements)
    this.record(elements)
  }

  seal(): ChoiceIndex {
    return this.inner.seal()
  }

  private record(elements: ArrayLike<unknown>): void {
    const groups = this.scratchGroups
    const counts = this.scratchCounts
    groups.length = 0
    counts.length = 0
    for (let index = 0; index < elements.length; index++) {
      const found = this.groupOf.get(elements[index])
      const group = found === undefined ? this.defaultGroup : found
      if (group === 0) continue
      let at = 0
      while (at < groups.length && groups[at] < group) at++
      if (at < groups.length && groups[at] === group) {
        counts[at]++
        continue
      }
      groups.splice(at, 0, group)
      counts.splice(at, 0, 1)
    }
    // Before the pushes rather than after: `Array.prototype.push` has its own
    // limit, and reaching that one first would raise its diagnostic instead of
    // the one that says which bound the corpus passed.
    assertWeightGroupsAddressable(this.groups.length + groups.length)
    for (let at = 0; at < groups.length; at++) {
      this.groups.push(groups[at])
      this.counts.push(counts[at])
    }
    this.offsets.push(this.groups.length)
  }
}

export function createWeightedTverskyIndexBuilder(
  alpha: number,
  beta: number,
  groupWeights: Float64Array,
  groupOf: ReadonlyMap<unknown, number>,
  defaultGroup: number,
): ChoiceIndexBuilder {
  return new WeightedTverskyIndexBuilder(alpha, beta, groupWeights, groupOf, defaultGroup)
}
