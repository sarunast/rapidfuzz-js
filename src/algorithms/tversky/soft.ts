import { maximumTransport, type SoftEdge } from './assignment.js'
import {
  elementScore,
  preparedElementScore,
  type CompiledElementSimilarity,
  type ElementKernels,
} from './elementSimilarity.js'
import { elementTableOf, type ElementTable, type Occurrence } from './occurrences.js'

/**
 * How many distinct fuzzy-comparable leftovers one *side* of a pair may have.
 *
 * A limit on the product of the two sides measures the wrong cost. Edge building
 * does cost `n × m` element scores, but the matching costs a shortest-path search
 * over `n + m` nodes repeated once per augmenting path, and a product bounds the
 * scoring while leaving the search free to grow: at a fixed product of 1024,
 * `32 × 32` searches 66 nodes and `1 × 1024` searches 1027, and the skinny shape
 * needed a thousand of those searches where the square one needed 33. Three
 * orders of magnitude between two pairs the product limit called equal.
 *
 * Sizing the two sides is all this limit does. It does *not* bound the number of
 * searches, because that is not a function of the element counts — see
 * {@link MAX_SOFT_AUGMENTATIONS}, which is the limit that does.
 *
 * A per-side limit of 32 implies the old product of 1024, so it is one number in
 * place of two, and the square shape it tops out at is the cheapest of the shapes
 * that product allowed. Measured end to end at the limit with every edge
 * surviving, which is the adverse case: `32 × 32` costs 0.72 ms a pair, 0.77 ms
 * once every element occurs 32 times over — 1024 occurrences a side — and
 * 1.41 ms once those occurrences are skewed rather than uniform, which is the
 * dearest shape this limit allows. A realistic pair is nowhere near any of them,
 * at 0.55 µs for one typo among three tokens, because a single candidate pairing
 * never reaches the residual network at all. Raising this later is not a breaking
 * change; lowering it is.
 *
 * Each figure is a recorded `bench/regression/baseline.json` median over the
 * loop its case runs — `bench/suites/tverskySoft.bench.ts` scores 10 adverse
 * pairs a call and 10 000 realistic ones — so it is checked in rather than
 * remembered from a run.
 *
 * Counted over leftovers an element scorer can actually see: three hundred
 * distinct objects against three hundred more produce no edges at all, and must
 * not be refused.
 */
const MAX_SOFT_ELEMENTS = 32

/**
 * How many augmenting paths the matching may walk for one pair.
 *
 * {@link MAX_SOFT_ELEMENTS} cannot stand in for this. Successive shortest paths
 * is pseudo-polynomial in the supplies rather than in the graph, so how often an
 * element repeats buys augmenting paths that counting distinct elements does not
 * see — `assignment.test.ts` holds the `2 × 2` matrix that takes one more of them
 * on skewed counts than on unit ones, with the same elements and the same edges.
 * Scaling every count *uniformly* is the case that is free, which is why a
 * benchmark that scales them uniformly settles nothing here.
 *
 * So this is a ceiling rather than a shape. No skewed `32 × 32` probed with
 * random profits and counts drawn up to a million reached 150 augmenting paths;
 * 512 leaves that room to spare and caps a pair at roughly 2.2 M node visits
 * rather than at nothing in particular. It throws rather than returning the best
 * matching so far, because a silent cutoff would make a score depend on how
 * skewed the occurrence counts happened to be.
 */
const MAX_SOFT_AUGMENTATIONS = 512

/** What exact matching reserved, and what it left for the element scorer. */
export interface ExactOverlap {
  readonly leftoverFirst: Uint32Array
  readonly leftoverSecond: Uint32Array
  /** Occurrences matched exactly, as an exact integer count. */
  readonly sharedCount: number
}

/**
 * One solved pairing, carrying the similarity behind its profit so that an
 * explanation can report it without scoring the pair a second time.
 */
export interface SoftPairing extends SoftEdge {
  readonly similarity: number
}

/** The three Tversky components, and the matching that produced them. */
export interface SoftComponents {
  readonly edges: readonly SoftPairing[]
  readonly units: Uint32Array
  readonly shared: number
  readonly firstOnly: number
  readonly secondOnly: number
}

/**
 * A choice prepared for soft matching: the occurrence walk, done once.
 *
 * The element scorer's own preparation is not held here — not because there is
 * nothing to amortize, since a matcher meets the same choice on every query,
 * but because holding it would retain a second, opaque representation of every
 * fuzzy-comparable token in the corpus for the life of the matcher. That trade
 * of memory against time needs its own measurement. The query side, which
 * repeats within a single scan and retains nothing beyond it, is held in
 * `ElementKernels`.
 */
export class SoftTverskyChoice {
  constructor(readonly occurrences: Occurrence[]) {}
}

export function preparedSoftChoice(value: unknown): SoftTverskyChoice {
  if (value instanceof SoftTverskyChoice) return value
  throw new TypeError('invalid prepared soft tversky choice')
}

/** Both sides as distinct elements, and what exact matching did with them. */
export interface SoftTables {
  readonly first: ElementTable
  readonly second: ElementTable
  readonly overlap: ExactOverlap
}

/**
 * The distinct-element view of a pair, and the exact reservation over it.
 *
 * Separate from {@link softComponentsOf} because the weighted path already has
 * its shared mass from `weightedComponents` and the plain path takes it from
 * `overlap.sharedCount`, which this pass has produced anyway. Splitting the two
 * keeps that count from being computed a second way.
 */
export function softTablesOf(
  first: readonly Occurrence[],
  second: readonly Occurrence[],
): SoftTables {
  const firstTable = elementTableOf(first)
  const secondTable = elementTableOf(second)
  return {
    first: firstTable,
    second: secondTable,
    overlap: exactOverlapOf(firstTable, secondTable),
  }
}

function exactOverlapOf(first: ElementTable, second: ElementTable): ExactOverlap {
  const leftoverFirst = new Uint32Array(first.entries.length)
  const leftoverSecond = new Uint32Array(second.entries.length)
  let sharedCount = 0
  for (let at = 0; at < second.entries.length; at++) {
    leftoverSecond[at] = second.entries[at].count
  }
  for (let at = 0; at < first.entries.length; at++) {
    const entry = first.entries[at]
    const other = second.indexOf.get(entry.canonical)
    if (other === undefined) {
      leftoverFirst[at] = entry.count
      continue
    }
    const partner = second.entries[other].count
    const overlap = entry.count < partner ? entry.count : partner
    sharedCount += overlap
    leftoverFirst[at] = entry.count - overlap
    leftoverSecond[other] = partner - overlap
  }
  return { leftoverFirst, leftoverSecond, sharedCount }
}

/** A leftover an element scorer can see, carrying the narrowing with it. */
interface Comparable {
  readonly at: number
  readonly operand: string
  readonly weight: number
}

function comparableLeftovers(table: ElementTable, leftover: Uint32Array): Comparable[] {
  const comparable: Comparable[] = []
  for (let at = 0; at < leftover.length; at++) {
    if (leftover[at] === 0) continue
    const entry = table.entries[at]
    const operand = entry.operand
    if (operand === null) continue
    comparable.push({ at, operand, weight: entry.weight })
  }
  return comparable
}

const EMPTY_COLUMNS: readonly unknown[] = []

function refuseOversized(rows: number, columns: number): void {
  if (rows > MAX_SOFT_ELEMENTS || columns > MAX_SOFT_ELEMENTS) {
    throw new RangeError(
      `elementSimilarity would compare ${rows} unmatched elements against ` +
        `${columns}, past the limit of ${MAX_SOFT_ELEMENTS} distinct ` +
        'fuzzy-comparable leftovers a side for one pair; compare shorter ' +
        'sequences, or reduce them to fewer distinct tokens before scoring',
    )
  }
}

/**
 * The soft components, or `null` where nothing was matched fuzzily and the
 * caller must return its exact score untouched.
 *
 * That `null` is what makes bit-identity with exact Tversky structural rather
 * than arithmetic: the two fold their penalties in different orders — this one
 * per element, the exact engines per weight group — so they agree on the real
 * number and need not agree on its last bit. Short-circuiting rather than
 * recomputing removes the question.
 *
 * Every term folded here is non-negative by construction. `leftover − used` is
 * integer, and `weight − profit` cannot go below zero because `profit` is
 * `min(wa, wb) · s` with `s ≤ 1` exactly, and multiplication is monotone under
 * round-to-nearest. Nothing is derived by subtracting one rounded aggregate
 * from another.
 *
 * `kernels` is the scan's held query side, and `null` where a pair is scored on
 * its own — which is a cost decision rather than a behavioural one, since a
 * prepared kernel and a direct score are the same number.
 */
export function softComponentsOf(
  tables: SoftTables,
  soft: CompiledElementSimilarity,
  exactShared: number,
  kernels: ElementKernels | null,
): SoftComponents | null {
  const first = tables.first
  const second = tables.second
  const overlap = tables.overlap
  const leftFirst = comparableLeftovers(first, overlap.leftoverFirst)
  const leftSecond = comparableLeftovers(second, overlap.leftoverSecond)
  if (leftFirst.length === 0 || leftSecond.length === 0) return null
  refuseOversized(leftFirst.length, leftSecond.length)

  const held =
    kernels !== null && kernels.earned(leftFirst.length * leftSecond.length)
      ? kernels
      : null
  const columns = held === null ? EMPTY_COLUMNS : held.columnsFor(leftSecond)
  const edges: SoftPairing[] = []
  for (const row of leftFirst) {
    const kernel = held === null ? null : held.kernelFor(row.operand)
    for (let at = 0; at < leftSecond.length; at++) {
      const column = leftSecond[at]
      const similarity =
        kernel === null
          ? elementScore(soft, row.operand, column.operand)
          : preparedElementScore(soft, kernel, columns[at])
      if (similarity < soft.threshold) continue
      // A weight several hundred exponents below its partner's can round the
      // shared mass away entirely; a zero-mass edge only widens the tie plateau.
      const profit =
        (row.weight < column.weight ? row.weight : column.weight) * similarity
      if (profit === 0) continue
      edges.push({ first: row.at, second: column.at, profit, similarity })
    }
  }
  if (edges.length === 0) return null

  const units = maximumTransport(
    edges,
    overlap.leftoverFirst,
    overlap.leftoverSecond,
    MAX_SOFT_AUGMENTATIONS,
  )

  const usedFirst = new Uint32Array(first.entries.length)
  const usedSecond = new Uint32Array(second.entries.length)
  let fuzzy = 0
  for (let at = 0; at < edges.length; at++) {
    const edge = edges[at]
    fuzzy += edge.profit * units[at]
    usedFirst[edge.first] += units[at]
    usedSecond[edge.second] += units[at]
  }

  return {
    edges,
    units,
    shared: exactShared + fuzzy,
    firstOnly: residualMass(first, overlap.leftoverFirst, usedFirst, edges, units, true),
    secondOnly: residualMass(
      second,
      overlap.leftoverSecond,
      usedSecond,
      edges,
      units,
      false,
    ),
  }
}

function residualMass(
  table: ElementTable,
  leftover: Uint32Array,
  used: Uint32Array,
  edges: readonly SoftEdge[],
  units: Uint32Array,
  isFirst: boolean,
): number {
  let mass = table.unmatchableMass
  for (let at = 0; at < leftover.length; at++) {
    mass += (leftover[at] - used[at]) * table.entries[at].weight
  }
  for (let at = 0; at < edges.length; at++) {
    const edge = edges[at]
    const weight = table.entries[isFirst ? edge.first : edge.second].weight
    mass += units[at] * (weight - edge.profit)
  }
  return mass
}
