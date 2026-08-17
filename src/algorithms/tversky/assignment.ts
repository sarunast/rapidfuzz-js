/**
 * One candidate pairing between a leftover element of each side, priced per
 * unit. `profit` is strictly positive — a zero-mass pairing is dropped before it
 * reaches the solver, where it would only widen the tie plateau.
 */
export interface SoftEdge {
  readonly first: number
  readonly second: number
  readonly profit: number
}

const NO_ARC = -1

/**
 * The units each edge carries in a maximum-profit one-to-one matching of
 * `supply` against `demand`.
 *
 * This is a transportation problem rather than an assignment problem: every
 * occurrence of one element carries the same weight and the same similarity to
 * any other element, so the occurrence-level problem collapses to distinct
 * elements with integer supplies. The transportation polytope is integral, so
 * the collapsed optimum *equals* the occurrence-level optimum.
 *
 * Successive maximum-profit augmenting paths, stopping at the first
 * non-positive augmentation: min-cost flow's cost is convex in the flow value,
 * so profit per augmentation is non-increasing and that rule is exact.
 *
 * Maximum *to within the rounding of a path cost*, and no further. A path costs
 * a sum of profits, so two flows whose exact totals differ by less than that
 * sum's own rounding are indistinguishable to the search, and either may come
 * back. `assignment.test.ts` pins how close that is, against an oracle that
 * compares exact rational totals rather than folded doubles. Closing the rest
 * needs exact summation of the path costs, which is a different solver rather
 * than a tie-break rule — an epsilon in the comparison only moves which
 * near-ties are decided wrongly.
 *
 * `budget` caps the augmenting paths, because their number is not bounded by the
 * element counts: successive shortest paths is pseudo-polynomial in the supplies,
 * so skewed occurrence counts buy augmentations that a limit on distinct elements
 * cannot see. The caller owns the number; this owns the counting.
 *
 * The total is deliberately not returned. The caller re-folds `profit × units`
 * itself, which keeps the solver's residual arithmetic out of the reported
 * number and keeps scoring and explanation agreeing to the bit.
 *
 * @throws {RangeError} If the matching takes more than `budget` augmenting paths.
 */
export function maximumTransport(
  edges: readonly SoftEdge[],
  supply: Uint32Array,
  demand: Uint32Array,
  budget: number,
): Uint32Array {
  const units = new Uint32Array(edges.length)
  if (edges.length === 0) return units
  // One candidate pairing has one answer: fill it, which is optimal because a
  // profit is strictly positive. Worth its own line because it is the shape a
  // realistic pair leaves — one typo among tokens that otherwise match exactly —
  // where the general path would build a residual network of twelve arrays and
  // run Dijkstra to move a single unit across a single arc.
  if (edges.length === 1) {
    const only = edges[0]
    const room = supply[only.first]
    const wanted = demand[only.second]
    units[0] = room < wanted ? room : wanted
    return units
  }
  if (supply.length === 1 || demand.length === 1) {
    return starTransport(edges, supply, demand, units)
  }
  let augmentations = 0

  const first = supply.length
  const second = demand.length
  const nodes = first + second + 2
  const sink = nodes - 1
  const arcs = 2 * (first + second + edges.length)

  // Forward arc `2k`, its residual partner `2k + 1`. Costs are negated profits,
  // so each augmentation is a shortest-path search.
  const from = new Int32Array(arcs)
  const to = new Int32Array(arcs)
  const capacity = new Uint32Array(arcs)
  const cost = new Float64Array(arcs)

  const connect = (
    slot: number,
    tail: number,
    target: number,
    room: number,
    gain: number,
  ) => {
    from[2 * slot] = tail
    to[2 * slot] = target
    capacity[2 * slot] = room
    cost[2 * slot] = -gain
    from[2 * slot + 1] = target
    to[2 * slot + 1] = tail
    cost[2 * slot + 1] = gain
  }

  for (let at = 0; at < first; at++) connect(at, 0, at + 1, supply[at], 0)
  for (let at = 0; at < second; at++) {
    connect(first + at, first + 1 + at, sink, demand[at], 0)
  }
  for (let at = 0; at < edges.length; at++) {
    const edge = edges[at]
    const room = Math.min(supply[edge.first], demand[edge.second])
    connect(
      first + second + at,
      edge.first + 1,
      first + 1 + edge.second,
      room,
      edge.profit,
    )
  }
  const middle = 2 * (first + second)

  // Adjacency in ascending arc order, which is what makes the chosen path a
  // function of the input alone.
  const firstArc = new Int32Array(nodes).fill(NO_ARC)
  const nextArc = new Int32Array(arcs)
  for (let arc = arcs - 1; arc >= 0; arc--) {
    nextArc[arc] = firstArc[from[arc]]
    firstArc[from[arc]] = arc
  }

  const potential = initialPotentials(edges, supply, demand, nodes, first, sink)
  const distance = new Float64Array(nodes)
  const ordering = new Float64Array(nodes)
  const settled = new Uint8Array(nodes)
  const reached = new Int32Array(nodes)

  for (;;) {
    // Dijkstra over reduced costs, but the label is the *raw* path cost and the
    // reduced one is derived from it per node rather than accumulated into it.
    // Folding `potential[node] - potential[next]` into the running label rounds
    // two paths whose raw costs differ by one ulp onto the same number, and the
    // strict `<` then keeps whichever arrived first — which is how a lower
    // profit wins. Every path to a node meets the same `potential[next]`, so
    // deriving the ordering key from the raw label cannot reorder them.
    //
    // A settled node's predecessor was settled strictly earlier, so `reached` is
    // a tree whatever the arithmetic does — the property this solver rests on.
    // Bellman-Ford cannot promise it: an arc and its own residual partner
    // evaluate to `fl(fl(x - p) + p) > x` often enough, which is a profitable
    // cycle to the relaxation and a predecessor cycle to the reconstruction.
    distance.fill(Number.POSITIVE_INFINITY)
    ordering.fill(Number.POSITIVE_INFINITY)
    distance[0] = 0
    ordering[0] = 0
    settled.fill(0)
    reached.fill(NO_ARC)
    for (;;) {
      let node = NO_ARC
      let nearest = Number.POSITIVE_INFINITY
      for (let at = 0; at < nodes; at++) {
        if (settled[at] === 0 && ordering[at] < nearest) {
          nearest = ordering[at]
          node = at
        }
      }
      if (node === NO_ARC) break
      settled[node] = 1
      const reach = distance[node]
      for (let arc = firstArc[node]; arc !== NO_ARC; arc = nextArc[arc]) {
        if (capacity[arc] === 0) continue
        const next = to[arc]
        if (settled[next] === 1) continue
        const candidate = reach + cost[arc]
        if (candidate < distance[next]) {
          distance[next] = candidate
          ordering[next] = candidate - potential[next]
          reached[next] = arc
        }
      }
    }
    if (reached[sink] === NO_ARC) return units
    // The label that chose the path decides whether to walk it. Re-adding the
    // arc costs backwards is the same sum in the opposite order, and the two
    // disagree on the sign of a last-bit gain often enough to stop on a path
    // Dijkstra had just found profitable.
    if (!(distance[sink] < 0)) return units
    if (++augmentations > budget) refuseBudget(first, second, budget)

    let bottleneck = Number.MAX_SAFE_INTEGER
    for (let node = sink; node !== 0; node = from[reached[node]]) {
      const arc = reached[node]
      if (capacity[arc] < bottleneck) bottleneck = capacity[arc]
    }

    for (let node = sink; node !== 0; node = from[reached[node]]) {
      const arc = reached[node]
      capacity[arc] -= bottleneck
      capacity[arc ^ 1] += bottleneck
      if (arc >= middle) {
        const edge = (arc - middle) >>> 1
        if ((arc & 1) === 0) units[edge] += bottleneck
        else units[edge] -= bottleneck
      }
    }
    // The label already is the raw shortest distance, so the usual
    // `potential += reduced distance` is this assignment with a rounding fewer.
    for (let at = 0; at < nodes; at++) {
      if (distance[at] !== Number.POSITIVE_INFINITY) potential[at] = distance[at]
    }
  }
}

/**
 * The optimum where one side holds a single distinct element, taken in
 * descending profit.
 *
 * A single row — or a single column — is a star rather than a transportation
 * problem: every unit leaves the same node, so there is nothing an augmenting
 * path could re-route, and taking the richest pairing first is exactly optimal.
 * No residual network, no potentials, and no budget: this walks the edges once.
 *
 * Ties go to the edge that arrived first, which keeps the answer a function of
 * the input alone. That the tie-breaking *matches* what the general path settles
 * on matters more than it looks: the caller prices each side's leftovers per
 * element, so two matchings carrying the same shared mass can still score
 * differently, and this has to be a fast path rather than a second opinion.
 */
function starTransport(
  edges: readonly SoftEdge[],
  supply: Uint32Array,
  demand: Uint32Array,
  units: Uint32Array,
): Uint32Array {
  const order = Array.from(edges, (_unused, at) => at)
  order.sort((left, right) => edges[right].profit - edges[left].profit || left - right)
  const freeSupply = Uint32Array.from(supply)
  const freeDemand = Uint32Array.from(demand)
  for (const at of order) {
    const edge = edges[at]
    const room = freeSupply[edge.first]
    const wanted = freeDemand[edge.second]
    const moved = room < wanted ? room : wanted
    if (moved === 0) continue
    units[at] = moved
    freeSupply[edge.first] -= moved
    freeDemand[edge.second] -= moved
  }
  return units
}

function refuseBudget(rows: number, columns: number, budget: number): never {
  throw new RangeError(
    `elementSimilarity needed more than ${budget} augmenting paths to match ` +
      `${rows} unmatched elements against ${columns}; how often those elements ` +
      'repeat is skewed far enough to cost unbounded work, so compare shorter ' +
      'sequences or even out the repeats before scoring',
  )
}

/**
 * Exact shortest distances through the zero-flow residual, which Dijkstra needs
 * as its starting potentials.
 *
 * No relaxation loop is required: before anything is pushed the network is a
 * four-layer DAG, so one pass in layer order is exact. A node no unit can reach
 * keeps potential `0` and never enters a relaxation, because the reachable set
 * only ever shrinks — an arc into an unreachable node gains capacity only when
 * an augmenting path leaves that node, and paths only leave reachable ones.
 */
function initialPotentials(
  edges: readonly SoftEdge[],
  supply: Uint32Array,
  demand: Uint32Array,
  nodes: number,
  first: number,
  sink: number,
): Float64Array {
  const potential = new Float64Array(nodes).fill(Number.POSITIVE_INFINITY)
  potential[0] = 0
  for (let at = 0; at < supply.length; at++) {
    if (supply[at] > 0) potential[at + 1] = 0
  }
  for (let at = 0; at < edges.length; at++) {
    const edge = edges[at]
    if (supply[edge.first] === 0 || demand[edge.second] === 0) continue
    const target = first + 1 + edge.second
    if (-edge.profit < potential[target]) potential[target] = -edge.profit
  }
  for (let at = 0; at < demand.length; at++) {
    const source = first + 1 + at
    if (demand[at] > 0 && potential[source] < potential[sink]) {
      potential[sink] = potential[source]
    }
  }
  for (let at = 0; at < nodes; at++) {
    if (potential[at] === Number.POSITIVE_INFINITY) potential[at] = 0
  }
  return potential
}
