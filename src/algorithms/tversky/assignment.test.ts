// The transport solver on its own, against an exhaustive oracle: no Tversky
// vocabulary reaches this file, so a failure here is the matching and nothing
// else.
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import { maximumTransport, type SoftEdge } from './assignment.js'

/** More augmenting paths than any fixture here needs, so only the budget test sees it. */
const PLENTY = 512

const bits = new DataView(new ArrayBuffer(8))

/** `steps` representable doubles away from `value`, so a one-ulp gap reads as one. */
function ulp(value: number, steps: number): number {
  bits.setFloat64(0, value)
  bits.setBigUint64(0, bits.getBigUint64(0) + BigInt(steps))
  return bits.getFloat64(0)
}

function foldTotal(edges: readonly SoftEdge[], units: Uint32Array): number {
  let total = 0
  for (let at = 0; at < edges.length; at++) total += edges[at].profit * units[at]
  return total
}

/** Every feasible unit count at every edge. Exponential, so fixtures stay tiny. */
function oracle(
  edges: readonly SoftEdge[],
  supply: Uint32Array,
  demand: Uint32Array,
): number {
  const freeSupply = Uint32Array.from(supply)
  const freeDemand = Uint32Array.from(demand)
  const walk = (at: number): number => {
    if (at === edges.length) return 0
    const edge = edges[at]
    const room = Math.min(freeSupply[edge.first], freeDemand[edge.second])
    let best = 0
    for (let take = 0; take <= room; take++) {
      freeSupply[edge.first] -= take
      freeDemand[edge.second] -= take
      const total = take * edge.profit + walk(at + 1)
      if (total > best) best = total
      freeSupply[edge.first] += take
      freeDemand[edge.second] += take
    }
    return best
  }
  return walk(0)
}

function expectFeasible(
  edges: readonly SoftEdge[],
  units: Uint32Array,
  supply: Uint32Array,
  demand: Uint32Array,
): void {
  const usedSupply = new Uint32Array(supply.length)
  const usedDemand = new Uint32Array(demand.length)
  for (let at = 0; at < edges.length; at++) {
    usedSupply[edges[at].first] += units[at]
    usedDemand[edges[at].second] += units[at]
  }
  for (let at = 0; at < supply.length; at++) {
    expect(usedSupply[at]).toBeLessThanOrEqual(supply[at])
  }
  for (let at = 0; at < demand.length; at++) {
    expect(usedDemand[at]).toBeLessThanOrEqual(demand[at])
  }
}

describe('maximumTransport', () => {
  it('returns no units when there are no edges', () => {
    const units = maximumTransport([], Uint32Array.of(3), Uint32Array.of(2), PLENTY)
    expect(units).toEqual(new Uint32Array(0))
  })

  // The case a `min(n, m) === 1` fast path gets wrong: one distinct element with
  // two occurrences has to reach two different counterparts.
  it('spreads one element across several counterparts', () => {
    const edges: SoftEdge[] = [
      { first: 0, second: 0, profit: 1 },
      { first: 0, second: 1, profit: 0.9 },
    ]
    const units = maximumTransport(edges, Uint32Array.of(2), Uint32Array.of(1, 1), PLENTY)
    expect(units).toEqual(Uint32Array.of(1, 1))
    expect(foldTotal(edges, units)).toBe(1.9)
  })

  it('lets one unit of demand go to the better of two competitors', () => {
    const edges: SoftEdge[] = [
      { first: 0, second: 0, profit: 0.5 },
      { first: 1, second: 0, profit: 0.75 },
    ]
    const units = maximumTransport(edges, Uint32Array.of(1, 1), Uint32Array.of(1), PLENTY)
    expect(units).toEqual(Uint32Array.of(0, 1))
  })

  it('reaches the optimum a greedy first pick would miss', () => {
    // Greedy takes 0↔0 at 1, stranding both remaining nodes; 0↔1 and 1↔0 pay 1.8.
    const edges: SoftEdge[] = [
      { first: 0, second: 0, profit: 1 },
      { first: 0, second: 1, profit: 0.9 },
      { first: 1, second: 0, profit: 0.9 },
    ]
    const units = maximumTransport(
      edges,
      Uint32Array.of(1, 1),
      Uint32Array.of(1, 1),
      PLENTY,
    )
    expect(foldTotal(edges, units)).toBe(1.8)
  })

  it('saturates a shared element before spilling to a weaker one', () => {
    const edges: SoftEdge[] = [
      { first: 0, second: 0, profit: 2 },
      { first: 1, second: 0, profit: 1 },
    ]
    const units = maximumTransport(edges, Uint32Array.of(1, 3), Uint32Array.of(3), PLENTY)
    expect(units).toEqual(Uint32Array.of(1, 2))
  })

  // Two augmentations in, `fl(fl(x - p) + p)` lands one ulp above `x`, so an
  // edge and its own residual partner look like a profitable cycle. A solver
  // whose predecessor pointers are not acyclic by construction walks that cycle
  // forever rather than returning.
  it('does not chase a residual round trip that only rounding made profitable', () => {
    const edges: SoftEdge[] = [
      { first: 0, second: 0, profit: 0.7197832760956315 },
      { first: 0, second: 1, profit: 0.7066570634420347 },
      { first: 1, second: 0, profit: 0.5000436248960742 },
      { first: 1, second: 1, profit: 0.053695831735475756 },
      { first: 2, second: 0, profit: 0.053695831735475756 },
      { first: 2, second: 1, profit: 0.05369583173547575 },
    ]
    const supply = Uint32Array.of(2, 1, 2)
    const demand = Uint32Array.of(2, 2)
    const units = maximumTransport(edges, supply, demand, PLENTY)
    expectFeasible(edges, units, supply, demand)
    expect(foldTotal(edges, units)).toBeCloseTo(oracle(edges, supply, demand), 12)
  })

  // Accumulating the *reduced* cost rounds the one-ulp gap between the two ways
  // out of row 0 onto a single label, and the strict `<` then keeps the cheaper
  // one. The units are the assertion because both flows fold to exactly 4: a
  // total cannot see this, which is why the bug survived an oracle.
  it('keeps a one-ulp better path the potential transform would round away', () => {
    const edges: SoftEdge[] = [
      { first: 0, second: 0, profit: 1 },
      { first: 0, second: 1, profit: 0.9999999999999999 },
      { first: 1, second: 0, profit: 3 },
    ]
    const supply = Uint32Array.of(1, 1)
    const demand = Uint32Array.of(2, 1)
    const units = maximumTransport(edges, supply, demand, PLENTY)
    expectFeasible(edges, units, supply, demand)
    expect(units).toEqual(Uint32Array.of(1, 0, 1))
  })

  // The same shape where one ulp is a whole unit of 2, so it is not an artefact
  // of the profits sitting near 1.
  it('keeps it at a scale where one ulp is 2', () => {
    const edges: SoftEdge[] = [
      { first: 0, second: 0, profit: 1e16 },
      { first: 0, second: 1, profit: 9999999999999998 },
      { first: 1, second: 0, profit: 3e16 },
    ]
    const units = maximumTransport(
      edges,
      Uint32Array.of(1, 1),
      Uint32Array.of(2, 1),
      PLENTY,
    )
    expect(units).toEqual(Uint32Array.of(1, 0, 1))
  })

  // Re-adding the arc costs backwards to decide whether to walk the path is the
  // same sum in the opposite order: here Dijkstra's label reads -5.55e-17 and the
  // reverse fold reads 0, so the walk stops on a path it had just found
  // profitable. Units again, because both flows fold to exactly 2.
  it('walks a path its own label found profitable', () => {
    const edges: SoftEdge[] = [
      { first: 0, second: 1, profit: 0.75 },
      { first: 1, second: 1, profit: 1 },
      { first: 1, second: 0, profit: 0.25000000000000006 },
    ]
    const supply = Uint32Array.of(3, 2)
    const demand = Uint32Array.of(3, 2)
    const units = maximumTransport(edges, supply, demand, PLENTY)
    expectFeasible(edges, units, supply, demand)
    expect(units).toEqual(Uint32Array.of(2, 0, 2))
  })

  // Why the budget cannot be replaced by counting distinct elements. Same
  // elements, same edges, same profits: skewing the counts alone buys another
  // augmenting path, so a limit that only sizes the two sides does not bound the
  // work. Successive shortest paths is pseudo-polynomial in the supplies.
  //
  // Two rows against two columns rather than a star, which would take the greedy
  // path and walk no augmenting paths at all.
  const skewable: SoftEdge[] = [
    { first: 0, second: 0, profit: 0.9 },
    { first: 0, second: 1, profit: 0.8 },
    { first: 1, second: 0, profit: 0.7 },
    { first: 1, second: 1, profit: 0.95 },
  ]

  it('spends more augmenting paths on skewed counts than on unit ones', () => {
    expect(() =>
      maximumTransport(skewable, Uint32Array.of(1, 1), Uint32Array.of(1, 1), 2),
    ).not.toThrow()
    expect(() =>
      maximumTransport(skewable, Uint32Array.of(2, 3), Uint32Array.of(5, 1), 2),
    ).toThrow(RangeError)
  })

  it('names the budget it ran out of', () => {
    expect(() =>
      maximumTransport(skewable, Uint32Array.of(2, 3), Uint32Array.of(5, 1), 2),
    ).toThrow(
      new RangeError(
        'elementSimilarity needed more than 2 augmenting paths to match 2 ' +
          'unmatched elements against 2; how often those elements repeat is ' +
          'skewed far enough to cost unbounded work, so compare shorter ' +
          'sequences or even out the repeats before scoring',
      ),
    )
  })

  // A single distinct element on a side is a star, and takes the greedy path
  // instead of a residual network. Optimality is what the shortcut has to keep,
  // so it answers to the same exact oracle as everything else.
  const star = fc
    .tuple(fc.integer({ min: 1, max: 5 }), fc.boolean())
    .chain(([many, oneRow]) => {
      const rows = oneRow ? 1 : many
      const columns = oneRow ? many : 1
      return fc.record({
        supply: fc.array(fc.integer({ min: 0, max: 4 }), {
          minLength: rows,
          maxLength: rows,
        }),
        demand: fc.array(fc.integer({ min: 0, max: 4 }), {
          minLength: columns,
          maxLength: columns,
        }),
        edges: fc.uniqueArray(
          fc.record({
            first: fc.integer({ min: 0, max: rows - 1 }),
            second: fc.integer({ min: 0, max: columns - 1 }),
            // Few enough values that ties are the norm rather than the exception.
            profit: fc.constantFrom(ulp(1, -1), 1, 0.5, 0.25, 3),
          }),
          { maxLength: 5, selector: (edge) => `${edge.first}:${edge.second}` },
        ),
      })
    })

  it('is exactly optimal on a star, whichever side is the single one', () => {
    fc.assert(
      fc.property(star, ({ supply, demand, edges }) => {
        const supplied = Uint32Array.from(supply)
        const demanded = Uint32Array.from(demand)
        const units = maximumTransport(edges, supplied, demanded, PLENTY)
        expectFeasible(edges, units, supplied, demanded)
        expect(exactTotal(edges, units)).toBe(exactOptimum(edges, supplied, demanded))
      }),
      { numRuns: 3000, seed: 0x5eed },
    )
  })

  it('gives a tie to the pairing that arrived first', () => {
    const edges: SoftEdge[] = [
      { first: 0, second: 0, profit: 0.5 },
      { first: 0, second: 1, profit: 0.5 },
    ]
    const units = maximumTransport(edges, Uint32Array.of(1), Uint32Array.of(1, 1), PLENTY)
    expect(units).toEqual(Uint32Array.of(1, 0))
  })

  // An exact oracle, because the interesting flows fold to the same double. Every
  // profit below is a positive normal double, so this is `significand << exponent`
  // with no cases: the exact value scaled by 2^1074, where sums compare exactly.
  function exactly(profit: number): bigint {
    bits.setFloat64(0, profit)
    const raw = bits.getBigUint64(0)
    const significand = (raw & 0xf_ffff_ffff_ffffn) | 0x10_0000_0000_0000n
    return significand << (((raw >> 52n) & 0x7ffn) - 1n)
  }

  function exactTotal(edges: readonly SoftEdge[], units: Uint32Array): bigint {
    let total = 0n
    for (let at = 0; at < edges.length; at++) {
      total += exactly(edges[at].profit) * BigInt(units[at])
    }
    return total
  }

  function exactOptimum(
    edges: readonly SoftEdge[],
    supply: Uint32Array,
    demand: Uint32Array,
  ): bigint {
    const freeSupply = Uint32Array.from(supply)
    const freeDemand = Uint32Array.from(demand)
    const walk = (at: number): bigint => {
      if (at === edges.length) return 0n
      const edge = edges[at]
      const room = Math.min(freeSupply[edge.first], freeDemand[edge.second])
      const unit = exactly(edge.profit)
      let best = 0n
      for (let take = 0; take <= room; take++) {
        freeSupply[edge.first] -= take
        freeDemand[edge.second] -= take
        const total = unit * BigInt(take) + walk(at + 1)
        if (total > best) best = total
        freeSupply[edge.first] += take
        freeDemand[edge.second] += take
      }
      return best
    }
    return walk(0)
  }

  // The measurement the "to within the rounding of a path cost" contract rests
  // on, run rather than quoted. Profits come from a ladder of doubles one ulp
  // apart because that is the only place a path ordering can invert — the
  // property tests below draw across the whole exponent range and never land
  // there. A ceiling rather than an equality: exactly 0 would be a better solver
  // and must not fail, while the accumulate-the-reduced-cost solver this replaced
  // scores 286 and must.
  it('is exactly optimal on all but a handful of one-ulp problems', () => {
    const ladder = [ulp(1, -2), ulp(1, -1), 1, ulp(1, 1), 3]
    let problems = 0
    let missed = 0
    for (let matrix = 0; matrix < ladder.length ** 4; matrix++) {
      const edges: SoftEdge[] = []
      let profits = matrix
      for (let at = 0; at < 4; at++) {
        edges.push({
          first: at >> 1,
          second: at & 1,
          profit: ladder[profits % ladder.length],
        })
        profits = Math.floor(profits / ladder.length)
      }
      for (let shape = 0; shape < 81; shape++) {
        let counts = shape
        const supply = new Uint32Array(2)
        const demand = new Uint32Array(2)
        for (let at = 0; at < 2; at++) {
          supply[at] = counts % 3
          counts = Math.floor(counts / 3)
        }
        for (let at = 0; at < 2; at++) {
          demand[at] = counts % 3
          counts = Math.floor(counts / 3)
        }
        problems++
        const units = maximumTransport(edges, supply, demand, PLENTY)
        if (exactTotal(edges, units) !== exactOptimum(edges, supply, demand)) missed++
      }
    }
    expect(problems).toBe(50_625)
    expect(missed).toBeLessThanOrEqual(32)
  })

  it('is deterministic across repeated calls', () => {
    const edges: SoftEdge[] = [
      { first: 0, second: 0, profit: 1 },
      { first: 0, second: 1, profit: 1 },
      { first: 1, second: 0, profit: 1 },
      { first: 1, second: 1, profit: 1 },
    ]
    const supply = Uint32Array.of(1, 1)
    const demand = Uint32Array.of(1, 1)
    const once = maximumTransport(edges, supply, demand, PLENTY)
    const twice = maximumTransport(edges, supply, demand, PLENTY)
    expect(once).toEqual(twice)
  })

  // Dyadic profits, so every partial sum is exact and the oracle's fold order
  // cannot disagree with the solver's in the last bit.
  const DYADIC = [0.25, 0.5, 0.75, 1, 2, 4]

  const problem = fc
    .tuple(fc.integer({ min: 1, max: 3 }), fc.integer({ min: 1, max: 3 }))
    .chain(([rows, columns]) =>
      fc.record({
        supply: fc.array(fc.integer({ min: 0, max: 2 }), {
          minLength: rows,
          maxLength: rows,
        }),
        demand: fc.array(fc.integer({ min: 0, max: 2 }), {
          minLength: columns,
          maxLength: columns,
        }),
        edges: fc.uniqueArray(
          fc.record({
            first: fc.integer({ min: 0, max: rows - 1 }),
            second: fc.integer({ min: 0, max: columns - 1 }),
            profit: fc.constantFrom(...DYADIC),
          }),
          {
            maxLength: 9,
            selector: (edge) => `${edge.first}:${edge.second}`,
          },
        ),
      }),
    )

  it('matches an exhaustive oracle exactly on dyadic profits', () => {
    fc.assert(
      fc.property(problem, ({ supply, demand, edges }) => {
        const supplied = Uint32Array.from(supply)
        const demanded = Uint32Array.from(demand)
        const units = maximumTransport(edges, supplied, demanded, PLENTY)
        expectFeasible(edges, units, supplied, demanded)
        expect(foldTotal(edges, units)).toBe(oracle(edges, supplied, demanded))
      }),
      { numRuns: 2000, seed: 0x5eed },
    )
  })

  // Dyadic profits are far friendlier than a real Indel or Jaro similarity: they
  // make every residual round trip cancel exactly, which is precisely the case
  // that hides a predecessor cycle.
  const messy = fc
    .tuple(fc.integer({ min: 1, max: 3 }), fc.integer({ min: 1, max: 3 }))
    .chain(([rows, columns]) =>
      fc.record({
        supply: fc.array(fc.integer({ min: 0, max: 2 }), {
          minLength: rows,
          maxLength: rows,
        }),
        demand: fc.array(fc.integer({ min: 0, max: 2 }), {
          minLength: columns,
          maxLength: columns,
        }),
        edges: fc.uniqueArray(
          fc.record({
            first: fc.integer({ min: 0, max: rows - 1 }),
            second: fc.integer({ min: 0, max: columns - 1 }),
            profit: fc.double({
              min: Number.MIN_VALUE,
              max: 1,
              noNaN: true,
              noDefaultInfinity: true,
            }),
          }),
          {
            maxLength: 9,
            selector: (edge) => `${edge.first}:${edge.second}`,
          },
        ),
      }),
    )

  // Relative rather than absolute, and a handful of ulps rather than 1e-12: the
  // solver is optimal only to within the rounding of a path cost, and an
  // absolute 1e-12 is 4500 ulps at a total of 1, which permits far more than the
  // arithmetic can lose. Worst measured over 200 000 random problems is 2.2e-16
  // of the total, and 0.77 ulps over this generator, so 8 is headroom.
  //
  // It bounds the loss; it does not catch a lost tie. Profits drawn across the
  // whole exponent range almost never land one ulp apart, which is the only
  // place the ordering can invert — the two unit cases above are where that is
  // pinned, and both fold to a total this assertion would wave through.
  const ULPS = 8 * Number.EPSILON

  it('matches an exhaustive oracle on arbitrary profits', () => {
    fc.assert(
      fc.property(messy, ({ supply, demand, edges }) => {
        const supplied = Uint32Array.from(supply)
        const demanded = Uint32Array.from(demand)
        const units = maximumTransport(edges, supplied, demanded, PLENTY)
        expectFeasible(edges, units, supplied, demanded)
        const total = foldTotal(edges, units)
        const exact = oracle(edges, supplied, demanded)
        expect(exact - total).toBeLessThanOrEqual(Math.abs(exact) * ULPS)
        expect(total - exact).toBeLessThanOrEqual(Math.abs(exact) * ULPS)
      }),
      { numRuns: 5000, seed: 0x5eed },
    )
  })

  it('finds the same total whatever order the edges arrive in', () => {
    fc.assert(
      fc.property(problem, ({ supply, demand, edges }) => {
        const supplied = Uint32Array.from(supply)
        const demanded = Uint32Array.from(demand)
        const forward = maximumTransport(edges, supplied, demanded, PLENTY)
        const backward = [...edges].reverse()
        const reversed = maximumTransport(backward, supplied, demanded, PLENTY)
        expect(foldTotal(backward, reversed)).toBe(foldTotal(edges, forward))
      }),
      { numRuns: 1000, seed: 0x5eed },
    )
  })
})
