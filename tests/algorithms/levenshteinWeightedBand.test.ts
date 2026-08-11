// Not a port. The generic weighted dynamic program bands itself from the
// cutoff: an alignment already owes `difference * deletion` for the two lengths
// alone, and every insertion past that has to be paid for with a deletion too,
// so the budget buys a bounded number of steps off the corridor between the two
// ends. Cells outside that are filled with a sentinel and never scored.
//
// A band that is one diagonal too narrow does not fail loudly — it returns a
// distance that is merely too large, on some inputs, under some cutoffs. So the
// test is a full unbanded dynamic program run beside it, over weights and
// cutoffs chosen to reach both kernels: whole numbers take the `Int32Array`
// one, fractions the `Float64Array` one.
import fc from 'fast-check'
import { expect, test } from 'vitest'

import {
  levenshteinDistance,
  levenshteinSimilarity,
} from '../../src/algorithms/levenshtein/metric.js'

/** The whole matrix, no band and no common-affix trimming. */
function reference(
  a: string,
  b: string,
  insertion: number,
  deletion: number,
  substitution: number,
): number {
  const cols = b.length
  let previous = Array.from({ length: cols + 1 }, (_unused, j) => j * insertion)
  for (let i = 1; i <= a.length; i++) {
    const current = [i * deletion]
    for (let j = 1; j <= cols; j++) {
      current.push(
        Math.min(
          previous[j] + deletion,
          current[j - 1] + insertion,
          previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : substitution),
        ),
      )
    }
    previous = current
  }
  return previous[cols]
}

const letters = fc.constantFrom('a', 'b', 'c')
const text = fc.string({ unit: letters, maxLength: 30 })
const whole = fc.integer({ min: 0, max: 9 })
const fractional = fc.constantFrom(0, 0.5, 1, 1.5, 2.25, 3.5, 7)

/**
 * Both orientations of the same pair. Swapping the arguments swaps insertion
 * and deletion, which is also what the kernel's own swap does when the second
 * input is the longer one — so this covers the corridor leaning either way.
 */
function agrees(
  a: string,
  b: string,
  insertion: number,
  deletion: number,
  substitution: number,
  scoreCutoff: number,
): void {
  const expected = reference(a, b, insertion, deletion, substitution)
  const forward = levenshteinDistance(a, b, {
    weights: { insertion, deletion, substitution },
    scoreCutoff,
  })
  const backward = levenshteinDistance(b, a, {
    weights: { insertion: deletion, deletion: insertion, substitution },
    scoreCutoff,
  })
  if (expected <= scoreCutoff) {
    expect(forward).toBe(expected)
    expect(backward).toBe(expected)
  } else {
    expect(forward).toBeGreaterThan(scoreCutoff)
    expect(backward).toBeGreaterThan(scoreCutoff)
  }
}

test('the banded weighted dynamic program matches a full one', () => {
  fc.assert(
    fc.property(
      text,
      text,
      whole,
      whole,
      whole,
      fc.integer({ min: 0, max: 200 }),
      agrees,
    ),
    { numRuns: 10_000 },
  )
})

test('the same, on the fractional-weight kernel', () => {
  fc.assert(
    fc.property(
      text,
      text,
      fractional,
      fractional,
      fractional,
      fc.double({ min: 0, max: 120, noNaN: true }),
      agrees,
    ),
    { numRuns: 10_000 },
  )
})

// Long enough that the band is a small fraction of the matrix, which is where a
// missing diagonal has room to change the answer.
test('the band holds on longer inputs under tight cutoffs', () => {
  fc.assert(
    fc.property(
      fc.string({ unit: letters, minLength: 40, maxLength: 120 }),
      fc.string({ unit: letters, minLength: 40, maxLength: 120 }),
      fc.integer({ min: 1, max: 9 }),
      fc.integer({ min: 1, max: 9 }),
      fc.integer({ min: 1, max: 9 }),
      fc.integer({ min: 0, max: 150 }),
      agrees,
    ),
    { numRuns: 5000 },
  )
})

// A length difference the corridor itself has to carry, rather than the
// excursion: `difference * deletion` is most of the budget here.
test('the band holds when one side is far longer than the other', () => {
  fc.assert(
    fc.property(
      fc.string({ unit: letters, minLength: 100, maxLength: 200 }),
      fc.string({ unit: letters, minLength: 1, maxLength: 60 }),
      whole,
      whole,
      whole,
      fc.integer({ min: 0, max: 400 }),
      agrees,
    ),
    { numRuns: 5000 },
  )
})

// The similarity scorer converts its cutoff into a distance budget, so it
// reaches the band with a bound the caller never wrote.
test('the similarity scorer bands to the same answers', () => {
  fc.assert(
    fc.property(
      text,
      text,
      whole,
      whole,
      whole,
      fc.integer({ min: 0, max: 60 }),
      (a, b, insertion, deletion, substitution, scoreCutoff) => {
        const distance = reference(a, b, insertion, deletion, substitution)
        const indel = a.length * deletion + b.length * insertion
        const maximum =
          a.length >= b.length
            ? Math.min(indel, b.length * substitution + (a.length - b.length) * deletion)
            : Math.min(indel, a.length * substitution + (b.length - a.length) * insertion)
        const similarity = maximum - distance
        const seen = levenshteinSimilarity(a, b, {
          weights: { insertion, deletion, substitution },
          scoreCutoff,
        })
        expect(seen).toBe(similarity >= scoreCutoff ? similarity : 0)
      },
    ),
    { numRuns: 5000 },
  )
})
