// Ported from RapidFuzz's own suite. The index is an acceleration strategy, so
// every question here is the same one: does it answer what the exhaustive
// scorer answers?
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import {
  CORPORA,
  exhaustive,
  exhaustiveScan,
  indexOf,
  LIMITS,
  METRICS,
  pairs,
  QUERIES,
  THRESHOLDS,
} from './support.js'

describe('an indexed search answers what the exhaustive one does', () => {
  it('matches key, score and order across the whole matrix', () => {
    let cases = 0
    for (const metric of METRICS) {
      for (const gramSize of [2, 3]) {
        for (const choices of CORPORA) {
          for (const query of QUERIES) {
            const index = indexOf(metric, gramSize, choices)
            for (const threshold of THRESHOLDS) {
              for (const limit of LIMITS) {
                expect(pairs(index.select(query, threshold, limit))).toEqual(
                  exhaustive(metric, gramSize, choices, query, threshold, limit),
                )
                cases++
              }
              expect(pairs(index.scan(query, threshold))).toEqual(
                exhaustiveScan(metric, gramSize, choices, query, threshold),
              )
              cases++
            }
          }
        }
      }
    }
    expect(cases).toBeGreaterThan(2000)
  })

  it('matches on randomised corpora', () => {
    const letters = fc.constantFrom('a', 'b', 'c', '😀', '\ud800', ' ')
    const text = fc.array(letters, { maxLength: 12 }).map((parts) => parts.join(''))
    fc.assert(
      fc.property(
        fc.array(text, { maxLength: 12 }),
        text,
        fc.constantFrom(...THRESHOLDS),
        fc.constantFrom(...LIMITS),
        fc.constantFrom(2, 3),
        fc.constantFrom(...METRICS),
        (choices, query, threshold, limit, gramSize, metric) => {
          const index = indexOf(metric, gramSize, choices)
          expect(pairs(index.select(query, threshold, limit))).toEqual(
            exhaustive(metric, gramSize, choices, query, threshold, limit),
          )
          expect(pairs(index.scan(query, threshold))).toEqual(
            exhaustiveScan(metric, gramSize, choices, query, threshold),
          )
          return true
        },
      ),
      { numRuns: 400, seed: 0x5eed },
    )
  })

  it('answers nothing when a caller asks for nothing', () => {
    // `limit: 0` is a supported answer rather than an excuse, and it is the one
    // call that leaves selection with no result array to insert into. The dense
    // corpus matters here: it puts every choice into the walk, so the empty
    // room is reached with candidates in hand rather than none.
    for (const metric of METRICS) {
      for (const choices of [
        ['node', 'nodes', 'noded', 'nodex', 'nodey', 'nodez', 'qq'],
        ['abc', 'abd'],
      ]) {
        const index = indexOf(metric, 3, choices)
        for (const threshold of THRESHOLDS) {
          expect(pairs(index.select('node', threshold, 0))).toEqual([])
          expect(pairs(index.select('node', threshold, 0))).toEqual(
            exhaustive(metric, 3, choices, 'node', threshold, 0),
          )
        }
      }
    }
  })
})

describe('ordering', () => {
  it('breaks a tie on the earlier id, whatever order the postings arrive in', () => {
    // The query's grams are `ab` then `bc`, so accumulation reaches choice 1
    // before choice 0 and the touched set is descending. Both score the same, so
    // the winner is decided by the tie rule alone — and dropping it answers
    // choice 1, which is the bug this pins.
    const choices = ['bc', 'ab']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      const top = pairs(index.select('abc', null, 1))
      expect(top).toEqual(exhaustive(metric, 2, choices, 'abc', null, 1))
      expect(top[0].id).toBe(0)
      expect(pairs(index.select('abc', null, 2))).toEqual(
        exhaustive(metric, 2, choices, 'abc', null, 2),
      )
    }
  })

  it('breaks a tie on the earlier id when the result is already full', () => {
    // The same shape with a third, worse choice in the way, so the tie is
    // resolved by displacing the last entry rather than by an empty slot.
    const choices = ['bc', 'ab', 'zz']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      expect(pairs(index.select('abc', null, 1))).toEqual(
        exhaustive(metric, 2, choices, 'abc', null, 1),
      )
    }
  })

  it('interleaves zero-scoring choices by id when scanning', () => {
    // The trap `scan` exists for: ranked order puts the matches first, and
    // collection order puts choice 0 first even though it scores nothing.
    const choices = ['zzzz', 'abcd', 'yyyy', 'abcd']
    for (const metric of METRICS) {
      const index = indexOf(metric, 3, choices)
      const scanned = pairs(index.scan('abcd', null))
      expect(scanned.map((row) => row.id)).toEqual([0, 1, 2, 3])
      expect(scanned).toEqual(exhaustiveScan(metric, 3, choices, 'abcd', null))
    }
  })

  it('confines a scan to the touched choices under a positive threshold', () => {
    const choices = ['zzzz', 'abcd', 'yyyy', 'abcd']
    for (const metric of METRICS) {
      const index = indexOf(metric, 3, choices)
      expect(pairs(index.scan('abcd', 0.5))).toEqual(
        exhaustiveScan(metric, 3, choices, 'abcd', 0.5),
      )
    }
  })
})

describe('reuse', () => {
  it('answers repeated queries from the same scratch', () => {
    const choices = ['node', 'nodes', 'noded', 'nodex', 'nodey', 'nodez', 'qq']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      for (const query of ['node', 'qq', 'nodes', 'node', 'zzzz']) {
        expect(pairs(index.select(query, null, 3))).toEqual(
          exhaustive(metric, 2, choices, query, null, 3),
        )
      }
      // A dense query, then a sparse one, then a dense one again: the sparse
      // walk has to see an accumulator the dense scan left clean.
      for (const query of ['node', 'qq', 'node']) {
        expect(pairs(index.scan(query, 0.1))).toEqual(
          exhaustiveScan(metric, 2, choices, query, 0.1),
        )
      }
    }
  })
})
