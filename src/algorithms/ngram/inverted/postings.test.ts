import { describe, expect, it } from 'vitest'

import {
  exhaustive,
  exhaustiveScan,
  indexOf,
  METRICS,
  pairs,
  THRESHOLDS,
} from '../../../../testing/invertedIndex.js'
import { createDiceIndexBuilder } from './dice.js'

describe('the posting representation', () => {
  it('stores no counts when nothing repeats, and widens when it does', () => {
    for (const metric of METRICS) {
      for (const repeats of [1, 2, 300, 70_000]) {
        const choices = ['ab', `${'a'.repeat(repeats + 1)}b`]
        const index = indexOf(metric, 2, choices)
        expect(pairs(index.select('aa', null, 2))).toEqual(
          exhaustive(metric, 2, choices, 'aa', null, 2),
        )
      }
    }
  })

  it('inverts a list that covers most of the corpus', () => {
    // `no` is in six of seven choices, past the two-thirds cutoff.
    const choices = ['node', 'nodes', 'noded', 'nodex', 'nodey', 'nodez', 'qq']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      for (const threshold of THRESHOLDS) {
        expect(pairs(index.select('node', threshold, 3))).toEqual(
          exhaustive(metric, 2, choices, 'node', threshold, 3),
        )
        expect(pairs(index.scan('node', threshold))).toEqual(
          exhaustiveScan(metric, 2, choices, 'node', threshold),
        )
      }
    }
  })

  it('inverts a list when no frequency anywhere exceeds one', () => {
    // Dense with `counts === null`: `ab` is in six of seven choices and no gram
    // repeats within any of them, so an exception can only be an absence.
    const choices = ['abc', 'abd', 'abe', 'abf', 'abg', 'abh', 'xyz']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      for (const threshold of THRESHOLDS) {
        expect(pairs(index.select('abc', threshold, 3))).toEqual(
          exhaustive(metric, 2, choices, 'abc', threshold, 3),
        )
        expect(pairs(index.scan('abc', threshold))).toEqual(
          exhaustiveScan(metric, 2, choices, 'abc', threshold),
        )
      }
    }
  })

  it('walks a sparse list under a scan a dense list already widened', () => {
    // `ab` is dense and `zq` is not, so one query reaches both and the sparse
    // walk runs with the touched set already abandoned.
    const choices = ['abc', 'abd', 'abe', 'abf', 'abg', 'abzq', 'zq']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      expect(pairs(index.select('abzq', null, null))).toEqual(
        exhaustive(metric, 2, choices, 'abzq', null, null),
      )
    }
  })

  it('walks a counted sparse list under a widened scan', () => {
    // The same, with a repeat somewhere so the whole index carries counts.
    const choices = ['abc', 'abd', 'abe', 'abf', 'abg', 'abzqzq', 'zq']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      // Both sides of the shared minimum: the query holds `zq` twice in the
      // first and once in the second, against a choice that holds it twice.
      for (const query of ['abzqzq', 'abzq']) {
        expect(pairs(index.select(query, null, null))).toEqual(
          exhaustive(metric, 2, choices, query, null, null),
        )
      }
    }
  })

  it('inverts a list whose members repeat the gram', () => {
    // Dense with a counts array: most choices hold `aa`, and some hold it twice.
    const choices = ['aab', 'aaab', 'aaac', 'aad', 'aae', 'aaf', 'zz']
    for (const metric of METRICS) {
      const index = indexOf(metric, 2, choices)
      expect(pairs(index.select('aab', null, null))).toEqual(
        exhaustive(metric, 2, choices, 'aab', null, null),
      )
    }
  })

  it('narrows posting ids to the corpus, and stays exact either side of the bound', () => {
    // A wrong bound here does not throw: it wraps an id and answers the wrong
    // choice, so both sizes around 65,536 are pinned with the only match last.
    for (const choiceCount of [0x1_0000, 0x1_0001]) {
      const choices: string[] = new Array<string>(choiceCount).fill('')
      const last = choiceCount - 1
      choices[last] = 'abc'
      const index = createDiceIndexBuilder(3)
      for (const choice of choices) index.add(choice)
      const sealed = index.seal()
      expect(pairs(sealed.select('abc', 0.5, 1))).toEqual([{ id: last, score: 1 }])
    }
  })
})
