import { describe, expect, it } from 'vitest'

import { QueryState } from './query.js'

// The retention cap is not exported: a consumer has no use for it, and a test
// that read it from the module could not tell a wrong constant from a right one.
const CAP = 1 << 16

describe('the result scratch a query state keeps', () => {
  function reserved(...sizes: readonly number[]): QueryState {
    const state = new QueryState()
    for (const size of sizes) state.reserve(size)
    return state
  }

  it('grows to what a query needs', () => {
    const state = reserved(64)

    expect(state.ids.length).toBe(64)
    expect(state.scores.length).toBe(64)
  })

  it('reuses what it holds when that is under the cap', () => {
    const state = reserved(CAP)
    const ids = state.ids
    const scores = state.scores

    state.reserve(1)
    expect(state.ids).toBe(ids)
    expect(state.scores).toBe(scores)
    expect(state.ids.length).toBe(CAP)
  })

  it('drops an oversized buffer for a query that no longer needs it', () => {
    const state = reserved(CAP + 1)
    const ids = state.ids
    const scores = state.scores

    state.reserve(1)
    expect(state.ids).not.toBe(ids)
    expect(state.scores).not.toBe(scores)
    expect(state.ids.length).toBe(1)
  })

  it('holds an oversized buffer while the next query could fill it again', () => {
    const state = reserved(CAP * 2)
    const ids = state.ids
    const scores = state.scores

    state.reserve(CAP + 1)
    expect(state.ids).toBe(ids)
    expect(state.scores).toBe(scores)
    expect(state.ids.length).toBe(CAP * 2)
  })

  // The two are one unit of capacity however they are reached: a score read
  // against a longer id array would be reading whatever the last query left,
  // and only replacing both together keeps that impossible.
  it.each([[64], [CAP], [CAP + 1]])(
    'keeps the pair capacity-aligned at %i slots',
    (size) => {
      const state = reserved(CAP + 1, 1, size)

      expect(state.ids.length).toBe(state.scores.length)
    },
  )
})
