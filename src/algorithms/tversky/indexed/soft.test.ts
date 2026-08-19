// The soft index builder's one-shot contract. `createIndexedMatcher` seals
// immediately, so the errors that guard a reused builder have no route through
// the public API and are driven here directly.
import { describe, expect, it } from 'vitest'

import { createScorer, scorerCompilation } from '#core/scoring/scorer.js'

import { normalizedSimilarity as indelSimilarity } from '../../indel/index.js'
import { softChoiceOf, softQueryOf } from '../soft.js'
import { createSoftTverskyIndexBuilder } from './soft.js'

const inner = scorerCompilation(createScorer(indelSimilarity))

function builderOf() {
  const candidates = inner.candidateChoices
  if (candidates === undefined) throw new Error('missing Indel candidate capability')
  return createSoftTverskyIndexBuilder(
    (choice) => softChoiceOf(choice, null),
    (query) => ({
      query: softQueryOf(query, null),
      scoreChoice: () => 0,
    }),
    candidates(),
    0.8,
  )
}

describe('the index answers the ChoiceIndex contract directly', () => {
  it('selects nothing for a zero limit', () => {
    const builder = builderOf()
    builder.add(['swisscom'])
    builder.add(['google'])
    const index = builder.seal()
    expect(index.select(['swisscom'], null, 0).length).toBe(0)
    expect(index.select(['swisscom'], 0.5, 0).length).toBe(0)
  })
})

describe('the soft index builder is one-shot', () => {
  it('refuses a choice added after sealing', () => {
    const builder = builderOf()
    builder.add(['swisscom'])
    builder.seal()
    expect(() => builder.add(['google'])).toThrow(TypeError)
  })

  it('refuses a second seal', () => {
    const builder = builderOf()
    builder.add(['swisscom'])
    builder.seal()
    expect(() => builder.seal()).toThrow(TypeError)
  })
})
