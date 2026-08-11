// `prepareQuery` and `prepareChoice`, against the loop each replaces.
//
// Its own file rather than more cases in `process.bench.ts`: a bench file is
// hashed into every one of its own baseline entries, so folding these in would
// have marked all 67 process cases "definition changed" and thrown away the
// comparison that proves `extract*` did not move. The two sets also answer
// different questions — that file measures `search`, this one measures scoring
// a pair without going through it.
//
// Every prepared case is paired with the direct call it stands in for, at the
// same shape, so the pair is readable as a ratio rather than needing the corpus
// held in your head. The handle construction is *inside* the measured body
// wherever a real caller would pay for it once per run.
import { describe } from 'vitest'

import { levenshteinDistance } from '../src/distance/levenshtein.js'
import { ratio, tokenSortRatio, wRatio } from '../src/_fuzz/legacy.js'
import { prepareChoice, prepareQuery } from '../src/search.js'
import { defaultProcess } from '../src/utils.js'
import { sentences, words } from './_corpus.js'
import { measure } from './_harness.js'

const choices = words(200, 12)
const query = 'abcdefghijkl'
const titles = sentences(200, 5)
const titleQuery = 'alpha bravo charlie delta echo'
const titleQueries = sentences(20, 5, 0x1122_3344)

describe('prepared query against many choices', () => {
  measure('200 choices, ratio, direct', () => {
    for (const choice of choices) ratio(query, choice)
  })
  measure('200 choices, ratio, prepared query', () => {
    const prepared = prepareQuery(query, { scorer: ratio })
    for (const choice of choices) prepared(choice)
  })

  measure('200 sentences, tokenSortRatio, direct', () => {
    for (const title of titles) tokenSortRatio(titleQuery, title)
  })
  measure('200 sentences, tokenSortRatio, prepared query', () => {
    const prepared = prepareQuery(titleQuery, { scorer: tokenSortRatio })
    for (const title of titles) prepared(title)
  })

  measure('200 sentences, wRatio, direct', () => {
    for (const title of titles) wRatio(titleQuery, title)
  })
  measure('200 sentences, wRatio, prepared query', () => {
    const prepared = prepareQuery(titleQuery, { scorer: wRatio })
    for (const title of titles) prepared(title)
  })
})

describe('prepared choice against many queries', () => {
  // The shape this handle exists for: the processor is spent once, and a
  // symmetric built-in is scored with the held side as its prepared query.
  measure('20 queries x 200 choices, ratio + defaultProcess, direct', () => {
    for (const q of titleQueries) {
      const processed = defaultProcess(q)
      for (const title of titles) ratio(processed, defaultProcess(title))
    }
  })
  measure('20 queries x 200 choices, ratio + defaultProcess, prepared choices', () => {
    const prepared = titles.map((title) =>
      prepareChoice(title, { scorer: ratio, processor: defaultProcess }),
    )
    for (const q of titleQueries) for (const choice of prepared) choice(q)
  })

  measure('20 queries x 200 choices, tokenSortRatio, direct', () => {
    for (const q of titleQueries) for (const title of titles) tokenSortRatio(q, title)
  })
  measure('20 queries x 200 choices, tokenSortRatio, prepared choices', () => {
    const prepared = titles.map((title) =>
      prepareChoice(title, { scorer: tokenSortRatio }),
    )
    for (const q of titleQueries) for (const choice of prepared) choice(q)
  })
})

describe('both halves prepared', () => {
  measure('20 x 200, tokenSortRatio, direct', () => {
    for (const q of titleQueries) for (const title of titles) tokenSortRatio(q, title)
  })
  measure('20 x 200, tokenSortRatio, composed', () => {
    const prepared = titles.map((title) =>
      prepareChoice(title, { scorer: tokenSortRatio }),
    )
    for (const q of titleQueries) {
      const preparedQ = prepareQuery(q, { scorer: tokenSortRatio })
      for (const choice of prepared) preparedQ(choice)
    }
  })

  measure('20 x 200, levenshteinDistance, direct', () => {
    for (const q of titleQueries) {
      for (const title of titles) levenshteinDistance(q, title)
    }
  })
  measure('20 x 200, levenshteinDistance, composed', () => {
    const prepared = titles.map((title) =>
      prepareChoice(title, { scorer: levenshteinDistance }),
    )
    for (const q of titleQueries) {
      const preparedQ = prepareQuery(q, { scorer: levenshteinDistance })
      for (const choice of prepared) preparedQ(choice)
    }
  })
})

describe('handle construction amortisation', () => {
  // Every case above builds its handles inside the measured body, because that
  // is what a caller pays. This pair holds the calls fixed and moves only the
  // construction: one handle built once, against one built per batch of 200
  // calls. The difference is what construction costs, spread over 200 — not
  // what a call costs, which this cannot see.
  //
  // Nothing here measures per-call handle overhead, and no case in this file
  // does. A prepared query called with a raw choice tests the choice's brand
  // and then runs the closure; the `WeakMap` is read only when a handle
  // consumes another handle. Isolating that would need a comparator holding
  // the same prepared state without a handle around it, and a caller cannot
  // build one — which is also why a "hand-written closure" case would only
  // re-measure prepared against direct under a name claiming otherwise.
  const reused = prepareQuery(query, { scorer: ratio })

  measure('200 choices, ratio, handle built once', () => {
    for (const choice of choices) reused(choice)
  })
  measure('200 choices, ratio, handle rebuilt each run', () => {
    const prepared = prepareQuery(query, { scorer: ratio })
    for (const choice of choices) prepared(choice)
  })
})
