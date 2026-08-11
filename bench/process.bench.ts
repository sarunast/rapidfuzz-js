import { describe } from 'vitest'

import { configure } from '../src/configure.js'
import { damerauLevenshteinDistance } from '../src/distance/damerauLevenshtein.js'
import { hammingDistance } from '../src/distance/hamming.js'
import { indelDistance, indelNormalizedSimilarity } from '../src/distance/indel.js'
import { jaroSimilarity } from '../src/distance/jaro.js'
import { jaroWinklerSimilarity } from '../src/distance/jaroWinkler.js'
import { lcsSeqNormalizedSimilarity, lcsSeqSimilarity } from '../src/distance/lcsSeq.js'
import {
  levenshteinDistance,
  levenshteinNormalizedSimilarity,
} from '../src/distance/levenshtein.js'
import { osaDistance } from '../src/distance/osa.js'
import { postfixSimilarity } from '../src/distance/postfix.js'
import { prefixSimilarity } from '../src/distance/prefix.js'
import {
  partialRatio,
  ratio,
  tokenRatio,
  tokenSetRatio,
  tokenSortRatio,
  wRatio,
} from '../src/_fuzz/legacy.js'
import { extract, extractOne, scoreMatrix, scorePairs } from '../src/search.js'
import { defaultProcess } from '../src/utils.js'
import { sentences, words } from './_corpus.js'
import { measure } from './_harness.js'

// The realistic shape: one query against a list of candidates.
const choices = words(2000, 12)
const manyChoices = words(20_000, 12, 0x2bad_cafe)
const query = 'abcdefghijkl'

// Lengths from 3 to 30, so most candidates differ from the 12-character query.
const variedChoices = [3, 6, 9, 12, 15, 20, 25, 30].flatMap((length) =>
  words(250, length, 0x51ed_2701 + length),
)

const shortChoices = words(200, 12)
const queries = words(50, 12, 0xdead_beef)

const titles = sentences(500, 5)
const titleQuery = 'alpha bravo charlie delta echo'
const longQuery = 'abcdefghijklmnopqrstuvwxyz'.repeat(10)
const longChoices = words(250, longQuery.length, 0x1357_2468)
const longPartialQuery = 'alpha beta gamma delta '.repeat(12)
const longPartialChoices = sentences(150, 55, 0x71a2_b3c4)
const tokenQueries = sentences(30, 5, 0x1122_3344)
const tokenChoices = sentences(120, 5, 0x5566_7788)
const looseQueries = words(12, 512, 0x1020_3040)
const looseChoices = words(60, 512, 0x5060_7080)
// Candidates sharing a long head with the query — the shape a title or job-ad
// search actually has, and the one a prepared scorer used to rescan in full.
// The short pair stays inside one 32-element word, the long pair does not.
const shortPrefixQueries = queries.slice(0, 40).map((value) => `abc ${value}`)
const shortPrefixChoices = shortChoices.map((value) => `abc ${value}`)
const longPrefixQueries = queries
  .slice(0, 40)
  .map((value) => `senior frontend engineer ${value}`)
const longPrefixChoices = shortChoices.map((value) => `senior frontend engineer ${value}`)
const astralQueries = queries.slice(0, 30).map((value) => `🦊${value}🚀`)
const astralChoices = shortChoices.slice(0, 100).map((value) => `🦊${value}🌍`)
const arrayQueries = queries.slice(0, 30).map((value) => Array.from(value))
const arrayChoices = shortChoices.slice(0, 100).map((value) => Array.from(value))
const typedQueries = queries
  .slice(0, 30)
  .map((value) => Uint16Array.from(value, (character) => character.charCodeAt(0)))
const typedChoices = shortChoices
  .slice(0, 100)
  .map((value) => Uint16Array.from(value, (character) => character.charCodeAt(0)))

// Narrower candidate lists for the scorers whose per-candidate cost would
// otherwise make one sample tens of milliseconds. A sample that long contains a
// garbage collection every time, which is exactly what the median cannot
// discard — see `_harness.ts`. The matrix shape in each name says which list it
// used, so the shape being measured stays legible.
const longPartialChoicesFew = longPartialChoices.slice(0, 20)
const longPartialChoicesSome = longPartialChoices.slice(0, 60)
const tokenChoicesFew = tokenChoices.slice(0, 40)
const looseChoicesFew = looseChoices.slice(0, 8)
const damerauChoices = shortChoices.slice(0, 40)

// One query per width, and two candidate lists per width: near-copies that
// clear a 0.9 cutoff, and independent strings that cannot. 256 elements is
// eight words and 1024 is thirty-two, so the band has room to be a small
// fraction of the pattern at the wider one and barely narrower at the other.
const bandQuery256 = words(1, 256, 0x0ba7_d256)[0]
const bandQuery1024 = words(1, 1024, 0x0ba7_d401)[0]
/** `source` with one element in twenty replaced, which scores just above 0.9. */
const nearCopies = (source: string, count: number): string[] =>
  Array.from({ length: count }, (_unused, run) =>
    Array.from(source, (character, index) =>
      (index + run) % 20 === 0 ? 'z' : character,
    ).join(''),
  )
const bandNear256 = nearCopies(bandQuery256, 40)
const bandNear1024 = nearCopies(bandQuery1024, 40)
const bandFar256 = words(40, 256, 0x0fa2_2560)
const bandFar1024 = words(40, 1024, 0x0fa2_1024)
// Near-copies cut short, so the query is the longer side by 60 to 100 elements
// — enough for the gate to refuse the held pattern, not enough for the cutoff
// to reject the pair before a kernel is picked.
const bandShorter1024 = nearCopies(bandQuery1024, 40).map((value, index) =>
  value.slice(0, 960 - index),
)
// The opposite shape: candidates that agree with the query on all but their
// last 32 elements, so the trimming kernel has almost nothing left to score.
const affixQuery = bandQuery1024.slice(0, 512)
const affixChoices = words(40, 32, 0x0aff_7a11).map(
  (tail) => affixQuery.slice(0, 480) + tail,
)
const astralChoicesFew = astralChoices.slice(0, 50)

describe('extractOne', () => {
  measure('2000 choices, wRatio (default)', () => {
    extractOne(query, choices)
  })
  measure('500 sentences, wRatio (default)', () => {
    extractOne(titleQuery, titles)
  })
  measure('2000 choices, ratio', () => {
    extractOne(query, choices, { scorer: ratio })
  })
  measure('2000 choices, ratio + scoreCutoff', () => {
    extractOne(query, choices, { scorer: ratio, scoreCutoff: 80 })
  })
  // Every choice above is the same length, so the length-difference prune can
  // never fire. These vary the length, which is the realistic shape and the
  // only one where the prune is observable at all.
  measure('2000 varied-length choices, ratio + scoreCutoff', () => {
    extractOne(query, variedChoices, { scorer: ratio, scoreCutoff: 80 })
  })
  measure('2000 varied-length choices, ratio', () => {
    extractOne(query, variedChoices, { scorer: ratio })
  })
  measure('2000 choices, defaultProcess', () => {
    extractOne(query, choices, { scorer: ratio, processor: defaultProcess })
  })
})

describe('extract', () => {
  measure('2000 choices, limit 5', () => {
    extract(query, choices, { scorer: ratio, limit: 5 })
  })
  measure('20000 choices, limit 5', () => {
    extract(query, manyChoices, { scorer: ratio, limit: 5 })
  })
  measure('250 x 260 chars, ratio + cutoff 70', () => {
    extract(longQuery, longChoices, { scorer: ratio, scoreCutoff: 70, limit: 5 })
  })
  measure('250 x 260 chars, ratio + cutoff 90', () => {
    extract(longQuery, longChoices, { scorer: ratio, scoreCutoff: 90, limit: 5 })
  })
})

describe('prepared ratio', () => {
  measure('250 x 260 chars, direct loose cutoff', () => {
    for (const choice of longChoices) ratio(longQuery, choice)
  })
  measure('250 x 260 chars, prepared tight cutoff', () => {
    extract(longQuery, longChoices, { scorer: ratio, scoreCutoff: 90, limit: null })
  })
})

// These scorers cache query-side state through process.extractOne. Direct fuzz
// benchmarks do not exercise that preparation path.
describe('prepared fuzz scorers', () => {
  measure('500 sentences, tokenSortRatio', () => {
    extractOne(titleQuery, titles, { scorer: tokenSortRatio })
  })
  measure('500 sentences, tokenSetRatio', () => {
    extractOne(titleQuery, titles, { scorer: tokenSetRatio })
  })
  measure('200 equal-length choices, partialRatio', () => {
    extractOne(query, shortChoices, { scorer: partialRatio })
  })
  // Eleven milliseconds even for twenty candidates: `partialRatio` scans O(n)
  // windows per candidate, and these are 55-word sentences. The longer window
  // buys back the samples that costs.
  measure(
    '20 long choices, partialRatio',
    () => {
      extractOne(longPartialQuery, longPartialChoicesFew, { scorer: partialRatio })
    },
    { time: 3000 },
  )
  measure('60 length-skewed choices, wRatio', () => {
    extractOne(longPartialQuery, longPartialChoicesSome, { scorer: wRatio })
  })
})

describe('token scorer matrices', () => {
  measure('30 x 40, tokenSortRatio', () => {
    scoreMatrix(tokenQueries, tokenChoicesFew, { scorer: tokenSortRatio })
  })
  measure('30 x 40, tokenSetRatio', () => {
    scoreMatrix(tokenQueries, tokenChoicesFew, { scorer: tokenSetRatio })
  })
  measure('30 x 40, tokenRatio', () => {
    scoreMatrix(tokenQueries, tokenChoicesFew, { scorer: tokenRatio })
  })
  measure('30 x 40, wRatio', () => {
    scoreMatrix(tokenQueries, tokenChoicesFew, { scorer: wRatio })
  })
})

describe('prepared long distance matrices', () => {
  measure('12 x 8, Levenshtein loose cutoff', () => {
    scoreMatrix(looseQueries, looseChoicesFew, { scorer: levenshteinDistance })
  })
  measure('12 x 8, Levenshtein tight cutoff', () => {
    scoreMatrix(looseQueries, looseChoicesFew, {
      scorer: levenshteinDistance,
      scoreCutoff: 8,
    })
  })
  measure('12 x 8, LCS loose cutoff', () => {
    scoreMatrix(looseQueries, looseChoicesFew, { scorer: lcsSeqSimilarity })
  })
  measure('12 x 8, Indel loose cutoff', () => {
    scoreMatrix(looseQueries, looseChoicesFew, { scorer: indelDistance })
  })
  measure('12 x 8, OSA multiword', () => {
    scoreMatrix(looseQueries, looseChoicesFew, { scorer: osaDistance })
  })
})

// Where a cutoff decides how many words of a held pattern the kernel touches.
//
// The matrices above score a prepared pattern with no cutoff at all, so none of
// them reaches the bounded kernel, and the only case that does — `'250 x 260
// chars, prepared tight cutoff'` — reaches it through `ratio` at one length.
// Two things have to be separated here that a single case cannot separate:
//
//   - a **near** list scores at or above the cutoff, so every candidate is
//     scanned to the end and an early exit can only cost;
//   - a **far** list is rejected, so an early exit is the only thing that pays.
//
// The two lists disagree about everything else, so a change that helps one and
// hurts the other is visible here and nowhere else.
//
// `ratio` and the LCS/Indel scorers are both here because they choose their
// kernel differently and neither substitutes for the other. `ratio` holds the
// pattern whenever the cutoff is tight enough to be worth it and reaches the
// bounded kernel at every width; the LCS and Indel scorers ask
// `preparedLengthWorthwhile` first, which at these widths says no and sends
// them to the trimming kernel instead. So the `ratio` cases measure the band
// and the scoreMatrix cases measure the gate that decides whether it is used.
describe('prepared LCS bands', () => {
  measure('40 x 256 chars, ratio near, cutoff 90', () => {
    extract(bandQuery256, bandNear256, { scorer: ratio, scoreCutoff: 90, limit: null })
  })
  measure('40 x 256 chars, ratio far, cutoff 90', () => {
    extract(bandQuery256, bandFar256, { scorer: ratio, scoreCutoff: 90, limit: null })
  })
  measure('40 x 1024 chars, ratio near, cutoff 90', () => {
    extract(bandQuery1024, bandNear1024, { scorer: ratio, scoreCutoff: 90, limit: null })
  })
  measure('40 x 1024 chars, ratio far, cutoff 90', () => {
    extract(bandQuery1024, bandFar1024, { scorer: ratio, scoreCutoff: 90, limit: null })
  })
  measure('40 x 256 chars, LCS near, cutoff 0.9', () => {
    scoreMatrix([bandQuery256], bandNear256, {
      scorer: lcsSeqNormalizedSimilarity,
      scoreCutoff: 0.9,
    })
  })
  measure('40 x 1024 chars, LCS near, cutoff 0.9', () => {
    scoreMatrix([bandQuery1024], bandNear1024, {
      scorer: lcsSeqNormalizedSimilarity,
      scoreCutoff: 0.9,
    })
  })
  measure('40 x 1024 chars, LCS far, cutoff 0.9', () => {
    scoreMatrix([bandQuery1024], bandFar1024, {
      scorer: lcsSeqNormalizedSimilarity,
      scoreCutoff: 0.9,
    })
  })
  measure('40 x 1024 chars, Indel near, cutoff 0.9', () => {
    scoreMatrix([bandQuery1024], bandNear1024, {
      scorer: indelNormalizedSimilarity,
      scoreCutoff: 0.9,
    })
  })
  // The shape the gate refuses outright: a query longer than every candidate
  // never reaches the held pattern, whatever the band would have cost. The
  // cutoff is looser here so the pairs are still worth scoring — at 0.9 a
  // choice this much shorter is out of reach before a kernel is chosen, and
  // what would be measured is the length prune in `search`.
  measure('40 x 1024 chars, LCS query longer, cutoff 0.85', () => {
    scoreMatrix([bandQuery1024], bandShorter1024, {
      scorer: lcsSeqNormalizedSimilarity,
      scoreCutoff: 0.85,
    })
  })
  // What the gate costs when it keeps the held pattern away. A candidate
  // sharing 480 of its 512 elements leaves the trimming kernel 32 to score,
  // where a held pattern reads all 512 however narrow the band. Dropping either
  // clause of `preparedLengthWorthwhile` was measured here at 3.2x slower —
  // against 1.6x to 7.7x faster on the affix-free lists above, which is the
  // whole difficulty: a tight cutoff produces a narrow band and a large affix
  // alike, and lengths cannot tell the two apart.
  //
  // `sharesAffix` is what now tells them apart, by looking at the elements
  // instead of the lengths, and this case is the price of asking: about 5%,
  // spent scanning a probe and then handing the pair to the trimming kernel
  // anyway. That is what the 1.35x to 7.0x above is bought with, and it is the
  // number a further relaxation has to improve on — 3.2x is history now, not
  // the bar.
  measure('40 x 512 chars, 480 shared, cutoff 0.9', () => {
    scoreMatrix([affixQuery], affixChoices, {
      scorer: lcsSeqNormalizedSimilarity,
      scoreCutoff: 0.9,
    })
  })
  // The same shape through Indel, whose gate `sharesAffix` also relaxes. Its
  // kernels are the LCS ones, but it reaches them through its own dispatch, so
  // the case above does not stand for it.
  measure('40 x 512 chars, 480 shared, Indel, cutoff 0.9', () => {
    scoreMatrix([affixQuery], affixChoices, {
      scorer: indelNormalizedSimilarity,
      scoreCutoff: 0.9,
    })
  })
})

describe('short metric matrices', () => {
  measure('50 x 200, Hamming distance', () => {
    scoreMatrix(queries, shortChoices, { scorer: hammingDistance })
  })
  measure('50 x 200, Prefix similarity', () => {
    scoreMatrix(queries, shortChoices, { scorer: prefixSimilarity })
  })
  measure('50 x 200, Postfix similarity', () => {
    scoreMatrix(queries, shortChoices, { scorer: postfixSimilarity })
  })
  measure('50 x 40, Damerau distance', () => {
    scoreMatrix(queries, damerauChoices, { scorer: damerauLevenshteinDistance })
  })
  // Hamming has no structure a kernel can exploit, so a cutoff is the only
  // thing that makes it sublinear. These two cover the bound reaching the scan
  // and the length difference rejecting a candidate before it starts.
  measure('50 x 200, Hamming distance, cutoff 3', () => {
    scoreMatrix(queries, shortChoices, { scorer: hammingDistance, scoreCutoff: 3 })
  })
  measure('30 x 250, Hamming distance, length skewed', () => {
    scoreMatrix(queries.slice(0, 30), longChoices, {
      scorer: hammingDistance,
      scoreCutoff: 8,
    })
  })
  // A single astral character used to push a whole converted sequence onto the
  // map-backed last-occurrence path, ASCII elements included.
  measure('30 x 50, Damerau distance, astral', () => {
    scoreMatrix(astralQueries, astralChoicesFew, { scorer: damerauLevenshteinDistance })
  })
})

describe('scoreMatrix', () => {
  measure('50 x 200, ratio', () => {
    scoreMatrix(queries, shortChoices, { scorer: ratio })
  })
  measure('50 x 200, indelNormalizedSimilarity', () => {
    scoreMatrix(queries, shortChoices, { scorer: indelNormalizedSimilarity })
  })
  measure('symmetric 200 x 200, ratio', () => {
    scoreMatrix(shortChoices, shortChoices, { scorer: ratio })
  })
  measure('30 x 100, astral ratio', () => {
    scoreMatrix(astralQueries, astralChoices, { scorer: ratio })
  })
  measure('30 x 100, array ratio', () => {
    scoreMatrix(arrayQueries, arrayChoices, { scorer: ratio })
  })
  measure('30 x 100, typed-array ratio', () => {
    scoreMatrix(typedQueries, typedChoices, { scorer: ratio })
  })
  measure('50 x 200, Levenshtein distance', () => {
    scoreMatrix(queries, shortChoices, { scorer: levenshteinDistance })
  })
  measure('50 x 200, Jaro similarity', () => {
    scoreMatrix(queries, shortChoices, { scorer: jaroSimilarity })
  })
  measure('50 x 200, Jaro-Winkler similarity', () => {
    scoreMatrix(queries, shortChoices, { scorer: jaroWinklerSimilarity })
  })
  measure('50 x 200, OSA distance', () => {
    scoreMatrix(queries, shortChoices, { scorer: osaDistance })
  })
  // A prepared query cannot re-base its match mask on a shared prefix, so it
  // marks those positions taken and starts past them instead. Prepared Jaro
  // used to rescan every leading character of every candidate.
  measure('40 x 200, Jaro similarity, short shared prefix', () => {
    scoreMatrix(shortPrefixQueries, shortPrefixChoices, { scorer: jaroSimilarity })
  })
  measure('40 x 200, Jaro similarity, long shared prefix', () => {
    scoreMatrix(longPrefixQueries, longPrefixChoices, { scorer: jaroSimilarity })
  })
})

// `process` substitutes the scorer's worst score for a missing cutoff, which is
// `0` for a similarity, so these reach the Levenshtein kernel with a finite
// budget on every candidate — a path `50 x 200, Levenshtein distance` above
// never takes.
describe('prepared Levenshtein similarity', () => {
  measure('250 x 260 chars, extractOne', () => {
    extractOne(longQuery, longChoices, { scorer: levenshteinNormalizedSimilarity })
  })
  measure('12 x 8 x 512 chars, scoreMatrix', () => {
    scoreMatrix(looseQueries, looseChoicesFew, {
      scorer: levenshteinNormalizedSimilarity,
    })
  })
  measure('50 x 200, scoreMatrix', () => {
    scoreMatrix(queries, shortChoices, { scorer: levenshteinNormalizedSimilarity })
  })
})

// A configured scorer wraps the one it configures, merging baked options into
// whatever the call supplies. Each pair below configures a scorer with *no*
// options and measures it against the same scorer bare, so the only difference
// is the wrapper — comparing weighted Levenshtein against plain would measure
// the generic DP against the bit-parallel kernel instead, which is a different
// amount of work and says nothing about `configure`.
//
// `scorePairs` is the one that matters most: it deliberately keeps the
// non-prepared per-pair loop, so a per-call merge is not amortised anywhere.
describe('configured scorers', () => {
  const wrapped = configure(ratio, {})
  const weighted = configure(levenshteinDistance, { weights: [1, 1, 2] })
  const pairQueries = queries.concat(queries.slice(0, 50)).slice(0, 100)
  const pairChoices = shortChoices.slice(0, 100)

  measure('50 x 200, ratio, wrapped', () => {
    scoreMatrix(queries, shortChoices, { scorer: wrapped })
  })
  measure('50 x 200, ratio, bare', () => {
    scoreMatrix(queries, shortChoices, { scorer: ratio })
  })

  measure('200 choices, ratio, wrapped', () => {
    extract(queries[0], shortChoices, { scorer: wrapped, limit: 5 })
  })
  measure('200 choices, ratio, bare', () => {
    extract(queries[0], shortChoices, { scorer: ratio, limit: 5 })
  })

  measure('100 pairs, ratio, wrapped', () => {
    scorePairs(pairQueries, pairChoices, { scorer: wrapped })
  })
  measure('100 pairs, ratio, bare', () => {
    scorePairs(pairQueries, pairChoices, { scorer: ratio })
  })

  // No bare comparator: weighted Levenshtein can only reach `process` through a
  // configured scorer now, so this is recorded as its own number.
  measure('50 x 200, Levenshtein weighted', () => {
    scoreMatrix(queries, shortChoices, { scorer: weighted })
  })
})
