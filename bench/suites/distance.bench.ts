import { damerauLevenshteinDistance } from '../../src/algorithms/damerauLevenshtein/implementation.js'
import {
  hammingDistance,
  hammingNormalizedSimilarity,
} from '../../src/algorithms/hamming/implementation.js'
import {
  indelDistance,
  indelNormalizedSimilarity,
} from '../../src/algorithms/indel/implementation.js'
import { jaroSimilarity } from '../../src/algorithms/jaro/implementation.js'
import { jaroWinklerSimilarity } from '../../src/algorithms/jaroWinkler/implementation.js'
import {
  lcsSeqEditops,
  lcsSeqNormalizedSimilarity,
} from '../../src/algorithms/lcs/implementation.js'
import { levenshteinEditops } from '../../src/algorithms/levenshtein/editops.js'
import {
  levenshteinDistance,
  levenshteinNormalizedSimilarity,
  levenshteinSimilarity,
} from '../../src/algorithms/levenshtein/metric.js'
import { osaDistance } from '../../src/algorithms/osa/implementation.js'
import {
  postfixNormalizedSimilarity,
  postfixSimilarity,
} from '../../src/algorithms/postfix/implementation.js'
import {
  prefixNormalizedSimilarity,
  prefixSimilarity,
} from '../../src/algorithms/prefix/implementation.js'
import {
  editedPairs,
  LATIN1,
  pairs,
  similarPairs,
  words,
  WORD_BOUNDARY_LENGTHS,
} from '../harness/corpus.js'
import { describe, measure } from '../harness/harness.js'

// Three length classes, because the algorithms have different crossover
// points: short strings are dominated by setup cost, long ones by the inner
// loop. A single size would hide whichever regression lives at the other end.
const short = similarPairs(200, 8)
const medium = similarPairs(200, 32)
const long = similarPairs(100, 128)
const veryLong = similarPairs(20, 1024)
const huge = similarPairs(5, 4096)
const alignmentHuge = similarPairs(1, 8192)
const alignmentDissimilar = pairs(words(2, 16_384, 0x3141_5926))
const alignmentSparseSource = 'a'.repeat(4096)
const alignmentSparseDestination = Array.from(alignmentSparseSource, (value, index) =>
  index % 256 === 0 ? 'b' : value,
).join('')
const dissimilar = pairs(words(200, 32))
// Pairs whose lengths alone put them out of reach of a cutoff: 1024 against
// 256 cannot lose fewer than 768 elements.
// The shorter side is a prefix of the longer, so the affix scan has 256
// elements to walk before it can conclude what the lengths already said.
const lengthSkewed = words(100, 1024, 0x0ba7_d101).map((value, index) =>
  index % 2 === 0 ? [value, value.slice(0, 256)] : [value.slice(0, 256), value],
)
// The mirror of `lengthSkewed`: the shorter side is a *suffix* of the longer,
// which is the shape the postfix scan pays for.
const suffixSkewed = words(100, 1024, 0x0ba7_d101).map((value, index) =>
  index % 2 === 0 ? [value, value.slice(-256)] : [value.slice(-256), value],
)
const weightedDissimilar = pairs(words(8, 512, 0x2718_2818))

// Prefixes of the corpora above, for the scorers that cost enough per pair to
// push a whole-corpus sample past a millisecond or two. Past that a sample is
// long enough to contain a garbage collection every time, and the median stops
// being able to reject the disturbed ones — see `harness/harness.ts`. These are the
// same pairs in the same order, so the inputs stay reproducible.
const veryLongFew = veryLong.slice(0, 8)
const veryLongOne = veryLong.slice(0, 1)
const hugeOne = huge.slice(0, 1)
const weightedDissimilarFew = weightedDissimilar.slice(0, 2)
const weightedDissimilarOne = weightedDissimilar.slice(0, 1)

// Every case below inlines its own loop rather than calling one shared
// `run(data, fn)` helper. V8 attaches an inline cache to a function literal
// rather than to a closure, so a shared helper gives every scorer in this file
// the same megamorphic call site — and measures whichever group ran first while
// that site was still monomorphic.

describe('indelDistance', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) indelDistance(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) indelDistance(a, b)
  })
  measure('32 chars, unrelated', () => {
    for (const [a, b] of dissimilar) indelDistance(a, b)
  })
  measure('128 chars, similar', () => {
    for (const [a, b] of long) indelDistance(a, b)
  })
  measure('1024 chars, similar', () => {
    for (const [a, b] of veryLong) indelDistance(a, b)
  })
})

describe('indelNormalizedSimilarity', () => {
  measure('128 chars, similar', () => {
    for (const [a, b] of long) indelNormalizedSimilarity(a, b)
  })
  measure('1024 chars, similar', () => {
    for (const [a, b] of veryLong) indelNormalizedSimilarity(a, b)
  })
})

describe('levenshteinDistance', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) levenshteinDistance(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) levenshteinDistance(a, b)
  })
  measure('32 chars, unrelated', () => {
    for (const [a, b] of dissimilar) levenshteinDistance(a, b)
  })
  measure('128 chars, similar', () => {
    for (const [a, b] of long) levenshteinDistance(a, b)
  })
  measure('1024 chars, similar', () => {
    for (const [a, b] of veryLong) levenshteinDistance(a, b)
  })
  measure('4096 chars, tight cutoff', () => {
    for (const [a, b] of huge) levenshteinDistance(a, b, { scoreCutoff: 4, scoreHint: 2 })
  })
  measure('4096 chars, cutoff 16', () => {
    for (const [a, b] of huge) {
      levenshteinDistance(a, b, { scoreCutoff: 16, scoreHint: 8 })
    }
  })
})

describe('levenshteinDistance, weighted', () => {
  const weightedLongA = 'a'.repeat(2048) + 'kitten' + 'z'.repeat(2048)
  const weightedLongB = 'a'.repeat(2048) + 'sitting' + 'z'.repeat(2048)

  measure('32 chars, similar', () => {
    for (const [a, b] of medium) levenshteinDistance(a, b, { weights: [1, 1, 2] })
  })
  measure('4096 chars, scaled Indel', () => {
    levenshteinDistance(weightedLongA, weightedLongB, { weights: [2, 2, 5] })
  })
  measure('4096 chars, asymmetric generic', () => {
    levenshteinDistance(weightedLongA, weightedLongB, { weights: [3, 7, 5] })
  })
  // The two cases above share a 2048-character head and tail, so the affix scan
  // reduces each to a seven-character problem and the generic dynamic program
  // never runs at size. These pairs have nothing to trim, which is the only way
  // to see the row loop itself.
  measure('512 chars, unrelated, asymmetric generic', () => {
    for (const [a, b] of weightedDissimilarFew) {
      levenshteinDistance(a, b, { weights: [3, 7, 5] })
    }
  })
  measure('512 chars, unrelated, fractional weights', () => {
    for (const [a, b] of weightedDissimilarOne) {
      levenshteinDistance(a, b, { weights: [1.5, 2.25, 3.5] })
    }
  })
})

describe('lcsSeqNormalizedSimilarity', () => {
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) lcsSeqNormalizedSimilarity(a, b)
  })
  measure('128 chars, similar', () => {
    for (const [a, b] of long) lcsSeqNormalizedSimilarity(a, b)
  })
  measure('4096 chars, cutoff 4000', () => {
    for (const [a, b] of huge)
      lcsSeqNormalizedSimilarity(a, b, { scoreCutoff: 4000 / 4096 })
  })
  // The pair every other case here lacks: one the cutoff rejects on length
  // alone. No affix is worth measuring and no kernel should run, so this is
  // entirely the cost of deciding not to score — which is most of what an
  // `extract` under a running cutoff does, and none of what the pairs above do.
  measure('1024 vs 256 chars, cutoff rejects on length', () => {
    for (const [a, b] of lengthSkewed)
      lcsSeqNormalizedSimilarity(a, b, { scoreCutoff: 900 / 1024 })
  })
})

// The Ukkonen band, on both sides of its cutoff. The case above is refused on
// length before a kernel runs, and `4096 chars, cutoff 4000` measures one width
// only — neither separates what the band costs a candidate that clears the
// cutoff from what it saves on one that cannot.
//
// The cutoffs are chosen from the corpora rather than picked: at 0.15 edits
// these pairs score 0.88 to 0.92, so 0.85 passes all of them, 0.95 none, and
// 0.9 splits them roughly in half — which is the mix an `extract` sees. Every
// case here reaches `lcsManyWordsBanded` with a band of 3 to 11 words out of 16
// or 32.
describe('lcsSeqNormalizedSimilarity, banded', () => {
  const half = similarPairs(40, 512, 0.15, 0x2f19_a3c1)

  measure('512 chars, cutoff 0.95, all rejected', () => {
    for (const [a, b] of half) lcsSeqNormalizedSimilarity(a, b, { scoreCutoff: 0.95 })
  })
  measure('1024 chars, cutoff 0.85, all accepted', () => {
    for (const [a, b] of veryLong) lcsSeqNormalizedSimilarity(a, b, { scoreCutoff: 0.85 })
  })
  measure('1024 chars, cutoff 0.9, half rejected', () => {
    for (const [a, b] of veryLong) lcsSeqNormalizedSimilarity(a, b, { scoreCutoff: 0.9 })
  })
})

describe('jaroSimilarity', () => {
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) jaroSimilarity(a, b)
  })
  measure('128 chars, similar', () => {
    for (const [a, b] of long) jaroSimilarity(a, b)
  })
  measure('1024 chars, similar', () => {
    for (const [a, b] of veryLong) jaroSimilarity(a, b)
  })
  measure('8 vs 377 chars, length skewed', () => {
    jaroSimilarity('01234567', '0'.repeat(170) + '7654321' + '0'.repeat(200))
  })
})

describe('jaroWinklerSimilarity', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) jaroWinklerSimilarity(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) jaroWinklerSimilarity(a, b)
  })
  measure('128 chars, similar', () => {
    for (const [a, b] of long) jaroWinklerSimilarity(a, b)
  })
  measure('32 chars, cutoff 0.92, half rejected', () => {
    for (const [a, b] of medium) jaroWinklerSimilarity(a, b, { scoreCutoff: 0.92 })
  })
})

describe('osaDistance', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) osaDistance(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) osaDistance(a, b)
  })
  measure('1024 chars, similar', () => {
    for (const [a, b] of veryLongFew) osaDistance(a, b)
  })
  measure('4096 chars, sparse substitutions', () => {
    osaDistance(alignmentSparseSource, alignmentSparseDestination)
  })
})

describe('damerauLevenshteinDistance', () => {
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) damerauLevenshteinDistance(a, b)
  })
  // One 1024-character pair is already six milliseconds of scalar dynamic
  // program — the only case in the suite that cannot be cut towards a
  // millisecond without changing the length it exists to cover.
  measure('1024 chars, similar', () => {
    for (const [a, b] of veryLongOne) damerauLevenshteinDistance(a, b)
  })
})

describe('levenshteinEditops', () => {
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) levenshteinEditops(a, b)
  })
  measure('128 chars, similar', () => {
    for (const [a, b] of long) levenshteinEditops(a, b)
  })
  measure('1024 chars, similar', () => {
    for (const [a, b] of veryLongFew) levenshteinEditops(a, b)
  })
  measure('4096 chars, sparse substitutions', () => {
    levenshteinEditops(alignmentSparseSource, alignmentSparseDestination)
  })
  // Forty-six milliseconds for the one pair, so the default window would leave
  // it around twenty samples. Five seconds buys a hundred, which is what the
  // median needs before it is worth reading.
  measure(
    '16384 chars, unrelated',
    () => {
      for (const [a, b] of alignmentDissimilar) levenshteinEditops(a, b)
    },
    { time: 5000 },
  )
})

// Alignment recovery runs on the matrix kernels in `lcs/internal/matrix.ts`
// and `levenshtein/internal/matrix.ts` rather than the score-only ones. Keep
// its scaling visible independently from similarity.
describe('lcsSeqEditops', () => {
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) lcsSeqEditops(a, b)
  })
  measure('128 chars, similar', () => {
    for (const [a, b] of long) lcsSeqEditops(a, b)
  })
  measure('1024 chars, similar', () => {
    for (const [a, b] of veryLongFew) lcsSeqEditops(a, b)
  })
  measure('8192 chars, similar', () => {
    for (const [a, b] of alignmentHuge) lcsSeqEditops(a, b)
  })
})

// The similarity-shaped Levenshtein scorers convert their cutoff into a
// distance budget, so they reach the kernel with a *finite* budget as soon as
// the caller supplies any cutoff at all — including the `0` that `process`
// substitutes for "no cutoff" on a similarity. `levenshteinDistance` above only
// sees a finite budget when the caller asks for one, so it never covers this.
describe('levenshteinSimilarity', () => {
  measure('128 chars, cutoff 0', () => {
    for (const [a, b] of long) levenshteinSimilarity(a, b, { scoreCutoff: 0 })
  })
  measure('1024 chars, cutoff 0', () => {
    for (const [a, b] of veryLong) levenshteinSimilarity(a, b, { scoreCutoff: 0 })
  })
  measure('4096 chars, cutoff 0', () => {
    for (const [a, b] of hugeOne) levenshteinSimilarity(a, b, { scoreCutoff: 0 })
  })
})

describe('levenshteinNormalizedSimilarity', () => {
  measure('1024 chars, no cutoff', () => {
    for (const [a, b] of veryLong) levenshteinNormalizedSimilarity(a, b)
  })
  measure('128 chars, cutoff 0', () => {
    for (const [a, b] of long) levenshteinNormalizedSimilarity(a, b, { scoreCutoff: 0 })
  })
  measure('1024 chars, cutoff 0', () => {
    for (const [a, b] of veryLong) {
      levenshteinNormalizedSimilarity(a, b, { scoreCutoff: 0 })
    }
  })
  measure('1024 chars, cutoff 0.9', () => {
    for (const [a, b] of veryLong) {
      levenshteinNormalizedSimilarity(a, b, { scoreCutoff: 0.9 })
    }
  })
  measure('4096 chars, cutoff 0', () => {
    for (const [a, b] of hugeOne) {
      levenshteinNormalizedSimilarity(a, b, { scoreCutoff: 0 })
    }
  })
})

// Recovery over inputs that fit a single machine word, which is where fuzzy
// matching mostly lives. The groups above reach the one-word matrix only after
// affix trimming shortens them into it, so neither shows what it costs on its
// own.
describe('editops, one word wide', () => {
  measure('8 chars, similar, Levenshtein', () => {
    for (const [a, b] of short) levenshteinEditops(a, b)
  })
  measure('8 chars, similar, LCS', () => {
    for (const [a, b] of short) lcsSeqEditops(a, b)
  })
  measure('32 chars, unrelated, Levenshtein', () => {
    for (const [a, b] of dissimilar) levenshteinEditops(a, b)
  })
  measure('32 chars, unrelated, LCS', () => {
    for (const [a, b] of dissimilar) lcsSeqEditops(a, b)
  })
})

// The alignment matrices index their match masks over the range their pattern's
// alphabet spans, so text drawn from the whole of Latin-1 builds the widest
// table ordinary input can. Eight such characters barely fill it and 32 of them
// use it, which is the trade the bound on that range decides — the corpora
// above are lowercase, and would never show it moving.
// Where the kernels change shape. A pattern of 32 elements or fewer takes a
// one-word kernel; 33 takes the multiword one, and each further word is another
// pass of its inner loop. The corpora above jump 8 → 32 → 128 → 1024, so a cost
// that appears only at a crossover falls in a gap between them and nothing
// notices.
//
// `editedPairs` rather than `similarPairs` here: both sides are exactly the
// length named, and consecutive lengths differ by their length alone rather
// than also by how many edits chance gave them.
describe('word-width boundaries', () => {
  // Bit-parallel work is about `length * words` per pair, so the count falls as
  // the length rises — otherwise the 129-character case would be sixteen times
  // the sample the 31-character one is, and only one of them could sit in the
  // range where a median means anything.
  const boundary = new Map(
    WORD_BOUNDARY_LENGTHS.map((length) => {
      const words = Math.ceil(length / 32)
      const count = Math.max(32, Math.round(80_000 / (length * words)))
      return [length, editedPairs(count, length, 2, 0x5bd1_e995 + length)]
    }),
  )

  for (const length of WORD_BOUNDARY_LENGTHS) {
    const data = boundary.get(length) ?? []
    measure(`${length} chars, Levenshtein`, () => {
      for (const [a, b] of data) levenshteinDistance(a, b)
    })
  }
})

// The same crossover for alignment recovery, which has a one-word matrix of its
// own and so changes shape at the same place for a different reason.
describe('word-width boundaries, editops', () => {
  const around = [31, 32, 33]
  const boundary = new Map(
    around.map((length) => [
      length,
      editedPairs(Math.round(6000 / length), length, 2, 0x5bd1_e995 + length),
    ]),
  )

  for (const length of around) {
    const data = boundary.get(length) ?? []
    measure(`${length} chars, Levenshtein`, () => {
      for (const [a, b] of data) levenshteinEditops(a, b)
    })
  }
})

describe('editops, Latin-1 alphabet', () => {
  const shortWide = similarPairs(200, 8, 0.15, 0x9e37_79b9, LATIN1)
  const mediumWide = similarPairs(200, 32, 0.15, 0x9e37_79b9, LATIN1)

  measure('8 chars, similar, Levenshtein', () => {
    for (const [a, b] of shortWide) levenshteinEditops(a, b)
  })
  measure('8 chars, similar, LCS', () => {
    for (const [a, b] of shortWide) lcsSeqEditops(a, b)
  })
  measure('32 chars, similar, Levenshtein', () => {
    for (const [a, b] of mediumWide) levenshteinEditops(a, b)
  })
  measure('32 chars, similar, LCS', () => {
    for (const [a, b] of mediumWide) lcsSeqEditops(a, b)
  })
})

// The three cheapest scorers in the package. They matter here out of
// proportion to their cost: the whole operation is a single scan, so anything
// wrapped around one — a closure, an extra `Math.max`, a subtraction — is a
// measurable fraction of it rather than noise against a kernel. Nothing
// covered these before, which meant a shared distance-family helper could have
// regressed them silently.
describe('hammingDistance', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) hammingDistance(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) hammingDistance(a, b)
  })
  measure('1024 chars, similar', () => {
    for (const [a, b] of veryLong) hammingDistance(a, b)
  })
})

describe('hammingNormalizedSimilarity', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) hammingNormalizedSimilarity(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) hammingNormalizedSimilarity(a, b)
  })
})

describe('prefixSimilarity', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) prefixSimilarity(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) prefixSimilarity(a, b)
  })
  // 256 shared elements before the two sides part, which is the shape the
  // scan actually costs something on.
  measure('1024 vs 256 chars, long shared prefix', () => {
    for (const [a, b] of lengthSkewed) prefixSimilarity(a, b)
  })
})

describe('prefixNormalizedSimilarity', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) prefixNormalizedSimilarity(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) prefixNormalizedSimilarity(a, b)
  })
})

describe('postfixSimilarity', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) postfixSimilarity(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) postfixSimilarity(a, b)
  })
  measure('1024 vs 256 chars, long shared suffix', () => {
    for (const [a, b] of suffixSkewed) postfixSimilarity(a, b)
  })
})

describe('postfixNormalizedSimilarity', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) postfixNormalizedSimilarity(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) postfixNormalizedSimilarity(a, b)
  })
  measure('1024 vs 256 chars, long shared suffix', () => {
    for (const [a, b] of suffixSkewed) postfixNormalizedSimilarity(a, b)
  })
})
