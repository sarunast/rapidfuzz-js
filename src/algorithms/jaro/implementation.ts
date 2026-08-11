import { preparePattern, type PatternMask } from '../shared/bitmask/pattern.js'
import {
  alignRepresentation,
  convPair,
  normSimCutoff,
  type ScorerOptions,
  type Sequence,
  NORMALIZED_SIMILARITY_FLAGS,
  withChoicePreparer,
  prepareScorerChoice,
  preparedScorerSequence,
  type PrepareScorer,
  type PreparedScorerFactory,
  type PreparedScore,
  withPreparedFlags,
  type Scorer,
} from '../shared/scorerSupport.js'

let patternFlags: Uint32Array | null = null
let textFlags: Uint32Array | null = null

/** Drop the retained flag buffers. Benchmark-only — see `resetSharedScratch`. */
export function resetJaroScratch(): void {
  patternFlags = null
  textFlags = null
}

function grown(buffer: Uint32Array | null, needed: number): Uint32Array {
  if (buffer !== null && buffer.length >= needed) return buffer
  let size = buffer === null ? 32 : buffer.length
  while (size < needed) size *= 2
  return new Uint32Array(size)
}

/**
 * Jaro similarity in `[0, 1]`.
 *
 * Two elements count as matching only if they are no further apart than
 * `floor(max(|s1|, |s2|) / 2) - 1` positions. Half-transpositions are counted
 * over the matched elements in order.
 */
export function jaroSimilarity_(
  pattern: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  scoreCutoff = 0,
): number {
  return jaroSimilarityCore(pattern, text, scoreCutoff)
}

/** Jaro similarity using a query mask retained by a process scorer. */
export function jaroSimilarityPrepared_(
  pattern: ArrayLike<unknown>,
  preparedPattern: PatternMask,
  text: ArrayLike<unknown>,
  scoreCutoff = 0,
): number {
  return jaroSimilarityCore(pattern, text, scoreCutoff, preparedPattern)
}

function jaroSimilarityCore(
  pattern: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  scoreCutoff: number,
  preparedPattern?: PatternMask,
): number {
  const lenP = pattern.length
  const lenT = text.length

  if (lenP === 0 && lenT === 0) return 1
  if (lenP === 0 || lenT === 0) return 0

  const minLength = Math.min(lenP, lenT)
  const ceiling = (minLength / lenP + minLength / lenT + 1) / 3
  if (ceiling < scoreCutoff) return 0

  const bound = Math.max(Math.floor(Math.max(lenP, lenT) / 2) - 1, 0)
  let patternEnd = lenP
  let textEnd = lenT
  if (lenT > lenP && lenT > lenP + bound) textEnd = lenP + bound
  else if (lenP > lenT && lenP > lenT + bound) patternEnd = lenT + bound
  let prefix = 0
  const activeShorter = Math.min(patternEnd, textEnd)
  if (typeof pattern === 'string' && typeof text === 'string') {
    while (
      prefix < activeShorter &&
      pattern.charCodeAt(prefix) === text.charCodeAt(prefix)
    ) {
      prefix++
    }
  } else {
    while (prefix < activeShorter && pattern[prefix] === text[prefix]) prefix++
  }

  // Where bit 0 of the match mask sits. A mask built here starts at the prefix,
  // so the prefix leaves the problem entirely. A mask retained by a process
  // scorer is indexed from the whole query and cannot be re-based, so instead
  // the prefix positions are marked as already taken and the scan starts past
  // them — the same state, without rebuilding the mask per candidate. That is
  // what lets a prepared Jaro profit from a shared prefix at all; before, it
  // rescanned every leading character of every candidate.
  const origin = preparedPattern === undefined ? prefix : 0
  const skip = prefix - origin
  const patternLength = patternEnd - origin
  const textLength = textEnd - origin
  const prepared = preparedPattern ?? preparePattern(pattern, prefix, patternLength)
  const patternWords = Math.max((patternLength + 31) >>> 5, 1)
  const textWords = Math.max((textLength + 31) >>> 5, 1)
  if (patternWords === 1 && textWords === 1) {
    return jaroOneWord(
      pattern,
      text,
      origin,
      prefix,
      patternLength,
      textLength,
      bound,
      prepared,
      lenP,
      lenT,
      scoreCutoff,
    )
  }
  let pFlag: Uint32Array
  let tFlag: Uint32Array
  patternFlags = grown(patternFlags, patternWords)
  textFlags = grown(textFlags, textWords)
  pFlag = patternFlags
  tFlag = textFlags
  pFlag.fill(0, 0, patternWords)
  tFlag.fill(0, 0, textWords)
  // Claim the prefix positions on the pattern side so a later text element
  // cannot match into them. The text side is left clear; the transposition pass
  // starts past the prefix on both sides instead.
  if (skip > 0) {
    const wholeWords = skip >>> 5
    pFlag.fill(0xffff_ffff, 0, wholeWords)
    const remainder = skip & 31
    if (remainder !== 0) pFlag[wholeWords] = (1 << remainder) - 1
  }
  let common = prefix
  const stringText = typeof text === 'string'
  const words = prepared.words
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

  for (let i = skip; i < textLength; i++) {
    const low = Math.max(0, i - bound)
    const high = Math.min(patternLength - 1, i + bound)
    const firstWord = low >>> 5
    const lastWord = high >>> 5
    const symbol = stringText ? text.charCodeAt(origin + i) : text[origin + i]
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < 256 &&
      (symbol | 0) === symbol
    ) {
      base = symbol * words
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * words
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }

    for (let word = firstWord; word <= lastWord; word++) {
      const matches = base < 0 ? 0 : masks[base + word]
      let available = matches & ~pFlag[word]
      if (word === firstWord) available &= -1 << (low & 31)
      if (word === lastWord && (high & 31) !== 31) {
        available &= (1 << ((high & 31) + 1)) - 1
      }
      if (available === 0) continue

      pFlag[word] |= available & -available
      tFlag[i >>> 5] |= 1 << (i & 31)
      common++
      break
    }
  }

  if (common === 0) return 0

  const commonCeiling = (common / lenP + common / lenT + 1) / 3
  if (commonCeiling < scoreCutoff) return 0

  const transpositions = countTranspositionsWords(
    pattern,
    text,
    origin,
    skip,
    textLength,
    pFlag,
    tFlag,
  )

  return finishJaro(lenP, lenT, common, transpositions, scoreCutoff)
}

/**
 * Half-transpositions between the matched positions of the two inputs.
 *
 * The matching phase already reads text through `charCodeAt`; this pass has to
 * read *both* sides, and doing it by index on a retained string allocates a
 * one-character string per side per matched position. Hoisting the
 * representation test out of the loop lets each variant compare integers or
 * values directly, with the branch paid once.
 */
function countTranspositionsWords(
  pattern: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  origin: number,
  skip: number,
  textLength: number,
  pFlag: Uint32Array,
  tFlag: Uint32Array,
): number {
  const textWords = (textLength + 31) >>> 5
  const firstWord = skip >>> 5
  // Both sides start past the prefix: those positions matched each other in
  // order and are equal by construction, so they contribute nothing and pairing
  // the remainder stays aligned. Masking the first word off at `skip` is also
  // what excludes the prefix bits claimed on the pattern side.
  const leading = -1 << (skip & 31)

  // Both sides are walked as cursors over their set bits, so a run of unmatched
  // positions costs nothing rather than a step each.
  let transpositions = 0
  let patternWord = firstWord
  let patternBits = pFlag[firstWord] & leading

  if (typeof pattern === 'string' && typeof text === 'string') {
    for (let word = firstWord; word < textWords; word++) {
      let bits = word === firstWord ? tFlag[word] & leading : tFlag[word]

      while (bits !== 0) {
        const lowest = bits & -bits
        bits ^= lowest
        const i = (word << 5) + 31 - Math.clz32(lowest)

        // No bound on `patternWord`: every match set one bit on each side, and
        // `leading` masks the same prefix positions off both, so the two words
        // hold the same number of set flags and the cursor cannot run out first.
        while (patternBits === 0) {
          patternWord++
          patternBits = pFlag[patternWord]
        }
        const patternLowest = patternBits & -patternBits
        patternBits ^= patternLowest
        const patternIndex = (patternWord << 5) + 31 - Math.clz32(patternLowest)

        if (text.charCodeAt(origin + i) !== pattern.charCodeAt(origin + patternIndex)) {
          transpositions++
        }
      }
    }

    return transpositions
  }

  for (let word = firstWord; word < textWords; word++) {
    let bits = word === firstWord ? tFlag[word] & leading : tFlag[word]

    while (bits !== 0) {
      const lowest = bits & -bits
      bits ^= lowest
      const i = (word << 5) + 31 - Math.clz32(lowest)

      // See the string path above: the two sides carry equally many set flags.
      while (patternBits === 0) {
        patternWord++
        patternBits = pFlag[patternWord]
      }
      const patternLowest = patternBits & -patternBits
      patternBits ^= patternLowest
      const patternIndex = (patternWord << 5) + 31 - Math.clz32(patternLowest)

      if (text[origin + i] !== pattern[origin + patternIndex]) transpositions++
    }
  }

  return transpositions
}

/** {@link countTranspositionsWords} for a pattern and text that each fit one word. */
function countTranspositionsOneWord(
  pattern: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  origin: number,
  skip: number,
  patternFlag: number,
  textFlag: number,
): number {
  const leading = -1 << (skip & 31)
  let transpositions = 0
  let bits = textFlag & leading
  let patternBits = patternFlag & leading

  if (typeof pattern === 'string' && typeof text === 'string') {
    while (bits !== 0 && patternBits !== 0) {
      const lowest = bits & -bits
      const patternLowest = patternBits & -patternBits
      bits ^= lowest
      patternBits ^= patternLowest

      if (
        text.charCodeAt(origin + 31 - Math.clz32(lowest)) !==
        pattern.charCodeAt(origin + 31 - Math.clz32(patternLowest))
      ) {
        transpositions++
      }
    }

    return transpositions
  }

  while (bits !== 0 && patternBits !== 0) {
    const lowest = bits & -bits
    const patternLowest = patternBits & -patternBits
    bits ^= lowest
    patternBits ^= patternLowest

    if (
      text[origin + 31 - Math.clz32(lowest)] !==
      pattern[origin + 31 - Math.clz32(patternLowest)]
    ) {
      transpositions++
    }
  }

  return transpositions
}

function finishJaro(
  patternLength: number,
  textLength: number,
  common: number,
  transpositions: number,
  scoreCutoff: number,
): number {
  const halfTranspositions = Math.floor(transpositions / 2)
  const result =
    (common / patternLength +
      common / textLength +
      (common - halfTranspositions) / common) /
    3
  return result >= scoreCutoff ? result : 0
}

function jaroOneWord(
  pattern: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  origin: number,
  prefix: number,
  patternLength: number,
  textLength: number,
  bound: number,
  prepared: PatternMask,
  fullPatternLength: number,
  fullTextLength: number,
  scoreCutoff: number,
): number {
  const skip = prefix - origin
  // See the multiword path: the prefix positions are claimed on the pattern
  // side, and the window state is opened where the scan starts rather than
  // walked there one position at a time.
  let patternFlag = skip >= 32 ? -1 : (1 << skip) - 1
  let textFlag = 0
  let common = prefix
  let low = Math.max(0, skip - bound)
  let high = Math.min(patternLength - 1, skip + bound)
  let firstMask = -1 << (low & 31)
  let lastMask = (high & 31) === 31 ? -1 : (1 << ((high & 31) + 1)) - 1
  const stringText = typeof text === 'string'
  const words = prepared.words
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

  for (let i = skip; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(origin + i) : text[origin + i]
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < 256 &&
      (symbol | 0) === symbol
    ) {
      base = symbol * words
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * words
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }
    const matches = base < 0 ? 0 : masks[base]
    const available = matches & ~patternFlag & firstMask & lastMask
    if (available !== 0) {
      patternFlag |= available & -available
      textFlag |= 1 << i
      common++
    }

    // Both masks shift without the wrap the multiword kernel needs, which moves
    // a word instead. This one serves a single word: `high` stops below
    // `patternLength <= 32`, and `low` rises at most to `textLength - bound`,
    // which for a `textLength` of at most 32 is 17. Neither reaches a multiple
    // of 32, and reopening the mask at one would be wrong here rather than
    // merely unnecessary — a window past the only word matches nothing, which
    // is what shifting the bits out already says.
    if (i >= bound) {
      low++
      firstMask <<= 1
    }
    if (high + 1 < patternLength) {
      high++
      lastMask = (lastMask << 1) | 1
    }
  }

  if (common === 0) return 0
  const commonCeiling = (common / fullPatternLength + common / fullTextLength + 1) / 3
  if (commonCeiling < scoreCutoff) return 0

  const transpositions = countTranspositionsOneWord(
    pattern,
    text,
    origin,
    skip,
    patternFlag,
    textFlag,
  )

  return finishJaro(
    fullPatternLength,
    fullTextLength,
    common,
    transpositions,
    scoreCutoff,
  )
}

export function prepareJaro(): PreparedScorerFactory {
  const prepare: PrepareScorer = (query) => {
    const a = preparedScorerSequence(prepareScorerChoice(query))
    const pattern = preparePattern(a, 0, a.length)

    const score: PreparedScore = (rawChoice, rawCutoff) => {
      const b = preparedScorerSequence(rawChoice)

      // The transposition pass compares the two sequences elementwise, so they
      // have to agree on how a character is spelled.
      const similarity = jaroSimilarityPrepared_(
        alignRepresentation(a, b),
        pattern,
        alignRepresentation(b, a),
        rawCutoff ?? 0,
      )
      return normSimCutoff(similarity, rawCutoff)
    }
    return score
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}

/**
 * Jaro similarity in `[0, 1]`, where `1` means identical.
 *
 * If the similarity is smaller than `scoreCutoff`, `0` is returned.
 *
 * Jaro's score is already normalised, so upstream reads `score_cutoff` as a
 * `double` in `[0, 1]` here exactly as it does for `normalized_similarity` —
 * there is no raw, element-counting score for it to bound. Hence the normalised
 * clamp: a fractional cutoff is meaningful rather than truncated, one outside
 * `[0, 1]` is refused, and a rejected pair reports the worst score rather than
 * `scoreCutoff + 1`.
 */
function jaroSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  return normSimCutoff(
    jaroSimilarity_(a, b, options.scoreCutoff ?? 0),
    options.scoreCutoff,
  )
}

export const jaroSimilarity: Scorer = /* @__PURE__ */ withPreparedFlags(
  jaroSimilarity_impl,
  NORMALIZED_SIMILARITY_FLAGS,
  prepareJaro(),
)
