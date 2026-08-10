import {
  alignRepresentation,
  conv,
  normDistCutoff,
  normSimCutoff,
  type ScorerOptions,
  type Sequence,
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  type MaybeSequence,
  isNone,
  asSequence,
  isSequence,
  PREPARE_CHOICE,
  prepareScorerChoice,
  preparedScorerSequence,
  scorerSequence,
  type PrepareScorer,
  type PreparedScore,
  withPreparedFlags,
  type NormalizedScorer,
  type Scorer,
} from '../_common.js'
import { preparePattern, type PatternMask } from './_bitVector/index.js'

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
    if (low > high) continue
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
    patternLength,
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
  patternLength: number,
  textLength: number,
  pFlag: Uint32Array,
  tFlag: Uint32Array,
): number {
  const patternWords = (patternLength + 31) >>> 5
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
  let patternBits = patternWords === 0 ? 0 : pFlag[firstWord] & leading

  if (typeof pattern === 'string' && typeof text === 'string') {
    for (let word = firstWord; word < textWords; word++) {
      let bits = word === firstWord ? tFlag[word] & leading : tFlag[word]

      while (bits !== 0) {
        const lowest = bits & -bits
        bits ^= lowest
        const i = (word << 5) + 31 - Math.clz32(lowest)

        while (patternBits === 0) {
          patternWord++
          if (patternWord >= patternWords) return transpositions
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

      while (patternBits === 0) {
        patternWord++
        if (patternWord >= patternWords) return transpositions
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

    if (i >= bound) {
      low++
      firstMask = (low & 31) === 0 ? -1 : firstMask << 1
    }
    if (high + 1 < patternLength) {
      high++
      lastMask = (high & 31) === 0 ? 1 : (lastMask << 1) | 1
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

type PreparedJaroKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

export function prepareJaro(kind: PreparedJaroKind): PrepareScorer {
  return (query) => {
    const a = preparedScorerSequence(prepareScorerChoice(query))
    if (a === null) throw new TypeError('expected a sequence')
    const pattern = preparePattern(a, 0, a.length)

    const score: PreparedScore = (rawChoice, rawCutoff) => {
      if (isNone(rawChoice)) {
        if (kind === 'normalizedDistance') return 1
        if (kind === 'normalizedSimilarity') return 0
      }
      let b = preparedScorerSequence(rawChoice)
      if (b === null) {
        if (!isSequence(rawChoice)) {
          throw new TypeError('expected a string or an array-like sequence')
        }
        b = scorerSequence(rawChoice)
      }

      const similarityCutoff =
        kind === 'distance' || kind === 'normalizedDistance'
          ? rawCutoff === null
            ? 0
            : 1 - rawCutoff
          : (rawCutoff ?? 0)
      // The transposition pass compares the two sequences elementwise, so they
      // have to agree on how a character is spelled.
      const similarity = jaroSimilarityPrepared_(
        alignRepresentation(a, b),
        pattern,
        alignRepresentation(b, a),
        similarityCutoff,
      )

      // Jaro's score is normalised, so `distance` and `normalizedDistance` are
      // the same metric read the same way — and likewise the two similarities.
      switch (kind) {
        case 'distance':
        case 'normalizedDistance':
          return normDistCutoff(1 - similarity, rawCutoff)
        case 'similarity':
        case 'normalizedSimilarity':
          return normSimCutoff(similarity, rawCutoff)
      }
    }
    Object.defineProperty(score, PREPARE_CHOICE, { value: prepareScorerChoice })
    return score
  }
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
  const [a, b] = conv(s1, s2, options.processor)
  return normSimCutoff(
    jaroSimilarity_(a, b, options.scoreCutoff ?? 0),
    options.scoreCutoff,
  )
}

/**
 * Jaro distance, i.e. `1 - jaroSimilarity(s1, s2)`.
 *
 * If the distance is greater than `scoreCutoff`, `1` is returned. See
 * {@link jaroSimilarity} for why the clamp is the normalised one.
 */
function jaroDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  const cutoff = options.scoreCutoff
  return normDistCutoff(
    1 - jaroSimilarity_(a, b, cutoff == null ? 0 : 1 - cutoff),
    cutoff,
  )
}

/**
 * Jaro distance. Identical to {@link jaroDistance} — the metric is already
 * normalised into `[0, 1]`.
 *
 * If the normalised distance is greater than `scoreCutoff`, `1` is returned.
 */
function jaroNormalizedDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 1

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  const cutoff = options.scoreCutoff
  return normDistCutoff(
    1 - jaroSimilarity_(a, b, cutoff == null ? 0 : 1 - cutoff),
    cutoff,
  )
}

/**
 * Jaro similarity. Identical to {@link jaroSimilarity} — the metric is already
 * normalised into `[0, 1]`.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function jaroNormalizedSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  return normSimCutoff(
    jaroSimilarity_(a, b, options.scoreCutoff ?? 0),
    options.scoreCutoff,
  )
}

// Scorer flags let `process` tell distances from similarities.
export const jaroSimilarity: Scorer = /* @__PURE__ */ withPreparedFlags(
  jaroSimilarity_impl,
  SIMILARITY_FLAGS,
  prepareJaro('similarity'),
)
export const jaroDistance: Scorer = /* @__PURE__ */ withPreparedFlags(
  jaroDistance_impl,
  DISTANCE_FLAGS,
  prepareJaro('distance'),
)
export const jaroNormalizedDistance: NormalizedScorer = /* @__PURE__ */ withPreparedFlags(
  jaroNormalizedDistance_impl,
  NORMALIZED_DISTANCE_FLAGS,
  prepareJaro('normalizedDistance'),
)
export const jaroNormalizedSimilarity: NormalizedScorer =
  /* @__PURE__ */ withPreparedFlags(
    jaroNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareJaro('normalizedSimilarity'),
  )
