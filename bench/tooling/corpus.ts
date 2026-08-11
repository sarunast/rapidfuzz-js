/**
 * Deterministic corpora for the benchmarks.
 *
 * No `Math.random`: a benchmark whose input changes between runs cannot be
 * compared against a stored baseline, which is the whole point of `bench:compare`.
 * The generator below is a small xorshift seeded per call, so every run sees
 * byte-identical input.
 */

/**
 * xorshift32 — deterministic, and good enough to avoid accidental structure.
 *
 * Zero is its one fixed point: every shift and xor of zero is zero, so a corpus
 * seeded with it would be a run of the alphabet's first character rather than
 * random text. Nothing passes zero today; the guard is here so nothing can add
 * it and spend a while wondering why one benchmark reads so strangely.
 */
function rng(seed: number): () => number {
  let state = seed >>> 0
  if (state === 0) throw new RangeError('xorshift32 seed must be non-zero')

  return () => {
    state ^= state << 13
    state >>>= 0
    state ^= state >>> 17
    state ^= state << 5
    state >>>= 0
    return state / 0x1_0000_0000
  }
}

/**
 * Alphabets are arrays of characters rather than strings.
 *
 * Indexing a string picks a UTF-16 code unit, while `for (const ch of source)`
 * below walks code points — so a string alphabet holding anything astral would
 * generate lone surrogate halves and then iterate over something else again.
 * That is a poor way for the corpus of a library built around Unicode
 * semantics to fail, and an array of characters cannot fail that way at all.
 */
const LOWER: readonly string[] = [...'abcdefghijklmnopqrstuvwxyz']

/**
 * Lowercase plus accented Latin-1, which spans from `a` to `ÿ`. The alignment
 * matrices index their match masks over the range an input's alphabet covers,
 * so this is the widest such table ordinary text builds — everything higher is
 * far enough away that they stop building one at all.
 */
export const LATIN1: readonly string[] = [...LOWER, ...'éàüñçÿ']

function draw(next: () => number, alphabet: readonly string[]): string {
  return alphabet[Math.floor(next() * alphabet.length)]
}

/**
 * A character from `alphabet` that is not the one already there.
 *
 * Drawing freely would let a substitution replace a character with itself,
 * which is one attempt in 26 over {@link LOWER} doing nothing at all. That
 * matters more than the rate suggests: the pairs it silently leaves unedited
 * are short ones, and a shorter input is likelier to come out identical and
 * take the equality shortcut rather than the kernel the case exists to measure.
 */
function replacement(
  next: () => number,
  alphabet: readonly string[],
  current: string,
): string {
  if (alphabet.length < 2) {
    throw new RangeError('alphabet needs two characters for a substitution')
  }

  let candidate = draw(next, alphabet)
  while (candidate === current) candidate = draw(next, alphabet)
  return candidate
}

function word(
  next: () => number,
  length: number,
  alphabet: readonly string[] = LOWER,
): string {
  let out = ''
  for (let i = 0; i < length; i++) out += draw(next, alphabet)
  return out
}

/** `count` random words of exactly `length` characters. */
export function words(count: number, length: number, seed = 0x2545_f491): string[] {
  const next = rng(seed)
  const out: string[] = []
  for (let i = 0; i < count; i++) out.push(word(next, length))
  return out
}

/**
 * The lengths either side of a machine word, where the kernels change shape.
 *
 * The bit-parallel kernels pack a pattern into 32-bit words — `words =
 * (patternLength + 31) >>> 5` — and a pattern of 32 or fewer takes a separate
 * one-word kernel entirely. So 32 and 33 are two different implementations, and
 * 64/65 and 128/129 are one more word of inner loop. Nothing else in the suite
 * sits near them: its lengths are 8, 32, 128, 1024, and a regression that lives
 * only at the crossover disappears into the gaps between those.
 *
 * Shared so that every case measuring a crossover measures the same one.
 */
export const WORD_BOUNDARY_LENGTHS: readonly number[] = [
  31, 32, 33, 63, 64, 65, 127, 128, 129,
]

/**
 * `count` pairs differing in exactly `edits` positions, each an equal length.
 *
 * {@link similarPairs} draws its edits at random, so what it produces is a
 * distribution: realistic, and the right thing for asking whether the library
 * got faster. It is the wrong thing for asking where a kernel's cost changes,
 * because the pair at length 32 and the pair at length 33 differ by more than
 * their length, and two lengths that should be one edit apart are not.
 *
 * These are substitutions only, so both sides are exactly `length` characters
 * and the distance between them is at most `edits` — which is what makes a
 * length sweep read as a length sweep, and a cutoff mean the same thing at
 * every point in it.
 *
 * ## The first and last positions are always among the edits
 *
 * Not an accident of the draw. Every scorer here trims the common prefix and
 * suffix before a kernel sees anything, so a pair differing only in the middle
 * is handed over as its middle: two 129-character strings with two edits
 * between them become a four-character problem, and a sweep meant to show what
 * each width costs measures the affix scan at every point instead. Measured,
 * that sweep was flat across 32/33 and 64/65 — the crossovers it existed to
 * find — because the kernel never ran at the width in the name.
 *
 * Pinning both ends leaves nothing to trim. It needs `edits >= 2`; a single
 * edit cannot avoid trimming, because whatever position it lands on is a common
 * prefix and a common suffix either side of it.
 */
export function editedPairs(
  count: number,
  length: number,
  edits: number,
  seed = 0x5bd1_e995,
  alphabet: readonly string[] = LOWER,
): Array<[string, string]> {
  if (edits > length) {
    throw new RangeError(`cannot make ${edits} edits in ${length} characters`)
  }
  if (edits < 2) {
    throw new RangeError('needs two edits to pin both ends and leave nothing to trim')
  }

  const next = rng(seed)
  const out: Array<[string, string]> = []

  for (let i = 0; i < count; i++) {
    const source = word(next, length, alphabet)
    const characters = [...source]

    // The two ends first, then the interior. A partial Fisher-Yates supplies
    // the rest: after `edits - 2` rounds the front of `interior` holds that
    // many distinct indices, uniformly drawn. Picking indices and rejecting
    // repeats would do the same thing, but would consume a number of draws
    // that depends on which ones came up — and a corpus whose generator
    // advances unpredictably is one no seed can reproduce.
    const chosen = [0, length - 1]
    const interior = characters.slice(1, -1).map((_, index) => index + 1)
    for (let k = 0; k < edits - 2; k++) {
      const pick = k + Math.floor(next() * (interior.length - k))
      const held = interior[k]
      interior[k] = interior[pick]
      interior[pick] = held
      chosen.push(interior[k])
    }

    for (const at of chosen) {
      characters[at] = replacement(next, alphabet, characters[at])
    }

    out.push([source, characters.join('')])
  }

  return out
}

/**
 * `count` pairs that are mostly similar — a copy of the source with roughly
 * `editRate` of its characters substituted, inserted or deleted.
 *
 * Near-identical inputs are the realistic case for fuzzy matching, and they
 * exercise different branches than random pairs do: common-affix trimming and
 * early exits only pay off when the inputs actually resemble each other.
 *
 * Whether a given pair gets any edit at all is left to chance rather than
 * forced. At eight characters and a 15% rate, around a quarter come out
 * identical — which is not a flaw to correct, because real candidate lists
 * contain identical and near-identical entries and the shortcuts that answer
 * them are part of what is being measured. What *is* corrected is a
 * substitution that only pretends to be one: see {@link replacement}.
 */
export function similarPairs(
  count: number,
  length: number,
  editRate = 0.15,
  seed = 0x9e37_79b9,
  alphabet: readonly string[] = LOWER,
): Array<[string, string]> {
  const next = rng(seed)
  const out: Array<[string, string]> = []

  for (let i = 0; i < count; i++) {
    const source = word(next, length, alphabet)
    let mutated = ''

    for (const ch of source) {
      if (next() >= editRate) {
        mutated += ch
        continue
      }

      // Each branch draws its own replacement, rather than one being drawn
      // ahead of the branch: a deletion draws nothing, and moving where the
      // generator is consumed would change every corpus this file produces.
      const kind = Math.floor(next() * 3)
      if (kind === 0) mutated += replacement(next, alphabet, ch)
      else if (kind === 1) mutated += ch + draw(next, alphabet)
      // kind === 2 deletes the character.
    }

    out.push([source, mutated])
  }

  return out
}

/** `count` sentences of `wordCount` words each, as `fuzz`'s token scorers see them. */
export function sentences(
  count: number,
  wordCount: number,
  seed = 0x1234_5678,
): string[] {
  const next = rng(seed)
  const out: string[] = []

  for (let i = 0; i < count; i++) {
    const parts: string[] = []
    for (let j = 0; j < wordCount; j++) {
      parts.push(word(next, 3 + Math.floor(next() * 6)))
    }
    out.push(parts.join(' '))
  }

  return out
}

/** Consecutive pairs of `items`, so a benchmark body has no index arithmetic. */
export function pairs(items: readonly string[]): Array<[string, string]> {
  const out: Array<[string, string]> = []
  for (let i = 1; i < items.length; i++) {
    const a = items[i - 1]
    const b = items[i]
    if (a !== undefined && b !== undefined) out.push([a, b])
  }
  return out
}
