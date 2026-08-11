/**
 * Affix helpers over scorer representations.
 *
 * `commonPrefix` and `commonSuffix` are exact operations and require
 * element-compatible representations; callers align first. The probes are
 * routing heuristics, and both routes they choose between are correct.
 */
const AFFIX_PROBE_LIMIT = 32

/**
 * Mixed representations report an affix unconditionally. Probing them with
 * `charCodeAt` against elements was measured (2026-08-11) at 1.11x on prepared
 * Levenshtein over affixed mixed pairs against 0.97x on unrelated ones — the
 * false positive is the cheaper error, and it only costs the trimming route.
 */
function hasAffix(a: ArrayLike<unknown>, b: ArrayLike<unknown>, probe: number): boolean {
  if (probe === 0) return true
  if (typeof a === 'string') {
    if (typeof b !== 'string') return true
    let i = 0
    while (i < probe && a.charCodeAt(i) === b.charCodeAt(i)) i++
    if (i === probe) return true
    const lastA = a.length - 1
    const lastB = b.length - 1
    let j = 0
    while (j < probe && a.charCodeAt(lastA - j) === b.charCodeAt(lastB - j)) j++
    return j === probe
  }
  if (typeof b === 'string') return true
  let i = 0
  while (i < probe && a[i] === b[i]) i++
  if (i === probe) return true
  const lastA = a.length - 1
  const lastB = b.length - 1
  let j = 0
  while (j < probe && a[lastA - j] === b[lastB - j]) j++
  return j === probe
}

/** Whether at least an eighth of either end agrees, capped at 32 elements. */
export function sharesAffix(a: ArrayLike<unknown>, b: ArrayLike<unknown>): boolean {
  return hasAffix(a, b, Math.min(Math.min(a.length, b.length) >>> 3, AFFIX_PROBE_LIMIT))
}

/** Wider Levenshtein probe: a quarter of either end, capped at 64 elements. */
export function sharesWideAffix(a: ArrayLike<unknown>, b: ArrayLike<unknown>): boolean {
  return hasAffix(
    a,
    b,
    Math.min(Math.min(a.length, b.length) >>> 2, 2 * AFFIX_PROBE_LIMIT),
  )
}

/** Number of equal leading elements. */
export function commonPrefix(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  const limit = Math.min(s1.length, s2.length)
  let i = 0
  if (typeof s1 === 'string' && typeof s2 === 'string') {
    while (i < limit && s1.charCodeAt(i) === s2.charCodeAt(i)) i++
    return i
  }
  while (i < limit && s1[i] === s2[i]) i++
  return i
}

/**
 * Length of the shared prefix and suffix, the suffix measured after the prefix.
 *
 * Fused rather than `commonPrefix` beside `commonSuffix`: the suffix has to stop
 * at the prefix, and two independent calls both report 2 on `'aa'` against
 * `'aa'` — four positions trimmed from a sequence holding two.
 */
export function commonAffix(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
): { prefixLen: number; suffixLen: number } {
  const shorter = Math.min(s1.length, s2.length)
  const end1 = s1.length - 1
  const end2 = s2.length - 1

  let prefixLen = 0
  let suffixLen = 0

  // Comparing two strings position by position allocates a one-character string
  // per side per step; reading the code units compares integers instead. Both
  // inputs share a representation by the time they arrive, so the branch is
  // taken once rather than per position.
  if (typeof s1 === 'string' && typeof s2 === 'string') {
    while (prefixLen < shorter && s1.charCodeAt(prefixLen) === s2.charCodeAt(prefixLen)) {
      prefixLen++
    }
    while (
      suffixLen < shorter - prefixLen &&
      s1.charCodeAt(end1 - suffixLen) === s2.charCodeAt(end2 - suffixLen)
    ) {
      suffixLen++
    }

    return { prefixLen, suffixLen }
  }

  while (prefixLen < shorter && s1[prefixLen] === s2[prefixLen]) prefixLen++
  while (
    suffixLen < shorter - prefixLen &&
    s1[end1 - suffixLen] === s2[end2 - suffixLen]
  ) {
    suffixLen++
  }

  return { prefixLen, suffixLen }
}

/** Number of equal trailing elements. */
export function commonSuffix(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  const limit = Math.min(s1.length, s2.length)
  const end1 = s1.length - 1
  const end2 = s2.length - 1
  let i = 0
  if (typeof s1 === 'string' && typeof s2 === 'string') {
    while (i < limit && s1.charCodeAt(end1 - i) === s2.charCodeAt(end2 - i)) i++
    return i
  }
  while (i < limit && s1[end1 - i] === s2[end2 - i]) i++
  return i
}
