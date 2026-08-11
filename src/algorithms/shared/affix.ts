const AFFIX_PROBE_LIMIT = 32

function hasAffix(a: ArrayLike<unknown>, b: ArrayLike<unknown>, probe: number): boolean {
  const lastA = a.length - 1
  const lastB = b.length - 1
  if (typeof a === 'string') {
    if (typeof b !== 'string') return true
    let i = 0
    while (i < probe && a.charCodeAt(i) === b.charCodeAt(i)) i++
    if (i === probe) return true
    let j = 0
    while (j < probe && a.charCodeAt(lastA - j) === b.charCodeAt(lastB - j)) j++
    return j === probe
  }
  if (typeof b === 'string') return true
  let i = 0
  while (i < probe && a[i] === b[i]) i++
  if (i === probe) return true
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
