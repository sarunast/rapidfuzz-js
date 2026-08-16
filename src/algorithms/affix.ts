// The probes are dispatch heuristics, not affix predicates: they read only
// enough leading and trailing elements to decide whether affix trimming may
// beat a prepared kernel, so false positives are intentional — a pair below
// the probe minimum passes unconditionally. They answer without converting or
// allocating anything. Callers provide scorer-normalized representations: BMP
// strings remain strings, character elements in non-string sequences are their
// numeric code points, and other elements retain their identity — which is why
// a `charCodeAt` compares directly against an element.
const AFFIX_PROBE_LIMIT = 32

function stringStringAffix(a: string, b: string, probe: number): boolean {
  let i = 0
  while (i < probe && a.charCodeAt(i) === b.charCodeAt(i)) i++
  if (i === probe) return true
  const lastA = a.length - 1
  const lastB = b.length - 1
  let j = 0
  while (j < probe && a.charCodeAt(lastA - j) === b.charCodeAt(lastB - j)) j++
  return j === probe
}

function stringArrayAffix(a: string, b: ArrayLike<unknown>, probe: number): boolean {
  let i = 0
  while (i < probe && a.charCodeAt(i) === b[i]) i++
  if (i === probe) return true
  const lastA = a.length - 1
  const lastB = b.length - 1
  let j = 0
  while (j < probe && a.charCodeAt(lastA - j) === b[lastB - j]) j++
  return j === probe
}

function arrayArrayAffix(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  probe: number,
): boolean {
  let i = 0
  while (i < probe && a[i] === b[i]) i++
  if (i === probe) return true
  const lastA = a.length - 1
  const lastB = b.length - 1
  let j = 0
  while (j < probe && a[lastA - j] === b[lastB - j]) j++
  return j === probe
}

function hasAffix(a: ArrayLike<unknown>, b: ArrayLike<unknown>, probe: number): boolean {
  if (probe === 0) return true
  if (typeof a === 'string') {
    return typeof b === 'string'
      ? stringStringAffix(a, b, probe)
      : stringArrayAffix(a, b, probe)
  }
  return typeof b === 'string'
    ? stringArrayAffix(b, a, probe)
    : arrayArrayAffix(a, b, probe)
}

export function passesAffixProbe(a: ArrayLike<unknown>, b: ArrayLike<unknown>): boolean {
  return hasAffix(a, b, Math.min(Math.min(a.length, b.length) >>> 3, AFFIX_PROBE_LIMIT))
}

export function passesWideAffixProbe(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
): boolean {
  return hasAffix(
    a,
    b,
    Math.min(Math.min(a.length, b.length) >>> 2, 2 * AFFIX_PROBE_LIMIT),
  )
}

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

export function commonAffix(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
): { prefixLen: number; suffixLen: number } {
  const shorter = Math.min(s1.length, s2.length)
  const end1 = s1.length - 1
  const end2 = s2.length - 1

  let prefixLen = 0
  let suffixLen = 0

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
