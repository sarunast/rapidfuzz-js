// Not ported from RapidFuzz — these are edges the upstream suite does not
// cover, but whose answers were taken from it. Expected values verified against
// rapidfuzz 3.14.5 (the C++ path) on 2026-08-08:
//
//   partial_ratio('', 'x'*200)            -> 0.0
//   partial_ratio_alignment('', 'x'*200)  -> (0.0, 0, 0, 0, 0)
//   partial_ratio('', '')                 -> 100.0
//   partial_ratio_alignment('', '')       -> (100.0, 0, 0, 0, 0)
//   partial_ratio('x'*200, '')            -> 0.0
//
// The empty needle is the one that had teeth. `partialRatioScan` sizes its
// interior search from the haystack, so an empty pattern against a long text
// used to allocate a `Uint32Array` as long as the text and bisect it, computing
// `1 - distance / (2 * len1)` as `0 / 0` at every step. The resulting `NaN`
// failed every comparison, so the score came out right by accident. The guard in
// `partialRatioScan` makes it right on purpose, and these assertions keep it so.
import { describe, expect, it } from 'vitest'

import { partialRatio, partialRatioAlignment, ratio } from '../src/fuzz.js'

const LONG = 'x'.repeat(200)

describe('partialRatio with an empty input', () => {
  it('scores an empty needle against a long haystack as 0', () => {
    expect(partialRatio('', LONG)).toBe(0)
    expect(partialRatio(LONG, '')).toBe(0)
  })

  it('reports the degenerate alignment for an empty needle', () => {
    expect(partialRatioAlignment('', LONG)).toEqual({
      score: 0,
      srcStart: 0,
      srcEnd: 0,
      destStart: 0,
      destEnd: 0,
    })
  })

  it('scores two empty inputs as a perfect match', () => {
    expect(partialRatio('', '')).toBe(100)
    expect(partialRatioAlignment('', '')).toEqual({
      score: 100,
      srcStart: 0,
      srcEnd: 0,
      destStart: 0,
      destEnd: 0,
    })
  })

  // The interior search only engages past 64 windows, so this is the length that
  // would have taken the bisection with a `NaN` bound.
  it('does not depend on how long the haystack is', () => {
    for (const length of [1, 63, 64, 65, 200, 5000]) {
      expect(partialRatio('', 'y'.repeat(length))).toBe(0)
    }
  })

  it('still answers null when a cutoff rejects the empty alignment', () => {
    expect(partialRatioAlignment('', LONG, { scoreCutoff: 1 })).toBeNull()
  })
})

// The window scan skips windows it can prove cannot reach the running cutoff:
// a window of `m` elements against a needle of `n` scores at most `2m / (n + m)`,
// so once that ceiling falls below the cutoff the suffix scan can stop and the
// prefix scan can start late. The prune is a claim about which windows *cannot*
// win, and the only way it can be wrong is by skipping one that could — which no
// assertion on a single score would show, since the answer would simply be a
// different, lower number that still looks like a score.
//
// So the oracle here is the window set itself: `ratio` against every window the
// scan is defined over, maximised. It shares no code with the scan.
describe('the window scan against every window, scored directly', () => {
  const windowsOf = (needle: string, haystack: string): string[] => {
    const out: string[] = []
    for (let i = 1; i < needle.length; i++) out.push(haystack.slice(0, i))
    for (let i = 0; i <= haystack.length - needle.length; i++) {
      out.push(haystack.slice(i, i + needle.length))
    }
    for (let i = haystack.length - needle.length; i < haystack.length; i++) {
      out.push(haystack.slice(i))
    }
    return out.filter((window) => window.length > 0)
  }

  // `partialRatio` scores the shorter input as the needle whichever way round it
  // is given, so the oracle has to make the same choice before enumerating.
  const bestWindow = (s1: string, s2: string): number => {
    const needle = s1.length <= s2.length ? s1 : s2
    const haystack = needle === s1 ? s2 : s1
    let best = 0
    for (const window of windowsOf(needle, haystack)) {
      const score = ratio(needle, window)
      if (score > best) best = score
    }
    return best
  }

  // A needle and haystacks at every relation to it the scan branches on: equal
  // length (no interior windows at all), a few interior windows, and enough to
  // take the bisection. Edit distance 1 and 2 from the needle are the shapes
  // where a good full-length window raises the cutoff far enough to prune most
  // of the rest, which is exactly where a wrong prune would show.
  const NEEDLE = 'abcdefghijklmnopqrstuvwxyz'
  const HAYSTACKS = [
    NEEDLE,
    'abcdefghijklMnopqrstuvwxyz',
    'abcdefghijklMnopqrsPuvwxyz',
    'zyxwvutsrqponmlkjihgfedcba',
    `qq${NEEDLE}qq`,
    `${'q'.repeat(80)}${NEEDLE}${'q'.repeat(80)}`,
    `${'q'.repeat(80)}abcdefghijklMnopqrstuvwxyz${'q'.repeat(80)}`,
    'abc',
    'a'.repeat(26),
  ]

  it('finds the same best window as scoring all of them', () => {
    for (const haystack of HAYSTACKS) {
      expect(partialRatio(NEEDLE, haystack)).toBeCloseTo(bestWindow(NEEDLE, haystack), 10)
    }
  })

  // Every cutoff the prune could act on, including ones landing exactly on the
  // answer: a cutoff at or below the best score must not change it, and one
  // above must reject. `minimumWindow` floors its estimate for this reason —
  // rounding up could skip the window that meets the cutoff exactly.
  it('answers the same score under every cutoff it should pass', () => {
    for (const haystack of HAYSTACKS) {
      const best = partialRatio(NEEDLE, haystack)
      for (const cutoff of [0, 1, 25, 50, 75, 90, 99, best, best + 1e-9, 100]) {
        const scored = partialRatio(NEEDLE, haystack, { scoreCutoff: cutoff })
        if (cutoff <= best) expect(scored).toBeCloseTo(best, 10)
        else expect(scored).toBe(0)
      }
    }
  })

  // The prune runs on the alignment path too, where it must not move the window
  // that gets reported — only a strictly better window replaces the one held, so
  // a skipped window that ties would be invisible in the score and visible here.
  it('reports the same alignment with and without a cutoff it passes', () => {
    for (const haystack of HAYSTACKS) {
      const plain = partialRatioAlignment(NEEDLE, haystack)
      expect(plain).not.toBeNull()
      if (plain === null) continue
      expect(
        partialRatioAlignment(NEEDLE, haystack, { scoreCutoff: plain.score }),
      ).toEqual(plain)
    }
  })
})
