// Not ported from RapidFuzz — upstream has no equivalent, because upstream does
// not agree with itself here.
//
// The token scorers split on whitespace, and "whitespace" has three candidate
// answers. Verified against rapidfuzz 3.14.5 on 2026-08-08, comparing
// `rapidfuzz.fuzz` (the C++ extension, which is what `import rapidfuzz` gets)
// against `rapidfuzz.fuzz_py` (the pure-Python fallback), for both `str` and
// `list` input:
//
//   code point    str.isspace()  cpp-str    py-str    cpp-list  py-list
//   U+0085        True           no split   split     split     split
//   U+00A0        True           no split   split     split     split
//   other spaces  True           split      split     split     split
//   U+FEFF        False          no split   no split  no split  no split
//
// So the C++ *string* path is the sole outlier, disagreeing with its own list
// path and with Python on exactly two code points. That is a bug rather than a
// documented conversion, so this port follows Python and splits on both. These
// assertions exist to stop anyone "fixing" `isSpaceCodePoint` toward C++.
//
// Upstream has no answer at all for a multi-character element:
// `token_sort_ratio(['a', '  ', 'b'], ...)` does not split under C++ and raises
// `ValueError: chr() arg not in range(0x110000)` under the Python path.
//
// Separators are written as code points rather than as literal characters: most
// of them are invisible, and one of them would end the line.
import { describe, expect, it } from 'vitest'

import { fuzzTokenSortRatio } from './tokenSortRatio.js'

/** Splits iff the separator is whitespace: sorting then makes the two equal. */
function splitsOn(cp: number): boolean {
  const separator = String.fromCodePoint(cp)
  return fuzzTokenSortRatio(`b${separator}a`, `a${separator}b`) === 100
}

const SPLITS: ReadonlyArray<readonly [string, number]> = [
  ['U+0009 tab', 0x09],
  ['U+000A line feed', 0x0a],
  ['U+000D carriage return', 0x0d],
  ['U+001C file separator', 0x1c],
  ['U+001F unit separator', 0x1f],
  ['U+0020 space', 0x20],
  ['U+1680 ogham space mark', 0x1680],
  ['U+2000 en quad', 0x2000],
  ['U+2028 line separator', 0x2028],
  ['U+202F narrow no-break space', 0x202f],
  ['U+3000 ideographic space', 0x3000],
]

describe('token splitting follows Python str.isspace(), not the C++ string path', () => {
  it.each(SPLITS)('splits on %s', (_label, cp) => {
    expect(splitsOn(cp)).toBe(true)
  })

  // The two the C++ string path gets wrong. `str.isspace()` is True for both,
  // and C++'s own list path splits on them.
  it.each([
    ['U+0085 next line', 0x85],
    ['U+00A0 no-break space', 0xa0],
  ])('splits on %s, where the C++ string path does not', (_label, cp) => {
    expect(splitsOn(cp)).toBe(true)
  })

  it.each([
    ['U+FEFF zero-width no-break space', 0xfeff],
    ['U+200B zero-width space', 0x200b],
  ])('does not split on %s, which is not Python whitespace', (_label, cp) => {
    expect(splitsOn(cp)).toBe(false)
  })

  // `trim()` would answer two of these the other way round: it strips U+FEFF and
  // leaves U+001C. `isSpaceElement` goes through the same code-point table as the
  // numeric path so that the two cannot drift apart.
  it('answers a single-character element the same way as the code point', () => {
    for (const cp of [0x1c, 0x85, 0xa0, 0xfeff, 0x200b, 0x20]) {
      const separator = String.fromCodePoint(cp)
      const asString = fuzzTokenSortRatio(`b${separator}a`, `a${separator}b`)
      const asList = fuzzTokenSortRatio(['b', separator, 'a'], ['a', separator, 'b'])
      expect(asList).toBe(asString)
    }
  })

  it('treats an all-whitespace multi-character element as a separator', () => {
    expect(fuzzTokenSortRatio(['b', '  ', 'a'], ['a', '  ', 'b'])).toBe(100)
    const tabThenNextLine = String.fromCodePoint(0x09, 0x85)
    expect(
      fuzzTokenSortRatio(['b', tabThenNextLine, 'a'], ['a', tabThenNextLine, 'b']),
    ).toBe(100)
  })

  it('does not treat a partly-whitespace element as a separator', () => {
    expect(fuzzTokenSortRatio(['b', ' x ', 'a'], ['a', ' x ', 'b'])).toBeLessThan(100)
  })

  // An empty element holds no whitespace, so it is an ordinary element inside a
  // token rather than a separator. `splitSequence` drops empty *tokens* — runs
  // of whitespace — which is a different thing.
  it('does not treat an empty element as a separator', () => {
    expect(fuzzTokenSortRatio(['b', '', 'a'], ['a', '', 'b'])).toBeLessThan(100)
  })
})
