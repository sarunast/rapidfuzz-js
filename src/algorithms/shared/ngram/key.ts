const RADIX_LADDER: readonly number[] = [0x100, 0x1_0000, 0x11_0000]

export function feasibleRadices(gramSize: number): readonly number[] {
  return RADIX_LADDER.filter(
    (radix) => Math.pow(radix, gramSize) <= Number.MAX_SAFE_INTEGER,
  )
}

export function canonicalRadix(gramSize: number): number | null {
  if (gramSize <= 2) return 0x11_0000
  if (gramSize === 3) return 0x1_0000
  if (gramSize <= 6) return 0x100
  return null
}

export function packGram(
  digits: ArrayLike<number>,
  start: number,
  gramSize: number,
  radix: number,
): number {
  let key = 0
  for (let offset = 0; offset < gramSize; offset++)
    key = key * radix + digits[start + offset]
  return key
}

export function unpackGram(
  key: number,
  gramSize: number,
  radix: number,
  digits: number[],
): void {
  let rest = key
  for (let offset = gramSize - 1; offset >= 0; offset--) {
    digits[offset] = rest % radix
    rest = Math.floor(rest / radix)
  }
}
