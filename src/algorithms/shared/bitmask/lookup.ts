export const DIRECT_LOOKUP_LIMIT = 256

export function isDirectSymbol(symbol: unknown): symbol is number {
  return (
    typeof symbol === 'number' &&
    symbol >= 0 &&
    symbol < DIRECT_LOOKUP_LIMIT &&
    (symbol | 0) === symbol
  )
}

export function isHighSymbol(symbol: unknown): symbol is number {
  return (
    typeof symbol === 'number' && symbol >= DIRECT_LOOKUP_LIMIT && (symbol | 0) === symbol
  )
}
