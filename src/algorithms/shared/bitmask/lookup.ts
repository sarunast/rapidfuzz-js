const DIRECT_LOOKUP_LIMIT = 256

/** Whether a symbol directly indexes the fixed Latin-1 mask region. */
export function isDirectSymbol(symbol: unknown): symbol is number {
  return (
    typeof symbol === 'number' &&
    symbol >= 0 &&
    symbol < DIRECT_LOOKUP_LIMIT &&
    (symbol | 0) === symbol
  )
}

/** Whether a symbol is an integer eligible for a prepared high-symbol window. */
export function isHighSymbol(symbol: unknown): symbol is number {
  return (
    typeof symbol === 'number' && symbol >= DIRECT_LOOKUP_LIMIT && (symbol | 0) === symbol
  )
}
