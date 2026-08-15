export function validGramSize(value: unknown): number {
  if (value == null) return 2
  if (typeof value !== 'number') throw new TypeError('gramSize must be a number')
  if (!(Number.isSafeInteger(value) && value >= 1)) {
    throw new RangeError('gramSize has to be a safe integer of at least 1')
  }
  return value
}

export function parseGramSize(options: Readonly<Record<string, unknown>>): number {
  return validGramSize(Reflect.get(options, 'gramSize'))
}
