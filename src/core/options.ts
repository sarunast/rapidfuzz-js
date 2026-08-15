export function assertOptionKeys(
  given: unknown,
  known: readonly string[],
  label: string,
): void {
  if (typeof given !== 'object' || given === null) {
    throw new TypeError(`${label} options must be an object`)
  }
  for (const key of Object.keys(given)) {
    if (!known.includes(key)) {
      throw new TypeError(`unknown ${label} option '${key}'`)
    }
  }
}
