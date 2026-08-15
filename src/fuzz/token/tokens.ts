import {
  convSequence,
  type ChoicePreparer,
  type Sequence,
} from '../../algorithms/shared/scorerSupport.js'

const SPACE = 32

function isSpaceCodePoint(cp: number): boolean {
  if (cp > 0x20) {
    if (cp < 0x85) return false

    return (
      cp === 0x85 ||
      cp === 0xa0 ||
      cp === 0x1680 ||
      (cp >= 0x2000 && cp <= 0x200a) ||
      cp === 0x2028 ||
      cp === 0x2029 ||
      cp === 0x202f ||
      cp === 0x205f ||
      cp === 0x3000
    )
  }

  return (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20)
}

function isSpaceElement(x: unknown): boolean {
  if (typeof x !== 'string' || x.length === 0) return false

  for (let i = 0; i < x.length; i++) {
    if (!isSpaceCodePoint(x.charCodeAt(i))) return false
  }
  return true
}

export function splitSequence(s: ArrayLike<unknown>): unknown[][] {
  const tokens: unknown[][] = []
  let current: unknown[] | null = null

  for (let i = 0; i < s.length; i++) {
    const element = s[i]
    const space =
      typeof element === 'number' ? isSpaceCodePoint(element) : isSpaceElement(element)

    if (space) {
      if (current !== null) {
        tokens.push(current)
        current = null
      }
    } else {
      if (current === null) current = []
      current.push(element)
    }
  }

  if (current !== null) tokens.push(current)
  return tokens
}

export function stringContainsWhitespace(s: string): boolean {
  for (let i = 0; i < s.length; i++) {
    if (isSpaceCodePoint(s.charCodeAt(i))) return true
  }
  return false
}

export function tokenForm(s: ArrayLike<unknown>): ArrayLike<unknown> {
  return typeof s === 'string' ? convSequence(s) : s
}

export function tokenPair(
  left: Sequence,
  right: Sequence,
): [ArrayLike<unknown>, ArrayLike<unknown>] {
  return [convSequence(left), convSequence(right)]
}

export function containsWhitespace(s: ArrayLike<unknown>): boolean {
  for (let i = 0; i < s.length; i++) {
    const element = s[i]
    if (
      typeof element === 'number' ? isSpaceCodePoint(element) : isSpaceElement(element)
    ) {
      return true
    }
  }
  return false
}

function joinedLength(tokens: readonly unknown[][]): number {
  if (tokens.length === 0) return 0

  let total = tokens.length - 1
  for (let i = 0; i < tokens.length; i++) total += tokens[i].length
  return total
}

export function joinTokens(
  tokens: readonly unknown[][],
  total: number = joinedLength(tokens),
): unknown[] {
  const separator = typeof tokens[0]?.[0] === 'string' ? ' ' : SPACE
  const out = new Array<unknown>(total)
  let n = 0

  for (let t = 0; t < tokens.length; t++) {
    if (n > 0) out[n++] = separator

    const token = tokens[t]
    for (let i = 0; i < token.length; i++) out[n++] = token[i]
  }

  return out
}

function typeOrder(x: unknown): number {
  if (x === null) return 1

  switch (typeof x) {
    case 'undefined':
      return 0
    case 'boolean':
      return 2
    case 'number':
      return 3
    case 'bigint':
      return 4
    case 'string':
      return 5
    case 'symbol':
      return 6
    default:
      return 7
  }
}

let identityOrdinals: WeakMap<object, number> | null = null
let nextIdentityOrdinal = 0

function isObjectLike(x: unknown): x is object {
  return (typeof x === 'object' && x !== null) || typeof x === 'function'
}

function identityOrdinal(x: object): number {
  const ordinals = (identityOrdinals ??= new WeakMap<object, number>())
  let ordinal = ordinals.get(x)
  if (ordinal === undefined) ordinals.set(x, (ordinal = ++nextIdentityOrdinal))
  return ordinal
}

function identityOrder(x: object, y: object): number {
  return identityOrdinal(x) - identityOrdinal(y)
}

let symbolOrdinals: Map<symbol, number> | null = null

function symbolIdentityOrder(x: symbol, y: symbol): number {
  const ordinals = (symbolOrdinals ??= new Map<symbol, number>())
  let left = ordinals.get(x)
  if (left === undefined) ordinals.set(x, (left = ++nextIdentityOrdinal))
  let right = ordinals.get(y)
  if (right === undefined) ordinals.set(y, (right = ++nextIdentityOrdinal))
  return left - right
}

function compareElements(x: unknown, y: unknown): number {
  if (isObjectLike(x)) {
    if (isObjectLike(y)) return identityOrder(x, y)
    return typeOrder(x) - typeOrder(y)
  }
  if (isObjectLike(y)) return typeOrder(x) - typeOrder(y)

  const byType = typeOrder(x) - typeOrder(y)
  if (byType !== 0) return byType

  if (typeof x === 'number' && typeof y === 'number') {
    if (Number.isNaN(x)) return Number.isNaN(y) ? 0 : 1
    return -1
  }
  if (typeof x === 'bigint' && typeof y === 'bigint') return x < y ? -1 : 1
  if (typeof x === 'string' && typeof y === 'string') return x < y ? -1 : 1
  if (typeof x === 'symbol' && typeof y === 'symbol') {
    const dx = String(x)
    const dy = String(y)
    if (dx !== dy) return dx < dy ? -1 : 1
    return symbolIdentityOrder(x, y)
  }

  return x ? 1 : -1
}

function compareTokens(a: readonly unknown[], b: readonly unknown[]): number {
  const limit = Math.min(a.length, b.length)

  for (let i = 0; i < limit; i++) {
    const x = a[i]
    const y = b[i]
    if (x === y) continue

    if (typeof x === 'number' && typeof y === 'number' && x === x && y === y) {
      return x - y
    }
    return compareElements(x, y)
  }

  return a.length - b.length
}

export function sortTokens(tokens: unknown[][]): unknown[][] {
  return tokens.sort(compareTokens)
}

const MAX_CODE_POINT = 0x10ffff
const BMP_KEY_PREFIX = '\u0011'

const BMP_KEY_PREFIX_CODE = 0x11

function isBmpToken(token: readonly unknown[]): token is readonly number[] {
  for (let i = 0; i < token.length; i++) {
    const x = token[i]
    if (typeof x !== 'number' || x < 0 || x > 0xffff || (x | 0) !== x) return false
  }
  return true
}

function tokenKey(token: readonly unknown[]): string {
  if (token.length <= 64 && isBmpToken(token)) {
    return BMP_KEY_PREFIX + String.fromCharCode(...token)
  }

  let key = ''

  for (let i = 0; i < token.length; i++) {
    const x = token[i]

    if (typeof x !== 'number' || x < 0 || x > MAX_CODE_POINT || (x | 0) !== x) {
      return mixedTokenKey(token)
    }

    key += String.fromCharCode(x >>> 16, x & 0xffff)
  }

  return key
}

function mixedTokenKey(token: readonly unknown[]): string {
  let key = ''

  for (let i = 0; i < token.length; i++) {
    if (i > 0) key += '\u0000'

    const x = token[i]
    key += isObjectLike(x)
      ? `${typeof x}:#${identityOrdinal(x)}`
      : `${typeof x}:${String(x)}`
  }

  return key
}

function equalTokens(a: readonly unknown[], b: readonly unknown[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

function isMixedTokenKey(key: string): boolean {
  return key.length > 0 && key.charCodeAt(0) > BMP_KEY_PREFIX_CODE
}

export class UniqueTokenSet {
  readonly packed: Map<string, unknown[]> = new Map<string, unknown[]>()
  readonly mixed: Map<string, unknown[][]> = new Map<string, unknown[][]>()
  size: number = 0

  add(key: string, token: unknown[]): void {
    if (!isMixedTokenKey(key)) {
      if (!this.packed.has(key)) {
        this.packed.set(key, token)
        this.size++
      }
      return
    }

    const bucket = this.mixed.get(key)
    if (bucket === undefined) {
      this.mixed.set(key, [token])
      this.size++
      return
    }

    for (let i = 0; i < bucket.length; i++) {
      if (equalTokens(bucket[i], token)) return
    }
    bucket.push(token)
    this.size++
  }

  has(key: string, token: readonly unknown[]): boolean {
    if (!isMixedTokenKey(key)) return this.packed.has(key)

    const bucket = this.mixed.get(key)
    if (bucket === undefined) return false
    for (let i = 0; i < bucket.length; i++) {
      if (equalTokens(bucket[i], token)) return true
    }
    return false
  }
}

export function uniqueTokens(tokens: readonly unknown[][]): UniqueTokenSet {
  const out = new UniqueTokenSet()

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]
    out.add(tokenKey(token), token)
  }

  return out
}

export class PreparedTokenChoice {
  split?: unknown[][]
  unique?: UniqueTokenSet
  sorted?: unknown[]
  hasWhitespace?: boolean
  canonicalLength?: number

  constructor(readonly sequence: ArrayLike<unknown>) {}
}

export function splitOf(choice: PreparedTokenChoice): unknown[][] {
  return (choice.split ??= splitSequence(choice.sequence))
}

export function uniqueOf(choice: PreparedTokenChoice): UniqueTokenSet {
  return (choice.unique ??= uniqueTokens(splitOf(choice)))
}

export function canonicalLengthOf(choice: PreparedTokenChoice): number {
  if (choice.canonicalLength !== undefined) return choice.canonicalLength
  if (choice.sorted !== undefined) return (choice.canonicalLength = choice.sorted.length)
  if (choice.split !== undefined) {
    return (choice.canonicalLength = joinedLength(choice.split))
  }

  return (choice.canonicalLength = canonicalLength(choice.sequence))
}

function canonicalLength(s: ArrayLike<unknown>): number {
  let elements = 0
  let tokens = 0
  let inToken = false

  for (let i = 0; i < s.length; i++) {
    const element = s[i]
    const space =
      typeof element === 'number' ? isSpaceCodePoint(element) : isSpaceElement(element)

    if (space) inToken = false
    else {
      if (!inToken) {
        tokens++
        inToken = true
      }
      elements++
    }
  }

  return tokens === 0 ? 0 : elements + tokens - 1
}

export function sortedOf(choice: PreparedTokenChoice): unknown[] {
  return (choice.sorted ??= joinTokens(sortTokens(splitOf(choice))))
}

export function hasWhitespaceOf(choice: PreparedTokenChoice): boolean {
  return (choice.hasWhitespace ??= containsWhitespace(choice.sequence))
}

export function tokenViewOf(sequence: ArrayLike<unknown>): PreparedTokenChoice {
  return new PreparedTokenChoice(sequence)
}
export function tokenChoicePreparer(): ChoicePreparer {
  return prepareTokenChoice
}

export function prepareTokenChoice(choice: Sequence): PreparedTokenChoice {
  return new PreparedTokenChoice(convSequence(choice))
}

export function preparedTokenChoice(value: unknown): PreparedTokenChoice {
  if (!(value instanceof PreparedTokenChoice)) {
    throw new TypeError('invalid prepared token choice')
  }

  return value
}

export function difference(a: UniqueTokenSet, b: UniqueTokenSet): unknown[][] {
  const out: unknown[][] = []

  for (const [key, token] of a.packed) if (!b.packed.has(key)) out.push(token)
  for (const [key, bucket] of a.mixed) {
    for (const token of bucket) if (!b.has(key, token)) out.push(token)
  }

  return out
}

export function intersects(a: UniqueTokenSet, b: UniqueTokenSet): boolean {
  const smaller = a.size <= b.size ? a : b
  const larger = smaller === a ? b : a

  for (const key of smaller.packed.keys()) if (larger.packed.has(key)) return true
  for (const [key, bucket] of smaller.mixed) {
    for (const token of bucket) if (larger.has(key, token)) return true
  }
  return false
}
