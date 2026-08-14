/**
 * The token engine: splitting, ordering, hashing, and the prepared-choice
 * records the token scorers read.
 *
 * Port of `_split_sequence` / `_join_splitted_sequence`, plus the machinery
 * around them that upstream gets from Python's `set` and tuple ordering for
 * free.
 *
 * ## Why this is one module and not five
 *
 * Splitting, the total order, the key packing and {@link UniqueTokenSet} are one
 * performance unit and one set of semantics. `tokenSetRatioConverted` walks a
 * set's two maps directly rather than through an interface, and every piece here
 * has to agree about what a generic JavaScript element *is* — how it splits, how
 * it orders, how it hashes, and now how its identity is spelled: for an object
 * or a function, {@link mixedTokenKey} and the sort order read the same
 * {@link identityOrdinal}. A symbol is the exception on both counts — it hashes
 * through `String(x)`, which two symbols of one description share, and it orders
 * through {@link symbolIdentityOrder}'s own table, because a `WeakMap` cannot key
 * one under this target's lib. Sharing a hash costs nothing there: a mixed key is
 * a bucket, and {@link equalTokens} separates its occupants. A boundary between
 * any two of these pieces would run through the middle of that agreement.
 *
 * `nextIdentityOrdinal` is shared between the object and symbol tiebreaks, but
 * that is convenience rather than necessity: {@link typeOrder} separates the two
 * kinds before either tiebreak is reached, so independent counters would order
 * everything identically. What cannot be separated is a counter from the map it
 * fills.
 *
 * Derived forms are *not* built here in one pass. {@link tokenChoicePreparer}
 * converts the sequence and stops; each form is left to the accessor that asks
 * for it, which is what lets a scoring branch pay only for what it reads. See
 * {@link PreparedTokenChoice}.
 *
 * Nothing here imports a public fuzz family: `../partialWindow.ts` is a peer,
 * not a dependency, and the basic and partial similarities must stay usable
 * without tokenising anything.
 */
import {
  convSequence,
  type ChoicePreparer,
  type Sequence,
} from '../../algorithms/shared/scorerSupport.js'

const SPACE = 32

/**
 * Whitespace per `str.isspace()`, over code points.
 *
 * The `0x21..0x84` reject comes first because that is where nearly every
 * character in real text lands, and it settles the question in two comparisons
 * instead of walking the whole chain.
 */
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

/**
 * Whitespace test for an element that is not a code point.
 *
 * Scored through {@link isSpaceCodePoint} rather than `trim()`, so that one
 * table answers the question however the element is spelled. JavaScript's
 * `trim()` is a different set from Python's `str.isspace()` in both directions:
 * it strips U+FEFF, which Python does not call whitespace, and it leaves
 * U+001C-U+001F, which Python does.
 *
 * Only reachable for the empty string and for elements of more than one code
 * point — sequence conversion has already turned every single-code-point
 * string into its code point by the time `splitSequence` runs, which is why
 * `'a\u0085b'` and `['a', '\u0085', 'b']` tokenise the same way either way.
 * Upstream has no answer to compare against here: C++ declines to split a
 * multi-character element and the pure-Python path raises `ValueError` on one.
 *
 * All of Python's whitespace is inside the BMP, so `charCodeAt` is enough.
 */
function isSpaceElement(x: unknown): boolean {
  if (typeof x !== 'string' || x.length === 0) return false

  for (let i = 0; i < x.length; i++) {
    if (!isSpaceCodePoint(x.charCodeAt(i))) return false
  }
  return true
}

/**
 * Split on runs of whitespace, dropping empty tokens.
 *
 * The code-point test is inlined rather than left to {@link isSpaceElement}:
 * this runs once per element of every input the token scorers see, and a
 * converted string is entirely numbers, so the branch is perfectly predicted.
 */
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

/**
 * {@link containsWhitespace} for a string, and deliberately not a branch inside
 * it.
 *
 * `s[i]` on a string allocates a one-character string per element, which is the
 * whole cost of the test on text holding no whitespace — the answer this is
 * asked for most. `charCodeAt` allocates nothing, and a code unit is enough:
 * all of Python's whitespace is inside the BMP, so neither half of a surrogate
 * pair can match any of it.
 *
 * The reason it is a second function rather than a `typeof` at the top of the
 * shared one: the two callers never see each other's representation — the
 * prepared path asks only about converted sequences — and folding this in
 * measured **1.07x in best-match search over 2000 choices**, reproducibly and
 * against a 1.006x null control. That path calls {@link containsWhitespace}
 * once per candidate over a dozen elements, where the call itself is the cost
 * and one more test is the difference between inlined and not.
 */
export function stringContainsWhitespace(s: string): boolean {
  for (let i = 0; i < s.length; i++) {
    if (isSpaceCodePoint(s.charCodeAt(i))) return true
  }
  return false
}

/**
 * Expand a sequence into the code points the token engine indexes.
 *
 * `ratio` and `partialRatio` read a BMP string as well as they read code
 * points — the bit-parallel kernels reach either through `charCodeAt`. The
 * token engine does not: it walks a string one character at a time, and a
 * one-character string fails {@link tokenKey}'s `isBmpToken` test, so every
 * token misses the packed key and lands in the identity map instead. Worse,
 * tokens of characters and tokens of code points do not compare elementwise, so
 * a query and a candidate must arrive here in the *same* form or their token
 * sets would never intersect.
 *
 * Expanding on the way into a token scorer is what lets `weightedRatio` leave a pair
 * alone for the whitespace-free majority that never gets here. A sequence that
 * is not a string was converted when it was formed, so this is a no-op for it —
 * never a second copy.
 */
export function tokenForm(s: ArrayLike<unknown>): ArrayLike<unknown> {
  return typeof s === 'string' ? convSequence(s) : s
}

/**
 * The pair a token scorer scores, in the one form its engine reads.
 *
 * `convPair` hands back two BMP strings unchanged, which is right for the edit
 * kernels — they read a string as fast as a typed array — and wrong here.
 * Tokenising a string costs 1.5-2x tokenising the same content as code points,
 * measured over the benchmark corpus at 585us against 390us for `tokenSort` and
 * 1113us against 564us for `tokenSet`. The prepared path never had the problem:
 * {@link prepareTokenChoice} has always converted, which is why only the
 * unprepared scorers were affected.
 *
 * Both sides convert unconditionally, so the pair stays in one representation.
 * That is the invariant token comparison depends on — a token of characters and
 * a token of code points never compare equal.
 */
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

/**
 * Element count {@link joinTokens} would produce, without building the array.
 *
 * Private: `tokenSetRatioConverted` accumulates the same total as it walks, and
 * hands it back through {@link joinTokens}'s `total` rather than calling here.
 */
function joinedLength(tokens: readonly unknown[][]): number {
  if (tokens.length === 0) return 0

  let total = tokens.length - 1
  for (let i = 0; i < tokens.length; i++) total += tokens[i].length
  return total
}

/**
 * Rejoin tokens with a single space element, matching the element type.
 *
 * `total` is the element count, for a caller that already knows it — sorting
 * does not change it, so a length measured before the sort still describes the
 * result. Left off, it is counted here. The default is in the signature rather
 * than at the call sites because JavaScript evaluates it only when the argument
 * is absent, which is what keeps the pass off the path that has the number.
 */
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

  // No trim afterwards: `splitSequence` opens a token by pushing an element
  // into it, so no token is empty, so every separator this sized room for is
  // written and `n` lands exactly on `out.length`.
  return out
}

/**
 * Order of a value's type, so that elements of different kinds still have a
 * defined position relative to one another.
 *
 * Python raises `TypeError` rather than ordering an `int` against a `dict`, so
 * there is no upstream answer to match here. What matters is that the order is
 * *total*: sorting is only worth doing because it canonicalises a token list,
 * and a comparator that contradicts itself leaves `[x, y]` and `[y, x]` sorted
 * differently — which is exactly what `tokenSortRatio` is trying to rule out.
 */
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

/**
 * Arbitrary but stable order for two values nothing else separates — two plain
 * objects, say, which every other rule sees as identical.
 *
 * Each object keeps the number it is given the first time it is compared, so
 * the order is consistent for the life of the process. That consistency is the
 * whole point: it is what makes `[x, y]` and `[y, x]` sort the same way round,
 * and so what lets `tokenSortRatio` see them as the same bag of words.
 */
let identityOrdinals: WeakMap<object, number> | null = null
let nextIdentityOrdinal = 0

function isObjectLike(x: unknown): x is object {
  return (typeof x === 'object' && x !== null) || typeof x === 'function'
}

/**
 * The number this object was given the first time it was seen, assigning one if
 * it has not been seen before.
 *
 * Shared by the ordering and by {@link mixedTokenKey}, which is what makes an
 * object's sort position and its hash agree.
 */
function identityOrdinal(x: object): number {
  const ordinals = (identityOrdinals ??= new WeakMap<object, number>())
  let ordinal = ordinals.get(x)
  if (ordinal === undefined) ordinals.set(x, (ordinal = ++nextIdentityOrdinal))
  return ordinal
}

/** Order two objects or functions that nothing else separates. */
function identityOrder(x: object, y: object): number {
  return identityOrdinal(x) - identityOrdinal(y)
}

/**
 * The same tiebreak for symbols, which a `WeakMap` cannot key under this
 * target's lib.
 *
 * Only symbols that collide on their description ever reach here — `Symbol('x')`
 * against another `Symbol('x')` — so what this holds is the rare case rather
 * than every symbol compared. It does hold them strongly, which is the price of
 * giving two otherwise indistinguishable values a stable order; without one
 * `[x, y]` and `[y, x]` sort differently and `tokenSortRatio` reports two
 * arrangements of the same tokens as unlike.
 */
let symbolOrdinals: Map<symbol, number> | null = null

function symbolIdentityOrder(x: symbol, y: symbol): number {
  const ordinals = (symbolOrdinals ??= new Map<symbol, number>())
  let left = ordinals.get(x)
  if (left === undefined) ordinals.set(x, (left = ++nextIdentityOrdinal))
  let right = ordinals.get(y)
  if (right === undefined) ordinals.set(y, (right = ++nextIdentityOrdinal))
  return left - right
}

/** Total order over two elements, matching Python's ordering where it has one. */
function compareElements(x: unknown, y: unknown): number {
  // Narrow the rank-7 case while its runtime proof is still visible to
  // TypeScript. Objects and functions sort after every primitive.
  if (isObjectLike(x)) {
    if (isObjectLike(y)) return identityOrder(x, y)
    return typeOrder(x) - typeOrder(y)
  }
  if (isObjectLike(y)) return typeOrder(x) - typeOrder(y)

  const byType = typeOrder(x) - typeOrder(y)
  if (byType !== 0) return byType

  if (typeof x === 'number' && typeof y === 'number') {
    // `NaN` is unordered by `<`, and `x - y` would hand `Array.prototype.sort`
    // a `NaN` it reads as "equal". Giving it a place of its own past every
    // real number keeps the comparison total.
    if (Number.isNaN(x)) return Number.isNaN(y) ? 0 : 1
    // `compareTokens` answers a pair of real numbers itself, so `y` is the
    // `NaN` here and sorts after every real number.
    return -1
  }
  if (typeof x === 'bigint' && typeof y === 'bigint') return x < y ? -1 : 1
  // `<` on strings orders by UTF-16 code unit, where Python orders by code
  // point; the two disagree once an astral character meets one in U+E000-U+FFFF.
  // There is nothing to match here, though. An element that is a single code
  // point has already become a number by way of `convElement`, so what reaches
  // this line is a string of two or more — and upstream cannot score one at all,
  // raising `ValueError: chr() arg not in range(0x110000)`. All that is required
  // of the order is that it be total and stable, which `<` is.
  if (typeof x === 'string' && typeof y === 'string') return x < y ? -1 : 1
  if (typeof x === 'symbol' && typeof y === 'symbol') {
    const dx = String(x)
    const dy = String(y)
    if (dx !== dy) return dx < dy ? -1 : 1
    // Distinct symbols can share a description, so the text does not separate
    // them and something else has to.
    return symbolIdentityOrder(x, y)
  }

  // What remains is two booleans. Equal `null` or `undefined` would also reach
  // this line, but the caller filters equal elements before calling us.
  return x ? 1 : -1
}

/** Elementwise comparison, matching Python's tuple and string ordering. */
function compareTokens(a: readonly unknown[], b: readonly unknown[]): number {
  const limit = Math.min(a.length, b.length)

  for (let i = 0; i < limit; i++) {
    const x = a[i]
    const y = b[i]
    if (x === y) continue

    // Fast path for the case every converted string takes: two real numbers,
    // where subtraction already gives the sign the comparator wants. `x === x`
    // rejects `NaN`, whose difference would be another `NaN`.
    if (typeof x === 'number' && typeof y === 'number' && x === x && y === y) {
      return x - y
    }
    return compareElements(x, y)
  }

  return a.length - b.length
}

/**
 * Sorts in place. Every call site passes a list built moments earlier — by
 * `splitSequence`, `difference` or `intersection` — so the copy a defensive
 * version would make is never observed by anything.
 */
export function sortTokens(tokens: unknown[][]): unknown[][] {
  return tokens.sort(compareTokens)
}

/** Largest value a code point can take, and so the fast path's upper bound. */
const MAX_CODE_POINT = 0x10ffff
/** Prefix for compact BMP keys; block keys can only open in `0x00..0x10`. */
const BMP_KEY_PREFIX = '\u0011'

/**
 * The same prefix as a code unit, written out rather than read back off the
 * string: {@link isMixedTokenKey} asks for it once per `add` and once per `has`,
 * which is twice per token of every token-set comparison.
 */
const BMP_KEY_PREFIX_CODE = 0x11

/**
 * Every element is an integer inside the BMP, so the whole token spreads into
 * `String.fromCharCode` unchanged.
 *
 * A predicate rather than a `bmp` flag the caller reads afterwards: the flag
 * proved the same thing, but only to a reader — the narrowing it implied had to
 * be asserted back onto `token`, which is exactly the assertion this project
 * does not allow. The loop below is the proof, and the return type carries it.
 */
function isBmpToken(token: readonly unknown[]): token is readonly number[] {
  for (let i = 0; i < token.length; i++) {
    const x = token[i]
    if (typeof x !== 'number' || x < 0 || x > 0xffff || (x | 0) !== x) return false
  }
  return true
}

/**
 * Stand-in for Python's tuple hashing inside a `set`.
 *
 * A converted string is a run of code points, and those pack into a key two
 * UTF-16 units at a time. Fixed width is what makes the packing injective: a
 * variable-width encoding would let `[0xd800, 0xdc00]` and `[0x10000]` — both
 * legitimate element sequences — collapse onto one key.
 *
 * The two key spaces cannot meet. A packed key opens with `cp >>> 16`, which is
 * at most `0x10`; a {@link mixedTokenKey} opens with a `typeof` name, so with a
 * lowercase letter. Packed keys are identities; mixed keys are only bucket
 * hashes and require an elementwise equality check.
 */
function tokenKey(token: readonly unknown[]): string {
  // The length test comes first and stays outside the predicate: it is what
  // bounds the spread below, and it is the cheaper of the two.
  if (token.length <= 64 && isBmpToken(token)) {
    return BMP_KEY_PREFIX + String.fromCharCode(...token)
  }

  let key = ''

  for (let i = 0; i < token.length; i++) {
    const x = token[i]

    // A non-integer or out-of-range number is not a code point, and packing one
    // would alias: `2.5` and `2` share their low 16 bits.
    if (typeof x !== 'number' || x < 0 || x > MAX_CODE_POINT || (x | 0) !== x) {
      return mixedTokenKey(token)
    }

    key += String.fromCharCode(x >>> 16, x & 0xffff)
  }

  return key
}

/**
 * Key for a token holding anything other than code points. The separator is a
 * NUL so that it cannot be spelled by an element's own text as readily as a
 * space could.
 *
 * ## Objects are keyed by identity, not by their text
 *
 * `String(x)` on an object runs the caller's `toString` or `Symbol.toPrimitive`,
 * which is wrong twice over. It is not stable — an object whose text changes
 * between two calls hashes into two buckets, and since {@link equalTokens}
 * compares with `===`, one identity is then counted as two distinct tokens. And
 * it is not total: `String(Object.create(null))` throws `TypeError`, so a
 * null-prototype element crashed the scorer outright.
 *
 * {@link identityOrdinal} answers both. It is the same number the sort order
 * uses, so an object's hash and its position agree, and reaching it runs no user
 * code. Two distinct objects always differ; one object always matches itself.
 *
 * `#` keeps the identity form clear of `String()`'s own output — `object:null`
 * and `object:#1` cannot collide — and every form still opens with a `typeof`
 * name, which is what {@link isMixedTokenKey} discriminates on.
 */
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

/** Whether `key` belongs to the collision-checked mixed-token key space. */
function isMixedTokenKey(key: string): boolean {
  return key.length > 0 && key.charCodeAt(0) > BMP_KEY_PREFIX_CODE
}

/**
 * Set-like token storage: a direct-keyed fast path for code-point tokens, and
 * collision buckets for arbitrary mixed ones.
 *
 * Not an allocation-free path — {@link tokenKey} builds a string for either kind
 * of token. What the fast path skips is everything after the key: no bucket
 * array, no elementwise {@link equalTokens}, and a `Map` probe that decides
 * membership on its own, because a packed key *is* the token's identity where a
 * mixed key is only its hash.
 */
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

/**
 * Deduplicate tokens by value, keeping the first of each.
 *
 * The tokens are stored rather than copied: they come straight from
 * `splitSequence`, and nothing downstream mutates a token.
 */
export function uniqueTokens(tokens: readonly unknown[][]): UniqueTokenSet {
  const out = new UniqueTokenSet()

  // Indexed, like every other loop over a token array here. `tokens` is always
  // the array `splitSequence` built, and the difference against `for…of` is
  // below what this machine can measure — this is for one reading convention
  // rather than for a number.
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]
    out.add(tokenKey(token), token)
  }

  return out
}

/**
 * A sequence together with whichever derived forms have been asked for so far.
 *
 * Every field but `sequence` is a memo: absent means "not built yet", and the
 * accessors below fill it on first use. Reach them through {@link splitOf},
 * {@link uniqueOf}, {@link sortedOf} and {@link hasWhitespaceOf} rather than
 * reading the fields, which is what keeps a form that no branch wants from ever
 * being built.
 *
 * The forms used to be built up front, from a per-scorer table of what that
 * scorer might read. That is one branch too coarse. `weightedRatio` may ask for the
 * unique set, the sorted form *and* the whitespace flag, but on any given pair
 * it takes one route through its ladder and usually wants far less: two inputs
 * of similar length with no whitespace need only `hasWhitespace`, and never
 * split at all. `tokenRatioConverted` is the same story one level down — it
 * returns on a perfect token-set score before it would sort anything, an
 * optimisation that eager preparation quietly cancelled.
 *
 * Mutating a record the caller holds is the point: `process` keeps one per
 * choice for the whole call, so the first row that needs a form pays for it and
 * every later row reuses it.
 */
export class PreparedTokenChoice {
  split?: unknown[][]
  unique?: UniqueTokenSet
  sorted?: unknown[]
  hasWhitespace?: boolean
  canonicalLength?: number

  constructor(readonly sequence: ArrayLike<unknown>) {}
}

/** The tokens of `choice`, split on first use. */
export function splitOf(choice: PreparedTokenChoice): unknown[][] {
  return (choice.split ??= splitSequence(choice.sequence))
}

/** The deduplicated tokens of `choice`, built on first use. */
export function uniqueOf(choice: PreparedTokenChoice): UniqueTokenSet {
  return (choice.unique ??= uniqueTokens(splitOf(choice)))
}

/**
 * Length {@link sortedOf} would return, counted straight off the sequence.
 *
 * Sorting permutes tokens and joining writes one separator between each pair,
 * so neither changes the count — which makes this answerable without splitting,
 * sorting or allocating anything. `tokenSortRatioConverted` asks for it to reject
 * a pair on the Indel length ceiling before it builds the forms that ceiling
 * would have been applied to.
 *
 * Whichever form is already built answers instead of the scan: this is reached
 * once per candidate, and a `weightedRatio` that already split for the token-set half
 * would otherwise pay a second pass over the same input.
 */
export function canonicalLengthOf(choice: PreparedTokenChoice): number {
  if (choice.canonicalLength !== undefined) return choice.canonicalLength
  if (choice.sorted !== undefined) return (choice.canonicalLength = choice.sorted.length)
  if (choice.split !== undefined) {
    return (choice.canonicalLength = joinedLength(choice.split))
  }

  return (choice.canonicalLength = canonicalLength(choice.sequence))
}

/**
 * The token-count arithmetic of {@link joinedLength}, over the raw sequence.
 *
 * A deliberate second copy of {@link splitSequence}'s whitespace walk, without
 * its allocations: the point of the caller is to answer before any token array
 * exists, so building one to measure it would defeat it.
 */
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

/**
 * The tokens of `choice` sorted and rejoined, built on first use.
 *
 * ## `split` has no ordering contract
 *
 * It is the token *multiset*, not the original textual order, so this sorts it
 * where it lies rather than copying the outer array first. Three things read it,
 * and none can observe the order:
 *
 * - {@link uniqueOf} keeps the first of each equal token — and "equal" here is
 *   elementwise `===`, so whichever instance it keeps holds the same contents.
 *   Its map insertion order does change, which reaches {@link difference}'s
 *   output order, but every one of the four places that consumes a difference
 *   sorts it before scoring.
 * - {@link intersects} answers a boolean.
 * - `partialTokenRatioConverted` reads `.length`.
 *
 * The sums in `tokenSetRatioConverted` — the shared count and payload — are
 * likewise order-free. `tests/tokenSplitOrder.test.ts` asserts the invariant
 * directly, by driving the accessors in both orders and requiring one answer.
 */
export function sortedOf(choice: PreparedTokenChoice): unknown[] {
  return (choice.sorted ??= joinTokens(sortTokens(splitOf(choice))))
}

/**
 * Whether `choice` holds any whitespace, tested on first use.
 *
 * Reads the sequence rather than the split, so `weightedRatio`'s shortcut past the
 * token scorers costs one pass and no tokenisation. `??=` is right even though
 * this memo is a boolean: `false` is not nullish, so a negative answer sticks.
 */
export function hasWhitespaceOf(choice: PreparedTokenChoice): boolean {
  return (choice.hasWhitespace ??= containsWhitespace(choice.sequence))
}

/**
 * A view over a sequence nobody prepared, so that a scorer reached directly can
 * share the memoisation above instead of rebuilding forms it needs twice.
 *
 * Deliberately not branded into {@link tokenChoices}: provenance is a claim
 * about a value that crossed the `process` boundary, and this one never leaves
 * the call that made it.
 */
export function tokenViewOf(sequence: ArrayLike<unknown>): PreparedTokenChoice {
  return new PreparedTokenChoice(sequence)
}
/**
 * Convert a choice once so every query row can reuse the result.
 *
 * Only the conversion happens here. Which derived forms get built, and whether
 * any do, is decided later by the branch that reads one — see
 * {@link PreparedTokenChoice}. `scoreMatrix` prepares every choice once per call,
 * so a form built here and never read is paid for on every choice in the list.
 */
export function tokenChoicePreparer(): ChoicePreparer {
  return prepareTokenChoice
}

/** Build a token choice after the caller has established the sequence contract. */
export function prepareTokenChoice(choice: Sequence): PreparedTokenChoice {
  return new PreparedTokenChoice(convSequence(choice))
}

/** Read the opaque choice returned by this module's private preparer. */
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

/**
 * Whether the two token sets share anything — the question several scorers ask
 * before they would otherwise build an intersection only to read its length.
 */
export function intersects(a: UniqueTokenSet, b: UniqueTokenSet): boolean {
  const smaller = a.size <= b.size ? a : b
  const larger = smaller === a ? b : a

  for (const key of smaller.packed.keys()) if (larger.packed.has(key)) return true
  for (const [key, bucket] of smaller.mixed) {
    for (const token of bucket) if (larger.has(key, token)) return true
  }
  return false
}
