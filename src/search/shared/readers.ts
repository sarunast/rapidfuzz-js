import { resolvePreparedChoice } from '../../core/scoring/preparedChoice.js'
import type { AnyBrand, PreparedChoice } from '../../core/scoring/preparedChoice.js'
import {
  normalizeSequence,
  snapshotSequence,
  validateSequence,
} from '../../core/sequence.js'
import type { MaybeSequence, Normalizer, Sequence } from '../../core/types.js'
import type { MissingItemsPolicy } from '../types.js'

export type SequenceReader<TItem> = (item: TItem) => Sequence | null

/**
 * What a reader consumes, which is not what a search was configured with: no
 * scorer, because a reader turns an item into a choice and never scores one.
 *
 * Both accessors at once, which the public union refuses and a JavaScript
 * caller can still pass. Reading them by value and refusing the combination is
 * `choiceReader`'s job, so the shape it takes has to be able to hold one.
 */
export interface ReaderOptions<TItem, TBrand = AnyBrand> {
  readonly getText?: ((item: TItem) => MaybeSequence) | undefined
  readonly getPrepared?: ((item: TItem) => PreparedChoice<NoInfer<TBrand>>) | undefined
  readonly normalize?: Normalizer | undefined
  readonly missingItems?: MissingItemsPolicy | undefined
}

/**
 * How a search reads its choices.
 *
 * `read` answers the prepared representation, or `null` for an item text mode
 * skips — a prepared choice is either resolved or an error. Its type is
 * `unknown` because the representation is erased at this boundary, which is
 * also why the type cannot say what the prose just did. `sequences` is the
 * text-mode reader itself, and `null` marks prepared mode for the one loop
 * that needs a `Sequence` rather than a prepared choice.
 */
export interface ChoiceReader<TItem> {
  readonly present: (item: TItem) => boolean
  readonly read: (item: TItem) => unknown
  readonly sequences: SequenceReader<TItem> | null
}

export function choiceReader<TItem, TBrand>(
  options: ReaderOptions<TItem, TBrand>,
  prepareChoice: (choice: Sequence) => unknown,
  preparedChoiceKey: object,
  own: boolean,
): ChoiceReader<TItem> {
  if (options.getPrepared === undefined) {
    // Two readers, built once: the loops read choices, and the null-query and
    // raw-score paths read sequences. Neither pays for the other.
    const prepare = own
      ? (value: Sequence): unknown => prepareChoice(snapshotSequence(value))
      : prepareChoice
    const readSequence = sequenceReader(options, own)
    return {
      present: (item) => readSequence(item) !== null,
      read: itemReader(options, prepare),
      sequences: readSequence,
    }
  }
  // Read by value rather than with `in`: the options type admits an explicit
  // `getText: undefined`, and a runtime stricter than the types would refuse a
  // call TypeScript accepted.
  if (options.getText !== undefined || options.missingItems !== undefined) {
    throw new TypeError('getPrepared cannot be combined with getText or missingItems')
  }
  const getPrepared = requireFunction(options.getPrepared, 'getPrepared')
  // The same normalizer the query goes through: a handle prepared under a
  // different one — or none — is refused rather than scored against a query it
  // was never comparable to.
  const normalize =
    options.normalize === undefined
      ? undefined
      : requireFunction(options.normalize, 'normalize')
  const read = (item: TItem): unknown =>
    resolvePreparedChoice(preparedChoiceKey, getPrepared(item), normalize)
  return {
    // Prepared mode has nothing to skip, so presence is the resolution itself:
    // a missing or foreign handle throws here as it does anywhere else.
    present: (item) => {
      read(item)
      return true
    },
    read,
    sequences: null,
  }
}

export function sequenceReader<TItem>(
  options: ReaderOptions<TItem>,
  own: boolean,
): SequenceReader<TItem> {
  const retain = own ? snapshotSequence : identity
  return itemReader(options, retain)
}

function identity(value: Sequence): Sequence {
  return value
}

/**
 * Checked where the reader is built rather than where it is called: an empty
 * collection never calls an accessor, so a search over one would otherwise
 * accept options no non-empty collection would.
 */
function requireFunction<TImplementation>(
  value: TImplementation,
  name: string,
): TImplementation {
  if (typeof value !== 'function') {
    throw new TypeError(`${name} must be a function`)
  }
  return value
}

/**
 * The four shapes a text-mode reader takes, in one place, with what to do with
 * the sequence it read left to the caller.
 *
 * `finish` rather than a wrapper around a `Sequence`-returning reader: a search
 * calls this once per candidate, and preparing a choice through a second
 * closure would add a call to that loop for every caller, including the ones
 * that never prepare anything.
 */
function itemReader<TItem, TResult>(
  options: ReaderOptions<TItem>,
  finish: (value: Sequence) => TResult,
): (item: TItem) => TResult | null {
  const policy = options.missingItems ?? 'skip'
  // Checked once, before the specialized reader exists: an unknown policy is a
  // typo, and reading it as 'throw' by falling through the 'skip' test would
  // answer one wrong argument with a different wrong behaviour.
  if (policy !== 'skip' && policy !== 'throw') {
    throw new TypeError("missingItems must be 'skip' or 'throw'")
  }
  const normalize =
    options.normalize === undefined
      ? undefined
      : requireFunction(options.normalize, 'normalize')

  if (options.getText === undefined) {
    if (normalize === undefined) {
      return (item) => {
        if (item == null) {
          if (policy === 'skip') return null
          throw new TypeError('source item is missing')
        }
        return finish(validateSequence(item))
      }
    }
    return (item) => {
      if (item == null) {
        if (policy === 'skip') return null
        throw new TypeError('source item is missing')
      }
      return finish(normalizeSequence(validateSequence(item), normalize))
    }
  }

  const getText = requireFunction(options.getText, 'getText')
  if (normalize === undefined) {
    return (item) => {
      if (item == null) {
        if (policy === 'skip') return null
        throw new TypeError('source item is missing')
      }
      const raw = getText(item)
      if (raw == null) {
        if (policy === 'skip') return null
        throw new TypeError('getText returned a missing value')
      }
      return finish(validateSequence(raw))
    }
  }
  return (item) => {
    if (item == null) {
      if (policy === 'skip') return null
      throw new TypeError('source item is missing')
    }
    const raw = getText(item)
    if (raw == null) {
      if (policy === 'skip') return null
      throw new TypeError('getText returned a missing value')
    }
    return finish(normalizeSequence(validateSequence(raw), normalize))
  }
}

export function normalizeQuery(
  query: MaybeSequence,
  normalize: Normalizer | undefined,
): Sequence | null {
  if (query == null) return null
  const valid = validateSequence(query)
  if (normalize === undefined) return valid
  return normalizeSequence(valid, normalize)
}
