import { resolvePreparedChoice } from '../core/prepared.js'
import {
  normalizeSequence,
  snapshotSequence,
  validateSequence,
} from '../core/sequence.js'
import type { Direction, MaybeSequence, Normalizer, Sequence } from '../core/types.js'
import type { AnyMatcherOptions, MatcherOptions } from './types.js'

export type SequenceReader<T> = (item: T) => Sequence | null

/**
 * How a search reads its choices. `read` answers `null` for an item the search
 * skips, which only text mode has — a prepared choice is either resolved or an
 * error. `sequences` is the text-mode reader itself, and `null` marks prepared
 * mode for the one loop that needs a `Sequence` rather than a prepared choice.
 */
export interface ChoiceReader<T> {
  readonly present: (item: T) => boolean
  readonly read: (item: T) => unknown
  readonly sequences: SequenceReader<T> | null
}

export function choiceReader<T, B>(
  options: AnyMatcherOptions<T, Direction, B>,
  prepareChoice: (choice: Sequence) => unknown,
  preparedChoiceKey: object,
  own: boolean,
): ChoiceReader<T> {
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
  const getPrepared = options.getPrepared
  const read = (item: T): unknown =>
    resolvePreparedChoice(preparedChoiceKey, getPrepared(item))
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

export function sequenceReader<T>(
  options: MatcherOptions<T, Direction>,
  own: boolean,
): SequenceReader<T> {
  const retain = own ? snapshotSequence : identity
  return itemReader(options, retain)
}

function identity(value: Sequence): Sequence {
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
function itemReader<T, R>(
  options: MatcherOptions<T, Direction>,
  finish: (value: Sequence) => R,
): (item: T) => R | null {
  const policy = options.missingItems ?? 'skip'
  // Checked once, before the specialized reader exists: an unknown policy is a
  // typo, and reading it as 'throw' by falling through the 'skip' test would
  // answer one wrong argument with a different wrong behaviour.
  if (policy !== 'skip' && policy !== 'throw') {
    throw new TypeError("missingItems must be 'skip' or 'throw'")
  }
  const normalize = options.normalize

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

  const getText = options.getText
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

export function optionalThreshold(value: number | undefined): number | null {
  if (value === undefined) return null
  if (!Number.isFinite(value)) throw new RangeError('threshold must be finite')
  return value
}

export function resultLimit(value: number | null | undefined): number | null {
  if (value === null) return null
  const limit = value ?? 5
  if (!Number.isSafeInteger(limit) || limit < 0) {
    throw new RangeError('limit must be null or a non-negative safe integer')
  }
  return limit
}
