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

export interface ReaderOptions<TItem, TBrand = AnyBrand> {
  readonly getText?: ((item: TItem) => MaybeSequence) | undefined
  readonly getPrepared?: ((item: TItem) => PreparedChoice<NoInfer<TBrand>>) | undefined
  readonly normalize?: Normalizer | undefined
  readonly missingItems?: MissingItemsPolicy | undefined
}

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
  if (options.getText !== undefined || options.missingItems !== undefined) {
    throw new TypeError('getPrepared cannot be combined with getText or missingItems')
  }
  const getPrepared = requireFunction(options.getPrepared, 'getPrepared')
  const normalize =
    options.normalize === undefined
      ? undefined
      : requireFunction(options.normalize, 'normalize')
  const read = (item: TItem): unknown =>
    resolvePreparedChoice(preparedChoiceKey, getPrepared(item), normalize)
  return {
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

function requireFunction<TImplementation>(
  value: TImplementation,
  name: string,
): TImplementation {
  if (typeof value !== 'function') {
    throw new TypeError(`${name} must be a function`)
  }
  return value
}

function itemReader<TItem, TResult>(
  options: ReaderOptions<TItem>,
  finish: (value: Sequence) => TResult,
): (item: TItem) => TResult | null {
  const policy = options.missingItems ?? 'skip'
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
