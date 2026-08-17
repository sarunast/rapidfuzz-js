import { isUnmatchableElement } from '#core/sequence.js'

/**
 * The ordinal an unmatchable element takes. Negative, so the gram extractor
 * recognises it without a second predicate, and outside the ordinal space,
 * which is dense from zero.
 */
export const UNMATCHABLE = -1

/**
 * Ordinals are only handed to elements that can reach a gram: those inside a
 * run of at least `gramSize` matchable elements. Everything in a shorter run
 * sits in a window an unmatchable element already poisons, so it can never name
 * a posting — and giving it an ordinal would hold it for the life of the index
 * and push every later element further up the radix ladder for nothing.
 */
export function ordinalizeChoice(
  elements: ArrayLike<unknown>,
  gramSize: number,
  table: Map<unknown, number>,
  output: number[],
): void {
  const length = elements.length
  output.length = length
  let start = 0
  while (start < length) {
    let end = start
    while (end < length && !isUnmatchableElement(elements[end])) end++
    if (end - start < gramSize) {
      for (let index = start; index < end; index++) output[index] = UNMATCHABLE
    } else {
      for (let index = start; index < end; index++) {
        const element = elements[index]
        const known = table.get(element)
        if (known !== undefined) {
          output[index] = known
          continue
        }
        const ordinal = table.size
        table.set(element, ordinal)
        output[index] = ordinal
      }
    }
    if (end < length) output[end] = UNMATCHABLE
    start = end + 1
  }
}

/**
 * Reads ordinals a vocabulary already holds, and never adds one. The radix an
 * index re-keys with follows from the size of that vocabulary, so an ordinal
 * assigned after it was chosen could be a digit the radix cannot hold and
 * `encodeGramKey` would spell a key that collides. `ReadonlyMap` states that
 * for the checker; the throw states it for a walk that ever stopped agreeing
 * with the one that assigned.
 *
 * @throws {Error} If the vocabulary has no ordinal for an element.
 */
export function resolveOrdinals(
  elements: ArrayLike<unknown>,
  table: ReadonlyMap<unknown, number>,
  output: number[],
): void {
  const length = elements.length
  output.length = length
  for (let index = 0; index < length; index++) {
    const ordinal = table.get(elements[index])
    if (ordinal === undefined) throw new Error('this element has no ordinal')
    output[index] = ordinal
  }
}

/**
 * An element the corpus holds takes its corpus ordinal; anything else takes a
 * query-local one from `table.size` upward, so it can never name a corpus gram
 * while repeats of it still share an ordinal and count as one gram.
 */
export function ordinalizeQuery(
  elements: ArrayLike<unknown>,
  table: ReadonlyMap<unknown, number>,
  unknown: Map<unknown, number>,
  output: number[],
): void {
  const length = elements.length
  output.length = length
  for (let index = 0; index < length; index++) {
    const element = elements[index]
    if (isUnmatchableElement(element)) {
      output[index] = UNMATCHABLE
      continue
    }
    const known = table.get(element)
    if (known !== undefined) {
      output[index] = known
      continue
    }
    const local = unknown.get(element)
    if (local !== undefined) {
      output[index] = local
      continue
    }
    const ordinal = table.size + unknown.size
    unknown.set(element, ordinal)
    output[index] = ordinal
  }
}
