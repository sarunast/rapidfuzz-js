import type { IndexedMatcherWorkload } from './shared.ts'

export const LATE_INVALID_ELEMENTS = 100_001

class InvalidElement {
  toString(): string {
    return 'late-invalid-sentinel'
  }
}

class LateInvalidQuery {
  [index: number]: unknown
  readonly length = LATE_INVALID_ELEMENTS

  constructor(sentinel: InvalidElement) {
    for (let index = 0; index < this.length - 1; index++) this[index] = index
    this[this.length - 1] = sentinel
  }
}

/** The temporary wrapper, sentinel, and thrown error die on return. */
export function runLateInvalidQuery(matcher: IndexedMatcherWorkload): void {
  try {
    matcher.best(new LateInvalidQuery(new InvalidElement()))
  } catch (error) {
    if (error instanceof TypeError && /integer elements only/.test(error.message)) return
    throw error
  }
  throw new Error('the late-invalid query unexpectedly succeeded')
}
