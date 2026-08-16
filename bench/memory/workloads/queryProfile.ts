import type { IndexedMatcherWorkload } from './shared.ts'

export const QUERY_PROFILE_ELEMENTS = 100_001

/** The temporary query and result both die on return. */
export function runQueryProfileSpike(matcher: IndexedMatcherWorkload): void {
  const query = Array.from({ length: QUERY_PROFILE_ELEMENTS }, (_, index) => index)
  matcher.best(query)
}
