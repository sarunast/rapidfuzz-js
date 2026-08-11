import process from 'node:process'

import { tokenSortSimilarity } from '../../dist/fuzz/index.js'
import { createMatcher, createScorer } from '../../dist/index.js'

const count = Number(process.argv[2] ?? 50_000)
if (!Number.isSafeInteger(count) || count <= 0) {
  throw new RangeError('item count must be a positive safe integer')
}
if (globalThis.gc === undefined) {
  throw new Error('run this benchmark with --expose-gc')
}

const items = Array.from({ length: count }, (_, index) => ({
  title: `catalog item ${index} alpha beta gamma`,
}))
const scorer = createScorer(tokenSortSimilarity)

function collect() {
  globalThis.gc?.()
  globalThis.gc?.()
  globalThis.gc?.()
}

function retainedBytes() {
  const usage = process.memoryUsage()
  return usage.heapUsed + usage.arrayBuffers
}

collect()
const before = retainedBytes()
const matcher = createMatcher(items, { scorer, getText: (item) => item.title })
collect()
const retained = retainedBytes() - before

if (matcher.size !== count) throw new Error('matcher did not retain every item')
process.stdout.write(
  `${JSON.stringify(
    {
      items: count,
      retainedBytes: retained,
      retainedBytesPerItem: retained / count,
    },
    null,
    2,
  )}\n`,
)
