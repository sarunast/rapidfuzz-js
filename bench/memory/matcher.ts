import process from 'node:process'

import { tokenSortRatio } from '../../dist/fuzz/index.js'
import { createMatcher, createScorer } from '../../dist/index.js'
import { collectGarbage, sampleMemory } from './harness.ts'

const count = Number(process.argv[2] ?? 50_000)
if (!Number.isSafeInteger(count) || count <= 0) {
  throw new RangeError('item count must be a positive safe integer')
}

const items = Array.from({ length: count }, (_, index) => ({
  title: `catalog item ${index} alpha beta gamma`,
}))
const scorer = createScorer(tokenSortRatio)

function retainedBytes() {
  return sampleMemory().retained
}

collectGarbage()
const before = retainedBytes()
const matcher = createMatcher(items, { scorer, getText: (item) => item.title })
collectGarbage()
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
