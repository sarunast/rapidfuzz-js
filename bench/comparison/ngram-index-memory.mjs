// @ts-check
/**
 * What each held structure costs in memory, over the corpus
 * `ngram-index.mjs` times: an indexed `Matcher`, a prepared `Matcher`, the gram
 * arrays `dice-coefficient` is handed, and a Fuse index.
 *
 * ```sh
 * pnpm build && node bench/comparison/ngram-index-memory.mjs
 * node bench/comparison/ngram-index-memory.mjs --max=1000000
 * ```
 *
 * **One arm per process, and that is the whole design.** Measuring several
 * structures in one heap has produced negative retained bytes here more than
 * once: whatever the first arm allocated is still being collected while the
 * second is measured, and a delta against a moving baseline is not a
 * measurement. So the parent process spawns a child per (size, arm) pair, and
 * each child builds the corpus, collects, reads the heap, builds exactly one
 * structure, collects, and reads it again. The difference is that structure and
 * nothing else — the corpus is on the wrong side of the baseline by
 * construction.
 *
 * The number to read is bytes per choice. It is what decides whether a
 * collection fits, and it is comparable across the ladder in a way that a total
 * is not.
 */

import { execFileSync } from 'node:child_process'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

import { buildCorpus } from './ladder-corpus.mjs'

const GRAM_SIZE = 2

/** Every arm, in the order the report prints them. */
const ARMS = ['indexed', 'matcher', 'grams', 'fuse', 'corpus']

/**
 * A prepared 1,000,000-choice collection retains 813 MB and its gram arrays a
 * gigabyte, both of which exceed a default old space. The child gets room for
 * them rather than the table getting a hole where the interesting number is.
 */
const CHILD_HEAP_MB = 8192

// ---------------------------------------------------------------- the child

function collect() {
  globalThis.gc?.()
  globalThis.gc?.()
  globalThis.gc?.()
}

function retainedBytes() {
  const usage = process.memoryUsage()
  return usage.heapUsed + usage.arrayBuffers
}

/**
 * Build one structure and report what it retains. `keep` is returned rather
 * than dropped so nothing is collected before the second reading, and the
 * caller prints something derived from it for the same reason.
 *
 * @param {string} arm
 * @param {number} count
 */
async function measureOne(arm, count) {
  const { similarity: diceSimilarity } =
    await import('../../dist/algorithms/dice/index.js')
  const { createIndexedMatcher, createMatcher, createScorer } =
    await import('../../dist/index.js')
  const scorer = createScorer(diceSimilarity, { gramSize: GRAM_SIZE })

  // The corpus is built before the baseline reading, so it is charged to the
  // baseline and not to the structure under measurement.
  const corpus = arm === 'corpus' ? null : buildCorpus(count)

  collect()
  const before = retainedBytes()

  let keep
  switch (arm) {
    case 'indexed':
      keep = createIndexedMatcher(corpus.choices, { scorer })
      break
    case 'matcher':
      keep = createMatcher(corpus.choices, { scorer })
      break
    case 'grams': {
      const { nGram } = await import('n-gram')
      const grams = nGram(GRAM_SIZE)
      keep = corpus.choices.map((choice) => grams(choice))
      break
    }
    case 'fuse': {
      const { default: Fuse } = await import('fuse.js')
      keep = new Fuse(corpus.choices, { includeScore: true, threshold: 0.4 })
      break
    }
    case 'corpus':
      keep = buildCorpus(count).choices
      break
    default:
      throw new RangeError(`unknown arm: ${arm}`)
  }

  collect()
  const retained = retainedBytes() - before

  // Reachable at the second reading, and proven so: a structure V8 collected
  // early would measure as free.
  const size = keep.size ?? keep.length ?? 0
  if (arm !== 'fuse' && size !== count) {
    throw new Error(`${arm} held ${size} of ${count} choices`)
  }
  return { arm, count, retained, perChoice: retained / count }
}

if (process.argv.includes('--child')) {
  const arm = process.argv
    .find((value) => value.startsWith('--arm='))
    ?.slice('--arm='.length)
  const count = Number(
    process.argv.find((value) => value.startsWith('--size='))?.slice('--size='.length),
  )
  if (globalThis.gc === undefined) throw new Error('run the child with --expose-gc')
  process.stdout.write(`${JSON.stringify(await measureOne(arm ?? '', count))}\n`)
} else {
  // -------------------------------------------------------------- the parent

  const sizes = [1_000, 10_000, 100_000]
  for (const argument of process.argv.slice(2)) {
    if (argument.startsWith('--max=')) {
      const max = Number(argument.slice('--max='.length))
      if (max >= 1_000_000) sizes.push(1_000_000)
    }
  }

  const here = fileURLToPath(import.meta.url)

  /**
   * @param {string} arm
   * @param {number} count
   */
  function runChild(arm, count) {
    const output = execFileSync(
      process.execPath,
      [
        '--expose-gc',
        `--max-old-space-size=${CHILD_HEAP_MB}`,
        here,
        '--child',
        `--arm=${arm}`,
        `--size=${count}`,
      ],
      { encoding: 'utf8', maxBuffer: 1 << 20 },
    )
    return JSON.parse(output)
  }

  const NAMES = {
    indexed: 'createIndexedMatcher',
    matcher: 'createMatcher',
    grams: 'dice-coefficient gram arrays',
    fuse: 'Fuse index',
    corpus: 'the choices themselves',
  }

  console.log(
    '\n  Retained bytes per choice, measured in a child process per arm.\n' +
      '  The corpus is charged to the baseline, so these are the structure alone.\n' +
      '  uFuzzy is absent because it holds nothing between queries.',
  )

  for (const count of sizes) {
    console.log(`\n  ${count.toLocaleString()} choices`)
    console.log(
      `  ${'held structure'.padEnd(32)}${'per choice'.padStart(12)}${'total'.padStart(14)}`,
    )
    for (const arm of ARMS) {
      const result = runChild(arm, count)
      const total = result.retained / 1e6
      console.log(
        `  ${NAMES[arm].padEnd(32)}${`${result.perChoice.toFixed(0)} B`.padStart(12)}` +
          `${`${total.toFixed(1)} MB`.padStart(14)}`,
      )
    }
  }

  console.log(
    `\n  Each child ran with a ${CHILD_HEAP_MB} MB old space, which is what the largest\n` +
      '  arms need: a prepared 1,000,000-choice collection does not fit in a default one.\n',
  )
}
