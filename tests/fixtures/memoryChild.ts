import process from 'node:process'

if (process.argv.includes('--invalid-json')) process.stdout.write('not json')
else if (process.argv.includes('--fail')) throw new Error('child failure')
else
  process.stdout.write(`${JSON.stringify({ gc: typeof globalThis.gc, child: true })}\n`)
