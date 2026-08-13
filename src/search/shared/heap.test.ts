import { describe, expect, it } from 'vitest'

import { pushHeap, replaceHeapRoot } from './heap.js'

const higher = (left: number, right: number): boolean => left > right

describe('binary heap primitives', () => {
  it('pushes above lower-priority parents and leaves higher ones in place', () => {
    const heap: number[] = []
    pushHeap(heap, 3, higher)
    pushHeap(heap, 1, higher)
    pushHeap(heap, 5, higher)
    expect(heap).toEqual([5, 1, 3])
  })

  it('restores through left, right, and single-child paths', () => {
    const left = [9, 8, 7, 6, 5]
    replaceHeapRoot(left, 0, higher)
    expect(left[0]).toBe(8)

    const right = [9, 7, 8, 5, 6]
    replaceHeapRoot(right, 0, higher)
    expect(right[0]).toBe(8)

    const settled = [9, 8, 7]
    replaceHeapRoot(settled, 8.5, higher)
    expect(settled).toEqual([8.5, 8, 7])
  })
})
