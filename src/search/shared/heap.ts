/** Whether `left` should be closer to the heap root than `right`. */
export type HigherPriority<TItem> = (left: TItem, right: TItem) => boolean

export function pushHeap<TItem>(
  heap: TItem[],
  value: TItem,
  higherPriority: HigherPriority<TItem>,
): void {
  let child = heap.length
  heap.push(value)
  while (child > 0) {
    // Unsigned: `>>` reads its operand as a signed 32-bit integer, so a child
    // index above 2^31 halves to a negative parent. `>>>` is the same
    // instruction and stays correct across every index an array can hold.
    const parent = (child - 1) >>> 1
    const parentValue = heap[parent]
    if (!higherPriority(value, parentValue)) break
    heap[child] = parentValue
    child = parent
  }
  heap[child] = value
}

/** Replaces the root of a non-empty heap and restores the heap invariant. */
export function replaceHeapRoot<TItem>(
  heap: TItem[],
  value: TItem,
  higherPriority: HigherPriority<TItem>,
): void {
  let parent = 0
  const length = heap.length
  while (true) {
    const left = parent * 2 + 1
    if (left >= length) break
    const right = left + 1
    let child = left
    let childValue = heap[left]
    if (right < length) {
      const rightValue = heap[right]
      if (higherPriority(rightValue, childValue)) {
        child = right
        childValue = rightValue
      }
    }
    if (!higherPriority(childValue, value)) break
    heap[parent] = childValue
    parent = child
  }
  heap[parent] = value
}
