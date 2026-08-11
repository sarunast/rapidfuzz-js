export type EditopTag = 'replace' | 'delete' | 'insert'
export type OpcodeTag = EditopTag | 'equal'

export interface Editop {
  readonly tag: EditopTag
  readonly srcPos: number
  readonly destPos: number
}

export interface Opcode {
  readonly tag: OpcodeTag
  readonly srcStart: number
  readonly srcEnd: number
  readonly destStart: number
  readonly destEnd: number
}

export interface MatchingBlock {
  readonly srcStart: number
  readonly destStart: number
  readonly length: number
}
