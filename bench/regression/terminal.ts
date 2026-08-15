/**
 * Painting a line for the terminal.
 *
 * Colour is decided once, from whether stdout is a TTY and whether `NO_COLOR`
 * is set, so nothing downstream has to ask.
 */

import process from 'node:process'

export function percent(value: number): string {
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`
}

const COLOUR = process.stdout.isTTY && !process.env['NO_COLOR']
export const paint = (code: number, text: string): string =>
  COLOUR ? `\u001B[${code}m${text}\u001B[0m` : text
export const dim = (text: string): string => paint(2, text)
export const green = (text: string): string => paint(32, text)
export const red = (text: string): string => paint(31, text)
export const yellow = (text: string): string => paint(33, text)
export const out = (text: string): void => {
  process.stdout.write(text)
}
