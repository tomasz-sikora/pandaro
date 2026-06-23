import type { Segment } from '@pandaro/shared-types'
import { computeEnergy } from '../audio/processor'

interface RawChunk {
  timestamp: [number, number]
  text: string
}

const PAUSE_SPEAKER_CHANGE_S = 1.5
const ENERGY_RATIO_THRESHOLD = 4.0
const MAX_SPEAKERS = 6

/**
 * Heuristic diarization: assign speaker labels based on pause length
 * and audio energy changes between consecutive Whisper chunks.
 */
export function diarize(
  chunks: RawChunk[],
  audioData: Float32Array,
  sampleRate: number,
): Segment[] {
  if (chunks.length === 0) return []

  const energies = chunks.map((c) =>
    computeEnergy(audioData, c.timestamp[0], c.timestamp[1], sampleRate),
  )

  let speakerIdx = 0
  const segments: Segment[] = []

  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i]
    if (i > 0) {
      const gap = c.timestamp[0] - chunks[i - 1].timestamp[1]
      const eRatio =
        energies[i] > 0 && energies[i - 1] > 0
          ? Math.max(energies[i], energies[i - 1]) /
            Math.min(energies[i], energies[i - 1])
          : 1

      if (gap >= PAUSE_SPEAKER_CHANGE_S || eRatio >= ENERGY_RATIO_THRESHOLD) {
        speakerIdx = (speakerIdx + 1) % MAX_SPEAKERS
      }
    }

    segments.push({
      id: i,
      start: c.timestamp[0],
      end: c.timestamp[1],
      text: c.text.trim(),
      speaker: `GŁOS_${String(speakerIdx + 1).padStart(2, '0')}`,
    })
  }

  return segments
}
