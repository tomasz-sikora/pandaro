import type { Segment, VectorEntry } from '@heimdall/shared-types'

export interface TextChunk {
  text: string
  segmentIds: number[]
  start: number
  end: number
  speaker?: string
}

const MAX_CHUNK_CHARS = 600
/** Each chunk must contain at least this many segments total. */
const MIN_CHUNK_SEGMENTS = 3
/** Number of segments shared between consecutive chunks (sliding window overlap). */
const OVERLAP_SEGMENTS = 2

export function chunkSegments(segments: Segment[]): TextChunk[] {
  const chunks: TextChunk[] = []
  let i = 0  // index of the first "new" (non-overlap) segment for this chunk

  while (i < segments.length) {
    const included: Segment[] = []
    let totalLen = 0

    // Reuse the tail of the previous chunk as leading context
    const overlapStart = Math.max(0, i - OVERLAP_SEGMENTS)
    for (let j = overlapStart; j < i; j++) {
      included.push(segments[j])
      totalLen += (segments[j].text ?? '').length
    }

    // Always consume at least ONE new segment — prevents infinite loop when
    // overlap text alone already fills MAX_CHUNK_CHARS.
    if (i < segments.length) {
      included.push(segments[i])
      totalLen += (segments[i].text ?? '').length
      i++
    }

    // Keep consuming until BOTH conditions are satisfied:
    //   • at least MIN_CHUNK_SEGMENTS total segments in the chunk
    //   • at least MAX_CHUNK_CHARS of text
    while (
      i < segments.length &&
      (included.length < MIN_CHUNK_SEGMENTS || totalLen < MAX_CHUNK_CHARS)
    ) {
      included.push(segments[i])
      totalLen += (segments[i].text ?? '').length
      i++
    }

    if (included.length > 0) {
      chunks.push({
        text: included.map((s) => s.text ?? '').join(' '),
        segmentIds: included.map((s) => s.id),
        start: included[0].start,
        end: included[included.length - 1].end,
        speaker: included[0].speaker,
      })
    }
  }

  return chunks
}

/** Convert chunks + embeddings to VectorEntry[] */
export function buildVectorEntries(
  chunks: TextChunk[],
  embeddings: number[][],
): VectorEntry[] {
  return chunks.map((c, i) => ({
    id: i,
    text: c.text,
    embedding: embeddings[i],
    metadata: {
      segmentIds: c.segmentIds,
      start: c.start,
      end: c.end,
      speaker: c.speaker,
    },
  }))
}
