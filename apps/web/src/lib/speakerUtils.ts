import type { SpeakerProfile } from '@heimdall/shared-types'

/**
 * Returns the best human-readable label for a speaker:
 *  1. display_name from the speaker profile (set by LLM identification)
 *  2. the raw speaker ID (e.g. GŁOS_01) as a fallback
 */
export function speakerDisplayName(
  speakerId: string,
  profiles: Record<string, SpeakerProfile>,
): string {
  return profiles[speakerId]?.display_name ?? speakerId
}
