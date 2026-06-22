/** Decode an audio file to 16 kHz mono Float32Array (Whisper format). */
export async function decodeAudioFile(file: File): Promise<{
  audioData: Float32Array
  duration: number
  sampleRate: number
}> {
  const arrayBuffer = await file.arrayBuffer()

  const tempCtx = new AudioContext()
  let audioBuffer: AudioBuffer
  try {
    audioBuffer = await tempCtx.decodeAudioData(arrayBuffer)
  } finally {
    await tempCtx.close()
  }

  const TARGET_SR = 16_000
  const duration = audioBuffer.duration

  // Mix to mono
  let mono: Float32Array
  if (audioBuffer.numberOfChannels === 1) {
    mono = audioBuffer.getChannelData(0).slice()
  } else {
    const left = audioBuffer.getChannelData(0)
    const right = audioBuffer.getChannelData(1)
    mono = new Float32Array(left.length)
    for (let i = 0; i < left.length; i++) {
      mono[i] = (left[i] + right[i]) / 2
    }
  }

  // Resample to 16 kHz if needed
  if (audioBuffer.sampleRate === TARGET_SR) {
    return { audioData: mono, duration, sampleRate: TARGET_SR }
  }

  const offlineCtx = new OfflineAudioContext(
    1,
    Math.ceil(duration * TARGET_SR),
    TARGET_SR,
  )

  // Put mono data into a buffer at original sample rate
  const srcBuf = offlineCtx.createBuffer(1, mono.length, audioBuffer.sampleRate)
  srcBuf.copyToChannel(new Float32Array(mono), 0)

  const src = offlineCtx.createBufferSource()
  src.buffer = srcBuf
  src.connect(offlineCtx.destination)
  src.start()

  const rendered = await offlineCtx.startRendering()
  return {
    audioData: rendered.getChannelData(0),
    duration,
    sampleRate: TARGET_SR,
  }
}

/** Compute RMS energy for a slice of audio */
export function computeEnergy(
  audio: Float32Array,
  startSec: number,
  endSec: number,
  sampleRate = 16_000,
): number {
  const start = Math.max(0, Math.floor(startSec * sampleRate))
  const end = Math.min(audio.length, Math.ceil(endSec * sampleRate))
  if (end <= start) return 0
  let sum = 0
  for (let i = start; i < end; i++) sum += audio[i] * audio[i]
  return Math.sqrt(sum / (end - start))
}
