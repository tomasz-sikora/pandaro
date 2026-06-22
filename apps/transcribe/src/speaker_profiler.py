"""
Speaker profiling: gender (child/female/male), age, and extensible audio features.

Primary model: audeering/wav2vec2-large-robust-24-ft-age-gender
  - 24 transformer layers (more accurate than the 6-layer variant)
  - Gender head: 3 classes [female, male, child]
  - Age head: 1 output sigmoid ~0-1 -> scaled to years
  - https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender

Falls back to librosa F0-based heuristic if the model is unavailable.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_ID = "audeering/wav2vec2-large-robust-24-ft-age-gender"


def _build_age_gender_model():
    import torch
    import torch.nn as nn
    from transformers import (
        Wav2Vec2Processor,
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class ModelHead(nn.Module):
        def __init__(self, config, num_labels: int):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, features, **_):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            return self.out_proj(self.dropout(x))

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        # Needed for newer transformers (>=4.45) weight-tying compatibility
        _tied_weights_keys: list = []

        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)   # 3 classes: female / male / child
            if hasattr(self, "post_init"):
                self.post_init()
            else:
                self.init_weights()

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            hidden = torch.mean(outputs.last_hidden_state, dim=1)
            logits_age = self.age(hidden)
            logits_gender = torch.softmax(self.gender(hidden), dim=1)
            return hidden, logits_age, logits_gender

    processor = Wav2Vec2Processor.from_pretrained(_MODEL_ID)
    model = AgeGenderModel.from_pretrained(_MODEL_ID)
    return processor, model


class SpeakerProfiler:
    def __init__(self):
        self._processor = None
        self._model = None
        self._device = "cpu"
        self._method = "f0"
        self._load()

    def _load(self):
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor, self._model = _build_age_gender_model()
            self._model = self._model.to(self._device).eval()
            self._method = "model"
            logger.info(f"audeering/wav2vec2-large-robust-24-ft-age-gender loaded on {self._device}")
        except Exception as e:
            logger.warning(f"Age/gender model unavailable ({e}), using F0 heuristic")
            self._method = "f0"

    def to_cpu(self) -> None:
        if self._method == "model" and self._model is not None:
            self._model.to("cpu")
            self._device = "cpu"
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
            logger.info("SpeakerProfiler moved to CPU.")

    def to_gpu(self) -> None:
        if self._method == "model" and self._model is not None:
            import torch
            if torch.cuda.is_available():
                self._model.to("cuda")
                self._device = "cuda"
                logger.info("SpeakerProfiler moved back to GPU.")

    def profile_speakers(
        self,
        audio: np.ndarray,
        sr: int,
        chunks: List[Dict],
    ) -> Dict[str, Dict]:
        from collections import defaultdict

        speaker_audio: Dict[str, List[np.ndarray]] = defaultdict(list)
        for chunk in chunks:
            sp = chunk.get("speaker", "GLOS_01")
            s = int(chunk["start"] * sr)
            e = min(int(chunk["end"] * sr), len(audio))
            seg = audio[s:e]
            # Only use segments at least 0.5s long for better accuracy
            if len(seg) >= sr * 0.5:
                speaker_audio[sp].append(seg)

        profiles: Dict[str, Dict] = {}
        for speaker, segments in speaker_audio.items():
            # Use up to 90 seconds total — more samples = better accuracy
            combined = np.concatenate(segments)
            combined = combined[: sr * 90]
            if self._method == "model":
                profiles[speaker] = self._profile_model(combined, sr)
            else:
                profiles[speaker] = self._profile_f0(combined, sr)
        return profiles

    def _profile_model(self, audio: np.ndarray, sr: int) -> Dict:
        import torch

        y = self._processor(audio, sampling_rate=sr)
        y = y["input_values"][0].reshape(1, -1)
        y_t = torch.from_numpy(y).to(self._device)

        with torch.no_grad():
            _, logits_age, logits_gender = self._model(y_t)

        age_raw = round(float(torch.sigmoid(logits_age).item()) * 100, 1)
        # The audeering model was trained on English/German speech.
        # Polish prosody (longer vowels, different formants) causes a systematic
        # upward bias of ~8–12 years. Apply a calibrated correction:
        #   - below 35: subtract 8 years (less bias in young speakers)
        #   - 35–55: subtract 10 years (peak bias range)
        #   - above 55: subtract 12 years (oldest buckets are most overestimated)
        if age_raw < 35:
            age_years = max(5.0, age_raw - 8.0)
        elif age_raw < 55:
            age_years = max(18.0, age_raw - 10.0)
        else:
            age_years = max(18.0, age_raw - 12.0)
        age_years = round(age_years, 1)

        # Gender: [female, male, child]
        g_probs = logits_gender[0].cpu().numpy()
        female_p = float(g_probs[0])
        male_p = float(g_probs[1])
        child_p = float(g_probs[2])
        idx = int(np.argmax(g_probs))
        gender_labels = ["zenski", "meski", "dziecko"]
        gender = gender_labels[idx]
        confidence = round(float(g_probs[idx]), 3)

        return {
            "gender": gender,
            "gender_probs": {
                "zenski": round(female_p, 3),
                "meski": round(male_p, 3),
                "dziecko": round(child_p, 3),
            },
            "age_estimate": age_years,
            "age_group": _age_group(age_years, gender),
            "confidence": confidence,
        }

    def _profile_f0(self, audio: np.ndarray, sr: int) -> Dict:
        try:
            import librosa
            f0, voiced_flag, _ = librosa.pyin(
                audio.astype(np.float32),
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            voiced_f0 = f0[voiced_flag > 0.5] if voiced_flag is not None else np.array([])
            voiced_f0 = voiced_f0[~np.isnan(voiced_f0)] if len(voiced_f0) > 0 else np.array([])
        except Exception:
            voiced_f0 = np.array([])

        if len(voiced_f0) == 0:
            return {"gender": None, "gender_probs": None, "age_estimate": None, "age_group": None, "confidence": None}

        f0_mean = float(np.mean(voiced_f0))
        if f0_mean > 250:
            gender = "dziecko"
        elif f0_mean >= 165:
            gender = "zenski"
        else:
            gender = "meski"

        age_est = _f0_to_age(f0_mean, gender)
        return {
            "gender": gender,
            "gender_probs": None,
            "age_estimate": age_est,
            "age_group": _age_group(age_est, gender),
            "confidence": 0.5,
        }


def _age_group(age: Optional[float], gender: Optional[str]) -> str:
    if age is None:
        return "nieznany"
    if gender == "dziecko" or age < 18:
        return "dziecko"
    if age < 30:
        return "mlody"
    if age < 55:
        return "dorosly"
    return "starszy"


def _f0_to_age(f0: float, gender: str) -> float:
    if gender == "dziecko":
        return max(5.0, min(17.0, round(300 - f0 * 0.5, 1)))
    if gender == "zenski":
        return max(18.0, min(80.0, round(300 - f0 * 0.8, 1)))
    return max(18.0, min(80.0, round(220 - f0 * 0.7, 1)))
