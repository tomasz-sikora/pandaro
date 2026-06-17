# 🐼 Pandaro

**Lokalne, efemeryczne narzędzie do analizy nagrań rozmów i spotkań** (telefon,
meeting). Transkrypcja (Whisper large‑v3), diaryzacja rozmówców, analiza głosu
(wiek / płeć / emocje), cechy akustyczne / OSINT, słowa kluczowe i encje,
tłumaczenie (UK/RU → PL), hierarchiczne podsumowanie oraz **hybrydowy RAG**
(odporny na warianty transliteracji cyrylicy) z czatem agenta i interpreterem
kodu Pyodide. Interfejs jest w języku polskim i działa w przeglądarce Chrome na
`http://localhost:8080`.

> **Prywatność / efemeryczność:** stan analizy żyje **wyłącznie w karcie
> przeglądarki**. Zamknięcie karty lub przycisk **„Wyczyść”** kasuje wszystko;
> backend jest bezstanowy i nie przechowuje nagrań ani wyników między
> uruchomieniami (poza pamięcią podręczną modeli).

---

## Architektura

```
Chrome SPA (po polsku)                         Python backend (FastAPI, GPU, Docker)
 ├─ stan efemeryczny (pamięć karty)             ├─ Orkiestrator faz (re-runnable, streaming)
 ├─ RAG: BM25 + fonetyka + wektory (RRF)        ├─ ASR  (WhisperX / faster-whisper, large-v3)
 ├─ czat agenta + interpreter Pyodide  ──REST/WS─┤─ Diaryzacja (pyannote → NeMo pluggable)
 └─ eksport/import .pandaro                      ├─ Paralingwistyka + akustyka (OSINT)
                                                 ├─ NER / słowa kluczowe / tłumaczenie
                                                 ├─ Proxy embeddingów → Ollama (bge-m3)
                                                 └─ Proxy LLM → Ollama (Gemma)
Zewnętrzne: Ollama (host) · HuggingFace (modele, HF_TOKEN, cache)
```

Ciężkie modele ML wymagają GPU, więc backend jest **bezstanowym silnikiem
obliczeniowym**, a przeglądarka jest **jedynym trwałym magazynem sesji**. Indeks
RAG i interpreter kodu działają po stronie klienta (WASM/TS), żeby spełnić
wymóg efemeryczności. Szczegóły projektu w sekcjach poniżej i w `backend/`.

### Fazy potoku (każda osobno uruchamialna ponownie)

`ingest → vad → asr → align → diarize → merge → speaker_id → paralinguistics →
acoustics → translate → keywords → summarize → rag → report`

---

## Wymagania

- **GPU NVIDIA** (zalecane RTX 3090 / 24 GB) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Docker** + Docker Compose (BuildKit)
- **[Ollama](https://ollama.com/) na hoście** z modelami LLM i embeddingów:
  ```bash
  ollama pull gemma3:27b      # lub „gemma4”, gdy dostępne na Twoim Ollama
  ollama pull bge-m3
  ```
- **`HF_TOKEN`** z [HuggingFace](https://huggingface.co/settings/tokens) z
  zaakceptowaną licencją modeli `pyannote/speaker-diarization-3.1` i
  `pyannote/segmentation-3.0` (diaryzacja jest gated).

---

## Uruchomienie (Docker — tryb docelowy)

```bash
cp config/.env.example .env          # uzupełnij HF_TOKEN
export HF_TOKEN=hf_...                # używany jako build-secret (nie trafia do warstwy obrazu)
docker compose up --build
```

Otwórz **http://localhost:8080** w Chrome.

- Modele HF są **wstępnie pobierane podczas budowy** (`docker/preload_models.py`)
  i trzymane w wolumenie `pandaro-cache` (`HF_HOME`), więc nie są pobierane
  ponownie przy kolejnych startach/buildach (cache mounts BuildKit + wolumen).
- `HF_TOKEN` jest przekazywany jako **sekret budowania**, nie jest zapisywany w
  żadnej warstwie obrazu.
- Kontener łączy się z **Ollamą na hoście** przez `host.docker.internal:11434`.

### Zarządzanie VRAM (24 GB współdzielone z Ollamą)

Tylko **jeden ciężki model** rezyduje na GPU naraz (blokada GPU + jawny offload
`to('cpu')`/`empty_cache()` między fazami). Ollama jest koordynowana przez
`keep_alive=0`, aby Gemma zwalniała pamięć, gdy potok potrzebuje VRAM na ASR.
Tryb `PANDARO_LOW_VRAM=true` włącza int8 i agresywny offload.

---

## Rozwój lokalny (bez Dockera)

**Backend** (działa też bez GPU — providerzy mają deterministyczne atrapy):

```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -e .
PANDARO_ASR_BACKEND=stub python -m pandaro.main      # API na :8080
# testy + lint:
pip install pytest ruff && pytest -q && ruff check .
```

> `PANDARO_ASR_BACKEND=stub` (oraz atrapy diaryzacji/paralingwistyki) pozwalają
> uruchomić **cały potok bez GPU i modeli** — przydatne do dewelopmentu i CI.

**Frontend** (Vite + React + TypeScript, UI po polsku):

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, proxy /api → :8080
npm run build      # produkcyjny build → frontend/dist (serwowany przez backend)
```

---

## Funkcje kluczowe

- **Precyzja ASR + pewność:** Whisper **large‑v3** (fp16), słowne znaczniki
  czasu i per‑słowo `confidence`; słowa o niskiej pewności są podkreślane i
  klikalne. Podpowiedzi słownika (nazwiska/terminy) z presetu biasują ASR.
- **RAG słowiański:** chunking po turach rozmówców; wyszukiwanie hybrydowe
  (gęste bge‑m3 + BM25 + **fonetyka/transliteracja** Double‑Metaphone‑like)
  łączone przez **RRF**; odpowiedzi agenta cytują fragmenty `[rozmówca, mm:ss]`
  jako dowód, z możliwością przeskoku do nagrania.
- **Eksport/import `.pandaro`:** pełna sesja (transkrypt, analizy, podsumowania,
  fragmenty + wektory RAG, presety) w jednym pliku; po imporcie nie trzeba
  ponownie uruchamiać potoku. Dodatkowo SRT/VTT/Markdown.
- **Re-run faz:** każdą fazę (np. „Budowa indeksu RAG”, „Diaryzacja”) można
  uruchomić ponownie z UI.
- **Asystent + Pyodide:** czat z RAG oraz interpreter kodu w przeglądarce do
  dalszej obróbki wyników (np. tabela kto powiedział X, wykres czasu mówienia).

---

## Konfiguracja

Wszystkie ustawienia są sterowane zmiennymi środowiskowymi (prefiks `PANDARO_`).
Pełna lista z opisami: [`config/.env.example`](config/.env.example).

| Zmienna | Domyślnie | Opis |
| --- | --- | --- |
| `PANDARO_LLM_MODEL` | `gemma4` | Model LLM w Ollama (fallback `gemma3:27b`). |
| `PANDARO_EMBEDDING_MODEL` | `bge-m3` | Model embeddingów (RAG). |
| `PANDARO_ASR_MODEL` | `large-v3` | Model Whisper. |
| `PANDARO_ASR_BACKEND` | `faster-whisper` | `faster-whisper` \| `whisperx` \| `stub`. |
| `PANDARO_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu`. |
| `PANDARO_LOW_VRAM` | `false` | int8 + agresywny offload. |
| `HF_TOKEN` | — | Token HuggingFace (modele gated). |

---

## Struktura repozytorium

```
backend/    FastAPI + orkiestrator + providerzy + logika tekstu (testy)
frontend/   SPA (React/TS, po polsku): transkrypt, fala, RAG, czat, eksport
docker/     Dockerfile (multi-stage CUDA) + preload modeli HF
config/     .env.example
compose.yaml
```

## Licencja

Zobacz [LICENSE](LICENSE).
