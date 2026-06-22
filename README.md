# Heimdall – Analiza Nagrań Audio

Aplikacja do analizy nagrań audio działająca **w całości w przeglądarce** (transkrypcja Whisper przez WebAssembly), z opcjonalną integracją z Ollama do analizy LLM i RAG.

## Funkcje

| Funkcja | Opis |
|---|---|
| **Transkrypcja** | Whisper (Xenova/whisper-small, multilingual) w Web Workerze – bez serwera |
| **Diaryzacja** | Automatyczna identyfikacja mówców na podstawie przerw i energii |
| **Tłumaczenie** | Na polski i angielski via Ollama |
| **Ekstrakcja encji** | Osoby, organizacje, miejsca, daty, słowa kluczowe via Ollama |
| **Podsumowanie** | Automatyczny raport z nagrania via Ollama |
| **RAG** | Wyszukiwanie w transkrypcji przez embeddingi + zadawanie pytań |
| **Chat AI** | Pytania i odpowiedzi o treść nagrania |

## Obsługiwane formaty

MP3, MP4, M4A, WAV

## Obsługiwane języki

| Język źródłowy | Tłumaczenie na |
|---|---|
| Polski, Angielski, Rosyjski, Ukraiński, Niemiecki | Polski, Angielski |

## Uruchomienie z Docker Compose

### Wymagania
- Docker i Docker Compose
- (opcjonalnie) Ollama działające na hoście: `http://localhost:11434`

### Szybki start

```bash
# Sklonuj repozytorium
git clone <repo-url>
cd heimdall

# Uruchom
docker compose up --build

# Otwórz w przeglądarce
open http://localhost:8080
```

### Zmienna środowiskowa OLLAMA_URL

Domyślnie proxy łączy się z Ollama pod adresem `http://host.docker.internal:11434`.
Możesz zmienić:

```bash
OLLAMA_URL=http://192.168.1.10:11434 docker compose up --build
```

### Modele Ollama (opcjonalne)

Aby korzystać z analizy LLM, zainstaluj modele w Ollama:

```bash
# Model do analizy / czatu
ollama pull llama3.2

# Model do embeddingów (RAG)
ollama pull nomic-embed-text
```

## Uruchomienie lokalne (dev)

```bash
# Wymagania: Node.js 22+, pnpm 9+
npm install -g pnpm

# Instalacja zależności
pnpm install

# Start dev server (web + proxy)
pnpm dev
```

Aplikacja dostępna pod `http://localhost:5173`.

## Architektura

```
┌─────────────────────────────────────────────────────┐
│                    Przeglądarka                      │
│                                                     │
│  React SPA ──► Web Worker (Whisper WASM/WebGPU)     │
│      │         Web Worker (Embeddings WASM)          │
│      │                                              │
│      └──► Ollama (przez nginx proxy)                │
└─────────────────────────────────────────────────────┘
        ↓              ↓
  [nginx:8080]   [proxy:3001]
        │              │
        └──────────────┴──► Ollama (host:11434)
```

### Stack

| Warstwa | Technologia |
|---|---|
| Frontend | React 18, TypeScript, Vite 6, Tailwind CSS 3 |
| Stan | Zustand 5 |
| AI w przeglądarce | @huggingface/transformers (Whisper, embeddingi) |
| LLM | Ollama (llama3.2 lub inny) |
| Embeddingi | Ollama (nomic-embed-text) lub Xenova/all-MiniLM-L6-v2 |
| Proxy | Hono + @hono/node-server |
| Konteneryzacja | Docker multi-stage + nginx |

### Przepływ przetwarzania

1. **Dekodowanie** – Web Audio API → Float32Array @ 16 kHz mono
2. **Transkrypcja** – Whisper w Web Workerze (model ~244 MB, cache w przeglądarce)
3. **Diaryzacja** – Heurystyka oparta o przerwy i energię sygnału
4. **Tłumaczenie** – Ollama (prompt do wybranego języka)
5. **Ekstrakcja encji** – Ollama (JSON z osobami, org., miejscami, datami, kluczowymi słowami)
6. **Podsumowanie** – Ollama (raport strukturalny)
7. **Embeddingi** – Ollama `/api/embed` lub Xenova/all-MiniLM-L6-v2 w przeglądarce
8. **RAG** – Cosine similarity w pamięci przeglądarki
9. **Chat** – Pytania → RAG retrieval → Ollama → odpowiedź strumieniowana

## Prywatność

- Transkrypcja dzieje się **lokalnie w przeglądarce** – plik audio nie opuszcza urządzenia
- Tekst transkrypcji jest przesyłany do Ollamy tylko jeśli jest ona skonfigurowana
- Brak kont, brak logowania, brak serwerowych baz danych

## Struktura projektu

```
heimdall/
├── apps/
│   ├── web/          # React SPA (Vite)
│   └── proxy/        # Hono proxy do Ollamy
├── packages/
│   └── shared-types/ # Wspólne typy TypeScript
├── Dockerfile.web
├── Dockerfile.proxy
├── docker-compose.yml
└── nginx.conf
```
