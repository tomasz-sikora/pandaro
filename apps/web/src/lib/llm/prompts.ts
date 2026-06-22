const LANG_NAMES: Record<string, string> = {
  pl: 'polski',
  en: 'angielski',
  ru: 'rosyjski',
  uk: 'ukraiński',
  de: 'niemiecki',
}

export function entityExtractionPrompt(text: string, sourceLang = 'pl'): string {
  const langNote =
    sourceLang !== 'pl'
      ? `Tekst pochodzi z języka ${LANG_NAMES[sourceLang] ?? sourceLang}. `
      : ''
  return `${langNote}Przeanalizuj poniższy tekst i wyodrębnij najważniejsze informacje. Odpowiedz WYŁĄCZNIE w formacie JSON (bez markdown, bez komentarzy).

Dla każdej encji podaj wartość w dwóch formach jeśli oryginalny język różni się od polskiego:
- format: "polska_forma (oryginalna_forma)" np. "Jan Kowalski (John Smith)"
- jeśli tekst jest po polsku, podaj tylko polską formę

Tekst (do 4000 znaków):
${text.slice(0, 4000)}

Odpowiedź JSON:
{
  "persons": ["lista imion i nazwisk — dwujęzycznie jeśli konieczne"],
  "organizations": ["lista firm, instytucji — dwujęzycznie jeśli konieczne"],
  "locations": ["lista miejsc, miast, krajów — dwujęzycznie jeśli konieczne"],
  "dates": ["lista dat i terminów"],
  "keywords": ["lista kluczowych słów i tematów po polsku (max 15)"]
}`
}

export function summaryPrompt(transcript: string, detectedLang: string): string {
  const langNote =
    detectedLang !== 'pl' && detectedLang !== 'auto'
      ? ` (język oryginalny: ${LANG_NAMES[detectedLang] ?? detectedLang})`
      : ''
  return `Stwórz szczegółowy raport z poniższej transkrypcji rozmowy${langNote}. Raport pisz po polsku.

Transkrypcja (do 6000 znaków):
${transcript.slice(0, 6000)}

Napisz raport w języku polskim zawierający:
## Streszczenie
(2-4 zdania)

## Główne tematy
(lista punktowana)

## Kluczowe decyzje i ustalenia
(lista lub "Brak")

## Uczestnicy rozmowy
(jeśli daje się zidentyfikować; podaj imię/rolę + język wypowiedzi)

## Następne kroki
(lista lub "Nie wspomniano")

Raport:`
}

export function ragQueryPrompt(context: string, question: string): string {
  return `Odpowiedz na pytanie na podstawie poniższych fragmentów transkrypcji (tekst przetłumaczony na polski). Jeśli odpowiedź nie wynika z transkrypcji, powiedz o tym wprost.

Fragmenty transkrypcji:
${context}

Pytanie: ${question}

Odpowiedź:`
}

/**
 * Reduce prompt: synthesise partial summaries from multiple windows of a long recording.
 */
export function summaryReducePrompt(partialSummaries: string, totalMinutes: number): string {
  return `Poniżej znajdują się częściowe podsumowania kolejnych fragmentów długiego nagrania (łącznie ~${totalMinutes} minut). Stwórz z nich jeden spójny, szczegółowy raport w języku polskim.

Częściowe podsumowania:
${partialSummaries}

Napisz raport w języku polskim zawierający:
## Streszczenie
(3-5 zdań)

## Główne tematy
(lista punktowana)

## Kluczowe decyzje i ustalenia
(lista lub "Brak")

## Uczestnicy rozmowy
(imiona/role)

## Następne kroki
(lista lub "Nie wspomniano")

Raport:`
}
