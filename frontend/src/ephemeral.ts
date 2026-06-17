// Gwarancje efemeryczności: stan analizy żyje wyłącznie w pamięci karty.
// Przy zamknięciu karty (pagehide/beforeunload) lub kliknięciu „Wyczyść”
// czyścimy sesję serwera (pliki tymczasowe) i ewentualne dane przeglądarki.

import { api } from "./api";

export function registerEphemeralWipe(getSessionId: () => string | null): () => void {
  const handler = () => {
    const sid = getSessionId();
    if (sid) {
      // sendBeacon przeżywa zamknięcie karty lepiej niż fetch.
      const url = `/api/sessions/${sid}`;
      try {
        navigator.sendBeacon?.(url, new Blob([], { type: "text/plain" }));
      } catch {
        /* ignore */
      }
      // DELETE jako uzupełnienie (sendBeacon nie obsługuje metody DELETE).
      api.clearSession(sid);
    }
    wipeBrowserStorage();
  };
  window.addEventListener("pagehide", handler);
  window.addEventListener("beforeunload", handler);
  return () => {
    window.removeEventListener("pagehide", handler);
    window.removeEventListener("beforeunload", handler);
  };
}

export async function wipeBrowserStorage(): Promise<void> {
  try {
    if (indexedDB && "databases" in indexedDB) {
      // @ts-expect-error databases() nie jest jeszcze w typach TS
      const dbs = (await indexedDB.databases()) as { name?: string }[];
      for (const d of dbs) if (d.name) indexedDB.deleteDatabase(d.name);
    }
  } catch {
    /* ignore */
  }
  try {
    sessionStorage.clear();
  } catch {
    /* ignore */
  }
}
