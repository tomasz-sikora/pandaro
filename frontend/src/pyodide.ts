// Interpreter kodu (Pyodide) dla czatu agenta — uruchamiany w przeglądarce
// (WASM), więc wyniki przetwarzania pozostają efemeryczne i prywatne.
//
// Pyodide ładowany jest leniwie z CDN przy pierwszym użyciu. Jeśli sieć jest
// niedostępna, funkcja zwraca czytelny błąd zamiast wywracać aplikację.

declare global {
  interface Window {
    loadPyodide?: (opts?: { indexURL?: string }) => Promise<any>;
  }
}

const PYODIDE_VERSION = "0.26.4";
const CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let pyodidePromise: Promise<any> | null = null;

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`Nie udało się załadować ${src}`));
    document.head.appendChild(s);
  });
}

export async function getPyodide(): Promise<any> {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      await loadScript(`${CDN}pyodide.js`);
      if (!window.loadPyodide) throw new Error("Pyodide niedostępne");
      return window.loadPyodide({ indexURL: CDN });
    })();
  }
  return pyodidePromise;
}

export interface CodeResult {
  ok: boolean;
  stdout: string;
  result: string | null;
  error: string | null;
}

export async function runPython(code: string, globals: Record<string, unknown> = {}): Promise<CodeResult> {
  try {
    const pyodide = await getPyodide();
    for (const [k, v] of Object.entries(globals)) {
      pyodide.globals.set(k, pyodide.toPy(v));
    }
    let stdout = "";
    pyodide.setStdout({ batched: (s: string) => (stdout += s + "\n") });
    const result = await pyodide.runPythonAsync(code);
    return {
      ok: true,
      stdout,
      result: result === undefined || result === null ? null : String(result),
      error: null,
    };
  } catch (e) {
    return { ok: false, stdout: "", result: null, error: String(e) };
  }
}
