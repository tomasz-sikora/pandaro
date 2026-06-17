// Slavic-aware text normalization + phonetic encoding.
// Lustrzane odbicie pandaro.text.translit / phonetic (Python) — dzięki temu
// zapytania w przeglądarce dopasowują się do indeksu zbudowanego po stronie
// serwera (np. warianty transliteracji cyrylicy i błędy ASR).

const CYRILLIC: Record<string, string> = {
  а: "a", б: "b", в: "v", г: "h", ґ: "g", д: "d", е: "e", є: "je", ё: "jo",
  ж: "zh", з: "z", и: "y", і: "i", ї: "ji", й: "j", к: "k", л: "l", м: "m",
  н: "n", о: "o", п: "p", р: "r", с: "s", т: "t", у: "u", ф: "f", х: "kh",
  ц: "c", ч: "ch", ш: "sh", щ: "shch", ъ: "", ы: "y", ь: "", э: "e", ю: "ju",
  я: "ja",
};

const POLISH: Record<string, string> = {
  ą: "a", ć: "c", ę: "e", ł: "l", ń: "n", ó: "o", ś: "s", ż: "z", ź: "z",
};

export function transliterate(text: string): string {
  let out = "";
  for (const ch of text.toLowerCase()) {
    if (ch in CYRILLIC) out += CYRILLIC[ch];
    else if (ch in POLISH) out += POLISH[ch];
    else out += ch;
  }
  // Strip remaining combining marks.
  return out.normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
}

export function normalizeToken(token: string): string {
  return transliterate(token).replace(/[^a-z0-9]+/g, "");
}

export function normalizeText(text: string): string {
  return text
    .trim()
    .split(/\s+/)
    .map(normalizeToken)
    .filter(Boolean)
    .join(" ");
}

const DIGRAPHS: [string, string][] = [
  ["shch", "S"], ["sch", "S"], ["sz", "S"], ["sh", "S"], ["cz", "C"],
  ["ch", "C"], ["kh", "H"], ["zh", "Z"], ["rz", "Z"], ["dz", "C"],
  ["ph", "F"], ["th", "T"], ["ck", "K"],
];

const CLASS: Record<string, string> = {
  b: "B", p: "B", w: "F", f: "F", v: "F", c: "C", z: "Z", s: "S", j: "J",
  g: "K", k: "K", q: "K", x: "KS", d: "T", t: "T", l: "L", r: "R", m: "M",
  n: "N", h: "H", y: "J",
};

const VOWELS = new Set(["a", "e", "i", "o", "u"]);

export function phoneticCode(word: string): string {
  let s = transliterate(word).replace(/[^a-z]/g, "");
  if (!s) return "";
  for (const [src, dst] of DIGRAPHS) s = s.split(src).join(dst.toLowerCase());
  const first = s[0];
  const head = /[a-z]/.test(first) ? first.toUpperCase() : first;
  const body: string[] = [];
  for (const ch of s.slice(1)) {
    if (ch >= "A" && ch <= "Z") body.push(ch);
    else if (VOWELS.has(ch)) continue;
    else body.push(CLASS[ch] ?? "");
  }
  const code = head + body.join("");
  let out = "";
  for (const c of code) if (out[out.length - 1] !== c) out += c;
  return out;
}

export function phoneticText(text: string): string {
  return text
    .trim()
    .split(/\s+/)
    .map(phoneticCode)
    .filter(Boolean)
    .join(" ");
}

export function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
