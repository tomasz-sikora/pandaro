import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Tag, Users, Building2, MapPin, Calendar, FileText, AlertCircle, UserCircle2, Quote } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useSessionStore } from '../../store/sessionStore'
import type { SpeakerProfile } from '@pandaro/shared-types'

function Section({
  icon: Icon,
  title,
  items,
  color,
}: {
  icon: React.ElementType
  title: string
  items: string[]
  color: string
}) {
  if (items.length === 0) return null
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className={`flex items-center gap-2 mb-3 text-sm font-semibold ${color}`}>
        <Icon className="w-4 h-4" />
        {title}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {items.map((item) => (
          <span
            key={item}
            className="text-xs bg-slate-50 text-slate-700 border border-slate-200 px-2.5 py-1 rounded-full"
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  )
}

function SpeakerProfileCard({ speaker, profile }: { speaker: string; profile: SpeakerProfile }) {
  const SPEAKER_COLORS: Record<string, string> = {
    GŁOS_01: 'bg-blue-100 text-blue-800 border-blue-200',
    GŁOS_02: 'bg-violet-100 text-violet-800 border-violet-200',
    GŁOS_03: 'bg-amber-100 text-amber-800 border-amber-200',
    GŁOS_04: 'bg-green-100 text-green-800 border-green-200',
    GŁOS_05: 'bg-rose-100 text-rose-800 border-rose-200',
    GŁOS_06: 'bg-cyan-100 text-cyan-800 border-cyan-200',
  }
  const colorCls = SPEAKER_COLORS[speaker] ?? 'bg-slate-100 text-slate-700 border-slate-200'

  const displayName = profile.display_name ?? speaker
  const hasDisplayName = displayName !== speaker

  const GENDER_LABELS: Record<string, string> = {
    zenski: 'żeński', meski: 'męski', dziecko: 'dziecko',
  }
  const EMOTION_LABELS: Record<string, string> = {
    anger: 'złość', happiness: 'radość', neutral: 'neutralny', sadness: 'smutek',
  }

  return (
    <div className={`rounded-xl border p-3 ${colorCls}`}>
      <div className="flex items-center gap-2 mb-2">
        <UserCircle2 className="w-4 h-4 shrink-0" />
        <div className="flex-1 min-w-0">
          <span className="font-semibold text-sm">{displayName}</span>
          {hasDisplayName && (
            <span className="ml-1.5 text-xs opacity-50">{speaker}</span>
          )}
        </div>
        {profile.confidence != null && (
          <span className="ml-auto text-xs opacity-50">pewność: {Math.round(profile.confidence * 100)}%</span>
        )}
      </div>
      <div className="space-y-0.5 text-xs">
        {profile.gender && (
          <p><span className="font-medium">Płeć:</span> {GENDER_LABELS[profile.gender] ?? profile.gender}</p>
        )}
        {profile.age_group && (
          <p>
            <span className="font-medium">Wiek:</span> {profile.age_group}
            {profile.age_estimate != null && ` (~${Math.round(profile.age_estimate)} lat)`}
          </p>
        )}
        {profile.emotion && (
          <p><span className="font-medium">Emocja:</span> {EMOTION_LABELS[profile.emotion] ?? profile.emotion}</p>
        )}
        {profile.speech_rate_label && (
          <p>
            <span className="font-medium">Tempo mowy:</span> {profile.speech_rate_label}
            {profile.speech_rate_syllables_per_sec != null &&
              ` (${profile.speech_rate_syllables_per_sec} syl/s)`}
          </p>
        )}
        {profile.snr_label && (
          <p>
            <span className="font-medium">Jakość audio:</span> {profile.snr_label}
            {profile.snr_db != null && ` (${profile.snr_db} dB)`}
          </p>
        )}
        {/* Gender probability breakdown */}
        {profile.gender_probs && (
          <div className="mt-1 flex gap-1 flex-wrap">
            {Object.entries(profile.gender_probs).map(([k, v]) => (
              <span key={k} className="text-xs bg-white bg-opacity-50 rounded px-1">
                {GENDER_LABELS[k] ?? k}: {Math.round(v * 100)}%
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default function AnalysisPage() {
  const { session } = useSessionStore()
  const navigate = useNavigate()

  useEffect(() => {
    if (!session) navigate('/')
  }, [session, navigate])

  if (!session) return null

  const { entities, report, summary, speakerProfiles, quotesAndFacts, topics } = session
  const profileEntries = Object.entries(speakerProfiles ?? {})

  const noData = !entities && !report && !summary && profileEntries.length === 0

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-6 py-4">
        <h1 className="font-semibold text-slate-900">Analiza nagrania</h1>
        <p className="text-sm text-slate-500 mt-0.5 truncate">{session.fileName}</p>
      </div>

      <div className="flex-1 overflow-auto p-6 space-y-4">
        {noData && (
          <div className="flex items-start gap-3 bg-amber-50 border border-amber-200 rounded-xl p-4">
            <AlertCircle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-amber-800">Brak danych analizy</p>
              <p className="text-sm text-amber-700 mt-0.5">
                Analiza LLM wymaga działającego Ollamy. Skonfiguruj URL w Ustawieniach.
              </p>
            </div>
          </div>
        )}

        {/* Quotes and facts */}
        {quotesAndFacts && quotesAndFacts.quotes.length > 0 && (
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-slate-700">
              <Quote className="w-4 h-4 text-brand-600" />
              Kluczowe cytaty
            </div>
            <div className="space-y-3">
              {quotesAndFacts.quotes.map((q, i) => (
                <blockquote key={i} className="border-l-2 border-brand-300 pl-3 py-0.5">
                  <p className="text-sm text-slate-800 italic">„{q.text}"</p>
                  <footer className="text-xs text-slate-500 mt-1">
                    {q.speaker}
                    {q.timestamp && <span className="ml-1 text-slate-400">@ {q.timestamp}</span>}
                    {q.significance && <span className="ml-2 text-brand-600">— {q.significance}</span>}
                  </footer>
                </blockquote>
              ))}
            </div>
          </div>
        )}

        {/* Facts / decisions */}
        {quotesAndFacts && (quotesAndFacts.facts.length > 0 || quotesAndFacts.decisions.length > 0) && (
          <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-3">
            {quotesAndFacts.facts.length > 0 && (
              <>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Fakty</p>
                <ul className="space-y-1">
                  {quotesAndFacts.facts.map((f, i) => (
                    <li key={i} className="text-sm text-slate-700 flex gap-2">
                      <span className="text-slate-400 shrink-0">{f.speaker}:</span>
                      <span>{f.text}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
            {quotesAndFacts.decisions.length > 0 && (
              <>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mt-2">Decyzje i ustalenia</p>
                <ul className="space-y-1 list-disc pl-4">
                  {quotesAndFacts.decisions.map((d, i) => (
                    <li key={i} className="text-sm text-slate-700">{d.text}</li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}

        {/* Topics timeline */}
        {topics && topics.length > 0 && (
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Tematy</p>
            <div className="space-y-1.5">
              {topics.map((t, i) => {
                const startMin = Math.floor(t.start_sec / 60)
                const startSec = Math.floor(t.start_sec % 60)
                return (
                  <div key={i} className="flex items-baseline gap-3 text-sm">
                    <span className="text-xs text-slate-400 font-mono shrink-0">
                      {startMin}:{startSec.toString().padStart(2, '0')}
                    </span>
                    <span className="text-slate-700">{t.topic}</span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Speaker profiles */}
        {profileEntries.length > 0 && (
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-slate-700">
              <Users className="w-4 h-4" />
              Profile mówców
            </div>
            <div className="grid grid-cols-2 gap-2">
              {profileEntries.sort(([a], [b]) => a.localeCompare(b)).map(([sp, profile]) => (
                <SpeakerProfileCard key={sp} speaker={sp} profile={profile} />
              ))}
            </div>
          </div>
        )}

        {/* Summary card */}
        {summary && (
          <div className="bg-brand-50 border border-brand-200 rounded-xl p-4">
            <p className="text-sm font-semibold text-brand-700 mb-2">Streszczenie</p>
            <p className="text-sm text-brand-900 leading-relaxed">{summary}</p>
          </div>
        )}

        {/* Entities */}
        {entities && (
          <>
            <Section icon={Users} title="Osoby" items={entities.persons} color="text-blue-700" />
            <Section icon={Building2} title="Organizacje" items={entities.organizations} color="text-violet-700" />
            <Section icon={MapPin} title="Miejsca" items={entities.locations} color="text-green-700" />
            <Section icon={Calendar} title="Daty i terminy" items={entities.dates} color="text-amber-700" />
            <Section icon={Tag} title="Słowa kluczowe" items={entities.keywords} color="text-slate-700" />
          </>
        )}

        {/* Full report */}
        {report && (
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-slate-700">
              <FileText className="w-4 h-4" />
              Pełny raport
            </div>
            <div className="prose prose-sm max-w-none text-slate-700 [&_h2]:text-base [&_h2]:font-semibold [&_h2]:mt-4 [&_h2]:mb-2 [&_h3]:text-sm [&_h3]:font-semibold [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-0.5 [&_p]:my-1.5 [&_strong]:font-semibold">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
