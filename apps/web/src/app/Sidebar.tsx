import { NavLink, useNavigate } from 'react-router-dom'
import {
  Upload,
  FileText,
  BarChart2,
  MessageSquare,
  Search,
  Settings,
  Mic,
  Trash2,
  Brain,
  RefreshCw,
} from 'lucide-react'
import { useSessionStore } from '../store/sessionStore'
import { useSettingsStore } from '../store/settingsStore'
import { useAgentPipeline } from '../hooks/useAgentPipeline'

const nav = [
  { to: '/', label: 'Nowe nagranie', icon: Upload, always: true },
  { to: '/transcript', label: 'Transkrypcja', icon: FileText, always: false },
  { to: '/analysis', label: 'Analiza', icon: BarChart2, always: false },
  { to: '/agent-log', label: 'Log agenta', icon: Brain, always: false },
  { to: '/search', label: 'Szukaj', icon: Search, always: false },
  { to: '/chat', label: 'Rozmowa z AI', icon: MessageSquare, always: false },
]

export function Sidebar() {
  const { session, clearSession } = useSessionStore()
  const { settings } = useSettingsStore()
  const { process, cancel } = useAgentPipeline()
  const navigate = useNavigate()
  const hasSession = session !== null

  const handleClear = () => {
    cancel()
    clearSession(settings.transcribeUrl)
    navigate('/')
  }

  const isProcessing =
    session !== null &&
    session.processing.step !== 'idle' &&
    session.processing.step !== 'done' &&
    session.processing.step !== 'error'

  const handleReprocess = () => {
    const file = (session as any)?.sourceFile as File | undefined
    if (!file) { navigate('/'); return }
    cancel()
    // Small delay so cancel() settles before new process starts
    setTimeout(() => process(file), 100)
    navigate('/transcript')
  }

  return (
    <aside className="w-56 flex flex-col bg-white border-r border-slate-200 shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-2 px-4 py-5 border-b border-slate-100">
        <div className="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center">
          <Mic className="w-4 h-4 text-white" />
        </div>
        <span className="font-semibold text-slate-800 text-sm">Pandaro</span>
      </div>

      {/* Session info + controls */}
      {session && (
        <div className="mx-2 mt-2 bg-brand-50 rounded-lg overflow-hidden">
          <div className="px-3 py-2">
            <p className="text-xs font-medium text-brand-700 truncate">
              {session.fileName}
            </p>
            {session.duration != null && (
              <p className="text-xs text-brand-500">
                {Math.round(session.duration)}s
              </p>
            )}
          </div>
          {/* Re-process button — only when not currently processing */}
          {!isProcessing && (session as any)?.sourceFile && (
            <button
              onClick={handleReprocess}
              className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-brand-700 hover:bg-brand-100 transition-colors border-t border-brand-100"
              title="Ponów przetwarzanie tego samego pliku z aktualnymi ustawieniami"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Ponów przetwarzanie
            </button>
          )}
          {/* Cancel button — only when processing */}
          {isProcessing && (
            <button
              onClick={cancel}
              className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-amber-700 hover:bg-amber-50 transition-colors border-t border-brand-100"
              title="Zatrzymaj przetwarzanie"
            >
              <span className="w-3.5 h-3.5 flex items-center justify-center">■</span>
              Zatrzymaj
            </button>
          )}
          <button
            onClick={handleClear}
            className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-red-600 hover:bg-red-50 transition-colors border-t border-brand-100"
            title="Wyczyść analizę i zacznij od nowa"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Wyczyść analizę
          </button>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex-1 px-2 py-3 space-y-0.5">
        {nav.map(({ to, label, icon: Icon, always }) => {
          const disabled = !always && !hasSession
          return (
            <NavLink
              key={to}
              to={disabled ? '#' : to}
              onClick={(e) => disabled && e.preventDefault()}
              className={({ isActive }) =>
                [
                  'flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                  disabled
                    ? 'text-slate-300 cursor-not-allowed'
                    : isActive
                    ? 'bg-brand-50 text-brand-700'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900',
                ].join(' ')
              }
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </NavLink>
          )
        })}
      </nav>

      {/* Settings */}
      <div className="px-2 py-3 border-t border-slate-100">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            [
              'flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
              isActive
                ? 'bg-brand-50 text-brand-700'
                : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900',
            ].join(' ')
          }
        >
          <Settings className="w-4 h-4 shrink-0" />
          Ustawienia
        </NavLink>
      </div>
    </aside>
  )
}
