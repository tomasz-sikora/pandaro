import { Suspense, lazy } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { AppShell } from './AppShell'

const UploadPage = lazy(() => import('../features/upload/UploadPage'))
const TranscriptPage = lazy(() => import('../features/transcript/TranscriptPage'))
const AnalysisPage = lazy(() => import('../features/analysis/AnalysisPage'))
const ChatPage = lazy(() => import('../features/chat/ChatPage'))
const SearchPage = lazy(() => import('../features/search/SearchPage'))
const SettingsPage = lazy(() => import('../features/settings/SettingsPage'))
const AgentLogPage = lazy(() => import('../features/agent-log/AgentLogPage'))

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center h-full min-h-[200px]">
      <div className="w-8 h-8 border-4 border-brand-200 border-t-brand-600 rounded-full animate-spin" />
    </div>
  )
}

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route
          path="/"
          element={
            <Suspense fallback={<LoadingSpinner />}>
              <UploadPage />
            </Suspense>
          }
        />
        <Route
          path="/transcript"
          element={
            <Suspense fallback={<LoadingSpinner />}>
              <TranscriptPage />
            </Suspense>
          }
        />
        <Route
          path="/analysis"
          element={
            <Suspense fallback={<LoadingSpinner />}>
              <AnalysisPage />
            </Suspense>
          }
        />
        <Route
          path="/chat"
          element={
            <Suspense fallback={<LoadingSpinner />}>
              <ChatPage />
            </Suspense>
          }
        />
        <Route
          path="/search"
          element={
            <Suspense fallback={<LoadingSpinner />}>
              <SearchPage />
            </Suspense>
          }
        />
        <Route
          path="/settings"
          element={
            <Suspense fallback={<LoadingSpinner />}>
              <SettingsPage />
            </Suspense>
          }
        />
        <Route
          path="/agent-log"
          element={
            <Suspense fallback={<LoadingSpinner />}>
              <AgentLogPage />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
