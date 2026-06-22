import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import { useSessionStore } from './store/sessionStore'

// Clear all session data on page unload / reload / close
window.addEventListener('beforeunload', () => {
  useSessionStore.getState().clearSession()
})

const root = document.getElementById('root')
if (!root) throw new Error('Root element not found')

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
