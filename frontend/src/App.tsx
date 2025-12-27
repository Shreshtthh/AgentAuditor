import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import Dashboard from './pages/Dashboard'
import AgentProfile from './pages/AgentProfile'
import AuditExplorer from './pages/AuditExplorer'
import SubmitAudit from './pages/SubmitAudit'
import Navbar from './components/Navbar'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
})

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('darkMode') === 'true' ||
        (!localStorage.getItem('darkMode') && window.matchMedia('(prefers-color-scheme: dark)').matches)
    }
    return false
  })

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('darkMode', String(darkMode))
  }, [darkMode])

  const toggleDarkMode = () => setDarkMode(!darkMode)

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen relative">
          {/* Background decoration */}
          <div className="fixed inset-0 -z-10 overflow-hidden">
            <div className="absolute -top-40 -right-40 w-80 h-80 bg-primary-500/10 dark:bg-primary-500/5 rounded-full blur-3xl" />
            <div className="absolute top-1/2 -left-40 w-80 h-80 bg-purple-500/10 dark:bg-purple-500/5 rounded-full blur-3xl" />
            <div className="absolute -bottom-40 right-1/3 w-80 h-80 bg-pink-500/10 dark:bg-pink-500/5 rounded-full blur-3xl" />
          </div>
          
          <Navbar darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
          
          <main className="relative">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/agents/:agentId" element={<AgentProfile />} />
              <Route path="/audits" element={<AuditExplorer />} />
              <Route path="/submit" element={<SubmitAudit />} />
            </Routes>
          </main>
          
          {/* Footer */}
          <footer className="mt-auto py-8 border-t border-slate-200/50 dark:border-dark-700/50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex flex-col md:flex-row justify-between items-center gap-4 text-sm text-slate-500 dark:text-slate-400">
                <p>© 2025 Cortensor Agent Auditor. Built for the Cortensor Hackathon.</p>
                <div className="flex items-center gap-4">
                  <span className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    Arbitrum Sepolia
                  </span>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </Router>
    </QueryClientProvider>
  )
}

export default App
