import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
  worker: {
    format: 'es',
  },
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
    proxy: {
      '/ollama': {
        target: 'http://localhost:11434',
        rewrite: (p) => p.replace(/^\/ollama/, ''),
        changeOrigin: true,
      },
      '/api': {
        target: 'http://localhost:3001',
        rewrite: (p) => p.replace(/^\/api/, ''),
        changeOrigin: true,
      },
    },
  },
  build: {
    target: 'esnext',
  },
})
