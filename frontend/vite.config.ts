import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server runs on 5173 and proxies API/WebSocket to the backend on :9090.
// In production the backend serves the built `dist/` directly.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:9090",
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
