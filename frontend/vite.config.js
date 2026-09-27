import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The backend mounts the whole API under /api and Prometheus under /metrics.
// During development Vite proxies exactly those paths, so the frontend code is
// identical whether it is served by Vite or by FastAPI in production.
const backend = process.env.VITE_BACKEND_URL || 'http://localhost:8000';
const proxy = {
  target: backend,
  changeOrigin: true,
};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': proxy,
      '/metrics': proxy,
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
  test: {
    // Node environment on purpose: these are source- and client-contract
    // tests that need no DOM, so they run identically on macOS, Windows, Linux
    // and in CI without pulling in a headless browser. Component tests would
    // require jsdom, which is deliberately not a dependency.
    environment: 'node',
    globals: true,
    include: ['src/**/*.test.{js,jsx}'],
    restoreMocks: true,
  },
});
