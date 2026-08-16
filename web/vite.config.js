import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build lands in web/dist and is copied into the API image, which serves it at /.
// One container, one command — adding a second service just to serve six static files
// would break the "fresh clone to working system in one documented command" promise
// for no benefit.
//
// In dev, `npm run dev` proxies the API so the page can be worked on with hot reload
// against the real backend rather than a mock.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/runs": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/version": "http://localhost:8000",
      "/watch": "http://localhost:8000",
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
