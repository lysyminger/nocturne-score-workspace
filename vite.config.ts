import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { alphaTab } from "@coderline/alphatab-vite";

export default defineConfig({
  plugins: [react(), alphaTab()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765"
    }
  }
});

