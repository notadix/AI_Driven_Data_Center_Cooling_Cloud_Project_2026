/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ashrae: {
          cold: '#00BFFF',     // < 18°C
          nominal: '#10B981',  // 18°C - 27°C (Optimal)
          warning: '#F59E0B',  // 27°C - 32°C (Warning)
          critical: '#EF4444', // >= 32°C (Critical SLA Breach)
        },
        cyber: {
          dark: '#0B0F19',
          card: '#111827',
          border: '#1E293B',
          glow: '#06B6D4',
          accent: '#10B981',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
        sans: ['Outfit', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'glow-cyan': '0 0 20px rgba(6, 182, 212, 0.35)',
        'glow-green': '0 0 20px rgba(16, 185, 129, 0.35)',
        'glow-red': '0 0 25px rgba(239, 68, 68, 0.45)',
      },
    },
  },
  plugins: [],
};
