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
          dark: '#000000',
          card: '#1C1C1E',
          border: '#2C2C2E',
          glow: '#32ADE6',
          accent: '#10B981',
        },
        // Overrides Tailwind's built-in cyan shades (the app's chrome accent) with a
        // cool ice-blue/teal scale (Apple's "Cyan" system accent), site-wide, without
        // touching every cyan-* class site. The ASHRAE heat-map colors above are untouched.
        cyan: {
          200: '#A8DFF5',
          300: '#6AC4DC',
          400: '#32ADE6',
          500: '#1C8FC4',
        },
        // `slate` is the actual dominant panel/border/text color everywhere in the app
        // (bg-slate-950, border-slate-800, text-slate-400, ...) - far more usages than
        // the shared .glass-panel class. Tailwind's slate is a cool blue-gray, which is
        // what kept the UI reading "bluish" after only the accent was retinted. This
        // swaps every used shade for Apple's neutral (no blue hue) system gray scale.
        slate: {
          100: '#F5F5F7',
          200: '#E5E5EA',
          300: '#C7C7CC',
          400: '#98989D',
          500: '#8E8E93',
          600: '#6E6E73',
          700: '#3A3A3C',
          800: '#2C2C2E',
          900: '#1C1C1E',
          950: '#000000',
        },
        // The 3 stray decorative icon accents (wind, water drop, fan-speed slider) were
        // the only remaining blue; neutral gray keeps them reading as plain SF-Symbols-
        // style icons instead of clashing with the new warm accent.
        blue: {
          400: '#98989D',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
        sans: ['-apple-system', 'BlinkMacSystemFont', '"SF Pro Text"', 'Outfit', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'glow-cyan': '0 0 20px rgba(50, 173, 230, 0.35)',
        'glow-green': '0 0 20px rgba(16, 185, 129, 0.35)',
        'glow-red': '0 0 25px rgba(239, 68, 68, 0.45)',
      },
    },
  },
  plugins: [],
};
