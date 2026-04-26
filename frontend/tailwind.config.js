/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,jsx,ts,tsx}",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  presets: [require("nativewind/preset")],
  theme: {
    extend: {
      colors: {
        // Listen brand palette (dark neon)
        bg: {
          DEFAULT: "#0B1020",
          elevated: "#111733",
          card: "#151B36",
          soft: "#1C2447",
        },
        border: {
          DEFAULT: "#2A3358",
          soft: "#1F2748",
        },
        text: {
          DEFAULT: "#E6EAF7",
          dim: "#8F9BC5",
          muted: "#5E6A95",
        },
        accent: {
          DEFAULT: "#5EE6FF",   // neon cyan
          soft: "#3EC7E0",
          deep: "#1E88A8",
        },
        brand: {
          purple: "#8A5CF6",
          pink: "#E06CF0",
        },
        ok: "#4EE6A8",
        warn: "#FFC061",
        err: "#FF6B8A",
      },
      fontFamily: {
        sans: ["System"],
      },
    },
  },
  plugins: [],
};
