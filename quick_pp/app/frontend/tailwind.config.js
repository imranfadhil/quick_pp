// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      // Define a custom color palette for a technical/professional look
      colors: {
        'primary-tech': '#1e3a8a', // Deep Blue
        'secondary-data': '#059669', // Emerald Green
        'background-dark': '#0f172a', // Slate 900 for sidebar/headers
        'text-light': '#cbd5e1', // Slate 300
      },
    },
  },
  plugins: [],
}