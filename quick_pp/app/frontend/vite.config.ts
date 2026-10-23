import tailwindcss from '@tailwindcss/vite';
import devtoolsJson from 'vite-plugin-devtools-json';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit(), devtoolsJson()],
	server: {
		host: '0.0.0.0',
		port: 5469,
		allowedHosts: ['localhost', '127.0.0.1', 'qpp-frontend', '.trycloudflare.com'],
		fs: {
			strict: true,
			deny: ['.env', '.env.*', '*.{crt,pem}', '**/.git/**']
		},
		watch: {
			usePolling: true
		}
	},
	ssr: {
		noExternal: true
	}
});

