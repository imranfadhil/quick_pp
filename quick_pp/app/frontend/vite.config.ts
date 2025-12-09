import tailwindcss from '@tailwindcss/vite';
import devtoolsJson from 'vite-plugin-devtools-json';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit(), devtoolsJson()],
	build: {
		rollupOptions: {
			output: {
				manualChunks(id) {
					if (id.includes('node_modules')) {
						if (id.includes('svelte')) {
							return 'vendor';
						}
						if (id.includes('bits-ui') || id.includes('@lucide') || id.includes('@tabler') || id.includes('vaul-svelte') || id.includes('svelte-sonner')) {
							// return 'ui';
						}
						if (id.includes('plotly') || id.includes('d3-')) {
							return 'charts';
						}
						if (id.includes('datatables')) {
							return 'tables';
						}
						if (id.includes('clsx') || id.includes('tailwind-merge') || id.includes('tailwind-variants') || id.includes('zod') || id.includes('@internationalized')) {
							return 'utils';
						}
						// All other node_modules go to vendor
						return 'vendor';
					}
				}
			}
		}
	}
});
