import adapterNode from '@sveltejs/adapter-node';
import adapterAuto from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		// Use adapter-auto for dev (automatic detection), adapter-node for production builds
		adapter: process.env.NODE_ENV === 'production' 
			? adapterNode({ out: 'build' }) 
			: adapterAuto()
	}
};

export default config;
