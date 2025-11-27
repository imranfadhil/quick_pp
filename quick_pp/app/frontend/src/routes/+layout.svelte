<script lang="ts">
	import './layout.css';
	import { onMount } from 'svelte';
	import favicon from '$lib/assets/favicon.ico';

	let { children } = $props();

	// Initialize backend DB connector once when the app mounts.
	const API_BASE = "http://localhost:6312";
	onMount(async () => {
		try {
			const res = await fetch(`${API_BASE}/quick_pp/database/init`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({}),
			});

			if (!res.ok) {
				const txt = await res.text();
				console.warn('DB init responded with non-OK status:', res.status, txt || res.statusText);
			} else {
				console.info('DB init completed successfully');
			}
		} catch (err) {
			console.warn('DB init failed:', err);
		}
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

{@render children()}
