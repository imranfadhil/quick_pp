<script lang="ts">
	import * as Sidebar from "$lib/components/ui/sidebar/index.js";
	import type { Icon } from "@tabler/icons-svelte";
	import { page } from '$app/stores';
	import { workspace, selectWell } from '$lib/stores/workspace';
	import { goto } from '$app/navigation';
	import { onDestroy } from 'svelte';

	const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

	let project: any = $state(null);
	let wells: string[] = $state([]);
	let loadingWells = $state(false);
	let selectedWell: any = $state(null);
	let selectedWellName: string = $state('');

	async function fetchWells(projectId: string | number) {
		loadingWells = true;
		wells = [];
		try {
			const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells`);
			if (res.ok) {
				const data = await res.json();
				wells = data?.wells || [];
			}
		} catch (e) {
			console.warn('Failed to fetch wells for sidebar', e);
		} finally {
			loadingWells = false;
		}
	}

	const unsubscribe = workspace.subscribe((w) => {
		if (w && w.project && w.project.project_id) {
			project = { ...w.project };
			selectedWell = w.selectedWell ?? null;
			selectedWellName = selectedWell?.name ?? '';
			// fetch wells for this project
			fetchWells(w.project.project_id);
		} else {
			project = null;
			wells = [];
			selectedWell = null;
			selectedWellName = '';
		}
	});

	onDestroy(() => unsubscribe());

		let { items }: { items: { title: string; url: string; icon?: Icon }[] } = $props();

		function isActive(url: string) {
			const path = $page.url.pathname;
			if (!url) return false;
			return path === url || (url !== '/' && path.startsWith(url));
		}

		function computeHref(itemUrl: string) {
			if (!project) return itemUrl;
			try {
				if (itemUrl && itemUrl.startsWith('/wells')) {
					const suffix = itemUrl.replace(/^\/wells/, '');
					if (selectedWell && selectedWell.name) {
						return `/wells/${project.project_id}/${encodeURIComponent(selectedWell.name)}${suffix}`;
					}
					return `/wells/${project.project_id}`;
				}
			} catch (e) {
				console.warn('Failed to compute href for nav item', e);
			}
			return itemUrl;
		}
</script>

<Sidebar.Group>	
	<Sidebar.GroupLabel>Well Analysis</Sidebar.GroupLabel>
	<Sidebar.GroupContent class="flex flex-col gap-2">
		{#if project}
			<div class="px-2 py-2">
				<div class="text-sm font-semibold">{project.name}</div>
				{#if loadingWells}
					<div class="text-sm">Loading wells…</div>
				{:else}
				{#if wells && wells.length}
					<div class="mt-2">
						<select id="well-select" class="input w-full mt-1 text-sm h-9" value={selectedWellName} onchange={(e) => {
							const name = (e.target as HTMLSelectElement).value;
							if (!name) return;
							selectedWellName = name;
							// update shared workspace and navigate
							selectWell({ id: name, name });
							goto(`/wells/${project.project_id}/${encodeURIComponent(String(name))}`);
						}}>
							<option value="">— select well —</option>
							{#each wells as w}
								<option value={w}>{w}</option>
							{/each}
						</select>
					</div>
				{:else}
					<div class="text-sm text-muted-foreground mt-2">No wells in this project.</div>
				{/if}
				{/if}
			</div>
		{:else}
			<div class="px-2 py-2 text-sm text-muted-foreground">No project selected <a href="/projects" class="ml-1">Open Projects</a></div>
		{/if}
		<Sidebar.Menu>
			
				{#each items as item (item.title)}
					<Sidebar.MenuItem>
						<Sidebar.MenuButton tooltipContent={item.title}>
							{#snippet child({ props })}
								<a href={computeHref(item.url)} {...props}
									class="{isActive(computeHref(item.url)) ? 'bg-panel-foreground/5 font-semibold' : ''} flex items-center gap-2 w-full"
									aria-current={isActive(computeHref(item.url)) ? 'page' : undefined}
											onclick={(e) => {
										e.preventDefault();
										const href = computeHref(item.url);
										// If navigating to a per-well page, ensure workspace selectedWell is set
										if (project && selectedWell && item.url.startsWith('/wells')) {
											selectWell({ id: selectedWell.id ?? selectedWell.name, name: selectedWell.name });
										}
										goto(href);
									}}
								>
									{#if item.icon}
										<item.icon />
									{/if}
									<span>{item.title}</span>
								</a>
							{/snippet}
						</Sidebar.MenuButton>
					</Sidebar.MenuItem>
				{/each}
		</Sidebar.Menu>
	</Sidebar.GroupContent>
</Sidebar.Group>
