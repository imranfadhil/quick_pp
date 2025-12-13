<script lang="ts">
	import * as Sidebar from "$lib/components/ui/sidebar/index.js";
	import type { Icon } from "@tabler/icons-svelte";
	import { page } from '$app/stores';
	import { workspace, selectWell, setDepthFilter, clearDepthFilter, setZoneFilter, clearZoneFilter } from '$lib/stores/workspace';
	import { goto } from '$app/navigation';
	import { onMount, onDestroy } from 'svelte';

	const API_BASE = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:6312';

	let project: any = $state(null);
	let wells: any[] = $state([]);
	let loadingWells = $state(false);
	let selectedWell: any = $state(null);
	let selectedWellName: string = $state('');
	
	// Depth filter state
	let depthFilterEnabled = $state(false);
	let minDepth: number | null = $state(null);
	let maxDepth: number | null = $state(null);
	
	// Zone filter state
	let zoneFilterEnabled = $state(false);
	let zones: string[] = $state([]);
	let selectedZones: string[] = $state([]);
	let loadingZones = $state(false);
	let zonesOpen = $state(false);
	let _lastSelectedWellName: string | null = null;
	let zonesWrapper: HTMLElement | null = $state(null);


	async function fetchWells(projectId: string | number) {
		loadingWells = true;
		wells = [];
		try {
			const res = await fetch(`${API_BASE}/quick_pp/database/projects/${projectId}/wells`);
			if (res.ok) {
				const data = await res.json();
				wells = data?.wells || [];
				wells.sort((a, b) => a.name.localeCompare(b.name));
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
				// fetch available zones only when the selected well changed
				const curWellName = w.selectedWell && w.selectedWell.name ? String(w.selectedWell.name) : null;
				if (curWellName !== _lastSelectedWellName) {
					_lastSelectedWellName = curWellName;
					if (curWellName) {
						fetchZones(w.project.project_id, curWellName);
					} else {
						zones = [];
						// keep selectedZones as-is (do not clear on unrelated changes)
					}
				}
		} else {
			project = null;
			wells = [];
			selectedWell = null;
			selectedWellName = '';
		}
		
		// Update depth filter state
		if (w?.depthFilter) {
			depthFilterEnabled = w.depthFilter.enabled;
			minDepth = w.depthFilter.minDepth;
			maxDepth = w.depthFilter.maxDepth;
		}

		// Update zone filter state
		if (w?.zoneFilter) {
			zoneFilterEnabled = w.zoneFilter.enabled;
			selectedZones = Array.isArray(w.zoneFilter.zones) ? [...w.zoneFilter.zones] : [];
		}
	});

	onDestroy(() => unsubscribe());

	onMount(() => {
		function onDocMouseDown(e: MouseEvent) {
			if (!zonesOpen) return;
			const el = zonesWrapper;
			if (!el) return;
			const target = e.target as Node | null;
			if (target && !el.contains(target)) {
				zonesOpen = false;
			}
		}

		function onDocKey(e: KeyboardEvent) {
			if (e.key === 'Escape' && zonesOpen) zonesOpen = false;
		}

		document.addEventListener('mousedown', onDocMouseDown);
		document.addEventListener('keydown', onDocKey);
		return () => {
			document.removeEventListener('mousedown', onDocMouseDown);
			document.removeEventListener('keydown', onDocKey);
		};
	});

		let { items }: { items: { title: string; url: string; icon?: Icon }[] } = $props();

		function isActive(url: string) {
			const path = $page.url.pathname;
			if (!url) return false;
			return path === url || (url !== '/' && path.startsWith(url));
		}

		function handleSelect(e: Event) {
			const name = (e.target as HTMLSelectElement).value;
			if (!name || !project) return;
			selectedWellName = name;
			// update shared workspace and navigate
			selectWell({ id: name, name });
			goto(`/wells/${project.project_id}/${encodeURIComponent(String(name))}`);
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
	
		function handleDepthFilterToggle() {
			if (depthFilterEnabled) {
				setDepthFilter(true, minDepth, maxDepth);
			} else {
				clearDepthFilter();
			}
		}
	
		function handleDepthChange() {
			if (depthFilterEnabled) {
				setDepthFilter(true, minDepth, maxDepth);
			}
		}
	
		function resetDepthFilter() {
			minDepth = null;
			maxDepth = null;
			clearDepthFilter();
		}

		// Zone helpers
		function extractZoneValue(row: Record<string, any>) {
			if (!row || typeof row !== 'object') return null;
			const candidates = ['name', 'zone', 'Zone', 'ZONE', 'formation', 'formation_name', 'formationName', 'FORMATION', 'formation_top', 'formationTop'];
			for (const k of candidates) {
				if (k in row && row[k] !== null && row[k] !== undefined && String(row[k]).trim() !== '') {
					return String(row[k]);
				}
			}
			for (const k of Object.keys(row)) {
				if (/zone|formation/i.test(k) && row[k] !== null && row[k] !== undefined && String(row[k]).trim() !== '') {
					return String(row[k]);
				}
			}
			return null;
		}

		async function fetchZones(projectId: string | number, wellName: string) {
			loadingZones = true;
			zones = [];
			try {
				const url = `${API_BASE}/quick_pp/database/projects/${projectId}/formation_tops?well_name=${encodeURIComponent(wellName)}`;
				const res = await fetch(url);
				if (!res.ok) return;
				const fd = await res.json();
				console.log('Fetched formation tops for zones:', fd);
				let dataArray: any[] = [];
				if (Array.isArray(fd)) dataArray = fd;
				else if (fd && Array.isArray(fd.tops)) dataArray = fd.tops;
				const setVals = new Set<string>();
				for (const r of dataArray) {
					const v = extractZoneValue(r);
					if (v !== null) setVals.add(v);
				}
				zones = Array.from(setVals).sort((a, b) => a.localeCompare(b));
			} catch (e) {
				console.warn('Failed to fetch zones for sidebar', e);
			} finally {
				loadingZones = false;
			}
		}

		function handleZoneFilterToggle() {
			if (zoneFilterEnabled) {
				setZoneFilter(true, selectedZones);
			} else {
				clearZoneFilter();
			}
		}

		function handleZonesChange() {
			if (zoneFilterEnabled) {
				setZoneFilter(true, selectedZones);
			}
		}

		function resetZoneFilter() {
			selectedZones = [];
			clearZoneFilter();
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
						<select id="well-select" class="input w-full mt-1 text-sm h-9" bind:value={selectedWellName} onchange={handleSelect}>
							<option value="">— select well —</option>
							{#each wells as w}
								<option value={w.name}>{w.name}</option>
							{/each}
						</select>
					</div>
				{:else}
					<div class="text-sm text-muted-foreground mt-2">No wells in this project.</div>
				{/if}
				{/if}
			</div>
			
			<!-- Depth Filter Controls -->
			{#if selectedWell}
				<div class="flex items-center gap-2 mb-2">
					<input 
						type="checkbox" 
						id="depth-filter" 
						bind:checked={depthFilterEnabled} 
						onchange={handleDepthFilterToggle}
						class="rounded"
					/>
					<label for="depth-filter" class="text-sm font-medium cursor-pointer">Filter by Depth</label>
				</div>
				
				{#if depthFilterEnabled}
					<div class="space-y-2">
						<div>
							<label class="text-xs text-muted-foreground" for="min-depth">Min Depth</label>
							<input 
								type="number" 
								id="min-depth" 
								bind:value={minDepth} 
								onchange={handleDepthChange}
								placeholder="e.g. 1000"
								class="input w-full text-sm h-8"
							/>
						</div>
						<div>
							<label class="text-xs text-muted-foreground" for="max-depth">Max Depth</label>
							<input 
								type="number" 
								id="max-depth" 
								bind:value={maxDepth} 
								onchange={handleDepthChange}
								placeholder="e.g. 2000"
								class="input w-full text-sm h-8"
							/>
						</div>
						<button 
							class="text-xs text-muted-foreground hover:text-foreground underline" 
							onclick={resetDepthFilter}
						>
							Clear Filter
						</button>
					</div>
				{/if}

				<!-- Zone Filter Controls -->
				<div class="flex items-center gap-2 mb-2">
					<input
						type="checkbox"
						id="zone-filter"
						bind:checked={zoneFilterEnabled}
						onchange={handleZoneFilterToggle}
						class="rounded"
					/>
					<label for="zone-filter" class="text-sm font-medium cursor-pointer">Filter by Zone</label>
				</div>
				{#if zoneFilterEnabled}
					{#if loadingZones}
						<div class="text-sm">Loading zones…</div>
					{:else}
						{#if zones && zones.length}
							<div>
								<div class="relative mt-1" bind:this={zonesWrapper}>
									<button type="button" class="input w-full text-sm h-9 flex items-center justify-between" onclick={() => zonesOpen = !zonesOpen} aria-haspopup="listbox" aria-expanded={zonesOpen}>
										<span>{selectedZones && selectedZones.length ? `${selectedZones.length} selected` : 'Choose zones'}</span>
										<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 ml-2" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 011.08 1.04l-4.25 4.25a.75.75 0 01-1.06 0L5.21 8.27a.75.75 0 01.02-1.06z" clip-rule="evenodd" /></svg>
									</button>
									{#if zonesOpen}
										<div class="absolute z-50 mt-1 bg-white dark:bg-slate-900 text-foreground border border-panel-foreground/10 p-2 rounded shadow w-full max-h-48 overflow-auto">
											{#each zones as z}
												<label class="flex items-center gap-2 text-sm py-1 text-foreground">
													<input type="checkbox" value={z} bind:group={selectedZones} onchange={handleZonesChange} />
													<span class="truncate">{z}</span>
												</label>
											{/each}
											<div class="flex items-center justify-between mt-2">
												<button class="text-xs text-muted-foreground hover:text-foreground underline" onclick={() => { zonesOpen = false; resetZoneFilter(); }}>Clear</button>
												<button class="text-xs font-medium" onclick={() => { zonesOpen = false; handleZonesChange(); }}>Apply</button>
											</div>
										</div>
									{/if}
								</div>
							</div>
						{:else}
							<div class="text-sm text-muted-foreground">No zones available for this well.</div>
						{/if}
					{/if}
				{/if}
			{/if}
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
