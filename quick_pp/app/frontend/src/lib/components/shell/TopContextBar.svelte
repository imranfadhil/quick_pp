<script lang="ts">
	import { authState } from '$lib/state/auth.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { downloadCsv, fmtDepth } from '$lib/utils/formatters';
	import { apiPost } from '$lib/api/client';

	let isResettingDemo = $state(false);
	async function resetDemoSession() {
		if (!workbench.projectId) return;
		isResettingDemo = true;
		try {
			await apiPost(`/quick_pp/workspace/projects/${workbench.projectId}/demo/reset`, {});
			await projectData.selectProject(workbench.projectId, workbench.wellId);
		} finally {
			isResettingDemo = false;
		}
	}

	let showFilterPopover = $state(false);
	let showNewProjectPopover = $state(false);
	let showExportMenu = $state(false);
	let showUserMenu = $state(false);
	let newProjName = $state('');
	let newProjDesc = $state('');
	let minDepthInput = $state<string>('');
	let maxDepthInput = $state<string>('');

	function handleProjectChange(e: Event) {
		const target = e.currentTarget as HTMLSelectElement;
		if (target.value) {
			projectData.selectProject(target.value);
		}
	}

	function handleWellChange(e: Event) {
		const target = e.currentTarget as HTMLSelectElement;
		if (target.value) {
			projectData.selectWell(target.value);
		}
	}

	async function createNewProject() {
		if (!newProjName.trim()) return;
		await projectData.createProject(newProjName.trim(), newProjDesc.trim());
		newProjName = '';
		newProjDesc = '';
		showNewProjectPopover = false;
	}

	function applyFilterInput() {
		const minV = parseFloat(minDepthInput);
		const maxV = parseFloat(maxDepthInput);
		if (Number.isFinite(minV) || Number.isFinite(maxV)) {
			const dr = wellData.depthRange;
			workbench.depthFilter = {
				min: Number.isFinite(minV) ? minV : dr.min,
				max: Number.isFinite(maxV) ? maxV : dr.max
			};
		} else {
			workbench.depthFilter = null;
		}
		showFilterPopover = false;
	}

	function toggleZone(zoneName: string) {
		if (workbench.selectedZones.includes(zoneName)) {
			workbench.selectedZones = workbench.selectedZones.filter((z) => z !== zoneName);
		} else {
			workbench.selectedZones = [...workbench.selectedZones, zoneName];
		}
	}

	function exportWellCurvesCsv() {
		const cols = Object.keys(wellData.curves);
		const indices = wellData.visibleIndices;
		const rows = indices.map((i) => {
			const r: Record<string, any> = {};
			for (const c of cols) {
				const v = wellData.curves[c]?.[i];
				r[c] = Number.isFinite(v) ? v : '';
			}
			return r;
		});
		downloadCsv(`${workbench.wellId || 'well'}_evaluated.csv`, cols, rows);
		showExportMenu = false;
	}

	function exportZonalSummaryCsv() {
		if (!wellData.zonalSummary.length) return;
		const headers = Object.keys(wellData.zonalSummary[0]);
		downloadCsv(`${workbench.wellId || 'well'}_reservoir_summary.csv`, headers, wellData.zonalSummary);
		showExportMenu = false;
	}

	let filterSummaryText = $derived.by(() => {
		const parts: string[] = [];
		if (workbench.selectedZones.length > 0) {
			parts.push(`Zones: ${workbench.selectedZones.join(', ')}`);
		}
		if (workbench.depthFilter) {
			parts.push(`${fmtDepth(workbench.depthFilter.min)}–${fmtDepth(workbench.depthFilter.max)}`);
		}
		return parts.length > 0 ? parts.join(' | ') : 'All Zones & Depths';
	});

	let hasActiveFilter = $derived(
		workbench.selectedZones.length > 0 || workbench.depthFilter !== null
	);
</script>

<header
	class="flex h-10 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-app)] px-3 select-none"
>
	<!-- Left Group: Brand, Project Combobox, Scope Toggle, Well Switcher -->
	<div class="flex items-center gap-2">
		<div class="flex items-center gap-1.5 pr-1">
			<div
				class="flex h-6 w-6 items-center justify-center rounded-sm bg-[var(--accent-primary)] text-[11px] font-bold text-white"
			>
				QP
			</div>
			<span class="text-xs font-bold tracking-tight text-[var(--text-primary)]">quick_pp</span>
			{#if authState.isDemoMode}
				<span
					title="Ephemeral Demo Session — Pre-seeded Volve & Costa SQLite DB is read-only"
					class="rounded-xs border border-emerald-300 bg-emerald-50 px-1.5 py-0.5 text-[9px] font-bold tracking-wider text-emerald-800 uppercase"
				>
					LIVE DEMO
				</span>
			{/if}
		</div>

		<span class="text-[var(--border-hairline)]">/</span>

		<!-- Project Selector -->
		<div class="relative flex items-center gap-1">
			<span class="text-[11px] font-medium text-[var(--text-muted)]">Project:</span>
			<select
				value={workbench.projectId ?? ''}
				onchange={handleProjectChange}
				class="h-7 min-w-[130px] rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0 text-xs font-medium text-[var(--text-primary)] focus:border-[var(--border-focus)] focus:outline-none"
			>
				{#if projectData.projects.length === 0}
					<option value="">No Projects</option>
				{/if}
				{#each projectData.projects as proj (proj.project_id)}
					<option value={String(proj.project_id)}>{proj.name}</option>
				{/each}
			</select>
			<button
				type="button"
				title="Create New Project"
				onclick={() => (showNewProjectPopover = !showNewProjectPopover)}
				class="flex h-7 w-7 items-center justify-center rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-xs font-semibold text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
			>
				+
			</button>

			{#if showNewProjectPopover}
				<div
					class="absolute top-8 left-0 z-50 w-64 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2.5 shadow-lg"
				>
					<div class="mb-1.5 text-xs font-semibold text-[var(--text-primary)]">
						Create New Project
					</div>
					<input
						type="text"
						placeholder="Project name (e.g. Volve)"
						bind:value={newProjName}
						class="mb-1.5 h-7 w-full rounded-sm border border-[var(--border-hairline)] px-2 text-xs"
					/>
					<input
						type="text"
						placeholder="Description (optional)"
						bind:value={newProjDesc}
						class="mb-2 h-7 w-full rounded-sm border border-[var(--border-hairline)] px-2 text-xs"
					/>
					<div class="flex justify-end gap-1.5">
						<button
							type="button"
							onclick={() => (showNewProjectPopover = false)}
							class="h-6 rounded-sm border border-[var(--border-hairline)] px-2 text-[11px]"
						>
							Cancel
						</button>
						<button
							type="button"
							onclick={createNewProject}
							class="h-6 rounded-sm bg-[var(--accent-primary)] px-2.5 text-[11px] font-medium text-white"
						>
							Create
						</button>
					</div>
				</div>
			{/if}
		</div>

		<span class="text-[var(--border-hairline)]">/</span>

		<!-- Scope Segmented Toggle -->
		<div
			class="flex h-7 items-center rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5"
		>
			<button
				type="button"
				onclick={() => (workbench.scope = 'single-well')}
				class="h-6 rounded-xs px-2 text-[11px] font-medium transition-colors {workbench.scope ===
				'single-well'
					? 'bg-[var(--bg-surface)] text-[var(--accent-primary)] shadow-2xs font-semibold'
					: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
			>
				Single Well
			</button>
			<button
				type="button"
				onclick={() => (workbench.scope = 'multi-well')}
				class="h-6 rounded-xs px-2 text-[11px] font-medium transition-colors {workbench.scope ===
				'multi-well'
					? 'bg-[var(--bg-surface)] text-[var(--accent-primary)] shadow-2xs font-semibold'
					: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
			>
				Multi-Well Field
			</button>
		</div>

		<!-- Well Switcher (< Well Dropdown >) -->
		{#if projectData.wells.length > 0}
			<span class="text-[var(--border-hairline)]">/</span>
			<div class="flex items-center gap-0.5">
				<button
					type="button"
					title="Previous Well"
					onclick={() => projectData.stepWell(-1)}
					class="flex h-7 w-6 items-center justify-center rounded-l-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
				>
					◄
				</button>
				<select
					value={workbench.wellId ?? ''}
					onchange={handleWellChange}
					class="mono-num h-7 min-w-[125px] border-y border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-0 text-xs font-semibold text-[var(--text-primary)] focus:outline-none"
				>
					{#each projectData.wells as w (w.name)}
						<option value={w.name}>{w.name}</option>
					{/each}
				</select>
				<button
					type="button"
					title="Next Well"
					onclick={() => projectData.stepWell(1)}
					class="flex h-7 w-6 items-center justify-center rounded-r-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
				>
					►
				</button>
			</div>
		{/if}
	</div>

	<!-- Center Group: Global Depth & Zone Filter Pill -->
	<div class="relative flex items-center gap-1">
		<button
			type="button"
			onclick={() => {
				minDepthInput = workbench.depthFilter ? String(workbench.depthFilter.min) : '';
				maxDepthInput = workbench.depthFilter ? String(workbench.depthFilter.max) : '';
				showFilterPopover = !showFilterPopover;
			}}
			class="flex h-7 items-center gap-1.5 rounded-sm border px-2.5 text-xs transition-colors {hasActiveFilter
				? 'border-blue-500 bg-blue-50 font-medium text-blue-800'
				: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
		>
			<span class="text-[10px] uppercase text-[var(--text-muted)]">Filter:</span>
			<span class="mono-num max-w-[240px] truncate">{filterSummaryText}</span>
		</button>

		{#if hasActiveFilter}
			<button
				type="button"
				title="Clear Depth & Zone Filters"
				onclick={() => workbench.clearFilters()}
				class="flex h-7 w-6 items-center justify-center rounded-sm border border-blue-200 bg-blue-50 text-xs font-bold text-blue-700 hover:bg-blue-100"
			>
				×
			</button>
		{/if}

		{#if showFilterPopover}
			<div
				class="absolute top-8 left-0 z-50 w-72 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-lg"
			>
				<div class="mb-2 flex items-center justify-between">
					<span class="text-xs font-semibold text-[var(--text-primary)]">
						Depth & Zone Filter
					</span>
					<button
						type="button"
						onclick={() => {
							workbench.clearFilters();
							minDepthInput = '';
							maxDepthInput = '';
							showFilterPopover = false;
						}}
						class="text-[11px] text-blue-600 hover:underline"
					>
						Reset All
					</button>
				</div>

				<div class="mb-2.5 grid grid-cols-2 gap-2">
					<label class="block">
						<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Min Depth (m)</span>
						<input
							type="number"
							step="any"
							placeholder={String(wellData.depthRange.min)}
							bind:value={minDepthInput}
							class="mono-num h-7 w-full rounded-sm border border-[var(--border-hairline)] px-2 text-xs"
						/>
					</label>
					<label class="block">
						<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Max Depth (m)</span>
						<input
							type="number"
							step="any"
							placeholder={String(wellData.depthRange.max)}
							bind:value={maxDepthInput}
							class="mono-num h-7 w-full rounded-sm border border-[var(--border-hairline)] px-2 text-xs"
						/>
					</label>
				</div>

				{#if wellData.tops.length > 0 || projectData.allZones.length > 0}
					<div class="mb-2">
						<span class="mb-1 block text-[10px] font-semibold uppercase text-[var(--text-muted)]">
							Formation Zones
						</span>
						<div class="max-h-32 space-y-1 overflow-y-auto border border-[var(--border-hairline)] p-1.5">
							{#each Array.from(new Set([...wellData.tops.map((t) => t.name), ...projectData.allZones])) as z}
								<label class="flex items-center gap-1.5 text-xs cursor-pointer">
									<input
										type="checkbox"
										checked={workbench.selectedZones.includes(z)}
										onchange={() => toggleZone(z)}
										class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
									/>
									<span class="truncate">{z}</span>
								</label>
							{/each}
						</div>
					</div>
				{/if}

				<div class="flex justify-end gap-1.5">
					<button
						type="button"
						onclick={() => (showFilterPopover = false)}
						class="h-6 rounded-sm border border-[var(--border-hairline)] px-2 text-[11px]"
					>
						Close
					</button>
					<button
						type="button"
						onclick={applyFilterInput}
						class="h-6 rounded-sm bg-[var(--accent-primary)] px-2.5 text-[11px] font-medium text-white"
					>
						Apply
					</button>
				</div>
			</div>
		{/if}
	</div>

	<!-- Right Group: Import Data, Export Dropdown, AI Assistant -->
	<div class="flex items-center gap-1.5">
		{#if authState.isDemoMode}
			<button
				type="button"
				disabled={isResettingDemo}
				onclick={resetDemoSession}
				title="Reset ephemeral session edits back to pre-seeded Volve & Costa state"
				class="h-7 rounded-sm border border-amber-300 bg-amber-50 px-2 text-[11px] font-semibold text-amber-800 hover:bg-amber-100 disabled:opacity-50"
			>
				{isResettingDemo ? 'Resetting…' : 'Reset Demo'}
			</button>
		{/if}
		<button
			type="button"
			onclick={() => (workbench.importerOpen = true)}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-medium text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
		>
			Import Data
		</button>

		<div class="relative">
			<button
				type="button"
				onclick={() => (showExportMenu = !showExportMenu)}
				class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-medium text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
			>
				Export ▾
			</button>
			{#if showExportMenu}
				<div
					class="absolute top-8 right-0 z-50 w-48 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-1 shadow-lg"
				>
					<button
						type="button"
						onclick={exportWellCurvesCsv}
						class="block w-full px-3 py-1.5 text-left text-xs text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
					>
						Evaluated Logs (.csv)
					</button>
					<button
						type="button"
						onclick={exportZonalSummaryCsv}
						disabled={wellData.zonalSummary.length === 0}
						class="block w-full px-3 py-1.5 text-left text-xs text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-40"
					>
						Zonal Summary (.csv)
					</button>
				</div>
			{/if}
		</div>

		<button
			type="button"
			onclick={() => (workbench.aiDrawerOpen = !workbench.aiDrawerOpen)}
			class="h-7 rounded-sm border px-2.5 text-xs font-medium transition-colors {workbench.aiDrawerOpen
				? 'border-blue-600 bg-blue-600 text-white'
				: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]'}"
		>
			AI Assistant
		</button>

		{#if authState.isAdmin}
			<button
				type="button"
				onclick={() => (authState.uamModalOpen = true)}
				class="h-7 rounded-sm border border-blue-200 bg-blue-50 px-2.5 text-xs font-semibold text-blue-800 hover:bg-blue-100"
			>
				UAM Console
			</button>
		{/if}

		{#if authState.user}
			<div class="relative">
				<button
					type="button"
					onclick={() => (showUserMenu = !showUserMenu)}
					class="flex h-7 items-center gap-1.5 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs font-medium text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
				>
					<span
						class="flex h-4 w-4 items-center justify-center rounded-xs bg-[var(--accent-primary)] text-[9px] font-bold text-white"
					>
						{authState.userInitials}
					</span>
					<span class="max-w-[96px] truncate font-semibold">{authState.user.username}</span>
					<span class="text-[10px] text-[var(--text-muted)]">▾</span>
				</button>

				{#if showUserMenu}
					<div
						class="absolute top-8 right-0 z-50 w-56 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-1 shadow-lg"
					>
						<div class="border-b border-[var(--border-hairline)] px-3 py-2">
							<div class="flex items-center justify-between">
								<span class="text-xs font-bold text-[var(--text-primary)]">
									{authState.user.full_name || authState.user.username}
								</span>
								<span
									class="rounded-xs bg-blue-100 px-1.5 py-0.5 text-[9px] font-bold text-blue-800 uppercase"
								>
									{authState.user.role}
								</span>
							</div>
							<div class="truncate text-[11px] text-[var(--text-muted)]">
								{authState.user.email}
							</div>
						</div>

						{#if authState.isAdmin}
							<button
								type="button"
								onclick={() => {
									showUserMenu = false;
									authState.uamModalOpen = true;
								}}
								class="block w-full px-3 py-1.5 text-left text-xs font-medium text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
							>
								User Access Management (UAM)
							</button>
						{/if}

						<button
							type="button"
							onclick={() => {
								showUserMenu = false;
								authState.accountModalOpen = true;
							}}
							class="block w-full px-3 py-1.5 text-left text-xs text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
						>
							Account Security & Password
						</button>

						<div class="my-1 border-t border-[var(--border-hairline)]"></div>

						<button
							type="button"
							onclick={() => {
								showUserMenu = false;
								authState.logout();
							}}
							class="block w-full px-3 py-1.5 text-left text-xs font-semibold text-red-600 hover:bg-red-50"
						>
							Sign Out
						</button>
					</div>
				{/if}
			</div>
		{/if}
	</div>
</header>
