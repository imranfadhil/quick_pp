<script lang="ts">
	import logo from '$lib/assets/logo.png';
	import { authState } from '$lib/state/auth.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { downloadCsv, fmtDepth } from '$lib/utils/formatters';
	import { apiPost, downloadHandoverPackage, downloadWellLas } from '$lib/api/client';

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
	let showManageProjectPopover = $state(false);
	let showManageWellPopover = $state(false);
	let showExportMenu = $state(false);
	let showUserMenu = $state(false);
	let newProjName = $state('');
	let newProjDesc = $state('');
	let editProjName = $state('');
	let editProjDesc = $state('');
	let confirmDeleteProject = $state(false);
	let editWellName = $state('');
	let confirmDeleteWell = $state(false);
	let mgmtError = $state<string | null>(null);
	let minDepthInput = $state<string>('');
	let maxDepthInput = $state<string>('');

	function openManageProjectPopover() {
		showNewProjectPopover = false;
		showManageWellPopover = false;
		confirmDeleteProject = false;
		mgmtError = null;
		const activeProj = projectData.projects.find(
			(p) => String(p.project_id) === String(workbench.projectId)
		);
		editProjName = activeProj?.name ?? projectData.summary?.name ?? '';
		editProjDesc = activeProj?.description ?? projectData.summary?.description ?? '';
		showManageProjectPopover = !showManageProjectPopover;
	}

	async function handleSaveProjectEdit() {
		if (!workbench.projectId || !editProjName.trim()) return;
		mgmtError = null;
		try {
			await projectData.renameProject(workbench.projectId, editProjName, editProjDesc);
			showManageProjectPopover = false;
		} catch (e: any) {
			mgmtError = e?.message || 'Failed to rename project';
		}
	}

	async function handleDeleteActiveProject() {
		if (!workbench.projectId) return;
		mgmtError = null;
		try {
			await projectData.deleteProject(workbench.projectId);
			confirmDeleteProject = false;
			showManageProjectPopover = false;
		} catch (e: any) {
			mgmtError = e?.message || 'Failed to delete project';
		}
	}

	function openManageWellPopover() {
		showNewProjectPopover = false;
		showManageProjectPopover = false;
		confirmDeleteWell = false;
		mgmtError = null;
		editWellName = workbench.wellId ?? '';
		showManageWellPopover = !showManageWellPopover;
	}

	async function handleSaveWellRename() {
		if (!workbench.wellId || !editWellName.trim()) return;
		mgmtError = null;
		try {
			await projectData.renameWell(workbench.wellId, editWellName);
			showManageWellPopover = false;
		} catch (e: any) {
			mgmtError = e?.message || 'Failed to rename well';
		}
	}

	async function handleDeleteActiveWell() {
		if (!workbench.wellId) return;
		mgmtError = null;
		try {
			await projectData.deleteWell(workbench.wellId);
			confirmDeleteWell = false;
			showManageWellPopover = false;
		} catch (e: any) {
			mgmtError = e?.message || 'Failed to delete well';
		}
	}

	function handleProjectChange(e: Event) {
		const target = e.currentTarget as HTMLSelectElement;
		if (target.value) {
			projectData.selectProject(target.value);
		} else {
			projectData.clearProject();
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

	async function applyFilterInput() {
		const minV = parseFloat(minDepthInput);
		const maxV = parseFloat(maxDepthInput);
		if (Number.isFinite(minV) || Number.isFinite(maxV)) {
			const dr = wellData.depthRange;
			await wellData.saveWellFocusDepth(
				Number.isFinite(minV) ? minV : dr.min,
				Number.isFinite(maxV) ? maxV : dr.max,
				true
			);
		} else {
			await wellData.saveWellFocusDepth(null, null, true);
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

	function toggleWellFilter(wellName: string) {
		if (projectData.selectedWellNames.includes(wellName)) {
			projectData.selectedWellNames = projectData.selectedWellNames.filter((w) => w !== wellName);
		} else {
			projectData.selectedWellNames = [...projectData.selectedWellNames, wellName];
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

	async function exportProjectHandoverZip() {
		if (!workbench.projectId) return;
		showExportMenu = false;
		const wellsFilter =
			hasWellFilter && projectData.selectedWellNames.length > 0
				? projectData.selectedWellNames
				: undefined;
		await downloadHandoverPackage(workbench.projectId, { wells: wellsFilter });
	}

	async function exportActiveWellHandoverZip() {
		if (!workbench.projectId || !workbench.wellId) return;
		showExportMenu = false;
		await downloadHandoverPackage(workbench.projectId, { wellName: workbench.wellId });
	}

	async function exportActiveWellLasFile() {
		if (!workbench.projectId || !workbench.wellId) return;
		showExportMenu = false;
		await downloadWellLas(workbench.projectId, workbench.wellId);
	}

	let hasWellFilter = $derived(
		projectData.wells.length > 0 &&
			projectData.selectedWellNames.length < projectData.wells.length
	);

	let filterSummaryText = $derived.by(() => {
		const parts: string[] = [];
		if (hasWellFilter) {
			const sel = projectData.selectedWellNames;
			parts.push(
				sel.length > 0 && sel.length <= 2
					? `Wells: ${sel.join(', ')}`
					: `Wells: ${sel.length}/${projectData.wells.length}`
			);
		}
		if (workbench.selectedZones.length > 0) {
			parts.push(`Zones: ${workbench.selectedZones.join(', ')}`);
		}
		if (workbench.depthFilter) {
			parts.push(`${fmtDepth(workbench.depthFilter.min)}–${fmtDepth(workbench.depthFilter.max)}`);
		}
		if (parts.length > 0) return parts.join(' | ');
		return workbench.scope === 'multi-well' ? 'All Wells, Zones & Depths' : 'All Zones & Depths';
	});

	let hasActiveFilter = $derived(
		hasWellFilter || workbench.selectedZones.length > 0 || workbench.depthFilter !== null
	);
</script>

<header
	class="flex h-10 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-app)] px-3 select-none"
>
	<!-- Left Group: Brand, Project Combobox, Scope Toggle, Well Switcher -->
	<div class="flex items-center gap-2">
		<div class="flex items-center gap-1.5 pr-1">
			<img
				src={logo}
				alt="quick_pp logo"
				class="h-6 w-6 shrink-0 object-contain"
			/>
			<span class="text-xs font-bold tracking-tight text-[var(--text-primary)]">quick_pp</span>
			{#if authState.appVersion}
				<span
					title="quick_pp v{authState.appVersion}"
					class="mono-num rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-1 py-0.5 text-[9px] font-medium text-[var(--text-muted)]"
				>
					v{authState.appVersion}
				</span>
			{/if}
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
				{:else}
					<option value="">— Select Project —</option>
				{/if}
				{#each projectData.projects as proj (proj.project_id)}
					<option value={String(proj.project_id)}>{proj.name}</option>
				{/each}
			</select>
			<button
				type="button"
				title="Create New Project"
				onclick={() => {
					showManageProjectPopover = false;
					showNewProjectPopover = !showNewProjectPopover;
				}}
				class="flex h-7 w-7 items-center justify-center rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-xs font-semibold text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
			>
				+
			</button>
			{#if workbench.projectId}
				<button
					type="button"
					title="Manage Project (Rename / Delete)"
					onclick={openManageProjectPopover}
					class="flex h-7 w-7 items-center justify-center rounded-sm border text-xs transition-colors {showManageProjectPopover
						? 'border-[var(--accent-primary)] bg-blue-50 text-[var(--accent-primary)]'
						: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
				>
					✎
				</button>
			{/if}

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

			{#if showManageProjectPopover && workbench.projectId}
				<div
					class="absolute top-8 left-0 z-50 w-72 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2.5 shadow-lg"
				>
					<div class="mb-1.5 flex items-center justify-between">
						<span class="text-xs font-semibold text-[var(--text-primary)]">
							Manage Project
						</span>
						<span class="mono-num text-[10px] text-[var(--text-muted)]">
							ID #{workbench.projectId}
						</span>
					</div>

					<label class="mb-1.5 block">
						<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Project Name</span>
						<input
							type="text"
							bind:value={editProjName}
							onkeydown={(e) => e.key === 'Enter' && handleSaveProjectEdit()}
							class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs font-medium"
						/>
					</label>

					<label class="mb-2 block">
						<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Description</span>
						<input
							type="text"
							placeholder="Field / asset description"
							bind:value={editProjDesc}
							onkeydown={(e) => e.key === 'Enter' && handleSaveProjectEdit()}
							class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs"
						/>
					</label>

					{#if mgmtError}
						<div class="mb-2 rounded-xs border border-red-200 bg-red-50 px-2 py-1 text-[10px] text-red-700">
							{mgmtError}
						</div>
					{/if}

					<div class="flex items-center justify-between border-t border-[var(--border-hairline)] pt-2">
						{#if !confirmDeleteProject}
							<button
								type="button"
								onclick={() => (confirmDeleteProject = true)}
								class="h-6 rounded-sm border border-red-200 bg-red-50 px-2 text-[10px] font-semibold text-red-700 hover:bg-red-100"
							>
								Delete Project...
							</button>
						{:else}
							<div class="flex items-center gap-1">
								<button
									type="button"
									onclick={handleDeleteActiveProject}
									class="h-6 rounded-sm bg-red-600 px-2 text-[10px] font-bold text-white hover:bg-red-700"
								>
									Confirm Delete
								</button>
								<button
									type="button"
									onclick={() => (confirmDeleteProject = false)}
									class="h-6 rounded-sm border border-[var(--border-hairline)] px-1.5 text-[10px]"
								>
									×
								</button>
							</div>
						{/if}

						<div class="flex items-center gap-1">
							<button
								type="button"
								onclick={() => (showManageProjectPopover = false)}
								class="h-6 rounded-sm border border-[var(--border-hairline)] px-2 text-[11px]"
							>
								Close
							</button>
							<button
								type="button"
								disabled={!editProjName.trim()}
								onclick={handleSaveProjectEdit}
								class="h-6 rounded-sm bg-[var(--accent-primary)] px-2.5 text-[11px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
							>
								Save
							</button>
						</div>
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
			<div class="relative flex items-center gap-0.5">
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
				{#if workbench.wellId}
					<button
						type="button"
						title="Manage Active Well (Rename / Delete)"
						onclick={openManageWellPopover}
						class="ml-0.5 flex h-7 w-7 items-center justify-center rounded-sm border text-xs transition-colors {showManageWellPopover
							? 'border-[var(--accent-primary)] bg-blue-50 text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
					>
						✎
					</button>
				{/if}

				{#if showManageWellPopover && workbench.wellId}
					<div
						class="absolute top-8 left-0 z-50 w-68 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2.5 shadow-lg"
					>
						<div class="mb-1.5 flex items-center justify-between">
							<span class="text-xs font-semibold text-[var(--text-primary)]">
								Manage Well
							</span>
							<span class="mono-num text-[10px] font-semibold text-[var(--accent-primary)]">
								{workbench.wellId}
							</span>
						</div>

						<label class="mb-2 block">
							<span class="mb-0.5 block text-[10px] text-[var(--text-muted)]">Well Name / UWI</span>
							<input
								type="text"
								bind:value={editWellName}
								onkeydown={(e) => e.key === 'Enter' && handleSaveWellRename()}
								class="mono-num h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs font-semibold"
							/>
						</label>

						{#if mgmtError}
							<div class="mb-2 rounded-xs border border-red-200 bg-red-50 px-2 py-1 text-[10px] text-red-700">
								{mgmtError}
							</div>
						{/if}

						<div class="flex items-center justify-between border-t border-[var(--border-hairline)] pt-2">
							{#if !confirmDeleteWell}
								<button
									type="button"
									onclick={() => (confirmDeleteWell = true)}
									class="h-6 rounded-sm border border-red-200 bg-red-50 px-2 text-[10px] font-semibold text-red-700 hover:bg-red-100"
								>
									Delete Well...
								</button>
							{:else}
								<div class="flex items-center gap-1">
									<button
										type="button"
										onclick={handleDeleteActiveWell}
										class="h-6 rounded-sm bg-red-600 px-2 text-[10px] font-bold text-white hover:bg-red-700"
									>
										Confirm Delete
									</button>
									<button
										type="button"
										onclick={() => (confirmDeleteWell = false)}
										class="h-6 rounded-sm border border-[var(--border-hairline)] px-1.5 text-[10px]"
									>
										×
									</button>
								</div>
							{/if}

							<div class="flex items-center gap-1">
								<button
									type="button"
									onclick={() => (showManageWellPopover = false)}
									class="h-6 rounded-sm border border-[var(--border-hairline)] px-2 text-[11px]"
								>
									Close
								</button>
								<button
									type="button"
									disabled={!editWellName.trim() || editWellName.trim() === workbench.wellId}
									onclick={handleSaveWellRename}
									class="h-6 rounded-sm bg-[var(--accent-primary)] px-2.5 text-[11px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
								>
									Rename
								</button>
							</div>
						</div>
					</div>
				{/if}
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
				title="Clear Well, Zone & Depth Filters"
				onclick={() => {
					workbench.clearFilters();
					projectData.selectedWellNames = projectData.wells.map((w) => w.name);
					wellData.saveWellFocusDepth(null, null, true);
				}}
				class="flex h-7 w-6 items-center justify-center rounded-sm border border-blue-200 bg-blue-50 text-xs font-bold text-blue-700 hover:bg-blue-100"
			>
				×
			</button>
		{/if}

		{#if showFilterPopover}
			<div
				class="absolute top-8 left-0 z-50 w-76 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3 shadow-lg"
			>
				<div class="mb-2 flex items-center justify-between">
					<span class="text-xs font-semibold text-[var(--text-primary)]">
						{workbench.scope === 'multi-well'
							? 'Multi-Well, Zone & Depth Filter'
							: 'Well Working Depth & Zone Filter'}
					</span>
					<button
						type="button"
						onclick={() => {
							workbench.clearFilters();
							projectData.selectedWellNames = projectData.wells.map((w) => w.name);
							wellData.saveWellFocusDepth(null, null, true);
							minDepthInput = '';
							maxDepthInput = '';
							showFilterPopover = false;
						}}
						class="text-[11px] text-blue-600 hover:underline"
					>
						Reset All
					</button>
				</div>

				{#if (workbench.scope === 'multi-well' || hasWellFilter) && projectData.wells.length > 0}
					<div class="mb-2.5">
						<div class="mb-1 flex items-center justify-between">
							<span class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
								Wells ({projectData.selectedWellNames.length}/{projectData.wells.length})
							</span>
							<div class="flex items-center gap-1.5 text-[10px]">
								<button
									type="button"
									onclick={() => {
										projectData.selectedWellNames = projectData.wells.map((w) => w.name);
									}}
									class="font-medium text-blue-600 hover:underline"
								>
									All
								</button>
								<span class="text-[var(--text-muted)]">·</span>
								<button
									type="button"
									onclick={() => {
										projectData.selectedWellNames = [];
									}}
									class="font-medium text-slate-600 hover:underline"
								>
									None
								</button>
							</div>
						</div>
						<div class="max-h-32 space-y-1 overflow-y-auto rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
							{#each projectData.wells as w (w.name)}
								<label class="flex cursor-pointer items-center justify-between gap-1.5 text-xs hover:bg-[var(--bg-subtle)] px-1 py-0.5 rounded-xs">
									<span class="flex items-center gap-1.5 min-w-0">
										<input
											type="checkbox"
											checked={projectData.selectedWellNames.includes(w.name)}
											onchange={() => toggleWellFilter(w.name)}
											class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
										/>
										<span class="mono-num truncate font-medium text-[var(--text-primary)]">{w.name}</span>
									</span>
									<span class="mono-num shrink-0 text-[10px] text-[var(--text-muted)]">
										{(w.depth_count ?? (w as any).row_count)?.toLocaleString() ?? ''}
									</span>
								</label>
							{/each}
						</div>
					</div>
				{/if}

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
					class="absolute top-8 right-0 z-50 w-64 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-1 shadow-lg"
				>
					<div class="px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
						Handover Bundles & LAS
					</div>
					<button
						type="button"
						disabled={!workbench.projectId || projectData.wells.length === 0}
						onclick={exportProjectHandoverZip}
						class="block w-full px-3 py-1.5 text-left text-xs font-medium text-[var(--accent-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-40"
					>
						Project Handover Package (.zip)
					</button>
					<button
						type="button"
						disabled={!workbench.projectId || !workbench.wellId}
						onclick={exportActiveWellHandoverZip}
						class="block w-full px-3 py-1.5 text-left text-xs text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-40"
					>
						Active Well Handover ({workbench.wellId || 'Well'} .zip)
					</button>
					<button
						type="button"
						disabled={!workbench.projectId || !workbench.wellId}
						onclick={exportActiveWellLasFile}
						class="block w-full px-3 py-1.5 text-left text-xs text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-40"
					>
						Active Well Logs + Params (.las)
					</button>
					<div class="my-1 border-t border-[var(--border-hairline)]"></div>
					<div class="px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
						Quick Flat Tables
					</div>
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

		{#if authState.isAdmin && !authState.isDemoMode}
			<button
				type="button"
				onclick={() => (authState.uamModalOpen = true)}
				class="h-7 rounded-sm border border-blue-200 bg-blue-50 px-2.5 text-xs font-semibold text-blue-800 hover:bg-blue-100"
			>
				UAM Console
			</button>
		{/if}

		{#if authState.user && !authState.isDemoMode}
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
