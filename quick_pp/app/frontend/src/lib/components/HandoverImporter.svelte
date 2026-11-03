<script lang="ts">
	import {
		commitHandoverPackage,
		inspectHandoverPackage,
		type HandoverCommitResult,
		type HandoverInspectionResult
	} from '$lib/api/client';
	import { authState } from '$lib/state/auth.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { appStatus } from '$lib/state/status.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	let fileInput = $state<HTMLInputElement | null>(null);
	let selectedFile = $state<File | null>(null);
	let isDragging = $state(false);

	let inspecting = $state(false);
	let committing = $state(false);
	let inspectionResult = $state<HandoverInspectionResult | null>(null);
	let commitResult = $state<HandoverCommitResult | null>(null);
	let errorMessage = $state<string | null>(null);

	let targetProjectName = $state('');
	let overwriteExisting = $state(true);

	function formatBytes(bytes: number): string {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
	}

	async function handleFile(file: File) {
		if (!file.name.toLowerCase().endsWith('.zip')) {
			errorMessage = 'Please select a valid Petrophysical Handover Package (.zip) archive.';
			return;
		}

		selectedFile = file;
		errorMessage = null;
		commitResult = null;
		inspecting = true;

		try {
			const res = await inspectHandoverPackage(file);
			inspectionResult = res;
			targetProjectName = res.suggested_name || res.project_name || 'Imported_Project';
		} catch (err: any) {
			inspectionResult = null;
			errorMessage = err?.message || 'Failed to inspect handover package archive.';
		} finally {
			inspecting = false;
		}
	}

	function onFileInputChange(e: Event) {
		const target = e.currentTarget as HTMLInputElement;
		if (target.files && target.files.length > 0) {
			handleFile(target.files[0]);
		}
	}

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		isDragging = false;
		if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
			handleFile(e.dataTransfer.files[0]);
		}
	}

	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		isDragging = true;
	}

	function handleDragLeave(e: DragEvent) {
		e.preventDefault();
		isDragging = false;
	}

	function resetFile() {
		selectedFile = null;
		inspectionResult = null;
		commitResult = null;
		errorMessage = null;
		targetProjectName = '';
		if (fileInput) {
			fileInput.value = '';
		}
	}

	async function handleCommit() {
		if (!selectedFile || !inspectionResult) return;
		if (!targetProjectName.trim()) {
			errorMessage = 'Target project name cannot be empty.';
			return;
		}

		committing = true;
		errorMessage = null;

		try {
			const res = await commitHandoverPackage(selectedFile, {
				projectName: targetProjectName.trim(),
				overwriteExisting
			});

			commitResult = res;
			appStatus.success(`Successfully imported Handover Package into "${res.project_name}"!`);

			// Refresh project inventory and select the newly imported project
			await projectData.loadProjects();
			if (res.project_id) {
				await projectData.selectProject(res.project_id);
			}

			// Automatically close modal after brief delay or user interaction
			setTimeout(() => {
				workbench.importerOpen = false;
			}, 1500);
		} catch (err: any) {
			errorMessage = err?.message || 'Failed to import handover package into project.';
		} finally {
			committing = false;
		}
	}
</script>

<div class="flex flex-col gap-4 p-4 text-xs">
	<!-- Top Intro Banner -->
	<div class="rounded-sm border border-blue-200 bg-blue-50/70 p-3 text-[var(--text-secondary)]">
		<div class="flex items-start gap-2">
			<span class="text-base text-blue-600">📦</span>
			<div>
				<h3 class="font-bold text-[var(--text-primary)]">
					Round-Trip Petrophysical Handover Package Importer
				</h3>
				<p class="mt-0.5 leading-relaxed text-[11px]">
					Import a complete quick_pp handover package archive (<code class="font-mono text-blue-900">.zip</code>).
					This restores evaluated LAS well logs, formation tops, fluid contacts, perforations, directional surveys,
					core RCA plugs, MDT pressure tests, and project CPI interpretation parameters into your workspace.
				</p>
			</div>
		</div>
	</div>

	<!-- Error Alert -->
	{#if errorMessage}
		<div class="flex items-start gap-2 rounded-sm border border-red-300 bg-red-50 p-2.5 text-red-800">
			<span class="text-sm font-bold">⚠</span>
			<div class="flex-1">
				<div class="font-semibold">Import Error</div>
				<div class="text-[11px]">{errorMessage}</div>
			</div>
			<button
				type="button"
				onclick={() => (errorMessage = null)}
				class="text-xs font-bold text-red-600 hover:text-red-900"
			>
				×
			</button>
		</div>
	{/if}

	<!-- Success Banner -->
	{#if commitResult}
		<div class="rounded-sm border border-emerald-300 bg-emerald-50 p-3 text-emerald-900">
			<div class="flex items-center gap-2 font-bold text-sm">
				<span>✓</span>
				<span>Handover Package Successfully Imported!</span>
			</div>
			<p class="mt-1 text-[11px]">
				Project <strong>{commitResult.project_name}</strong> (ID #{commitResult.project_id}) has been created with
				<strong>{commitResult.wells_imported}</strong> well(s). Switching active project now...
			</p>
			{#if commitResult.warnings && commitResult.warnings.length > 0}
				<div class="mt-2 text-[10px] text-amber-800">
					<strong>Notices:</strong>
					<ul class="list-disc pl-4">
						{#each commitResult.warnings as w}
							<li>{w}</li>
						{/each}
					</ul>
				</div>
			{/if}
		</div>
	{/if}

	{#if !inspectionResult}
		<!-- File Upload / Dropzone -->
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			onclick={() => fileInput?.click()}
			ondragover={handleDragOver}
			ondragleave={handleDragLeave}
			ondrop={handleDrop}
			class="flex cursor-pointer flex-col items-center justify-center rounded-sm border-2 border-dashed p-8 text-center transition-colors {isDragging
				? 'border-blue-500 bg-blue-50/50'
				: 'border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-slate-400 hover:bg-slate-100/60'}"
		>
			<input
				type="file"
				accept=".zip"
				bind:this={fileInput}
				onchange={onFileInputChange}
				class="hidden"
			/>
			{#if inspecting}
				<div class="flex flex-col items-center gap-2">
					<div class="h-6 w-6 animate-spin rounded-full border-2 border-blue-600 border-t-transparent"></div>
					<span class="font-medium text-[var(--text-primary)]">
						Inspecting handover package archive &amp; manifest...
					</span>
					<span class="text-[11px] text-[var(--text-muted)]">
						Scanning LAS logs, CSV tables, and CPI parameters
					</span>
				</div>
			{:else}
				<div class="mb-2 text-2xl">📦</div>
				<div class="font-semibold text-[var(--text-primary)]">
					Drag &amp; drop Petrophysical Handover Archive (.zip) here
				</div>
				<div class="mt-1 text-[11px] text-[var(--text-muted)]">
					or click to browse from your computer
				</div>
				<div class="mt-3 inline-flex items-center gap-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 py-1 text-[10px] font-medium text-[var(--text-secondary)]">
					Standard format: exported by quick_pp Handover Package export
				</div>
			{/if}
		</div>
	{:else}
		<!-- Inspected Package Preview Card -->
		<div class="flex flex-col gap-3 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-3.5 shadow-2xs">
			<!-- Header with File Name & Change button -->
			<div class="flex items-center justify-between border-b border-[var(--border-hairline)] pb-2.5">
				<div>
					<div class="flex items-center gap-2">
						<span class="font-bold text-sm text-[var(--text-primary)]">
							{inspectionResult.project_name}
						</span>
						<span class="rounded-xs bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-700">
							{inspectionResult.wells_count} Well{inspectionResult.wells_count === 1 ? '' : 's'}
						</span>
						{#if inspectionResult.quick_pp_version}
							<span class="rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700">
								v{inspectionResult.quick_pp_version}
							</span>
						{/if}
					</div>
					{#if inspectionResult.description}
						<p class="mt-0.5 text-[11px] text-[var(--text-muted)]">
							{inspectionResult.description}
						</p>
					{/if}
				</div>
				<button
					type="button"
					onclick={resetFile}
					class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-1 text-[11px] font-medium text-[var(--text-secondary)] hover:bg-slate-200"
				>
					Choose Different File
				</button>
			</div>

			<!-- Well Names Pills -->
			{#if inspectionResult.well_names && inspectionResult.well_names.length > 0}
				<div>
					<span class="text-[10px] font-semibold text-[var(--text-muted)] uppercase tracking-wide">
						Wells in Package ({inspectionResult.well_names.length}):
					</span>
					<div class="mt-1 flex flex-wrap gap-1">
						{#each inspectionResult.well_names as w}
							<span class="mono-num rounded-xs border border-slate-300 bg-slate-50 px-1.5 py-0.5 text-[10px] font-medium text-slate-800">
								{w}
							</span>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Inventory Counts Grid -->
			<div>
				<span class="text-[10px] font-semibold text-[var(--text-muted)] uppercase tracking-wide">
					Package Ancillary Content:
				</span>
				<div class="mt-1.5 grid grid-cols-4 gap-2">
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
						<div class="text-[10px] text-[var(--text-muted)]">Formation Tops</div>
						<div class="mono-num font-bold text-sm text-[var(--text-primary)]">
							{inspectionResult.inventory_counts.formation_tops ?? 0}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
						<div class="text-[10px] text-[var(--text-muted)]">Fluid Contacts</div>
						<div class="mono-num font-bold text-sm text-[var(--text-primary)]">
							{inspectionResult.inventory_counts.fluid_contacts ?? 0}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
						<div class="text-[10px] text-[var(--text-muted)]">Perforations</div>
						<div class="mono-num font-bold text-sm text-[var(--text-primary)]">
							{inspectionResult.inventory_counts.perforations ?? 0}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
						<div class="text-[10px] text-[var(--text-muted)]">Well Surveys</div>
						<div class="mono-num font-bold text-sm text-[var(--text-primary)]">
							{inspectionResult.inventory_counts.well_surveys ?? 0}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
						<div class="text-[10px] text-[var(--text-muted)]">Core RCA Plugs</div>
						<div class="mono-num font-bold text-sm text-[var(--text-primary)]">
							{inspectionResult.inventory_counts.core_rca_samples ?? 0}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
						<div class="text-[10px] text-[var(--text-muted)]">MDT Pressure Tests</div>
						<div class="mono-num font-bold text-sm text-[var(--text-primary)]">
							{inspectionResult.inventory_counts.pressure_tests ?? 0}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
						<div class="text-[10px] text-[var(--text-muted)]">SCAL (Pc / RelPerm)</div>
						<div class="mono-num font-bold text-sm text-[var(--text-primary)]">
							{(inspectionResult.inventory_counts.scal_pc_points ?? 0) +
								(inspectionResult.inventory_counts.scal_relperm_points ?? 0)}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
						<div class="text-[10px] text-[var(--text-muted)]">CPI Parameters</div>
						<div class="font-bold text-sm text-emerald-700">
							{inspectionResult.has_manifest ? 'Preserved' : 'Default'}
						</div>
					</div>
				</div>
			</div>

			<!-- Ingestion Configuration Form -->
			<div class="mt-2 rounded-xs border border-slate-200 bg-slate-50/70 p-3">
				<div class="font-semibold text-[11px] text-[var(--text-primary)] mb-2">
					Import Target Configuration
				</div>
				<div class="flex flex-col gap-2.5">
					<label class="block">
						<span class="mb-1 block text-[10px] font-medium text-[var(--text-secondary)]">
							Target Project Name:
						</span>
						<input
							type="text"
							bind:value={targetProjectName}
							placeholder="e.g. Volve Field Handover"
							class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs font-semibold"
						/>
						{#if inspectionResult.is_duplicate_name}
							<p class="mt-1 text-[10px] text-amber-700">
								ℹ A project named <strong>"{inspectionResult.project_name}"</strong> already exists. We have suggested <strong>"{inspectionResult.suggested_name}"</strong>, or you can retain the original name to merge data.
							</p>
						{/if}
					</label>

					<label class="flex items-center gap-2 cursor-pointer text-[11px] text-[var(--text-primary)]">
						<input
							type="checkbox"
							bind:checked={overwriteExisting}
							class="rounded-xs text-blue-600 focus:ring-0"
						/>
						<span>Overwrite existing continuous well curves and ancillary tables if matching well exists</span>
					</label>
				</div>
			</div>

			<!-- Commit Action Bar -->
			<div class="mt-2 flex items-center justify-between border-t border-[var(--border-hairline)] pt-3">
				<div class="text-[10px] text-[var(--text-muted)]">
					{#if selectedFile}
						File: <span class="font-medium text-slate-700">{selectedFile.name}</span> ({formatBytes(selectedFile.size)})
					{/if}
				</div>
				<div class="flex items-center gap-2">
					<button
						type="button"
						onclick={resetFile}
						disabled={committing}
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-medium hover:bg-slate-100 disabled:opacity-50"
					>
						Cancel
					</button>
					<button
						type="button"
						onclick={handleCommit}
						disabled={committing || !targetProjectName.trim()}
						class="flex h-7 items-center gap-1.5 rounded-sm bg-[var(--accent-primary)] px-4 text-xs font-semibold text-white shadow-xs hover:bg-blue-700 disabled:opacity-50"
					>
						{#if committing}
							<div class="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
							<span>Ingesting Package...</span>
						{:else}
							<span>📥 Ingest &amp; Load Project</span>
						{/if}
					</button>
				</div>
			</div>
		</div>
	{/if}
</div>

