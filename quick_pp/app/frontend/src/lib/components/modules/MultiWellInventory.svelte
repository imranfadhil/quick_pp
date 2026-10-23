<script lang="ts">
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtDepth } from '$lib/utils/formatters';

	const KEY_CURVES = ['GR', 'RT', 'NPHI', 'RHOB', 'PHIE', 'PERM', 'SWT', 'ROCK_FLAG', 'SHF'];

	// Project & Well Management state
	let isEditingProject = $state(false);
	let projNameDraft = $state('');
	let projDescDraft = $state('');
	let confirmDeleteProj = $state(false);

	let editingWellName = $state<string | null>(null);
	let wellNameDraft = $state('');
	let confirmDeleteWellName = $state<string | null>(null);

	function startEditProject() {
		projNameDraft = projectData.summary?.name ?? '';
		projDescDraft = projectData.summary?.description ?? '';
		confirmDeleteProj = false;
		isEditingProject = true;
	}

	async function saveProjectEdit() {
		if (!workbench.projectId || !projNameDraft.trim()) return;
		projectData.error = null;
		try {
			await projectData.renameProject(workbench.projectId, projNameDraft, projDescDraft);
			isEditingProject = false;
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to rename project';
		}
	}

	async function executeDeleteProject() {
		if (!workbench.projectId) return;
		projectData.error = null;
		try {
			await projectData.deleteProject(workbench.projectId);
			confirmDeleteProj = false;
			isEditingProject = false;
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to delete project';
		}
	}

	function startRenameWell(wellName: string) {
		confirmDeleteWellName = null;
		editingWellName = wellName;
		wellNameDraft = wellName;
	}

	async function saveWellRename(oldName: string) {
		if (!wellNameDraft.trim() || wellNameDraft.trim() === oldName) {
			editingWellName = null;
			return;
		}
		projectData.error = null;
		try {
			await projectData.renameWell(oldName, wellNameDraft);
			editingWellName = null;
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to rename well';
		}
	}

	async function executeDeleteWell(wellName: string) {
		projectData.error = null;
		try {
			await projectData.deleteWell(wellName);
			confirmDeleteWellName = null;
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to delete well';
		}
	}

	async function openWellInSingleMode(wellName: string) {
		await projectData.selectWell(wellName);
		workbench.scope = 'single-well';
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection
			title="Field Overview"
			badge={`${projectData.wells.length} Wells`}
		>
			<div class="mb-2 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2 text-xs">
				{#if !isEditingProject}
					<div class="flex items-start justify-between gap-2">
						<div class="min-w-0">
							<div class="text-[10px] uppercase text-[var(--text-muted)]">Active Project</div>
							<div class="truncate font-semibold text-[var(--text-primary)]">
								{projectData.summary?.name || 'No Project Selected'}
							</div>
							{#if projectData.summary?.description}
								<div class="mt-0.5 truncate text-[10px] text-[var(--text-secondary)]">
									{projectData.summary.description}
								</div>
							{/if}
						</div>
						{#if workbench.projectId}
							<div class="flex shrink-0 items-center gap-1">
								<button
									type="button"
									title="Rename Project or Edit Description"
									onclick={startEditProject}
									class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
								>
									✎ Rename
								</button>
								{#if !confirmDeleteProj}
									<button
										type="button"
										title="Delete Project"
										onclick={() => (confirmDeleteProj = true)}
										class="h-6 rounded-xs border border-red-200 bg-red-50 px-1.5 text-[10px] font-medium text-red-700 hover:bg-red-100"
									>
										Delete
									</button>
								{:else}
									<button
										type="button"
										onclick={executeDeleteProject}
										class="h-6 rounded-xs bg-red-600 px-1.5 text-[10px] font-bold text-white hover:bg-red-700"
									>
										Confirm?
									</button>
									<button
										type="button"
										onclick={() => (confirmDeleteProj = false)}
										class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[10px]"
									>
										×
									</button>
								{/if}
							</div>
						{/if}
					</div>
				{:else}
					<div class="space-y-1.5">
						<div class="text-[10px] font-semibold text-[var(--text-secondary)]">
							Edit Project Details
						</div>
						<input
							type="text"
							placeholder="Project Name"
							bind:value={projNameDraft}
							onkeydown={(e) => e.key === 'Enter' && saveProjectEdit()}
							class="h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs font-medium"
						/>
						<input
							type="text"
							placeholder="Description (optional)"
							bind:value={projDescDraft}
							onkeydown={(e) => e.key === 'Enter' && saveProjectEdit()}
							class="h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
						/>
						<div class="flex justify-end gap-1">
							<button
								type="button"
								onclick={() => (isEditingProject = false)}
								class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[10px]"
							>
								Cancel
							</button>
							<button
								type="button"
								onclick={saveProjectEdit}
								class="h-6 rounded-xs bg-[var(--accent-primary)] px-2 text-[10px] font-semibold text-white"
							>
								Save
							</button>
						</div>
					</div>
				{/if}
			</div>

			<div class="grid grid-cols-2 gap-1.5 text-xs">
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
					<div class="text-[10px] text-[var(--text-muted)]">Field Min Depth</div>
					<div class="mono-num font-semibold text-[var(--text-primary)]">
						{fmtDepth(projectData.summary?.min_depth)}
					</div>
				</div>
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
					<div class="text-[10px] text-[var(--text-muted)]">Field Max Depth</div>
					<div class="mono-num font-semibold text-[var(--text-primary)]">
						{fmtDepth(projectData.summary?.max_depth)}
					</div>
				</div>
			</div>

			<button
				type="button"
				onclick={() => (workbench.importerOpen = true)}
				class="mt-2 h-7 w-full rounded-sm bg-[var(--accent-primary)] px-2.5 text-xs font-semibold text-white hover:bg-blue-700"
			>
				+ Batch Import LAS / Ancillary Files
			</button>
		</ParameterSection>

		<ParameterSection title="Well & Curve Availability Matrix">
			<div class="overflow-x-auto border border-[var(--border-hairline)]">
				<table class="w-full border-collapse text-left text-[11px]">
					<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
						<tr class="h-6 border-b border-[var(--border-hairline)]">
							<th class="px-1.5 font-semibold">Well</th>
							<th class="px-1 text-right font-semibold">Rows</th>
							{#each KEY_CURVES as kc}
								<th class="px-1 text-center font-semibold text-[9px]">{kc}</th>
							{/each}
						</tr>
					</thead>
					<tbody class="divide-y divide-[var(--border-hairline)]">
						{#if projectData.wells.length === 0}
							<tr>
								<td colspan={KEY_CURVES.length + 2} class="p-3 text-center text-xs text-[var(--text-muted)]">
									No wells in project. Click Batch Import above.
								</td>
							</tr>
						{:else}
							{#each projectData.wells as w (w.well_id)}
								<tr
									onclick={() => openWellInSingleMode(w.name)}
									title="Click to inspect well in Single-Well mode"
									class="h-6 cursor-pointer hover:bg-[var(--bg-subtle)] {workbench.wellId ===
									w.name
										? 'bg-blue-50/70'
										: ''}"
								>
									<td class="mono-num px-1.5 font-semibold text-[var(--accent-primary)]">
										{#if editingWellName === w.name}
											<!-- svelte-ignore a11y_click_events_have_key_events -->
											<!-- svelte-ignore a11y_no_static_element_interactions -->
											<div
												onclick={(e) => e.stopPropagation()}
												class="flex items-center gap-0.5 py-0.5"
											>
												<input
													type="text"
													bind:value={wellNameDraft}
													onkeydown={(e) => {
														if (e.key === 'Enter') saveWellRename(w.name);
														if (e.key === 'Escape') editingWellName = null;
													}}
													class="mono-num h-5 w-24 rounded-xs border border-[var(--accent-primary)] bg-white px-1 text-[11px] font-semibold text-[var(--text-primary)]"
												/>
												<button
													type="button"
													title="Save Well Name"
													onclick={() => saveWellRename(w.name)}
													class="flex h-5 w-5 items-center justify-center rounded-xs bg-[var(--accent-primary)] text-[10px] font-bold text-white"
												>
													✓
												</button>
												<button
													type="button"
													title="Cancel Rename"
													onclick={() => (editingWellName = null)}
													class="flex h-5 w-5 items-center justify-center rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[10px]"
												>
													×
												</button>
											</div>
										{:else}
											<div class="group flex items-center justify-between gap-1">
												<span class="truncate">{w.name}</span>
												<!-- svelte-ignore a11y_click_events_have_key_events -->
												<!-- svelte-ignore a11y_no_static_element_interactions -->
												<span
													onclick={(e) => e.stopPropagation()}
													class="flex shrink-0 items-center gap-0.5 opacity-75 group-hover:opacity-100"
												>
													{#if confirmDeleteWellName === w.name}
														<button
															type="button"
															title="Confirm Delete Well"
															onclick={() => executeDeleteWell(w.name)}
															class="h-4 rounded-xs bg-red-600 px-1 text-[9px] font-bold text-white hover:bg-red-700"
														>
															Del?
														</button>
														<button
															type="button"
															title="Cancel"
															onclick={() => (confirmDeleteWellName = null)}
															class="h-4 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-[9px] text-[var(--text-secondary)]"
														>
															×
														</button>
													{:else}
														<button
															type="button"
															title="Rename Well {w.name}"
															onclick={() => startRenameWell(w.name)}
															class="flex h-4 w-4 items-center justify-center rounded-xs text-[10px] text-[var(--text-muted)] hover:bg-blue-100 hover:text-[var(--accent-primary)]"
														>
															✎
														</button>
														<button
															type="button"
															title="Delete Well {w.name}"
															onclick={() => (confirmDeleteWellName = w.name)}
															class="flex h-4 w-4 items-center justify-center rounded-xs text-[10px] text-[var(--text-muted)] hover:bg-red-100 hover:text-red-700"
														>
															×
														</button>
													{/if}
												</span>
											</div>
										{/if}
									</td>
									<td class="mono-num px-1 text-right text-[var(--text-secondary)]">
										{w.depth_count ?? (w as any).row_count ?? 0}
									</td>
									{#each KEY_CURVES as kc}
										{@const has = (w.curves ?? (w as any).available_curves ?? []).includes(kc)}
										<td class="px-1 text-center">
											<span
												class="inline-block h-2 w-2 rounded-full {has
													? 'bg-emerald-500'
													: 'bg-slate-200'}"
											></span>
										</td>
									{/each}
								</tr>
							{/each}
						{/if}
					</tbody>
				</table>
			</div>
		</ParameterSection>

		{#if projectData.error}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{projectData.error}
			</div>
		{/if}
	</div>

	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<span class="text-[11px] text-[var(--text-muted)]">
			Zones: {projectData.allZones.length} | Curves: {projectData.allCurves.length}
		</span>
		<button
			type="button"
			onclick={() => {
				if (workbench.projectId) projectData.selectProject(workbench.projectId);
			}}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-medium text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
		>
			Refresh Field
		</button>
	</div>
</div>


