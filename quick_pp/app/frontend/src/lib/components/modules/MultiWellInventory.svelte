<script lang="ts">
	import { postJson } from '$lib/api/client';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtDepth, fmtNum } from '$lib/utils/formatters';

	const KEY_CURVES = ['GR', 'RT', 'NPHI', 'RHOB', 'PHIE', 'PERM', 'SWT', 'ROCK_FLAG', 'SHF'];

	// Multi-well Stratigraphic Normalization state
	let normMethod = $state<'quantile' | 'histogram'>('quantile');
	let normKeyWell = $state<string>('');
	let normLogs = $state<string[]>(['GR', 'NPHI']);
	let normReport = $state<Record<string, any>[]>([]);

	async function openWellInSingleMode(wellName: string) {
		await projectData.selectWell(wellName);
		workbench.scope = 'single-well';
	}

	function toggleNormLog(mnem: string) {
		if (normLogs.includes(mnem)) {
			if (normLogs.length > 1) normLogs = normLogs.filter((c) => c !== mnem);
		} else {
			normLogs = [...normLogs, mnem];
		}
	}

	async function runStratNormalization(persist: boolean = false) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const res = await postJson<{
				report: Record<string, any>[];
				saved_wells: number;
				normalized_columns: string[];
			}>(`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/stratigraphic_normalization`, {
				well_names: projectData.selectedWellNames.length ? projectData.selectedWellNames : null,
				logs: normLogs,
				key_well: normKeyWell || null,
				method: normMethod,
				save: persist
			});
			normReport = res.report || [];
			if (persist) {
				wellData.lastSavedMessage = `Saved ${res.normalized_columns.join(', ')} (${res.saved_wells} wells) ✓`;
				if (workbench.wellId) {
					await wellData.loadWellBundle(workbench.projectId, workbench.wellId);
				}
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Stratigraphic normalization error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection
			title="Field Overview"
			badge={`${projectData.wells.length} Wells`}
		>
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
										{w.name}
									</td>
									<td class="mono-num px-1 text-right text-[var(--text-secondary)]">
										{w.depth_count}
									</td>
									{#each KEY_CURVES as kc}
										{@const has = w.curves.includes(kc)}
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

		<ParameterSection
			title="Multi-Well Stratigraphic Normalization"
			badge={normReport.length ? `${normReport.length} fits` : 'Zone-Aware'}
		>
			<div class="grid grid-cols-2 gap-1">
				{#each [['quantile', 'Quantile (P10–P90)'], ['histogram', 'Histogram CDF']] as [m, lbl]}
					<button
						type="button"
						onclick={() => (normMethod = m as any)}
						class="h-6 rounded-sm border text-[11px] font-medium {normMethod === m
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
					>
						{lbl}
					</button>
				{/each}
			</div>

			<div class="mt-1.5 flex items-center justify-between text-[11px]">
				<span class="text-[var(--text-secondary)]">Reference Key Well:</span>
				<select
					bind:value={normKeyWell}
					class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px] font-medium"
				>
					<option value="">Pooled Field Distribution</option>
					{#each projectData.wells as w}
						<option value={w.name}>{w.name}</option>
					{/each}
				</select>
			</div>

			<div class="mt-1.5 flex items-center gap-1">
				<span class="text-[11px] text-[var(--text-secondary)]">Logs:</span>
				{#each ['GR', 'NPHI', 'RHOB', 'DTC'] as mnem}
					<button
						type="button"
						onclick={() => toggleNormLog(mnem)}
						class="mono-num h-5 rounded-xs border px-1.5 text-[10px] font-semibold {normLogs.includes(
							mnem
						)
							? 'border-blue-600 bg-blue-600 text-white'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
					>
						{mnem}
					</button>
				{/each}
			</div>

			<div class="mt-1.5 flex gap-1.5">
				<button
					type="button"
					disabled={projectData.isComputingMultiWell || projectData.wells.length === 0}
					onclick={() => runStratNormalization(false)}
					class="h-6 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-[11px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
				>
					Analyze Normalization
				</button>
				<button
					type="button"
					disabled={projectData.isComputingMultiWell || projectData.wells.length === 0}
					onclick={() => runStratNormalization(true)}
					class="h-6 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
				>
					Save *_NORM
				</button>
			</div>

			{#if normReport.length > 0}
				<div class="mt-1.5 max-h-40 overflow-y-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[10px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-5 border-b border-[var(--border-hairline)]">
								<th class="px-1 font-semibold">Well</th>
								<th class="px-1 font-semibold">Zone</th>
								<th class="px-1 font-semibold">Curve</th>
								<th class="px-1 text-right font-semibold">Shift</th>
								<th class="px-1 text-right font-semibold">Scale</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each normReport as r}
								<tr class="h-5">
									<td class="mono-num px-1 font-medium">{r.WELL_NAME}</td>
									<td class="px-1 text-[var(--text-secondary)]">{r.ZONE}</td>
									<td class="mono-num px-1 font-semibold">{r.LOG}</td>
									<td class="mono-num px-1 text-right">{fmtNum(r.SHIFT, 2)}</td>
									<td class="mono-num px-1 text-right">{fmtNum(r.SCALE, 3)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
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


