<script lang="ts">
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtDepth } from '$lib/utils/formatters';

	const KEY_CURVES = ['GR', 'RT', 'NPHI', 'RHOB', 'PHIE', 'PERM', 'SWT', 'ROCK_FLAG', 'SHF'];

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

