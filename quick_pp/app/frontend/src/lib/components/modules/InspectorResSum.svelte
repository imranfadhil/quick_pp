<script lang="ts">
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { downloadCsv } from '$lib/utils/formatters';

	let vshMax = $state(0.4);
	let phitMin = $state(0.05);
	let swtMax = $state(0.75);
	let includeUpscaling = $state(true);
	let cellSize = $state(2.0);
	let upscaledCellCount = $state<number | null>(null);

	async function executeResSum(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		wellData.isCalculating = true;
		wellData.lastError = null;
		try {
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/reservoir_summary',
				{
					params: {
						path: { project_id: workbench.projectId, well_name: workbench.wellId }
					},
					body: {
						cut_offs: {
							VSHALE: vshMax,
							PHIT: phitMin,
							SWT: swtMax
						},
						include_upscaling: includeUpscaling,
						cell_size: cellSize,
						min_depth: workbench.depthFilter?.min ?? null,
						max_depth: workbench.depthFilter?.max ?? null,
						zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
						save: persist
					}
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Reservoir summary failed'));
			}
			if (data.flags?.data) {
				wellData.mergeComputedCurves(data.flags.data);
			}
			wellData.zonalSummary = data.summary || [];
			upscaledCellCount = data.upscaling?.n_cells ?? null;
			if (persist) {
				wellData.lastSavedMessage = 'Pay Flags Saved ✓';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Reservoir summary error';
		} finally {
			wellData.isCalculating = false;
		}
	}

	function exportSummaryCsv() {
		if (!wellData.zonalSummary.length) return;
		const headers = Object.keys(wellData.zonalSummary[0]);
		downloadCsv(
			`${workbench.wellId || 'well'}_reservoir_summary.csv`,
			headers,
			wellData.zonalSummary
		);
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection title="Net Rock & Net Pay Cutoffs" badge="Live Flag Strip">
			<NumericScrubber
				label="Max Shale (Vsh,max)"
				bind:value={vshMax}
				unit="v/v"
				min={0.05}
				max={1.0}
				step={0.02}
				onrun={() => executeResSum(false)}
			/>
			<NumericScrubber
				label="Min Porosity (Φ,min)"
				bind:value={phitMin}
				unit="v/v"
				min={0.0}
				max={0.35}
				step={0.005}
				onrun={() => executeResSum(false)}
			/>
			<NumericScrubber
				label="Max Water Sat (Sw,max)"
				bind:value={swtMax}
				unit="v/v"
				min={0.1}
				max={1.0}
				step={0.02}
				onrun={() => executeResSum(false)}
			/>
		</ParameterSection>

		<ParameterSection
			title="Regular Grid Upscaling"
			badge={upscaledCellCount ? `${upscaledCellCount} cells` : undefined}
		>
			<label class="flex items-center gap-2 py-0.5 text-xs font-medium cursor-pointer">
				<input
					type="checkbox"
					bind:checked={includeUpscaling}
					class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
				/>
				<span>Compute Upscaled Grid Statistics</span>
			</label>

			{#if includeUpscaling}
				<NumericScrubber
					label="Grid Cell Thickness"
					bind:value={cellSize}
					unit="m"
					min={0.25}
					max={50.0}
					step={0.5}
					onrun={() => executeResSum(false)}
				/>
			{/if}
		</ParameterSection>

		<ParameterSection title="Zonal Summary Export" badge={`${wellData.zonalSummary.length} zones`}>
			<p class="text-[11px] text-[var(--text-secondary)]">
				Zonal Net-to-Gross, PHIE, PERM, and Hydrocarbon Pore Volume (HPV) are displayed in the Right Pane Data Grid.
			</p>
			<button
				type="button"
				disabled={wellData.zonalSummary.length === 0}
				onclick={exportSummaryCsv}
				class="mt-1.5 h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-xs font-semibold text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-40"
			>
				Export Zonal Table (.csv)
			</button>
		</ParameterSection>

		{#if wellData.lastError}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{wellData.lastError}
			</div>
		{/if}
	</div>

	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<button
			type="button"
			disabled={wellData.isCalculating}
			onclick={() => executeResSum(false)}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{wellData.isCalculating ? 'Computing...' : 'Run Summary (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={wellData.isCalculating}
			onclick={() => executeResSum(true)}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
		>
			Save Flags
		</button>
	</div>
</div>

