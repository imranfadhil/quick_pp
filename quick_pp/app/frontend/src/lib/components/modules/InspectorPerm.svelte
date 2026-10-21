<script lang="ts">
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	let method = $state<'choo' | 'timur' | 'tixier' | 'coates' | 'kozeny_carman'>('choo');
	let m = $state(2.0);
	let A = $state(1000000);
	let B = $state(3.0);
	let C = $state(2.0);
	let s = $state(0.01);
	let swirr = $state(0.2);

	async function executePerm(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		wellData.isCalculating = true;
		wellData.lastError = null;
		const deptArr = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const phitArr = wellData.curves['PHIT'] ?? wellData.curves['PHIE'] ?? [];
		const vclayArr = wellData.curves['VCLAY'] ?? wellData.curves['VSHALE'] ?? [];
		const vsiltArr = wellData.curves['VSILT'] ?? [];
		const cleanArr = (arr: number[]) =>
			arr.length > 0 ? arr.map((v) => (Number.isFinite(v) ? v : null)) : undefined;
		try {
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/permeability',
				{
					params: {
						path: { project_id: workbench.projectId, well_name: workbench.wellId }
					},
					body: {
						method,
						m,
						A,
						B,
						C,
						s,
						swirr,
						depths: cleanArr(deptArr) as number[] | undefined,
						phit: cleanArr(phitArr),
						vclay: cleanArr(vclayArr),
						vsilt: cleanArr(vsiltArr),
						min_depth: workbench.depthFilter?.min ?? null,
						max_depth: workbench.depthFilter?.max ?? null,
						zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
						save: persist
					} as any
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Permeability calculation failed'));
			}
			wellData.mergeComputedCurves(data.data, data.stats);
			wellData.permCoreMetrics = data.core_metrics;
			if (persist) {
				wellData.lastSavedMessage = 'Permeability Saved ✓';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Permeability error';
		} finally {
			wellData.isCalculating = false;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection title="Permeability Model" badge={method.toUpperCase()}>
			<div class="grid grid-cols-[1fr_118px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Empirical Equation</span>
				<select
					bind:value={method}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="choo">Choo (Sand-Silt-Clay)</option>
					<option value="kozeny_carman">Kozeny-Carman</option>
					<option value="timur">Timur (Swirr)</option>
					<option value="tixier">Tixier</option>
					<option value="coates">Coates-Dumanoir</option>
				</select>
			</div>

			{#if method === 'choo'}
				<NumericScrubber
					label="Cementation m"
					bind:value={m}
					min={1.0}
					max={4.0}
					step={0.05}
					onrun={() => executePerm(false)}
				/>
				<NumericScrubber
					label="Constant A"
					bind:value={A}
					min={100}
					max={1e8}
					step={10000}
					decimals={0}
					onrun={() => executePerm(false)}
				/>
				<NumericScrubber
					label="Clay Exponent B"
					bind:value={B}
					min={0.5}
					max={8.0}
					step={0.1}
					onrun={() => executePerm(false)}
				/>
				<NumericScrubber
					label="Silt Exponent C"
					bind:value={C}
					min={0.5}
					max={8.0}
					step={0.1}
					onrun={() => executePerm(false)}
				/>
			{:else if method === 'kozeny_carman'}
				<NumericScrubber
					label="Surface Area Factor s"
					bind:value={s}
					min={0.0001}
					max={1.0}
					step={0.001}
					decimals={4}
					onrun={() => executePerm(false)}
				/>
			{:else}
				<NumericScrubber
					label="Irreducible Sw (Swirr)"
					bind:value={swirr}
					unit="v/v"
					min={0.02}
					max={0.8}
					step={0.01}
					onrun={() => executePerm(false)}
				/>
			{/if}
		</ParameterSection>

		{#if wellData.permCoreMetrics}
			<ParameterSection
				title="Core Permeability Calibration (log10 mD)"
				badge={`n=${wellData.permCoreMetrics.count}`}
			>
				<div class="grid grid-cols-3 gap-1.5 text-center">
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
						<div class="text-[10px] text-[var(--text-muted)]">log10 RMSE</div>
						<div class="mono-num text-xs font-bold text-[var(--text-primary)]">
							{fmtNum(wellData.permCoreMetrics.rmse, 3)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
						<div class="text-[10px] text-[var(--text-muted)]">log10 MAE</div>
						<div class="mono-num text-xs font-bold text-[var(--text-primary)]">
							{fmtNum(wellData.permCoreMetrics.mae, 3)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
						<div class="text-[10px] text-[var(--text-muted)]">R²</div>
						<div class="mono-num text-xs font-bold text-[var(--status-ok)]">
							{fmtNum(wellData.permCoreMetrics.r2, 3)}
						</div>
					</div>
				</div>
			</ParameterSection>
		{/if}

		<ParameterSection title="Multi-Well FZI Shortcut">
			<p class="text-[11px] text-[var(--text-secondary)]">
				Need Rock-Type-based permeability transforms across multiple wells?
			</p>
			<button
				type="button"
				onclick={() => {
					workbench.scope = 'multi-well';
					workbench.multiWellStep = 'perm-transform';
				}}
				class="mt-1 h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-xs font-medium text-[var(--accent-primary)] hover:bg-blue-50"
			>
				Open Field Perm Transform →
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
			onclick={() => executePerm(false)}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{wellData.isCalculating ? 'Running...' : 'Run (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={wellData.isCalculating}
			onclick={() => executePerm(true)}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
		>
			Save
		</button>
	</div>
</div>

