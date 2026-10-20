<script lang="ts">
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	async function executeSaturation(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		wellData.isCalculating = true;
		wellData.lastError = null;
		const sp = workbench.saturationParams;

		try {
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/saturation',
				{
					params: {
						path: { project_id: workbench.projectId, well_name: workbench.wellId }
					},
					body: {
						method: sp.method,
						meas_system: sp.measSystem,
						water_salinity: sp.waterSalinity,
						rw: sp.rwPickett,
						a: sp.a,
						m: sp.m,
						n: sp.n,
						temp_gradient: sp.tempGradient,
						surface_temp: sp.surfaceTemp,
						b_method: sp.bMethod,
						qv_method: sp.qvMethod,
						cec_clay: sp.cecClay,
						rho_dry_clay: sp.rhoDryClay,
						min_depth: workbench.depthFilter?.min ?? null,
						max_depth: workbench.depthFilter?.max ?? null,
						zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
						save: persist
					} as any
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Saturation calculation failed'));
			}
			wellData.mergeComputedCurves(data.data, data.stats);
			if (persist) {
				wellData.lastSavedMessage = 'Saturation Saved ✓';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Saturation error';
		} finally {
			wellData.isCalculating = false;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection
			title="Water Saturation Model"
			badge={workbench.saturationParams.method === 'waxman_smits' ? 'WAXMAN-SMITS' : 'ARCHIE'}
		>
			<div class="grid grid-cols-[1fr_118px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Primary Solver</span>
				<select
					bind:value={workbench.saturationParams.method}
					onchange={() => executeSaturation(false)}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="waxman_smits">Waxman-Smits</option>
					<option value="archie">Archie</option>
				</select>
			</div>

			<div class="grid grid-cols-[1fr_118px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Temp Units</span>
				<select
					bind:value={workbench.saturationParams.measSystem}
					onchange={() => executeSaturation(false)}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="metric">Metric (°C / km)</option>
					<option value="imperial">Imperial (°F / kft)</option>
				</select>
			</div>
		</ParameterSection>

		<ParameterSection title="Electrical Constants & Pickett Plot" badge="Live Pickett Line">
			<NumericScrubber
				label="Tortuosity Factor (a)"
				bind:value={workbench.saturationParams.a}
				min={0.4}
				max={2.5}
				step={0.05}
				onrun={() => executeSaturation(false)}
			/>
			<NumericScrubber
				label="Cementation Exp (m)"
				bind:value={workbench.saturationParams.m}
				min={1.1}
				max={3.5}
				step={0.02}
				onrun={() => executeSaturation(false)}
			/>
			<NumericScrubber
				label="Saturation Exp (n)"
				bind:value={workbench.saturationParams.n}
				min={1.1}
				max={3.5}
				step={0.02}
				onrun={() => executeSaturation(false)}
			/>
			<NumericScrubber
				label="Pickett Rw Intercept"
				bind:value={workbench.saturationParams.rwPickett}
				unit="ohm.m"
				min={0.005}
				max={2.0}
				step={0.005}
				decimals={3}
				onrun={() => executeSaturation(false)}
			/>
		</ParameterSection>

		<ParameterSection title="Salinity & Geothermal Gradient">
			<NumericScrubber
				label="Water Salinity"
				bind:value={workbench.saturationParams.waterSalinity}
				unit="ppm"
				min={500}
				max={250000}
				step={500}
				decimals={0}
				onrun={() => executeSaturation(false)}
			/>
			<NumericScrubber
				label="Temp Gradient"
				bind:value={workbench.saturationParams.tempGradient}
				unit="°/km"
				min={10}
				max={80}
				step={1.0}
				onrun={() => executeSaturation(false)}
			/>
			<NumericScrubber
				label="Surface Temp"
				bind:value={workbench.saturationParams.surfaceTemp}
				unit="°C"
				min={0}
				max={50}
				step={1.0}
				onrun={() => executeSaturation(false)}
			/>
		</ParameterSection>

		{#if workbench.saturationParams.method === 'waxman_smits'}
			<ParameterSection title="Shale Conductivity (B & Qv)">
				<NumericScrubber
					label="Clay CEC"
					bind:value={workbench.saturationParams.cecClay}
					unit="meq/g"
					min={0.01}
					max={1.5}
					step={0.01}
					onrun={() => executeSaturation(false)}
				/>
				<NumericScrubber
					label="Dry Clay Density"
					bind:value={workbench.saturationParams.rhoDryClay}
					unit="g/cc"
					min={2.4}
					max={2.95}
					step={0.01}
					onrun={() => executeSaturation(false)}
				/>
			</ParameterSection>
		{/if}

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
			onclick={() => executeSaturation(false)}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{wellData.isCalculating ? 'Running...' : 'Run (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={wellData.isCalculating}
			onclick={() => executeSaturation(true)}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
		>
			Save
		</button>
	</div>
</div>
