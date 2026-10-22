<script lang="ts">
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	// Live preview of formation temperature and Rw at median depth from Salinity & Temp Gradient
	let medianDepth = $derived.by(() => {
		const dr = wellData.depthRange;
		return 0.5 * (dr.min + dr.max);
	});

	let estimatedRwAtMedianDepth = $derived.by(() => {
		const sp = workbench.saturationParams;
		const d = medianDepth;
		let tempC: number;
		if (sp.measSystem === 'imperial') {
			const tvdKft = (d * 3.28084) / 1000.0;
			const tempF = sp.surfaceTemp + sp.tempGradient * tvdKft;
			tempC = Math.max(5, Math.min(250, ((tempF - 32.0) * 5.0) / 9.0));
		} else {
			tempC = Math.max(5, Math.min(250, sp.surfaceTemp + sp.tempGradient * (d / 1000.0)));
		}
		const tempF = (tempC * 9.0) / 5.0 + 32.0;
		const sal = Math.max(100, sp.waterSalinity);
		const rw = Math.pow(400000.0 / (sal * tempF), 0.88);
		return {
			tempC: Number(tempC.toFixed(1)),
			rw: Number(rw.toFixed(4))
		};
	});

	async function executeSaturation(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		wellData.isCalculating = true;
		wellData.lastError = null;
		const sp = workbench.saturationParams;
		const deptArr = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const phitArr = wellData.curves['PHIT'] ?? wellData.curves['PHIE'] ?? [];
		const vclayArr = wellData.curves['VCLAY'] ?? wellData.curves['VSHALE'] ?? [];
		const cleanArr = (arr: number[]) =>
			arr.length > 0 ? arr.map((v) => (Number.isFinite(v) ? v : null)) : undefined;

		try {
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/saturation',
				{
					params: {
						path: { project_id: workbench.projectId, well_name: workbench.wellId }
					},
					body: {
						method: sp.method,
						rw_mode: sp.rwMode,
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
						depths: cleanArr(deptArr) as number[] | undefined,
						phit: cleanArr(phitArr),
						vclay: cleanArr(vclayArr),
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
			wellData.markModuleComputed('saturation', data.data);
			// If using Salinity & Temp Gradient mode, sync the median computed Rw to rwPickett for the Pickett Plot line
			if (sp.rwMode === 'salinity_temp') {
				const rwStat = (data.stats as any)?.RW;
				const medRw = rwStat?.p50 ?? rwStat?.mean;
				if (typeof medRw === 'number' && Number.isFinite(medRw) && medRw > 0) {
					workbench.saturationParams.rwPickett = Number(medRw.toFixed(4));
				}
			}
			if (persist) {
				wellData.lastSavedMessage = 'Saturation Saved ✓';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Saturation error';
		} finally {
			wellData.isCalculating = false;
		}
	}

	async function saveSaturation() {
		await wellData.saveModuleComputed('saturation', 'Saturation Saved ✓');
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection
			title="Water Saturation Model"
			badge={workbench.saturationParams.method === 'waxman_smits' ? 'WAXMAN-SMITS' : 'ARCHIE'}
		>
			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Primary Solver</span>
				<select
					bind:value={workbench.saturationParams.method}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="waxman_smits">Waxman-Smits</option>
					<option value="archie">Archie</option>
				</select>
			</div>

			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Rw Source Mode</span>
				<select
					bind:value={workbench.saturationParams.rwMode}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="salinity_temp">Salinity & Temp Rw(z)</option>
					<option value="pickett">Pickett Constant Rw</option>
				</select>
			</div>

			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Temp Units</span>
				<select
					bind:value={workbench.saturationParams.measSystem}
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
			/>
			<NumericScrubber
				label="Cementation Exp (m)"
				bind:value={workbench.saturationParams.m}
				min={1.1}
				max={3.5}
				step={0.02}
			/>
			<NumericScrubber
				label="Saturation Exp (n)"
				bind:value={workbench.saturationParams.n}
				min={1.1}
				max={3.5}
				step={0.02}
			/>
			<NumericScrubber
				label={workbench.saturationParams.rwMode === 'pickett'
					? 'Pickett Rw (Active)'
					: 'Pickett Line Rw'}
				bind:value={workbench.saturationParams.rwPickett}
				unit="ohm.m"
				min={0.005}
				max={2.0}
				step={0.005}
				decimals={3}
			/>
		</ParameterSection>

		<ParameterSection
			title="Salinity & Geothermal Gradient"
			badge={workbench.saturationParams.rwMode === 'salinity_temp'
				? `Rw ≈ ${estimatedRwAtMedianDepth.rw} Ω·m`
				: 'Temp B(T) Only'}
		>
			<NumericScrubber
				label="Water Salinity"
				bind:value={workbench.saturationParams.waterSalinity}
				unit="ppm"
				min={500}
				max={250000}
				step={500}
				decimals={0}
			/>
			<NumericScrubber
				label="Temp Gradient"
				bind:value={workbench.saturationParams.tempGradient}
				unit={workbench.saturationParams.measSystem === 'imperial' ? '°F/kft' : '°C/km'}
				min={5}
				max={80}
				step={1.0}
			/>
			<NumericScrubber
				label="Surface Temp"
				bind:value={workbench.saturationParams.surfaceTemp}
				unit={workbench.saturationParams.measSystem === 'imperial' ? '°F' : '°C'}
				min={0}
				max={120}
				step={1.0}
			/>
			<div class="mt-1 flex items-center justify-between rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10px] text-[var(--text-secondary)]">
				<span>
					At {medianDepth.toFixed(0)} m: <strong class="mono-num">{estimatedRwAtMedianDepth.tempC}°C</strong> →
					<strong class="mono-num text-[var(--accent-primary)]">Rw = {estimatedRwAtMedianDepth.rw} ohm.m</strong>
				</span>
				{#if workbench.saturationParams.rwMode === 'pickett'}
					<button
						type="button"
						onclick={() => {
							workbench.saturationParams.rwPickett = estimatedRwAtMedianDepth.rw;
						}}
						class="font-semibold text-blue-700 hover:underline"
					>
						Copy to Rw
					</button>
				{/if}
			</div>
		</ParameterSection>

		{#if workbench.saturationParams.method === 'waxman_smits'}
			<ParameterSection
				title="Shale Conductivity (B & Qv)"
				badge={workbench.saturationParams.qvMethod === 'qvn' ? 'JUHASZ Qvn' : 'CLAY CEC Qv'}
			>
				<div class="mb-1 grid grid-cols-[1fr_130px] items-center gap-x-2 py-0.5">
					<span class="text-xs font-medium text-[var(--text-primary)]">Qv Method</span>
					<select
						bind:value={workbench.saturationParams.qvMethod}
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
					>
						<option value="cec">Clay CEC & Density</option>
						<option value="qvn">Juhasz Normalized Qvn</option>
					</select>
				</div>

				{#if workbench.saturationParams.qvMethod !== 'qvn'}
					<NumericScrubber
						label="Clay CEC"
						bind:value={workbench.saturationParams.cecClay}
						unit="meq/g"
						min={0.01}
						max={1.5}
						step={0.01}
					/>
					<NumericScrubber
						label="Dry Clay Density"
						bind:value={workbench.saturationParams.rhoDryClay}
						unit="g/cc"
						min={2.4}
						max={2.95}
						step={0.01}
					/>
				{:else}
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10.5px] text-[var(--text-secondary)]">
						Computes normalized <strong class="mono-num">Qvn = VCLAY × PHIT_SH / PHIT</strong> from log-derived shale porosity (Juhasz, 1981).
					</div>
				{/if}
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
			disabled={wellData.isCalculating || !wellData.hasPendingComputed('saturation')}
			onclick={saveSaturation}
			title={wellData.hasPendingComputed('saturation')
				? 'Save computed Water Saturation curves directly to the well'
				: 'Run the calculation first to enable saving'}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
		>
			Save
		</button>
	</div>
</div>
