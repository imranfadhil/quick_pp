<script lang="ts">
	import { onMount } from 'svelte';
	import { postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench, type VolumetricZoneParamConfig } from '$lib/state/workbench.svelte';
	import { downloadCsv, fmtNum } from '$lib/utils/formatters';

	const mv = workbench.multiWellVolumetrics;

	let isCalculating = $state(false);
	let lastError = $state<string | null>(null);
	let saveMessage = $state<string | null>(null);

	// Get available zones from project wells or current result
	let availableZones = $derived.by(() => {
		const zones = new Set<string>();
		if (projectData.volumetricResult?.zonal_summary) {
			for (const z of projectData.volumetricResult.zonal_summary) {
				if (z.zone && z.zone !== 'ALL') zones.add(z.zone);
			}
		}
		for (const z of wellData.availableZones) {
			if (z && z !== 'ALL') zones.add(z);
		}
		return Array.from(zones).sort();
	});

	// Get or initialize config for active selected zone
	function getZoneConfig(zone: string): VolumetricZoneParamConfig {
		if (!mv.zoneConfigs[zone]) {
			mv.zoneConfigs[zone] = {
				areaMean: 1200.0,
				areaStd: 300.0,
				grvMean: 50.0,
				grvStd: 15.0,
				thickMean: 25.0,
				thickStd: 5.0,
				ntgMean: 0.65,
				ntgStd: 0.10,
				poroMean: 0.18,
				swMean: 0.35,
				boP50: 1.25,
				bgP50: 0.005,
				rs: 500.0,
				rfOil: 0.30,
				rfGas: 0.70
			};
		}
		return mv.zoneConfigs[zone] as VolumetricZoneParamConfig;
	}

	let activeZoneCfg = $derived(
		mv.selectedZone === 'ALL'
			? getZoneConfig('ALL')
			: getZoneConfig(mv.selectedZone)
	);

	async function runVolumetrics(persist: boolean = false) {
		if (!workbench.projectId) return;
		isCalculating = true;
		lastError = null;
		saveMessage = null;

		try {
			// Convert zoneConfigs into zone_params payload
			const zoneParamsPayload: Record<string, any> = {};
			for (const [zKey, cfg] of Object.entries(mv.zoneConfigs)) {
				const zEntry: Record<string, any> = {
					bo: [Math.max(1.01, (cfg.boP50 ?? 1.25) - 0.15), (cfg.boP50 ?? 1.25) + 0.20, cfg.boP50 ?? 1.25, 0.05],
					bg: [Math.max(0.001, (cfg.bgP50 ?? 0.005) - 0.0015), (cfg.bgP50 ?? 0.005) + 0.002, cfg.bgP50 ?? 0.005, 0.0006],
					rs: cfg.rs ?? 500.0,
					rf_oil: [Math.max(0.05, (cfg.rfOil ?? 0.30) - 0.15), Math.min(0.70, (cfg.rfOil ?? 0.30) + 0.15), cfg.rfOil ?? 0.30, 0.05],
					rf_gas: [Math.max(0.20, (cfg.rfGas ?? 0.70) - 0.20), Math.min(0.95, (cfg.rfGas ?? 0.70) + 0.15), cfg.rfGas ?? 0.70, 0.06]
				};
				if (mv.mode === 'grv') {
					const gMean = cfg.grvMean ?? 50.0;
					const gStd = cfg.grvStd ?? 15.0;
					zEntry.grv = [Math.max(1.0, gMean - 2.0 * gStd), gMean + 2.5 * gStd, gMean, gStd];
				} else {
					const aMean = cfg.areaMean ?? 1200.0;
					const aStd = cfg.areaStd ?? 300.0;
					zEntry.area = [Math.max(10.0, aMean - 2.0 * aStd), aMean + 2.5 * aStd, aMean, aStd];
				}
				zoneParamsPayload[zKey] = zEntry;
			}

			const payload = {
				cut_offs: {
					VSHALE: mv.vshMax,
					PHIT: mv.phitMin,
					SWT: mv.swtMax
				},
				mode: mv.mode,
				compute_oil: mv.computeOil,
				compute_gas: mv.computeGas,
				well_names: projectData.activeWells.length > 0 ? projectData.activeWells : null,
				zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
				zone_params: zoneParamsPayload,
				n_samples: mv.nSamples,
				seed: mv.seed,
				save_to_project: persist
			};

			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/multi_well_volumetrics`,
				payload
			);

			projectData.volumetricResult = res;

			if (persist) {
				saveMessage = 'Volumetric Evaluation & Parameters Saved to Project ✓';
				setTimeout(() => {
					saveMessage = null;
				}, 4000);
			}
		} catch (e: any) {
			lastError = e?.message || 'Failed to compute multi-well volumetrics.';
		} finally {
			isCalculating = false;
		}
	}

	function exportZonalCsv() {
		const res = projectData.volumetricResult;
		if (!res?.zonal_summary?.length) return;
		const rows = res.zonal_summary.map((z) => ({
			Zone: z.zone,
			Wells: z.well_count,
			Gross_m: z.mean_gross,
			Net_Pay_m: z.mean_net_pay,
			NTG_pay: z.ntg_pay,
			PHIE_pay: z.mean_phie,
			Sw_pay: z.mean_sw,
			HCPT_m: z.mean_hcpt,
			Mode: z.mode,
			Area_Acres: z.area_mean ?? '',
			GRV_10e6_m3: z.grv_mean ?? '',
			STOIIP_1P_MMbbl: z.stoiip?.p90 ?? 0,
			STOIIP_2P_MMbbl: z.stoiip?.p50 ?? 0,
			STOIIP_3P_MMbbl: z.stoiip?.p10 ?? 0,
			STOIIP_Mean_MMbbl: z.stoiip?.mean ?? 0,
			Rec_Oil_2P_MMbbl: z.rec_oil?.p50 ?? 0,
			GIIP_1P_Bcf: z.giip?.p90 ?? 0,
			GIIP_2P_Bcf: z.giip?.p50 ?? 0,
			GIIP_3P_Bcf: z.giip?.p10 ?? 0,
			Rec_Gas_2P_Bcf: z.rec_gas?.p50 ?? 0,
			Assoc_Gas_2P_Bcf: z.assoc_gas?.p50 ?? 0
		}));
		downloadCsv('MultiWell_Zonal_Volumetrics.csv', Object.keys(rows[0]), rows);
	}

	function exportWellMatrixCsv() {
		const res = projectData.volumetricResult;
		if (!res?.well_summary?.length) return;
		downloadCsv(
			'MultiWell_Zonal_Well_Matrix.csv',
			Object.keys(res.well_summary[0]),
			res.well_summary
		);
	}

	onMount(() => {
		if (!projectData.volumetricResult && workbench.projectId) {
			runVolumetrics(false);
		}
	});
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['ressum']} />

		{#if saveMessage}
			<div class="m-2 rounded-sm border border-emerald-300 bg-emerald-50 px-2.5 py-1.5 text-[11px] font-medium text-emerald-800">
				{saveMessage}
			</div>
		{/if}

		{#if lastError}
			<div class="m-2 rounded-sm border border-red-300 bg-red-50 px-2.5 py-1.5 text-[11px] font-medium text-red-800">
				{lastError}
			</div>
		{/if}

		<!-- Net Pay Cutoffs Section -->
		<ParameterSection title="Net Rock & Net Pay Cut-offs" badge="All Wells">
			<NumericScrubber
				label="Max Shale (Vsh,max)"
				bind:value={mv.vshMax}
				unit="v/v"
				min={0.05}
				max={0.90}
				step={0.02}
				onrun={() => runVolumetrics(false)}
			/>
			<NumericScrubber
				label="Min Porosity (Φ,min)"
				bind:value={mv.phitMin}
				unit="v/v"
				min={0.01}
				max={0.35}
				step={0.005}
				onrun={() => runVolumetrics(false)}
			/>
			<NumericScrubber
				label="Max Water Sat (Sw,max)"
				bind:value={mv.swtMax}
				unit="v/v"
				min={0.10}
				max={0.95}
				step={0.02}
				onrun={() => runVolumetrics(false)}
			/>
		</ParameterSection>

		<!-- Structural Calculation Mode & Fluids -->
		<ParameterSection title="Structural Geometry & Fluid Mode" badge={mv.mode.toUpperCase()}>
			<div class="flex items-center justify-between py-1 text-xs">
				<span class="font-medium text-[var(--text-secondary)]">Geometry Mode:</span>
				<div class="flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5 text-[10.5px]">
					<button
						type="button"
						class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.mode === 'area'
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-xs'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => {
							mv.mode = 'area';
							runVolumetrics(false);
						}}
					>
						Drainage Area (Acres)
					</button>
					<button
						type="button"
						class="rounded-xs px-2 py-0.5 font-medium transition-colors {mv.mode === 'grv'
							? 'bg-[var(--bg-surface)] font-semibold text-[var(--accent-primary)] shadow-xs'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => {
							mv.mode = 'grv';
							runVolumetrics(false);
						}}
					>
						GRV (10⁶ m³)
					</button>
				</div>
			</div>

			<div class="mt-1 flex items-center justify-between border-t border-[var(--border-hairline)] pt-1 text-xs">
				<span class="font-medium text-[var(--text-secondary)]">Phases:</span>
				<div class="flex items-center gap-3">
					<label class="flex items-center gap-1 text-[11px] font-medium cursor-pointer">
						<input
							type="checkbox"
							bind:checked={mv.computeOil}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>Oil (STOIIP)</span>
					</label>
					<label class="flex items-center gap-1 text-[11px] font-medium cursor-pointer">
						<input
							type="checkbox"
							bind:checked={mv.computeGas}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>Gas (GIIP)</span>
					</label>
				</div>
			</div>
		</ParameterSection>

		<!-- Zone-Specific Parameters Tuning -->
		<ParameterSection
			title="Zonal Geological & PVT Distributions"
			badge={mv.selectedZone === 'ALL' ? 'Field Default' : mv.selectedZone}
		>
			<div class="mb-1.5 flex items-center justify-between text-xs">
				<span class="text-[11px] font-medium text-[var(--text-secondary)]">Edit Formation:</span>
				<select
					bind:value={mv.selectedZone}
					class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-6 pl-1.5 text-[11px] font-medium"
				>
					<option value="ALL">All Formations (Default)</option>
					{#each availableZones as z}
						<option value={z}>{z}</option>
					{/each}
				</select>
			</div>

			{#if mv.mode === 'area'}
				<div class="grid grid-cols-2 gap-x-1.5">
					<NumericScrubber
						label="Area Mean"
						bind:value={activeZoneCfg.areaMean}
						unit="ac"
						min={50}
						max={25000}
						step={50}
						onrun={() => runVolumetrics(false)}
					/>
					<NumericScrubber
						label="Area StdDev"
						bind:value={activeZoneCfg.areaStd}
						unit="ac"
						min={10}
						max={5000}
						step={25}
						onrun={() => runVolumetrics(false)}
					/>
				</div>
			{:else}
				<div class="grid grid-cols-2 gap-x-1.5">
					<NumericScrubber
						label="GRV Mean"
						bind:value={activeZoneCfg.grvMean}
						unit="10⁶m³"
						min={1}
						max={1000}
						step={2}
						onrun={() => runVolumetrics(false)}
					/>
					<NumericScrubber
						label="GRV StdDev"
						bind:value={activeZoneCfg.grvStd}
						unit="10⁶m³"
						min={0.5}
						max={200}
						step={1}
						onrun={() => runVolumetrics(false)}
					/>
				</div>
			{/if}

			<div class="mt-1 border-t border-[var(--border-hairline)] pt-1">
				<span class="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
					PVT & Recovery Factor
				</span>
				<div class="grid grid-cols-2 gap-x-1.5 mt-0.5">
					{#if mv.computeOil}
						<NumericScrubber
							label="Oil Bo"
							bind:value={activeZoneCfg.boP50}
							unit="bbl/STB"
							min={1.02}
							max={2.20}
							step={0.02}
							onrun={() => runVolumetrics(false)}
						/>
						<NumericScrubber
							label="Solution GOR (Rs)"
							bind:value={activeZoneCfg.rs}
							unit="scf/STB"
							min={0}
							max={5000}
							step={50}
							onrun={() => runVolumetrics(false)}
						/>
						<NumericScrubber
							label="Oil RF"
							bind:value={activeZoneCfg.rfOil}
							unit="fraction"
							min={0.05}
							max={0.65}
							step={0.02}
							onrun={() => runVolumetrics(false)}
						/>
					{/if}

					{#if mv.computeGas}
						<NumericScrubber
							label="Gas Bg"
							bind:value={activeZoneCfg.bgP50}
							unit="rcf/scf"
							min={0.001}
							max={0.03}
							step={0.0005}
							onrun={() => runVolumetrics(false)}
						/>
						<NumericScrubber
							label="Gas RF"
							bind:value={activeZoneCfg.rfGas}
							unit="fraction"
							min={0.20}
							max={0.90}
							step={0.02}
							onrun={() => runVolumetrics(false)}
						/>
					{/if}
				</div>
			</div>
		</ParameterSection>

		<!-- Simulation Controls -->
		<ParameterSection title="Latin Hypercube Simulation" badge={`N=${mv.nSamples}`}>
			<NumericScrubber
				label="Realizations (N)"
				bind:value={mv.nSamples}
				min={500}
				max={20000}
				step={500}
				onrun={() => runVolumetrics(false)}
			/>
			<NumericScrubber
				label="Random Seed"
				bind:value={mv.seed}
				min={1}
				max={9999}
				step={1}
				onrun={() => runVolumetrics(false)}
			/>

			<div class="mt-2 flex flex-col gap-1.5">
				<button
					type="button"
					onclick={() => runVolumetrics(false)}
					disabled={isCalculating}
					class="flex h-7 w-full items-center justify-center gap-1.5 rounded-sm bg-[var(--accent-primary)] text-xs font-semibold text-white shadow-xs hover:bg-blue-600 disabled:opacity-50"
				>
					{#if isCalculating}
						<span class="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent"></span>
						<span>Running Monte Carlo...</span>
					{:else}
						<span>Run Field Volumetrics (LHS)</span>
					{/if}
				</button>

				<button
					type="button"
					onclick={() => runVolumetrics(true)}
					disabled={isCalculating}
					class="flex h-6 w-full items-center justify-center gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[11px] font-medium text-[var(--text-primary)] hover:bg-slate-200"
				>
					<span>Save Volumetric Parameters to Project</span>
				</button>
			</div>
		</ParameterSection>

		<!-- Exports -->
		<ParameterSection title="Data & Summary Exports">
			<div class="flex flex-col gap-1">
				<button
					type="button"
					onclick={exportZonalCsv}
					disabled={!projectData.volumetricResult}
					class="h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[11px] font-medium hover:bg-slate-200 disabled:opacity-40"
				>
					Export Zonal Volumetrics (.CSV)
				</button>
				<button
					type="button"
					onclick={exportWellMatrixCsv}
					disabled={!projectData.volumetricResult}
					class="h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[11px] font-medium hover:bg-slate-200 disabled:opacity-40"
				>
					Export Well-by-Well Matrix (.CSV)
				</button>
			</div>
		</ParameterSection>
	</div>
</div>
