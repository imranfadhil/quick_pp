<script lang="ts">
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import {
		fillMissingCurve,
		fitBqvCrossplot,
		fitBqvCrossplotLine,
		getBqvFitIndices
	} from '$lib/utils/saturation';

	// Default reference depth is the bottom of the active filtered interval (or well max depth)
	let defaultRefDepth = $derived.by(() => {
		const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const indices = wellData.visibleIndices;
		if (indices.length > 0 && dept.length > 0) {
			let maxD = -Infinity;
			for (const i of indices) {
				const d = dept[i];
				if (typeof d === 'number' && Number.isFinite(d) && d > maxD) {
					maxD = d;
				}
			}
			if (Number.isFinite(maxD)) return Number(maxD.toFixed(1));
		}
		if (workbench.depthFilter && Number.isFinite(workbench.depthFilter.max)) {
			return Number(workbench.depthFilter.max.toFixed(1));
		}
		return Number(wellData.depthRange.max.toFixed(1));
	});

	let refDepth = $state<number>(0);

	$effect(() => {
		refDepth = defaultRefDepth;
	});

	function estimateTempAndRwAtDepth(depthM: number) {
		const sp = workbench.saturationParams;
		const d = Number.isFinite(depthM) ? depthM : 0;
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
			rw: Number(rw.toFixed(4)),
			tempCExact: tempC,
			rwExact: rw
		};
	}

	let estimatedRwAtSurface = $derived.by(() => estimateTempAndRwAtDepth(0));
	let estimatedRwAtRefDepth = $derived.by(() => estimateTempAndRwAtDepth(refDepth));
	let bqvReferenceRw = $derived.by(() => {
		const md = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const tvd =
			wellData.curves['TVDSS'] ?? wellData.curves['TVD'] ?? wellData.curves['DEPTH'] ?? md;
		const fallbackTvd = wellData.curves['DEPTH'] ?? md;
		let closestIndex = -1;
		let closestDistance = Infinity;
		for (const index of wellData.visibleIndices) {
			const measuredDepth = md[index];
			if (
				Number.isFinite(measuredDepth) &&
				Math.abs(measuredDepth - refDepth) < closestDistance
			) {
				closestIndex = index;
				closestDistance = Math.abs(measuredDepth - refDepth);
			}
		}
		const temperatureDepth =
			closestIndex >= 0
				? Number.isFinite(tvd[closestIndex])
					? tvd[closestIndex]
					: (fallbackTvd[closestIndex] ?? refDepth)
				: refDepth;
		return estimateTempAndRwAtDepth(temperatureDepth);
	});
	let bqvAutoFit = $derived.by(() => {
		const rawRt = wellData.curves['RT'] ?? [];
		const rawPhit = wellData.curves['PHIT'] ?? [];
		const rawVclay = wellData.curves['VCLAY'] ?? wellData.curves['VSHALE'] ?? [];
		const phitSand = wellData.curves['PHIT_SAND'];
		const useTs =
			workbench.tsParams.useInSaturation &&
			Boolean(phitSand?.some((value) => Number.isFinite(value)));
		return fitBqvCrossplot(
			useTs ? fillMissingCurve(wellData.curves['RSAND_TS'], rawRt) : rawRt,
			useTs ? fillMissingCurve(phitSand, rawPhit) : rawPhit,
			useTs ? fillMissingCurve(wellData.curves['VSH_DIS'], rawVclay) : rawVclay,
			wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [],
			getBqvFitIndices(wellData.visibleIndices, workbench.brushedDepthIndices),
			workbench.saturationParams.m
		);
	});
	let bqvFit = $derived.by(() =>
		workbench.saturationParams.bqvFitMode === 'manual'
			? fitBqvCrossplotLine(
					bqvAutoFit.points,
					workbench.saturationParams.bqvManualSlope,
					workbench.saturationParams.bqvManualIntercept
				)
			: bqvAutoFit
	);
	let crossplotCalibration = $derived.by(() => {
		const fit = bqvFit;
		if (fit.slope === null || fit.slope <= 0) return null;
		const sp = workbench.saturationParams;
		const rw = sp.rwMode === 'pickett' ? sp.rwPickett : bqvReferenceRw.rwExact;
		const tempC = bqvReferenceRw.tempCExact;
		const b =
			(-1.28 + 0.225 * tempC - 0.0004059 * tempC ** 2) /
			(1 + (0.045 * tempC - 0.27) * Math.pow(rw, 1.23));
		const cec = (fit.slope * sp.a) / (b * sp.rhoDryClay);
		return Number.isFinite(b) && b > 0 && Number.isFinite(cec) && cec > 0 ? { b, cec } : null;
	});
	let isCrossplotCalibrationStale = $derived.by(() => {
		const sp = workbench.saturationParams;
		if (sp.qvMethod !== 'crossplot') return false;
		const fit = bqvFit;
		return (
			fit.slope === null ||
			!crossplotCalibration ||
			sp.bqvAppliedSlope === null ||
			sp.bqvAppliedRefDepth === null ||
			sp.bqvAppliedRhoClay === null ||
			Math.abs(fit.slope - sp.bqvAppliedSlope) > Math.max(1e-6, Math.abs(fit.slope) * 1e-4) ||
			Math.abs(refDepth - sp.bqvAppliedRefDepth) > 0.05 ||
			Math.abs(sp.rhoDryClay - sp.bqvAppliedRhoClay) > 1e-6 ||
			Math.abs(crossplotCalibration.cec - sp.cecClay) > 1e-4
		);
	});
	const saturationModelLabels: Record<string, string> = {
		archie: 'ARCHIE',
		waxman_smits: 'WAXMAN-SMITS',
		simandoux: 'SIMANDOUX',
		indonesian: 'INDONESIAN',
		dual_water: 'DUAL WATER'
	};

	let hasTsAvailable = $derived.by(() => {
		const ps = wellData.curves['PHIT_SAND'];
		return Boolean(
			workbench.tsParams.enabled || (ps && ps.some((v) => Number.isFinite(v)))
		);
	});

	async function executeSaturation(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		if (workbench.saturationParams.qvMethod === 'crossplot' && isCrossplotCalibrationStale) {
			wellData.lastError =
				'Apply the current BQv crossplot fit before calculating Waxman-Smits saturation.';
			return;
		}
		wellData.isCalculating = true;
		wellData.lastError = null;
		const sp = workbench.saturationParams;
		const deptArr = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const phitArr = wellData.curves['PHIT'] ?? [];
		const vclayArr = wellData.curves['VCLAY'] ?? wellData.curves['VSHALE'] ?? [];
		const phitSandArr = wellData.curves['PHIT_SAND'] ?? [];
		const vshLamArr = wellData.curves['VSH_LAM'] ?? [];
		const vshDisArr = wellData.curves['VSH_DIS'] ?? [];
		const rsandTsArr = wellData.curves['RSAND_TS'] ?? [];
		const useTs = hasTsAvailable && workbench.tsParams.useInSaturation;
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
						rsh: sp.rsh,
						swb: sp.swb,
						rwb: sp.rwb,
						use_thomas_stieber: useTs,
						depths: cleanArr(deptArr) as number[] | undefined,
						phit: cleanArr(phitArr),
						vclay: cleanArr(vclayArr),
						phit_sand: useTs ? cleanArr(phitSandArr) : undefined,
						vsh_lam: useTs ? cleanArr(vshLamArr) : undefined,
						vsh_dis: useTs ? cleanArr(vshDisArr) : undefined,
						rsand_ts: useTs ? cleanArr(rsandTsArr) : undefined,
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

	function applyCrossplotCalibration() {
		const fit = bqvFit;
		const calibration = crossplotCalibration;
		if (fit.slope === null || !calibration) return;
		const sp = workbench.saturationParams;
		sp.cecClay = Number(calibration.cec.toFixed(4));
		sp.bqvAppliedSlope = fit.slope;
		sp.bqvAppliedRefDepth = refDepth;
		sp.bqvAppliedRhoClay = sp.rhoDryClay;
		sp.qvMethod = 'crossplot';
	}

	function useManualBqvFit() {
		const sp = workbench.saturationParams;
		if (sp.bqvFitMode === 'manual') return;
		if (bqvAutoFit.slope !== null) sp.bqvManualSlope = bqvAutoFit.slope;
		if (bqvAutoFit.intercept !== null) sp.bqvManualIntercept = bqvAutoFit.intercept;
		sp.bqvFitMode = 'manual';
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['saturation']} />
		<ParameterSection
			title="Water Saturation Model"
			badge={saturationModelLabels[workbench.saturationParams.method]}
		>
			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Primary Solver</span>
				<select
					bind:value={workbench.saturationParams.method}
					onchange={() => executeSaturation(false)}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="archie">Archie</option>
					<option value="waxman_smits">Waxman-Smits</option>
					<option value="simandoux">Simandoux</option>
					<option value="indonesian">Indonesian</option>
					<option value="dual_water">Dual Water</option>
				</select>
			</div>
			{#if workbench.saturationParams.method === 'simandoux' ||
				workbench.saturationParams.method === 'indonesian'}
				<NumericScrubber
					label="Shale Resistivity (Rsh)"
					bind:value={workbench.saturationParams.rsh}
					unit="ohm.m"
					min={0.01}
					max={1000}
					step={0.1}
					decimals={2}
				/>
				<div class="text-[10px] text-[var(--text-muted)]">
					Indonesian uses bulk PHIT with the shale correction from VCLAY.
				</div>
			{:else if workbench.saturationParams.method === 'dual_water'}
				<NumericScrubber
					label="Bound Water Saturation (Swb)"
					bind:value={workbench.saturationParams.swb}
					unit="v/v"
					min={0}
					max={1}
					step={0.01}
					decimals={2}
				/>
				<NumericScrubber
					label="Bound Water Resistivity (Rwb)"
					bind:value={workbench.saturationParams.rwb}
					unit="ohm.m"
					min={0.001}
					max={10}
					step={0.01}
					decimals={3}
				/>
				<div class="text-[10px] text-[var(--text-muted)]">
					Dual Water uses constant bound-water Swb and Rwb values.
				</div>
			{/if}

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

			{#if hasTsAvailable}
				<div class="my-1.5 rounded-xs border border-teal-200 bg-teal-50/70 px-2 py-1.5 text-[10.5px]">
					<label class="flex items-center justify-between gap-2 font-medium text-teal-950 cursor-pointer">
						<span class="flex items-center gap-1.5">
							<input
								type="checkbox"
								bind:checked={workbench.tsParams.useInSaturation}
								onchange={() => executeSaturation(false)}
								class="h-3.5 w-3.5 rounded-xs border-teal-400 text-teal-700"
							/>
							<span>Use Thomas-Stieber Thin-Bed Inputs</span>
						</span>
						<span class="rounded-2xs bg-teal-100 px-1 py-0.5 font-mono text-[9px] font-bold text-teal-800">
							{workbench.tsParams.useInSaturation ? 'TS Active' : 'Bulk'}
						</span>
					</label>
					<div class="mt-0.5 text-[9.5px] leading-snug text-teal-800">
						{#if workbench.tsParams.useInSaturation}
							Solves sand-lamina <span class="mono-num">Sw</span> from <span class="mono-num">PHIT_SAND</span>, dispersed <span class="mono-num">VSH_DIS</span>, and de-shaled <span class="mono-num">RSAND_TS</span>, then maps back to bulk <span class="mono-num">SWT</span> conserving HCPV.
						{:else}
							Using standard bulk <span class="mono-num">RT</span>, <span class="mono-num">PHIT</span>, and total <span class="mono-num">VCLAY</span>.
						{/if}
					</div>
				</div>
			{/if}
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
				? `Rw ≈ ${estimatedRwAtRefDepth.rw} Ω·m`
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
			<div class="mt-1 space-y-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1.5 text-[10px] text-[var(--text-secondary)]">
				<div class="flex items-center justify-between gap-1.5">
					<span>
						At Surface: <strong class="mono-num">{estimatedRwAtSurface.tempC}°C</strong> →
						<strong class="mono-num text-[var(--accent-primary)]">Rw = {estimatedRwAtSurface.rw} ohm.m</strong>
					</span>
					{#if workbench.saturationParams.rwMode === 'pickett'}
						<button
							type="button"
							onclick={() => {
								workbench.saturationParams.rwPickett = estimatedRwAtSurface.rw;
							}}
							class="shrink-0 font-semibold text-blue-700 hover:underline"
						>
							Copy to Rw
						</button>
					{/if}
				</div>
				<div class="flex items-center justify-between gap-1.5 border-t border-[var(--border-hairline)] pt-1">
					<span class="flex flex-wrap items-center gap-1">
						<span>At</span>
						<input
							type="number"
							step="1"
							min="0"
							bind:value={refDepth}
							title="Reference depth for Rw calculation (defaults to bottom of filtered interval)"
							class="mono-num h-5 w-16 rounded-2xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 py-0 text-right text-[10px] font-semibold text-[var(--text-primary)] focus:border-[var(--border-focus)] focus:ring-1 focus:ring-[var(--border-focus)] focus:outline-none"
						/>
						<span>m:</span>
						<strong class="mono-num">{estimatedRwAtRefDepth.tempC}°C</strong>
						<span>→</span>
						<strong class="mono-num text-[var(--accent-primary)]">Rw = {estimatedRwAtRefDepth.rw} ohm.m</strong>
						{#if Math.abs((refDepth ?? 0) - defaultRefDepth) > 0.1}
							<button
								type="button"
								onclick={() => {
									refDepth = defaultRefDepth;
								}}
								title="Reset to bottom of filtered interval ({defaultRefDepth} m)"
								class="ml-0.5 text-[9.5px] font-medium text-[var(--text-muted)] hover:text-[var(--accent-primary)]"
							>
								↺
							</button>
						{/if}
					</span>
					{#if workbench.saturationParams.rwMode === 'pickett'}
						<button
							type="button"
							onclick={() => {
								workbench.saturationParams.rwPickett = estimatedRwAtRefDepth.rw;
							}}
							class="shrink-0 font-semibold text-blue-700 hover:underline"
						>
							Copy to Rw
						</button>
					{/if}
				</div>
			</div>
		</ParameterSection>

		{#if workbench.saturationParams.method === 'waxman_smits'}
			<ParameterSection
				title="Shale Conductivity (B & Qv)"
				badge={workbench.saturationParams.qvMethod === 'qvn'
					? 'JUHASZ Qvn'
					: workbench.saturationParams.qvMethod === 'crossplot'
						? 'CROSSPLOT CEC'
						: 'CLAY CEC Qv'}
			>
				<div class="mb-1 grid grid-cols-[1fr_130px] items-center gap-x-2 py-0.5">
					<span class="text-xs font-medium text-[var(--text-primary)]">Qv Method</span>
					<select
						bind:value={workbench.saturationParams.qvMethod}
						class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
					>
						<option value="cec">Clay CEC & Density</option>
						<option value="qvn">Juhasz Normalized Qvn</option>
						<option value="crossplot">Crossplot-calibrated CEC</option>
					</select>
				</div>

				{#if workbench.saturationParams.qvMethod === 'cec'}
					<NumericScrubber
						label="Clay CEC"
						bind:value={workbench.saturationParams.cecClay}
						unit="meq/g"
						min={0.01}
						max={1.5}
						step={0.01}
					/>
				{:else if workbench.saturationParams.qvMethod === 'crossplot'}
					<div
						class="rounded-xs border border-blue-200 bg-blue-50/70 p-1.5 text-[10.5px] text-blue-950"
					>
						<div class="font-semibold">
							Applied clay CEC: {workbench.saturationParams.cecClay.toFixed(4)} meq/g
						</div>
						{#if isCrossplotCalibrationStale}
							<div class="mt-0.5 text-amber-800">
								The fit, reference depth, or clay density has changed. Apply the
								current fit again before running saturation.
							</div>
						{/if}
					</div>
				{:else}
					<div
						class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10.5px] text-[var(--text-secondary)]"
					>
						Computes normalized <strong class="mono-num"
							>Qvn = VCLAY × PHIT_SH / PHIT</strong
						> from log-derived shale porosity (Juhasz, 1981).
					</div>
				{/if}

				{#if workbench.saturationParams.qvMethod !== 'qvn'}
					<NumericScrubber
						label="Dry Clay Density"
						bind:value={workbench.saturationParams.rhoDryClay}
						unit="g/cc"
						min={2.4}
						max={2.95}
						step={0.01}
					/>
				{/if}

				<div
					class="mt-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px] text-[var(--text-secondary)]"
				>
					<div class="mb-1 flex items-center justify-between gap-2">
						<span class="font-semibold text-[var(--text-primary)]">Fit interval and line</span>
						<div class="flex items-center gap-1">
							<button
								type="button"
								onclick={() => (workbench.saturationParams.bqvFitMode = 'auto')}
								class="rounded-xs border px-1.5 py-0.5 font-semibold {workbench.saturationParams.bqvFitMode ===
								'auto'
									? 'border-blue-600 bg-blue-600 text-white'
									: 'border-[var(--border-hairline)] bg-white text-[var(--text-secondary)]'}"
							>
								Autofit
							</button>
							<button
								type="button"
								disabled={bqvAutoFit.points.length < 2}
								onclick={useManualBqvFit}
								class="rounded-xs border px-1.5 py-0.5 font-semibold {workbench.saturationParams.bqvFitMode ===
								'manual'
									? 'border-blue-600 bg-blue-600 text-white'
									: 'border-[var(--border-hairline)] bg-white text-[var(--text-secondary)]'} disabled:cursor-not-allowed disabled:opacity-45"
							>
								Manual
							</button>
						</div>
					</div>
					{#if workbench.saturationParams.bqvFitMode === 'manual'}
						<NumericScrubber
							label="Fit slope (BQv/a)"
							bind:value={workbench.saturationParams.bqvManualSlope}
							min={0}
							step={0.001}
							decimals={5}
						/>
						<NumericScrubber
							label="Y-intercept (1/aRw)"
							bind:value={workbench.saturationParams.bqvManualIntercept}
							step={0.001}
							decimals={5}
						/>
					{/if}
					<div>
						{#if bqvFit.slope !== null && bqvFit.intercept !== null}
							{workbench.saturationParams.bqvFitMode === 'manual'
								? 'Manual fit'
								: 'Autofit'}:
							<strong class="mono-num"
								>{bqvFit.slope.toFixed(4)}</strong
							>
							(<span class="mono-num">BQv/a</span>),
							intercept <strong class="mono-num">{bqvFit.intercept?.toFixed(4)}</strong>,
							<strong class="mono-num"
								>a × slope = {(bqvFit.slope * workbench.saturationParams.a).toFixed(
									4
								)}</strong
							>,
							<strong class="mono-num"
								>R² = {bqvFit.r2 === null ? '—' : bqvFit.r2.toFixed(3)}</strong
							>,
							<strong class="mono-num">n = {bqvFit.points.length}</strong>.
							{#if crossplotCalibration}
								At {refDepth.toFixed(1)} m, B = {crossplotCalibration.b.toFixed(3)} and
								implied CEC = {crossplotCalibration.cec.toFixed(4)} meq/g.
							{:else}
								A positive slope and valid reference-depth inputs are required for CEC.
							{/if}
						{:else}
							Select an interval with at least two valid crossplot samples and a
							positive fitted slope.
						{/if}
					</div>
					<div class="mt-1 flex items-start justify-between gap-2">
						<span class="text-[9.5px]">
							Lasso water-bearing and clean-shale samples on the BQv plot; depth/zone filters
							also constrain the interval. The intercept sets the plotted line only and does
							not override the configured Rw.
						</span>
						<button
							type="button"
							disabled={!bqvFit ||
								!crossplotCalibration ||
								crossplotCalibration.cec < 0.01 ||
								crossplotCalibration.cec > 1.5}
							onclick={applyCrossplotCalibration}
							class="shrink-0 rounded-xs border border-blue-300 bg-white px-1.5 py-0.5 font-semibold text-blue-800 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-45"
							title="Apply BQv slope as effective CEC at the reference depth"
						>
							Apply fit
						</button>
					</div>
					{#if crossplotCalibration && (crossplotCalibration.cec < 0.01 || crossplotCalibration.cec > 1.5)}
						<div class="mt-1 text-[var(--status-error)]">
							Implied CEC is outside the supported 0.01–1.5 meq/g range; adjust the
							fit interval or reference depth.
						</div>
					{/if}
				</div>
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
			disabled={wellData.isCalculating || wellData.isSaving}
			onclick={() => executeSaturation(false)}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{wellData.isCalculating ? 'Running...' : 'Run (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={wellData.isCalculating || wellData.isSaving || !wellData.hasPendingComputed('saturation')}
			onclick={saveSaturation}
			title={wellData.hasPendingComputed('saturation')
				? wellData.getPendingSaveTooltip(
						'saturation',
						'Save computed Water Saturation curves directly to the well'
					)
				: 'Run the calculation first to enable saving'}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
		>
			{wellData.isSaving ? 'Saving...' : 'Save'}
		</button>
	</div>
</div>
