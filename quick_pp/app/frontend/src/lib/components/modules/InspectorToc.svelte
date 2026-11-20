<script lang="ts">
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { api } from '$lib/api/client';
	import { appStatus } from '$lib/state/status.svelte';

	const tp = workbench.tocParams;

	let isRunning = $state(false);
	let isDetecting = $state(false);
	let calibrationResult = $state<{
		success: boolean;
		slope?: number;
		intercept?: number;
		r2?: number;
		rmse?: number;
		sample_count?: number;
		implied_lom?: number;
		error?: string;
	} | null>(null);

	// Find available curve candidates
	let availableResCurves = $derived.by(() => {
		const cols = Object.keys(wellData.curves);
		const candidates = ['RD', 'RT', 'ILD', 'LLD', 'AHT90', 'AT90', 'RES_DEEP', 'RD_NORM', 'RT_ENV_CORR'];
		const matched = candidates.filter((c) => cols.includes(c));
		const others = cols.filter(
			(c) => (c.startsWith('R') || c.includes('RES')) && !matched.includes(c) && !c.includes('RW')
		);
		const all = [...matched, ...others];
		return all.length > 0 ? all : ['RD', 'RT'];
	});

	let availablePoroCurves = $derived.by(() => {
		const cols = Object.keys(wellData.curves);
		const ptype = tp.porosityType;
		if (ptype === 'sonic') {
			const candidates = ['DT', 'DTC', 'DT_CO', 'DT_SPLICE', 'DTC_SPLICE'];
			const matched = candidates.filter((c) => cols.includes(c));
			const others = cols.filter((c) => c.startsWith('DT') && !matched.includes(c));
			return [...matched, ...others].length > 0 ? [...matched, ...others] : ['DT', 'DTC'];
		} else if (ptype === 'density') {
			const candidates = ['RHOB_HC', 'RHOB', 'RHOB_NORM', 'RHOB_ENV_CORR', 'RHOB_ENH', 'DEN'];
			const matched = candidates.filter((c) => cols.includes(c));
			const others = cols.filter((c) => c.includes('RHOB') && !matched.includes(c));
			return [...matched, ...others].length > 0 ? [...matched, ...others] : ['RHOB', 'RHOB_HC'];
		} else {
			const candidates = ['NPHI_HC', 'NPHI', 'NPHI_NORM', 'NPHI_ENV_CORR', 'NPHI_ENH', 'NEU'];
			const matched = candidates.filter((c) => cols.includes(c));
			const others = cols.filter((c) => c.includes('NPHI') && !matched.includes(c));
			return [...matched, ...others].length > 0 ? [...matched, ...others] : ['NPHI', 'NPHI_HC'];
		}
	});

	// Ensure selected curves default to sensible candidates
	$effect(() => {
		if (!tp.resCurve || !availableResCurves.includes(tp.resCurve)) {
			tp.resCurve = availableResCurves[0] ?? 'RD';
		}
	});

	$effect(() => {
		if (!availablePoroCurves.includes(tp.poroCurve)) {
			tp.poroCurve = availablePoroCurves[0] ?? (tp.porosityType === 'sonic' ? 'DT' : tp.porosityType === 'density' ? 'RHOB' : 'NPHI');
		}
	});

	// Synchronize maturity LOM and Ro
	function handleLomChange(val: number) {
		tp.lom = Number(val.toFixed(2));
		// Ro = exp((LOM - 10.59) / 4.54)
		const roCalc = Math.exp((tp.lom - 10.59) / 4.54);
		tp.ro = Number(Math.max(0.2, Math.min(5.0, roCalc)).toFixed(2));
	}

	function handleRoChange(val: number) {
		tp.ro = Number(val.toFixed(2));
		// LOM = 10.59 + 4.54 * ln(Ro)
		const lomCalc = 10.59 + 4.54 * Math.log(Math.max(0.2, tp.ro));
		tp.lom = Number(Math.max(4.0, Math.min(16.0, lomCalc)).toFixed(2));
	}

	function handlePorosityTypeChange(newType: 'sonic' | 'density' | 'neutron') {
		tp.porosityType = newType;
		if (newType === 'sonic') {
			tp.poroCurve = 'DT';
			tp.poroBaseline = 80.0;
			tp.scalingCoeff = null;
		} else if (newType === 'density') {
			tp.poroCurve = 'RHOB';
			tp.poroBaseline = 2.50;
			tp.scalingCoeff = null;
		} else {
			tp.poroCurve = 'NPHI';
			tp.poroBaseline = 0.35;
			tp.scalingCoeff = null;
		}
	}

	function resetDefaultScaling() {
		tp.scalingCoeff = null;
	}

	let effectiveCoeff = $derived.by(() => {
		if (tp.scalingCoeff !== null && Number.isFinite(tp.scalingCoeff)) {
			return tp.scalingCoeff;
		}
		if (tp.porosityType === 'sonic') return 0.02;
		if (tp.porosityType === 'density') return -2.50;
		return 4.0;
	});

	let maturityStage = $derived.by(() => {
		const lom = tp.lom;
		if (lom < 7.5) return { label: 'Immature', color: 'bg-emerald-100 text-emerald-800 border-emerald-300' };
		if (lom < 9.5) return { label: 'Early Oil Window', color: 'bg-amber-100 text-amber-800 border-amber-300' };
		if (lom < 11.0) return { label: 'Peak Oil Window', color: 'bg-orange-100 text-orange-800 border-orange-300' };
		if (lom < 12.5) return { label: 'Condensate / Wet Gas', color: 'bg-sky-100 text-sky-800 border-sky-300' };
		return { label: 'Dry Gas / Post-Mature', color: 'bg-purple-100 text-purple-800 border-purple-300' };
	});

	// Evaluate TOC statistics from currently loaded curve
	let tocSummary = $derived.by(() => {
		const arr = wellData.curves['TOC'];
		if (!arr || arr.length === 0) return null;
		const vis = wellData.visibleIndices;
		const validVals: number[] = [];
		let richCount = 0;
		let goodCount = 0;
		let total = 0;

		for (const i of vis) {
			const v = arr[i];
			if (typeof v === 'number' && Number.isFinite(v)) {
				validVals.push(v);
				total++;
				if (v >= 2.0) richCount++;
				if (v >= 1.0) goodCount++;
			}
		}

		if (validVals.length === 0) return null;
		validVals.sort((a, b) => a - b);

		const mean = validVals.reduce((sum, x) => sum + x, 0) / validVals.length;
		const min = validVals[0];
		const max = validVals[validVals.length - 1];
		const p50 = validVals[Math.floor(validVals.length * 0.5)];
		const netRichRatio = total > 0 ? (richCount / total) * 100 : 0;
		const netGoodRatio = total > 0 ? (goodCount / total) * 100 : 0;

		return {
			count: validVals.length,
			mean: Number(mean.toFixed(2)),
			min: Number(min.toFixed(2)),
			max: Number(max.toFixed(2)),
			p50: Number(p50.toFixed(2)),
			netRichPct: Number(netRichRatio.toFixed(1)),
			netGoodPct: Number(netGoodRatio.toFixed(1))
		};
	});

	async function executeToc(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		const selectionIndices = wellData.getBrushedCalculationIndices();
		const selectionMode = selectionIndices !== null;

		isRunning = true;
		try {
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/source_rock' as any,
				{
					params: {
						project_id: workbench.projectId,
						well_name: workbench.wellId
					},
					body: {
						min_depth: workbench.depthFilter?.min,
						max_depth: workbench.depthFilter?.max,
						zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : undefined,
						res_curve: tp.resCurve,
						poro_curve: tp.poroCurve,
						porosity_type: tp.porosityType,
						r_baseline: tp.rBaseline,
						poro_baseline: tp.poroBaseline,
						scaling_coeff: tp.scalingCoeff,
						maturity_mode: tp.maturityMode,
						lom: tp.lom,
						ro: tp.ro,
						toc_baseline: tp.tocBaseline,
						clamp_negative: tp.clampNegative,
						save: persist && !selectionMode
					} as any
				} as any
			);

			if (error || !data) {
				const errMsg = (error as any)?.detail || 'Failed to compute Source Rock TOC';
				appStatus.error(errMsg, workbench.wellId);
				return;
			}

			const payload = data as any;
			if (payload.core_calibration) {
				calibrationResult = payload.core_calibration;
			}

			// Merge computed curves
			const computed: Record<string, number[]> = {};
			if (payload.DELTA_LOG_R) computed['DELTA_LOG_R'] = payload.DELTA_LOG_R;
			if (payload.TOC) computed['TOC'] = payload.TOC;
			if (payload.TOC_PASSEY) computed['TOC_PASSEY'] = payload.TOC_PASSEY;

			wellData.mergeComputedCurves(computed, payload.stats, selectionIndices);
			wellData.markModuleComputed('source_rock', Object.keys(computed));

			if (persist && !selectionMode) {
				await wellData.saveModuleComputed('source_rock', 'Source Rock & TOC Saved ✓');
			} else {
				appStatus.success(`TOC computed via Passey ${tp.porosityType.toUpperCase()} (LOM: ${tp.lom})`, workbench.wellId);
			}
		} catch (err: any) {
			appStatus.error(err.message || 'Error running TOC calculation', workbench.wellId);
		} finally {
			isRunning = false;
		}
	}

	async function autoDetectBaseline() {
		if (!workbench.projectId || !workbench.wellId) return;
		isDetecting = true;
		try {
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/source_rock' as any,
				{
					params: {
						project_id: workbench.projectId,
						well_name: workbench.wellId
					},
					body: {
						min_depth: workbench.depthFilter?.min,
						max_depth: workbench.depthFilter?.max,
						zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : undefined,
						res_curve: tp.resCurve,
						poro_curve: tp.poroCurve,
						porosity_type: tp.porosityType,
						r_baseline: tp.rBaseline,
						poro_baseline: tp.poroBaseline,
						auto_detect_baseline: true,
						save: false
					} as any
				} as any
			);

			const payload = data as any;
			if (payload?.baseline_detection) {
				const bd = payload.baseline_detection;
				if (bd.r_baseline) tp.rBaseline = bd.r_baseline;
				if (bd.poro_baseline) tp.poroBaseline = bd.poro_baseline;
				appStatus.success(
					`Baseline picked from shale: R_base=${bd.r_baseline} Ω·m, Poro_base=${bd.poro_baseline}`,
					workbench.wellId
				);
			}
		} catch (err: any) {
			appStatus.error(err.message || 'Error detecting baseline', workbench.wellId);
		} finally {
			isDetecting = false;
		}
	}

	async function saveToc() {
		await wellData.saveModuleComputed('source_rock', 'Source Rock & TOC Saved ✓');
	}
</script>

<div class="flex h-full flex-col overflow-y-auto bg-[var(--bg-surface)] text-[var(--text-primary)]">
	<!-- Saved Parameters Indicator -->
	<SavedParamsIndicator modules={['source_rock']} />

	<!-- Overview Header -->
	<div class="border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)]/50 p-2.5">
		<div class="flex items-center justify-between">
			<span class="text-[11px] font-bold text-[var(--text-primary)] uppercase tracking-wide">
				Source Rock · Passey ΔlogR
			</span>
			<span class="rounded-xs border px-1.5 py-0.5 text-[9px] font-semibold {maturityStage.color}">
				{maturityStage.label}
			</span>
		</div>
		<p class="mt-1 text-[10px] leading-tight text-[var(--text-muted)]">
			Estimates Total Organic Carbon (TOC wt%) by overlaying deep resistivity and porosity logs (sonic, density, or neutron) in lean shale.
		</p>
	</div>

	<!-- Parameter Section 1: Log Curves & Method Selection -->
	<ParameterSection title="1. Log Curves & Method Selection" defaultOpen={true}>
		<div class="flex flex-col gap-2 p-2">
			<!-- Porosity Log Type -->
			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2">
				<label for="toc-poro-type" class="text-xs font-medium text-[var(--text-secondary)]">Porosity Log Type</label>
				<select
					id="toc-poro-type"
					value={tp.porosityType}
					onchange={(e) => handlePorosityTypeChange(e.currentTarget.value as any)}
					class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs font-semibold text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-hidden"
				>
					<option value="sonic">Sonic (Δt, µs/ft)</option>
					<option value="density">Density (ρb, g/cc)</option>
					<option value="neutron">Neutron (φN, v/v)</option>
				</select>
			</div>

			<!-- Resistivity Curve -->
			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2">
				<label for="toc-res-curve" class="text-xs font-medium text-[var(--text-secondary)]">Resistivity Curve</label>
				<select
					id="toc-res-curve"
					bind:value={tp.resCurve}
					class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-hidden"
				>
					{#each availableResCurves as c}
						<option value={c}>{c}</option>
					{/each}
				</select>
			</div>

			<!-- Porosity Curve -->
			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2">
				<label for="toc-poro-curve" class="text-xs font-medium text-[var(--text-secondary)]">Porosity Curve</label>
				<select
					id="toc-poro-curve"
					bind:value={tp.poroCurve}
					class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-hidden"
				>
					{#each availablePoroCurves as c}
						<option value={c}>{c}</option>
					{/each}
				</select>
			</div>
		</div>
	</ParameterSection>

	<!-- Parameter Section 2: Lean Shale Baseline Picking -->
	<ParameterSection title="2. Lean Shale Baseline Picking" defaultOpen={true}>
		<div class="flex flex-col gap-1 p-2">
			<div class="flex items-center justify-between pb-1">
				<span class="text-[10px] text-[var(--text-muted)]">Non-source lean shale overlay values</span>
				<button
					type="button"
					onclick={autoDetectBaseline}
					disabled={isDetecting}
					class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--accent-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
				>
					{isDetecting ? 'Detecting...' : 'Auto-Pick Shale'}
				</button>
			</div>

			<NumericScrubber
				label="R_baseline (Shale)"
				bind:value={tp.rBaseline}
				unit="Ω·m"
				min={0.1}
				max={50}
				step={0.1}
				decimals={2}
				hint="Resistivity in organic-lean, water-saturated shale"
				onrun={() => executeToc(false)}
			/>

			{#if tp.porosityType === 'sonic'}
				<NumericScrubber
					label="Δt_baseline (Shale)"
					bind:value={tp.poroBaseline}
					unit="µs/ft"
					min={40}
					max={160}
					step={0.5}
					decimals={1}
					hint="Sonic transit time in lean shale baseline (std ~70-90 µs/ft)"
					onrun={() => executeToc(false)}
				/>
			{:else if tp.porosityType === 'density'}
				<NumericScrubber
					label="ρb_baseline (Shale)"
					bind:value={tp.poroBaseline}
					unit="g/cm³"
					min={1.8}
					max={3.0}
					step={0.01}
					decimals={2}
					hint="Bulk density in lean shale baseline (std ~2.40-2.60 g/cc)"
					onrun={() => executeToc(false)}
				/>
			{:else}
				<NumericScrubber
					label="φN_baseline (Shale)"
					bind:value={tp.poroBaseline}
					unit="v/v"
					min={0.05}
					max={0.65}
					step={0.01}
					decimals={2}
					hint="Neutron porosity in lean shale baseline (std ~0.30-0.45 v/v)"
					onrun={() => executeToc(false)}
				/>
			{/if}

			<div class="flex items-center justify-between pt-1">
				<span class="text-[10px] text-[var(--text-secondary)]">
					Scale Coeff: <strong>{effectiveCoeff}</strong>
				</span>
				{#if tp.scalingCoeff !== null}
					<button
						type="button"
						onclick={resetDefaultScaling}
						class="text-[9px] text-[var(--accent-primary)] hover:underline"
					>
						Reset Default
					</button>
				{/if}
			</div>
		</div>
	</ParameterSection>

	<!-- Parameter Section 3: Thermal Maturity & Background TOC -->
	<ParameterSection title="3. Thermal Maturity & Background TOC" defaultOpen={true}>
		<div class="flex flex-col gap-1.5 p-2">
			<!-- Maturity Mode Toggle -->
			<div class="grid grid-cols-[1fr_130px] items-center gap-x-2">
				<label for="toc-maturity-mode" class="text-xs font-medium text-[var(--text-secondary)]">Maturity Scale</label>
				<select
					id="toc-maturity-mode"
					bind:value={tp.maturityMode}
					class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-hidden"
				>
					<option value="lom">LOM (Hood Scale)</option>
					<option value="ro">Vitrinite Reflectance (% Ro)</option>
				</select>
			</div>

			{#if tp.maturityMode === 'lom'}
				<NumericScrubber
					label="LOM (Maturity)"
					value={tp.lom}
					unit=""
					min={6.0}
					max={14.0}
					step={0.1}
					decimals={1}
					hint="Level of Organic Metamorphism (Hood et al. 1975)"
					onchange={handleLomChange}
					onrun={() => executeToc(false)}
				/>
				<div class="text-[9.5px] text-[var(--text-muted)] italic">
					Equivalent Vitrinite Reflectance: <strong>{tp.ro}% Ro</strong>
				</div>
			{:else}
				<NumericScrubber
					label="Vitrinite Reflectance"
					value={tp.ro}
					unit="% Ro"
					min={0.4}
					max={2.5}
					step={0.02}
					decimals={2}
					hint="Mean random vitrinite reflectance Ro"
					onchange={handleRoChange}
					onrun={() => executeToc(false)}
				/>
				<div class="text-[9.5px] text-[var(--text-muted)] italic">
					Converted Hood Maturity: <strong>LOM {tp.lom}</strong>
				</div>
			{/if}

			<NumericScrubber
				label="TOC_baseline"
				bind:value={tp.tocBaseline}
				unit="wt%"
				min={0.0}
				max={3.0}
				step={0.1}
				decimals={2}
				hint="Background organic carbon content in non-source shale"
				onrun={() => executeToc(false)}
			/>

			<label class="mt-1 flex items-center gap-2 text-xs text-[var(--text-secondary)] cursor-pointer">
				<input
					type="checkbox"
					bind:checked={tp.clampNegative}
					class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
				/>
				<span>Clamp TOC at baseline when ΔlogR &lt; 0</span>
			</label>
		</div>
	</ParameterSection>

	<!-- Parameter Section 4: Source Rock Summary & Calibration -->
	<ParameterSection title="4. Source Rock Statistics & Calibration" defaultOpen={true}>
		<div class="flex flex-col gap-2 p-2">
			{#if tocSummary}
				<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2 text-xs">
					<div class="grid grid-cols-2 gap-y-1 text-[11px]">
						<div>Mean TOC: <strong>{tocSummary.mean} wt%</strong></div>
						<div>Peak TOC: <strong>{tocSummary.max} wt%</strong></div>
						<div>Median (P50): <strong>{tocSummary.p50} wt%</strong></div>
						<div>Samples: <strong>{tocSummary.count}</strong></div>
					</div>
					<div class="mt-2 border-t border-[var(--border-hairline)] pt-1.5 text-[10.5px]">
						<div>Rich Source (&ge; 2.0 wt%): <strong>{tocSummary.netRichPct}%</strong> of interval</div>
						<div>Potential Source (&ge; 1.0 wt%): <strong>{tocSummary.netGoodPct}%</strong> of interval</div>
					</div>
				</div>
			{:else}
				<div class="text-center py-2 text-[10.5px] text-[var(--text-muted)] italic">
					Run calculation to view organic richness and interval statistics
				</div>
			{/if}

			{#if calibrationResult}
				<div class="rounded-xs border border-blue-200 bg-blue-50/70 p-2 text-[10.5px] text-blue-900">
					<div class="font-bold">Core TOC Calibration Fit</div>
					<div>R²: <strong>{calibrationResult.r2}</strong> (RMSE: {calibrationResult.rmse} wt%)</div>
					<div>Implied LOM: <strong>{calibrationResult.implied_lom ?? '—'}</strong> ({calibrationResult.sample_count} core plugs)</div>
					{#if calibrationResult.implied_lom}
						<button
							type="button"
							onclick={() => {
								if (calibrationResult?.implied_lom) {
									handleLomChange(calibrationResult.implied_lom);
									executeToc(false);
								}
							}}
							class="mt-1.5 rounded-xs bg-blue-600 px-2 py-0.5 text-[10px] font-semibold text-white hover:bg-blue-700"
						>
							Adopt Calibrated LOM ({calibrationResult.implied_lom})
						</button>
					{/if}
				</div>
			{/if}
		</div>
	</ParameterSection>

	<!-- Footer Action Buttons -->
	<div class="mt-auto border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5">
		<div class="flex items-center gap-2">
			<button
				type="button"
				onclick={() => executeToc(false)}
				disabled={isRunning}
				class="flex h-7 flex-1 items-center justify-center rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
			>
				{isRunning ? 'Calculating...' : 'Run TOC (Passey)'}
			</button>

			<button
				type="button"
				onclick={saveToc}
				disabled={!wellData.hasPendingComputed('source_rock') || wellData.isSaving}
				class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
			>
				{wellData.isSaving ? 'Saving...' : 'Save'}
			</button>
		</div>
	</div>
</div>
