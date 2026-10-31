<script lang="ts">
	import { onMount } from 'svelte';
	import { postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { projectParams } from '$lib/state/projectParams.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	// Multi-well FWL & Contact Optimizer state
	// Optimizer settings live in shared project state so they are saved/restored per zone selection
	const so = projectData.shfOptParams;
	let fwlOptResult = $state<any>(null);

	let calculatedZones = $derived.by(() => {
		const sr = projectData.shfResult;
		const allowedZones =
			workbench.selectedZones.length > 0 ? new Set(workbench.selectedZones) : null;
		const minD = workbench.depthFilter?.min ?? -Infinity;
		const maxD = workbench.depthFilter?.max ?? Infinity;
		const found = new Set<string>();

		if (sr?.zones && sr?.shf) {
			for (let i = 0; i < sr.zones.length; i++) {
				const zn = sr.zones[i]?.trim();
				if (!zn || zn === 'Unknown' || zn === 'None' || zn === 'nan') continue;
				if (allowedZones && !allowedZones.has(zn)) continue;
				const d = sr.depths?.[i];
				if (typeof d === 'number' && (d < minD || d > maxD)) continue;
				const shfVal = sr.shf[i];
				if (typeof shfVal === 'number' && Number.isFinite(shfVal)) {
					found.add(zn);
				}
			}
		}

		for (const zn of Object.keys(projectData.shfParams.fwlByZone)) {
			if (!allowedZones || allowedZones.has(zn)) {
				found.add(zn);
			}
		}

		return Array.from(found).sort();
	});

	const MODEL_META: Record<
		'leverett_j' | 'skelt_harrison' | 'cuddy',
		{ label: string; badge: string; fitTitle: string; formula: string }
	> = {
		leverett_j: {
			label: 'Leverett-J',
			badge: 'J-Curve Fit',
			fitTitle: 'SCAL Leverett-J Curve Fit',
			formula: 'J = a · Swn^(-b)'
		},
		skelt_harrison: {
			label: 'Skelt-Harrison',
			badge: 'Pc Curve Fit',
			fitTitle: 'SCAL Skelt-Harrison Pc Fit',
			formula: 'Sw = 1 - a·exp(-(b/(h+d))^c)'
		},
		cuddy: {
			label: 'Cuddy Bvw',
			badge: 'BVW Fit',
			fitTitle: 'SCAL Cuddy BVW Fit',
			formula: 'BVW = φ·Sw = a · Pc^(-b)'
		}
	};

	const ROCK_COLORS = ['#2563eb', '#16a34a', '#d97706', '#dc2626', '#9333ea', '#0891b2'];

	onMount(() => {
		// Initialize sensible default FWL (TVDSS) based on current well TVDSS range or contact if still at default 2500m
		if (wellData.depthRange.max > 0 && projectData.shfParams.fwl === 2500.0) {
			const tvdssArr = (wellData.curves['TVDSS'] ?? wellData.curves['TVD'] ?? []).filter(
				(v): v is number => typeof v === 'number' && Number.isFinite(v)
			);
			const contactFwl = wellData.contacts.find((c) => c.type === 'FWL' || c.type === 'OWC');
			if (contactFwl) {
				const deptArr = wellData.deptCurve;
				const rawTvdss = wellData.curves['TVDSS'] ?? wellData.curves['TVD'];
				let contactTvdss = contactFwl.depth;
				if (rawTvdss && deptArr.length === rawTvdss.length && deptArr.length > 0) {
					let bestIdx = -1;
					let bestDist = Infinity;
					for (let i = 0; i < deptArr.length; i++) {
						const d = deptArr[i];
						const t = rawTvdss[i];
						if (typeof d === 'number' && typeof t === 'number' && Number.isFinite(t)) {
							const dist = Math.abs(d - contactFwl.depth);
							if (dist < bestDist) {
								bestDist = dist;
								bestIdx = i;
							}
						}
					}
					if (bestIdx >= 0) {
						contactTvdss = Number(
							(rawTvdss[bestIdx] + (contactFwl.depth - deptArr[bestIdx])).toFixed(1)
						);
					}
				}
				projectData.shfParams.fwl = contactTvdss;
			} else if (tvdssArr.length >= 2) {
				const minTvdss = Math.min(...tvdssArr);
				const maxTvdss = Math.max(...tvdssArr);
				projectData.shfParams.fwl = Number(
					(minTvdss + 0.72 * (maxTvdss - minTvdss)).toFixed(1)
				);
			} else {
				projectData.shfParams.fwl = Number(
					(
						wellData.depthRange.min +
						0.72 * (wellData.depthRange.max - wellData.depthRange.min)
					).toFixed(1)
				);
			}
		}
		if (
			!projectParams.status['shf']?.matched &&
			(projectData.swirrResult ||
				projectData.swirrApplyResult ||
				Object.keys(projectData.swirrConfig.rules ?? {}).length > 0)
		) {
			projectData.shfParams.swirrSource = 'poro_perm';
		}
		if (!projectData.shfResult && projectData.swirrResult) {
			projectData.recomputeScalFitsFromPlugs(false);
		}
	});

	function selectShfModel(m: 'leverett_j' | 'skelt_harrison' | 'cuddy') {
		if (projectData.shfParams.model === m && projectData.shfResult?.model === m) return;
		projectData.shfParams.model = m;
		if (projectData.swirrResult && !projectData.shfResult?.shf?.length) {
			projectData.recomputeScalFitsFromPlugs(false);
		}
	}

	function updateZoneFwl(zone: string, rawVal: string) {
		const num = Number(rawVal);
		if (!Number.isFinite(num)) return;
		projectData.shfParams.fwlByZone = {
			...projectData.shfParams.fwlByZone,
			[zone]: Number(num.toFixed(2))
		};
	}

	function clearZoneFwls() {
		projectData.shfParams.fwlByZone = {};
	}

	async function executeShf(resetFits: boolean = false, markDirty: boolean = true) {
		if (!workbench.projectId) return;
		const validTvdWells = projectData.wellsWithTvd.map((w) => w.name);
		if (validTvdWells.length === 0) {
			projectData.error =
				'TVD is required for Saturation Height Function (SHF). None of the selected wells have a TVD curve or directional survey (TVD is never assumed equal to MD).';
			return;
		}
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const sp = projectData.shfParams;
			const jSatMode = sp.jSatMode ?? 'swn';
			// Restored SHF fits (from saved project params) are used once on the next non-reset run
			const saved = projectData.shfSavedFits;
			projectData.shfSavedFits = null;
			const restoredFits =
				!resetFits &&
				saved &&
				saved.model === sp.model &&
				(sp.model !== 'leverett_j' || (saved.j_sat_mode ?? 'swn') === jSatMode) &&
				Object.keys(saved.fits ?? {}).length > 0
					? saved.fits
					: null;
			const customFits =
				restoredFits ??
				(!resetFits &&
				projectData.shfResult?.model === sp.model &&
				(sp.model !== 'leverett_j' || (projectData.shfResult?.j_sat_mode ?? 'swn') === jSatMode) &&
				projectData.shfResult?.fits &&
				Object.keys(projectData.shfResult.fits).length > 0
					? projectData.shfResult.fits
					: null);
			const fwlByZonePayload =
				sp.fwlByZone && Object.keys(sp.fwlByZone).length > 0 ? sp.fwlByZone : null;
			const data = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/shf/compute`,
				{
					well_names: validTvdWells,
					model: sp.model,
					j_sat_mode: jSatMode,
					unit: sp.unit ?? 'metric',
					drt_method: projectData.drtMethod,
					cutoffs: [...projectData.activeDrtCutoffs].sort((a, b) => a - b),
					fwl: sp.fwl,
					fwl_by_zone: fwlByZonePayload,
					ift: sp.ift,
					theta: sp.theta,
					gw: sp.gw,
					ghc: sp.ghc,
					swirr_source: sp.swirrSource ?? 'lab_min',
					swirr_config: projectData.swirrConfig,
					fits: customFits,
					save: false
				}
			);
			projectData.shfResult = {
				model: data.model ?? sp.model,
				j_sat_mode: data.j_sat_mode ?? jSatMode,
				fits: data.fits ?? {},
				fits_swn: data.fits_swn ?? (jSatMode === 'swn' ? data.fits : undefined),
				fits_sw: data.fits_sw ?? (jSatMode === 'sw' ? data.fits : undefined),
				fwl_by_zone: data.fwl_by_zone ?? sp.fwlByZone,
				scal_data: data.scal_data,
				...data.data
			};
			if (markDirty) {
				projectData.markMultiWellComputed('shf');
			}

			// Live-merge SHF curve into active well so LogTrackCanvas Track 6 updates in real time!
			if (workbench.wellId && data.data?.well_names) {
				const wnArr = data.data.well_names;
				const dArr = data.data.depths;
				const shfArr = data.data.shf;
				const wDepths: number[] = [];
				const wShf: (number | null)[] = [];
				for (let i = 0; i < wnArr.length; i++) {
					if (wnArr[i] === workbench.wellId) {
						wDepths.push(dArr[i]);
						wShf.push(shfArr[i] ?? null);
					}
				}
				if (wDepths.length > 0) {
					wellData.mergeComputedCurves({
						DEPTH: wDepths,
						SHF: wShf
					});
				}
			}
			void projectData.loadStickPlot(true);
		} catch (e: any) {
			projectData.error = e?.message || 'SHF error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function saveShf() {
		await projectData.saveMultiWellComputed('shf');
	}

	async function runOptimizeFwl() {
		if (!workbench.projectId) return;
		const validTvdWells = projectData.wellsWithTvd.map((w) => w.name);
		if (validTvdWells.length === 0) {
			projectData.error =
				'TVD is required for FWL optimization. None of the selected wells have a TVD curve or directional survey (TVD is never assumed equal to MD).';
			return;
		}
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const sp = projectData.shfParams;
			const activeZonesPayload =
				workbench.selectedZones.length > 0
					? workbench.selectedZones
					: calculatedZones.length > 0
						? calculatedZones
						: null;
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/optimize_fwl`,
				{
					well_names: validTvdWells,
					zones: activeZonesPayload,
					model: sp.model,
					shm_params: projectData.shfResult?.fits ?? null,
					loss: so.optLoss,
					solve_per_zone: so.solvePerZone,
					ift: sp.ift,
					theta: sp.theta,
					gw: sp.gw,
					ghc: sp.ghc,
					mdt_fwl: so.useMdtPrior ? so.mdtFwl : null,
					mdt_weight: so.mdtWeight,
					pore_volume_weighted: so.poreVolumeWeighted,
					equal_well_weights: so.equalWellWeights
				}
			);
			fwlOptResult = res;
			if (typeof res?.fwl === 'number' && Number.isFinite(res.fwl)) {
				projectData.shfParams.fwl = Number(res.fwl.toFixed(2));
			}
			if (
				so.solvePerZone &&
				res?.fwl_by_zone &&
				typeof res.fwl_by_zone === 'object' &&
				Object.keys(res.fwl_by_zone).length > 0
			) {
				projectData.shfParams.fwlByZone = { ...res.fwl_by_zone };
			} else {
				projectData.shfParams.fwlByZone = {};
			}
		} catch (e: any) {
			projectData.error = e?.message || 'FWL optimization error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['shf']} />
		{#if projectData.activeWells.length > 0 && projectData.wellsWithoutTvd.length > 0}
			{#if projectData.wellsWithTvd.length === 0}
				<div
					class="m-2 rounded-sm border border-red-300 bg-red-50 p-2 text-[11px] leading-snug text-red-900"
				>
					<div class="font-bold">TVDSS Required — SHF &amp; Stick Plot Disabled</div>
					<div class="mt-0.5 text-[10.5px] text-red-800">
						None of the {projectData.activeWells.length} selected well(s) have a
						<span class="mono-num font-semibold">TVDSS</span> / <span class="mono-num font-semibold">TVD</span> curve or directional survey
						(&ge; 2 stations). True Vertical Depth Subsea is never assumed equal to Measured Depth (MD).
						Import directional surveys or TVDSS/TVD curves to enable SHF &amp; FWL optimization.
					</div>
				</div>
			{:else}
				<div
					class="m-2 rounded-sm border border-amber-300 bg-amber-50 p-2 text-[10.5px] leading-snug text-amber-900"
				>
					<div class="font-semibold">
						{projectData.wellsWithoutTvd.length} well(s) without TVDSS excluded (TVDSS &ne; MD):
					</div>
					<div class="mono-num mt-0.5 text-[10px] text-amber-800">
						{projectData.wellsWithoutTvd.map((w) => w.name).join(', ')}
					</div>
				</div>
			{/if}
		{/if}

		<ParameterSection
			title="SHF Method & Multi-Well FWL Optimizer"
			badge={fwlOptResult ? `FWL=${fmtNum(fwlOptResult.fwl, 1)}m` : MODEL_META[projectData.shfParams.model].badge}
		>
			<div class="grid grid-cols-3 gap-1">
				{#each [['leverett_j', 'Leverett-J'], ['skelt_harrison', 'Skelt-Harrison'], ['cuddy', 'Cuddy Bvw']] as [m, lbl]}
					<button
						type="button"
						onclick={() => selectShfModel(m as any)}
						class="h-6 rounded-sm border text-[10px] font-medium {projectData.shfParams.model === m
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
					>
						{lbl}
					</button>
				{/each}
			</div>

			<div class="mt-1.5 flex items-center justify-between text-[11px]">
				<span class="text-[var(--text-secondary)]">Loss Objective:</span>
				<select
					bind:value={so.optLoss}
					class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0 pr-6 pl-1.5 text-[11px] font-medium"
				>
					<option value="rmse">RMSE (L2)</option>
					<option value="huber">Huber Robust</option>
					<option value="mae">MAE (L1)</option>
					<option value="bvw">Bulk Volume Water (BVW)</option>
				</select>
			</div>

			<div class="mt-1 flex flex-wrap items-center justify-between gap-y-1 text-[11px]">
				<label class="flex items-center gap-1 cursor-pointer">
					<input
						type="checkbox"
						bind:checked={so.solvePerZone}
						onchange={() => {
							if (!so.solvePerZone && Object.keys(projectData.shfParams.fwlByZone).length > 0) {
								void clearZoneFwls();
							}
						}}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span class="font-semibold text-[var(--text-primary)]">Per-Zone FWL</span>
				</label>
				<label class="flex items-center gap-1 cursor-pointer">
					<input
						type="checkbox"
						bind:checked={so.poreVolumeWeighted}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Φ-Weighted</span>
				</label>
				<label class="flex items-center gap-1 cursor-pointer">
					<input
						type="checkbox"
						bind:checked={so.equalWellWeights}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Equal Well Wt</span>
				</label>
				<label class="flex items-center gap-1 cursor-pointer">
					<input
						type="checkbox"
						bind:checked={so.useMdtPrior}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>MDT Prior</span>
				</label>
			</div>

			{#if so.useMdtPrior}
				<div class="grid grid-cols-2 gap-x-2">
					<NumericScrubber
						label="MDT FWL (TVDSS)"
						bind:value={so.mdtFwl}
						unit="m TVDSS"
						min={0}
						max={10000}
						step={5.0}
					/>
					<NumericScrubber
						label="Prior Weight"
						bind:value={so.mdtWeight}
						min={0.1}
						max={10.0}
						step={0.2}
					/>
				</div>
			{/if}

			<button
				type="button"
				disabled={projectData.isComputingMultiWell || projectData.wellsWithTvd.length === 0}
				onclick={runOptimizeFwl}
				title={projectData.wellsWithTvd.length === 0
					? 'Disabled: No selected wells have TVDSS/TVD or directional survey (TVDSS is never assumed equal to MD)'
					: 'Optimize Free Water Level (FWL) & fluid contacts in TVDSS using wells with valid TVDSS'}
				class="mt-1.5 h-7 w-full rounded-sm border border-blue-600 bg-blue-50 px-2 text-xs font-semibold text-blue-800 hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50"
			>
				{so.solvePerZone ? 'Optimize Per-Zone FWL & Contacts (TVDSS)' : 'Optimize Field FWL & Contacts (TVDSS)'}
			</button>

			{#if fwlOptResult}
				<div class="mt-1.5 grid grid-cols-2 gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]">
					<div>
						<span class="text-[var(--text-muted)]">Field Fallback FWL (TVDSS):</span>
						<div class="mono-num font-bold text-blue-700">{fmtNum(fwlOptResult.fwl, 2)} m TVDSS</div>
					</div>
					<div>
						<span class="text-[var(--text-muted)]">OWC / TTZ (TVDSS):</span>
						<div class="mono-num font-semibold">
							{fmtNum(fwlOptResult.owc, 1)} / {fmtNum(fwlOptResult.ttz, 1)} m
						</div>
					</div>
					<div>
						<span class="text-[var(--text-muted)]">Transition Zone:</span>
						<div class="mono-num font-semibold">
							{fmtNum(fwlOptResult.transition_zone_thickness, 1)} m
						</div>
					</div>
					<div>
						<span class="text-[var(--text-muted)]">Fit RMSE / R²:</span>
						<div class="mono-num font-semibold">
							{fmtNum(fwlOptResult.rmse, 3)} / {fmtNum(fwlOptResult.r2 ?? fwlOptResult.r_squared, 3)}
						</div>
					</div>
				</div>
			{/if}

			{#if (so.solvePerZone || Object.keys(projectData.shfParams.fwlByZone).length > 0) && ((fwlOptResult?.zone_metrics && fwlOptResult.zone_metrics.length > 0) || calculatedZones.length > 0)}
				{@const zoneRows =
					fwlOptResult?.zone_metrics && fwlOptResult.zone_metrics.length > 0
						? fwlOptResult.zone_metrics.filter(
								(zr: any) =>
									calculatedZones.length === 0 || calculatedZones.includes(zr.zone)
							)
						: calculatedZones.map((zn) => ({
								zone: zn,
								fwl: projectData.shfParams.fwlByZone[zn] ?? projectData.shfParams.fwl,
								owc: null,
								ttz: null,
								rmse: null
							}))}
				{#if zoneRows.length > 0}
					<div class="mt-2">
						<div class="mb-1 flex items-center justify-between text-[10px] text-[var(--text-secondary)]">
							<span class="font-semibold text-[var(--text-primary)]">Per-Zone FWL & Contacts (TVDSS)</span>
							{#if Object.keys(projectData.shfParams.fwlByZone).length > 0}
								<button
									type="button"
									onclick={clearZoneFwls}
									class="text-[10px] font-medium text-blue-700 hover:underline"
								>
									Use Single Global FWL
								</button>
							{/if}
						</div>
						<div class="overflow-x-auto border border-[var(--border-hairline)]">
							<table class="w-full border-collapse text-left text-[10px]">
								<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
									<tr class="h-5 border-b border-[var(--border-hairline)]">
										<th class="px-1.5 font-semibold">Zone</th>
										<th class="px-1 text-right font-semibold">FWL (m TVDSS)</th>
										<th class="px-1 text-right font-semibold">OWC / TTZ (TVDSS)</th>
										<th class="px-1.5 text-right font-semibold">RMSE</th>
									</tr>
								</thead>
								<tbody class="divide-y divide-[var(--border-hairline)]">
									{#each zoneRows as zRow}
										{@const activeZfwl =
											projectData.shfParams.fwlByZone[zRow.zone] ??
											zRow.fwl ??
											projectData.shfParams.fwl}
										<tr class="h-6">
											<td class="px-1.5 font-semibold truncate max-w-[90px]" title={zRow.zone}>
												{zRow.zone}
											</td>
											<td class="px-1 text-right">
												<input
													type="number"
													step="1.0"
													value={activeZfwl}
													onchange={(e) =>
														updateZoneFwl(zRow.zone, (e.currentTarget as HTMLInputElement).value)}
													class="mono-num h-5 w-16 rounded-xs border border-[var(--border-hairline)] px-1 text-right text-[10px] font-semibold text-blue-700"
												/>
											</td>
											<td class="mono-num px-1 text-right text-[var(--text-secondary)]">
												{zRow.owc != null ? `${fmtNum(zRow.owc, 0)} / ${fmtNum(zRow.ttz, 0)}` : '—'}
											</td>
											<td class="mono-num px-1.5 text-right">
												{zRow.rmse != null ? fmtNum(zRow.rmse, 3) : '—'}
											</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					</div>
				{/if}
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Capillary Pressure & Fluid Parameters"
			badge={MODEL_META[projectData.shfParams.model].label}
		>
			<NumericScrubber
				label="Free Water Level — FWL (TVDSS)"
				bind:value={projectData.shfParams.fwl}
				unit="m TVDSS"
				min={0}
				max={10000}
				step={1.0}
			/>
			<NumericScrubber
				label="Interfacial Tension (IFT)"
				bind:value={projectData.shfParams.ift}
				unit="dyn/cm"
				min={1.0}
				max={80.0}
				step={0.5}
			/>
			<NumericScrubber
				label="Contact Angle (θ)"
				bind:value={projectData.shfParams.theta}
				unit="deg"
				min={0}
				max={90}
				step={1.0}
			/>
			<NumericScrubber
				label="Water Gradient (gw)"
				bind:value={projectData.shfParams.gw}
				unit="g/cc"
				min={0.9}
				max={1.3}
				step={0.01}
			/>
			<NumericScrubber
				label="HC Gradient (ghc)"
				bind:value={projectData.shfParams.ghc}
				unit="g/cc"
				min={0.1}
				max={0.98}
				step={0.01}
			/>

			{#if projectData.shfParams.model === 'leverett_j' && (projectData.shfParams.jSatMode ?? 'swn') === 'swn'}
				<div class="mt-2 border-t border-[var(--border-hairline)] pt-2">
					<div class="mb-1 flex items-center justify-between text-[10px]">
						<span class="font-semibold text-[var(--text-primary)]">
							Swirr Normalization Method
						</span>
						<span class="mono-num text-[9.5px] text-[var(--text-muted)]">
							{(projectData.shfParams.swirrSource ?? 'lab_min') === 'poro_perm'
								? 'Swn uses modeled Swirr(k, φ)'
								: 'Swn uses plug min(Sw)'}
						</span>
					</div>
					<div class="grid grid-cols-2 gap-1">
						{#each [['lab_min', 'Lab Terminal Sw (Per-Plug)'], ['poro_perm', 'Poro-Perm Correlation (Modeled)']] as [srcKey, srcLabel]}
							<button
								type="button"
								onclick={() =>
									projectData.setShfSwirrSource(srcKey as 'lab_min' | 'poro_perm')}
								class="h-6 rounded-sm border px-1.5 text-[10px] font-medium {(projectData.shfParams
									.swirrSource ?? 'lab_min') === srcKey
									? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
									: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-50'}"
							>
								{srcLabel}
							</button>
						{/each}
					</div>
				</div>
			{/if}
		</ParameterSection>

		{#if true}
			{@const activeModel = projectData.shfResult?.model ?? projectData.shfParams.model}
			{@const meta = MODEL_META[activeModel]}
			{@const isSwMode = (projectData.shfParams.jSatMode ?? 'swn') === 'sw'}
			{@const activeFormula =
				activeModel === 'leverett_j'
					? isSwMode
						? 'J = a · Sw^(-b)'
						: 'J = a · Swn^(-b)'
					: meta.formula}
			{@const fitEntries = Object.entries(projectData.shfResult?.fits ?? {})}
			<ParameterSection title="Resultant SCAL Fit Parameters" badge={activeFormula}>
				<div class="mb-1.5 flex items-center justify-between text-[10px] text-[var(--text-secondary)]">
					<span>
						SCAL points:
						<strong class="mono-num text-[var(--text-primary)]">
							{projectData.shfResult?.scal_data?.pc?.length ?? 0}
						</strong>
					</span>
				</div>

				{#if fitEntries.length > 0}
					<div class="overflow-x-auto border border-[var(--border-hairline)]">
						<table class="w-full border-collapse text-left text-[11px]">
							<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
								<tr class="h-6 border-b border-[var(--border-hairline)]">
									<th class="px-1.5 font-semibold">RT</th>
									<th class="px-1.5 text-right font-semibold">a</th>
									<th class="px-1.5 text-right font-semibold">b</th>
									{#if activeModel === 'skelt_harrison'}
										<th class="px-1.5 text-right font-semibold">c</th>
										<th class="px-1.5 text-right font-semibold">d</th>
									{/if}
									<th class="px-1.5 text-right font-semibold">RMSE</th>
									{#if activeModel !== 'skelt_harrison'}
										<th class="px-1.5 text-right font-semibold">R²</th>
									{/if}
								</tr>
							</thead>
							<tbody class="divide-y divide-[var(--border-hairline)]">
								{#each fitEntries as [rf, fit]}
									{@const rtColor = ROCK_COLORS[(Number(rf) - 1) % ROCK_COLORS.length] ?? '#2563eb'}
									<tr class="h-6">
										<td class="px-1.5 font-semibold whitespace-nowrap">
											<span
												class="mr-1 inline-block h-2 w-2 rounded-full"
												style="background:{rtColor}"
											></span>RT {rf}
										</td>
										<td class="mono-num px-1.5 text-right text-[10px] font-medium text-[var(--text-primary)]">
											{fmtNum(fit.a, 4)}
										</td>
										<td class="mono-num px-1.5 text-right text-[10px] font-medium text-[var(--text-primary)]">
											{fmtNum(fit.b, 3)}
										</td>
										{#if activeModel === 'skelt_harrison'}
											<td class="mono-num px-1.5 text-right text-[10px] font-medium text-[var(--text-primary)]">
												{fmtNum(fit.c ?? 1.5, 2)}
											</td>
											<td class="mono-num px-1.5 text-right text-[10px] font-medium text-[var(--text-primary)]">
												{fmtNum(fit.d ?? 1.5, 2)}
											</td>
										{/if}
										<td class="mono-num px-1.5 text-right text-[10px]">{fmtNum(fit.rmse, 3)}</td>
										{#if activeModel !== 'skelt_harrison'}
											<td class="mono-num px-1.5 text-right text-[10px]">{fmtNum(fit.r2, 3)}</td>
										{/if}
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{:else}
					<div class="rounded-xs border border-dashed border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2 text-[10px] text-[var(--text-secondary)]">
						No SCAL curve fits available yet. Fit Swirr &amp; J/Pc curves in the <strong>[SCAL] SCAL &amp; Rel-Perm</strong> standalone module, or run <strong>Compute SHF</strong> to auto-fit from project SCAL data.
					</div>
				{/if}
			</ParameterSection>
		{/if}

		{#if projectData.error}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{projectData.error}
			</div>
		{/if}
	</div>

	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell || projectData.wellsWithTvd.length === 0}
			onclick={() => executeShf(false, true)}
			title={projectData.wellsWithTvd.length === 0
				? 'Disabled: No selected wells have TVD or directional survey (TVD is never assumed equal to MD)'
				: `Compute SHF across ${projectData.wellsWithTvd.length} well(s) with valid TVD`}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
		>
			{projectData.isComputingMultiWell
				? 'Computing...'
				: projectData.wellsWithTvd.length === 0
					? 'Compute SHF (No TVD)'
					: 'Compute SHF (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell || !projectData.hasPendingMultiWell('shf')}
			onclick={saveShf}
			title={projectData.hasPendingMultiWell('shf')
				? 'Save computed SHF curves across all project wells'
				: 'Compute or adjust SHF parameters first to enable saving'}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
		>
			Save All Wells
		</button>
	</div>
</div>
