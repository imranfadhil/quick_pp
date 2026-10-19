<script lang="ts">
	import { postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { downloadCsv, fmtNum } from '$lib/utils/formatters';

	let vshMax = $state(0.4);
	let phitMin = $state(0.05);
	let swtMax = $state(0.75);

	// Upscaling & Log Blocking state
	let includeUpscaling = $state(true);
	let byZone = $state(false);
	let cellSize = $state(2.0);
	let permMethod = $state<'geometric' | 'arithmetic' | 'harmonic' | 'power_law'>('geometric');
	let omega = $state(0.33);
	let upscaledCellCount = $state<number | null>(null);
	let blockedCells = $state<Record<string, any>[]>([]);
	let permBounds = $state<{
		k_harm: number;
		k_geom: number;
		k_power: number;
		k_arith: number;
		kv_kh_ratio: number;
	} | null>(null);

	// Log-level & Zonal Monte Carlo Uncertainty state
	let includeUncertainty = $state(false);
	let nSamples = $state(300);
	let zoneUncertaintySummary = $state<Record<string, any>[]>([]);

	// Volumetric LHS STOIIP Monte Carlo state
	let areaMean = $state(640.0);
	let areaStd = $state(80.0);
	let thickMean = $state(25.0);
	let thickStd = $state(4.0);
	let ntgMean = $state(0.65);
	let poroMean = $state(0.18);
	let swMean = $state(0.28);
	let boP50 = $state(1.22);
	let stoiipMcResult = $state<{
		p10: number;
		p50: number;
		p90: number;
		mean: number;
		std: number;
	} | null>(null);

	async function executeResSum(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		wellData.isCalculating = true;
		wellData.lastError = null;
		try {
			const data = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/wells/${encodeURIComponent(workbench.wellId)}/reservoir_summary`,
				{
					cut_offs: {
						VSHALE: vshMax,
						PHIT: phitMin,
						SWT: swtMax
					},
					include_upscaling: includeUpscaling,
					cell_size: cellSize,
					by_zone: byZone,
					perm_method: permMethod,
					omega,
					include_uncertainty: includeUncertainty,
					n_samples: nSamples,
					min_depth: workbench.depthFilter?.min ?? null,
					max_depth: workbench.depthFilter?.max ?? null,
					zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
					save: persist
				}
			);

			if (data.flags?.data) {
				wellData.mergeComputedCurves(data.flags.data);
			}
			if (data.upscaling?.step_log?.data) {
				wellData.mergeComputedCurves(data.upscaling.step_log.data);
			}
			if (data.uncertainty?.cpi?.data) {
				wellData.mergeComputedCurves(data.uncertainty.cpi.data);
			}
			wellData.zonalSummary = data.summary || [];
			upscaledCellCount = data.upscaling?.n_cells ?? null;
			blockedCells = data.upscaling?.blocked_cells ?? [];
			zoneUncertaintySummary = data.uncertainty?.summary ?? [];

			// Also compute Cardwell-Parsons permeability bounds if PERM curve exists
			const permArr = (wellData.curves['PERM'] ?? []).filter(
				(v) => Number.isFinite(v) && v > 0
			);
			if (permArr.length > 1) {
				const boundsRes = await postJson<any>('/quick_pp/upscaling/permeability_bounds', {
					perm: permArr,
					omega
				});
				permBounds = boundsRes;
			}

			if (persist) {
				wellData.lastSavedMessage = 'Pay Flags Saved ✓';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Reservoir summary error';
		} finally {
			wellData.isCalculating = false;
		}
	}

	async function runVolumetricMc() {
		wellData.isCalculating = true;
		wellData.lastError = null;
		try {
			const netThickMeanFt = thickMean * ntgMean * 3.28084;
			const netThickStdFt = thickStd * ntgMean * 3.28084;
			const res = await postJson<any>('/quick_pp/uncertainty/volumetric', {
				area_bound: [
					Math.max(10, areaMean - 2.5 * areaStd),
					areaMean + 2.5 * areaStd,
					areaMean,
					areaStd
				],
				thickness_bound: [
					Math.max(1, netThickMeanFt - 2.5 * netThickStdFt),
					netThickMeanFt + 2.5 * netThickStdFt,
					netThickMeanFt,
					netThickStdFt
				],
				porosity_bound: [
					Math.max(0.02, poroMean - 0.08),
					Math.min(0.4, poroMean + 0.08),
					poroMean,
					0.03
				],
				water_saturation_bound: [
					Math.max(0.05, swMean - 0.15),
					Math.min(0.95, swMean + 0.15),
					swMean,
					0.06
				],
				volume_factor_bound: [Math.max(1.01, boP50 - 0.08), boP50 + 0.12],
				n_try: 5000,
				seed: 42
			});
			const pct = res?.percentiles ?? {};
			const scale = (res?.mean ?? 0) > 1e4 ? 1e-6 : 1.0;
			stoiipMcResult = {
				p10: (pct.P90 ?? 0) * scale,
				p50: (pct.P50 ?? 0) * scale,
				p90: (pct.P10 ?? 0) * scale,
				mean: (res?.mean ?? 0) * scale,
				std: (res?.std ?? 0) * scale
			};
		} catch (e: any) {
			wellData.lastError = e?.message || 'Volumetric Monte Carlo error';
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

	function exportBlockedCsv() {
		if (!blockedCells.length) return;
		const headers = Object.keys(blockedCells[0]);
		downloadCsv(
			`${workbench.wellId || 'well'}_upscaled_grid.csv`,
			headers,
			blockedCells
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
			title="Log Blocking & Geomodeling Upscaling"
			badge={upscaledCellCount ? `${upscaledCellCount} cells` : undefined}
		>
			<div class="flex items-center justify-between py-0.5 text-xs">
				<label class="flex items-center gap-1.5 font-medium cursor-pointer">
					<input
						type="checkbox"
						bind:checked={includeUpscaling}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Block Logs to Grid</span>
				</label>

				{#if includeUpscaling}
					<label class="flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)] cursor-pointer">
						<input
							type="checkbox"
							bind:checked={byZone}
							onchange={() => executeResSum(false)}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
						/>
						<span>By Zone</span>
					</label>
				{/if}
			</div>

			{#if includeUpscaling}
				{#if !byZone}
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

				<div class="mt-1 flex items-center justify-between text-[11px]">
					<span class="text-[var(--text-secondary)]">Perm Averaging:</span>
					<select
						bind:value={permMethod}
						onchange={() => executeResSum(false)}
						class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px] font-medium"
					>
						<option value="geometric">Geometric (2D Iso)</option>
						<option value="arithmetic">Arithmetic (kH Max)</option>
						<option value="harmonic">Harmonic (kV Min)</option>
						<option value="power_law">Cardwell-Parsons (ω)</option>
					</select>
				</div>

				{#if permMethod === 'power_law'}
					<NumericScrubber
						label="Power-Law Exponent (ω)"
						bind:value={omega}
						min={-1.0}
						max={1.0}
						step={0.05}
						onrun={() => executeResSum(false)}
					/>
				{/if}

				{#if permBounds}
					<div class="mt-1.5 grid grid-cols-3 gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]">
						<div>
							<span class="text-[var(--text-muted)]">kH / kV (mD)</span>
							<div class="mono-num font-semibold">
								{fmtNum(permBounds.k_arith, 1)} / {fmtNum(permBounds.k_harm, 1)}
							</div>
						</div>
						<div>
							<span class="text-[var(--text-muted)]">kGeom (mD)</span>
							<div class="mono-num font-semibold">{fmtNum(permBounds.k_geom, 1)}</div>
						</div>
						<div>
							<span class="text-[var(--text-muted)]">kV/kH Ratio</span>
							<div class="mono-num font-semibold">{fmtNum(permBounds.kv_kh_ratio, 3)}</div>
						</div>
					</div>
				{/if}

				{#if blockedCells.length > 0}
					<button
						type="button"
						onclick={exportBlockedCsv}
						class="mt-1.5 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[11px] font-semibold text-[var(--text-primary)] hover:bg-slate-200"
					>
						Export Blocked Grid ({blockedCells.length} cells .csv)
					</button>
				{/if}
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Probabilistic Uncertainty (Monte Carlo LHS)"
			badge={includeUncertainty ? `N=${nSamples}` : 'P10/P50/P90'}
		>
			<label class="flex items-center gap-2 py-0.5 text-xs font-medium cursor-pointer">
				<input
					type="checkbox"
					bind:checked={includeUncertainty}
					onchange={() => executeResSum(false)}
					class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
				/>
				<span>Propagate Log CPI Uncertainty (P10/P50/P90)</span>
			</label>

			{#if includeUncertainty}
				<NumericScrubber
					label="LHS Realizations per Depth"
					bind:value={nSamples}
					min={50}
					max={1000}
					step={50}
					onrun={() => executeResSum(false)}
				/>
				{#if zoneUncertaintySummary.length > 0}
					<div class="mt-1 overflow-x-auto border border-[var(--border-hairline)]">
						<table class="w-full border-collapse text-left text-[10px]">
							<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
								<tr class="h-5 border-b border-[var(--border-hairline)]">
									<th class="px-1 font-semibold">Zone</th>
									<th class="px-1 text-right font-semibold">1P (P90)</th>
									<th class="px-1 text-right font-semibold">2P (P50)</th>
									<th class="px-1 text-right font-semibold">3P (P10)</th>
								</tr>
							</thead>
							<tbody class="divide-y divide-[var(--border-hairline)]">
								{#each zoneUncertaintySummary as z}
									<tr class="h-5">
										<td class="px-1 font-medium">{z.ZONE}</td>
										<td class="mono-num px-1 text-right">{fmtNum(z.STOIIP_MMBBL_P90, 2)}</td>
										<td class="mono-num px-1 text-right font-semibold">{fmtNum(z.STOIIP_MMBBL_P50, 2)}</td>
										<td class="mono-num px-1 text-right">{fmtNum(z.STOIIP_MMBBL_P10, 2)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{/if}
			{/if}

			<div class="mt-2 border-t border-[var(--border-hairline)] pt-1.5">
				<div class="text-[10px] font-semibold uppercase text-[var(--text-muted)]">
					Prospect Volumetric STOIIP (5,000 LHS)
				</div>
				<div class="grid grid-cols-2 gap-x-2">
					<NumericScrubber
						label="Area Mean"
						bind:value={areaMean}
						unit="ac"
						min={40}
						max={10000}
						step={40}
						onrun={runVolumetricMc}
					/>
					<NumericScrubber
						label="Area StdDev"
						bind:value={areaStd}
						unit="ac"
						min={5}
						max={2000}
						step={10}
						onrun={runVolumetricMc}
					/>
					<NumericScrubber
						label="Net Pay h"
						bind:value={thickMean}
						unit="m"
						min={1}
						max={250}
						step={1}
						onrun={runVolumetricMc}
					/>
					<NumericScrubber
						label="NTG Mode"
						bind:value={ntgMean}
						unit="v/v"
						min={0.1}
						max={1.0}
						step={0.02}
						onrun={runVolumetricMc}
					/>
					<NumericScrubber
						label="Porosity Φ"
						bind:value={poroMean}
						unit="v/v"
						min={0.04}
						max={0.35}
						step={0.01}
						onrun={runVolumetricMc}
					/>
					<NumericScrubber
						label="Water Sat Sw"
						bind:value={swMean}
						unit="v/v"
						min={0.05}
						max={0.85}
						step={0.02}
						onrun={runVolumetricMc}
					/>
				</div>
				<button
					type="button"
					onclick={runVolumetricMc}
					class="mt-1 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[11px] font-semibold text-[var(--text-primary)] hover:bg-slate-200"
				>
					Run STOIIP Monte Carlo (5,000 LHS)
				</button>
				{#if stoiipMcResult}
					<div class="mt-1.5 grid grid-cols-3 gap-1 rounded-sm border border-blue-200 bg-blue-50/70 p-1.5 text-center text-[10px]">
						<div>
							<div class="text-[var(--text-muted)]">1P (P90)</div>
							<div class="mono-num font-bold text-slate-800">{fmtNum(stoiipMcResult.p90, 2)} MMbbl</div>
						</div>
						<div>
							<div class="text-[var(--text-muted)]">2P (P50)</div>
							<div class="mono-num font-bold text-blue-800">{fmtNum(stoiipMcResult.p50, 2)} MMbbl</div>
						</div>
						<div>
							<div class="text-[var(--text-muted)]">3P (P10)</div>
							<div class="mono-num font-bold text-emerald-800">{fmtNum(stoiipMcResult.p10, 2)} MMbbl</div>
						</div>
					</div>
				{/if}
			</div>
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


