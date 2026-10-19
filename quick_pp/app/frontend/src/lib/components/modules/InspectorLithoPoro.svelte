<script lang="ts">
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	const AVAILABLE_MINERALS = ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE', 'HALITE'];

	function toggleMineral(m: string) {
		if (workbench.lithoParams.minerals.includes(m)) {
			workbench.lithoParams.minerals = workbench.lithoParams.minerals.filter((x) => x !== m);
		} else {
			workbench.lithoParams.minerals = [...workbench.lithoParams.minerals, m];
		}
	}

	async function executeLithoPoro(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		wellData.isCalculating = true;
		wellData.lastError = null;
		const lp = workbench.lithoParams;
		const filterArgs = {
			min_depth: workbench.depthFilter?.min ?? null,
			max_depth: workbench.depthFilter?.max ?? null,
			zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
			save: persist
		};

		try {
			// 1. Optional Hydrocarbon Correction
			if (lp.enableHcCorrection) {
				const { data: hcData, error: hcErr } = await api.POST(
					'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/hc_correction',
					{
						params: {
							path: { project_id: workbench.projectId, well_name: workbench.wellId }
						},
						body: {
							dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
							dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
							water_point: [lp.fluidNphi, lp.fluidRhob],
							rho_water: lp.rhoWater,
							rho_hc: lp.rhoHc,
							hi_hc: lp.hiHc,
							buffer: lp.hcBuffer,
							...filterArgs
						}
					}
				);
				if (hcErr) throw new Error(String((hcErr as any)?.detail ?? 'HC Correction failed'));
				if (hcData) {
					wellData.mergeComputedCurves(hcData.data, hcData.stats);
				}
			}

			// 2. Lithology Estimation
			const { data: lithData, error: lithErr } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/lithology',
				{
					params: {
						path: { project_id: workbench.projectId, well_name: workbench.wellId }
					},
					body: {
						model: lp.model,
						use_hc_corrected: lp.enableHcCorrection && lp.useHcCorrected,
						dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
						dry_silt_point: [lp.drySiltNphi, lp.drySiltRhob],
						dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
						fluid_point: [lp.fluidNphi, lp.fluidRhob],
						silt_line_angle: lp.siltLineAngle,
						minerals: lp.minerals,
						...filterArgs
					}
				}
			);
			if (lithErr) throw new Error(String((lithErr as any)?.detail ?? 'Lithology failed'));
			if (lithData) {
				wellData.mergeComputedCurves(lithData.data, lithData.stats);
			}

			// 3. Porosity Estimation
			const { data: poroData, error: poroErr } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/porosity',
				{
					params: {
						path: { project_id: workbench.projectId, well_name: workbench.wellId }
					},
					body: {
						method: lp.porosityMethod,
						use_hc_corrected: lp.enableHcCorrection && lp.useHcCorrected,
						dry_sand_point: [lp.drySandNphi, lp.drySandRhob],
						dry_silt_point: [lp.drySiltNphi, lp.drySiltRhob],
						dry_clay_point: [lp.dryClayNphi, lp.dryClayRhob],
						fluid_point: [lp.fluidNphi, lp.fluidRhob],
						silt_line_angle: lp.siltLineAngle,
						...filterArgs
					}
				}
			);
			if (poroErr) throw new Error(String((poroErr as any)?.detail ?? 'Porosity failed'));
			if (poroData) {
				wellData.mergeComputedCurves(poroData.data, poroData.stats);
				wellData.poroCoreMetrics = poroData.core_metrics;
			}

			if (persist) {
				wellData.lastSavedMessage = 'Litho & Poro Saved ✓';
			}
		} catch (e: any) {
			wellData.lastError = e?.message || 'Lithology/Porosity error';
		} finally {
			wellData.isCalculating = false;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection title="Lithology & Porosity Model" badge={workbench.lithoParams.model.toUpperCase()}>
			<div class="grid grid-cols-[1fr_118px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Lithology Model</span>
				<select
					bind:value={workbench.lithoParams.model}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="ssc">Sand-Silt-Clay</option>
					<option value="multi_mineral">Multi-Mineral</option>
					<option value="vsh_gr">Vsh from GR</option>
				</select>
			</div>

			<div class="grid grid-cols-[1fr_118px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Porosity Method</span>
				<select
					bind:value={workbench.lithoParams.porosityMethod}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="neu_den">NPHI-RHOB Xplot</option>
					<option value="den">Density Only</option>
				</select>
			</div>
		</ParameterSection>

		<ParameterSection title="Endpoint Parameters (NPHI / RHOB)" badge="Linked to Crossplot">
			{#if workbench.lithoParams.model === 'multi_mineral'}
				<div class="mb-1.5 text-[11px] font-medium text-[var(--text-secondary)]">
					Active Minerals:
				</div>
				<div class="mb-2 grid grid-cols-2 gap-1">
					{#each AVAILABLE_MINERALS as minName}
						<label class="flex items-center gap-1.5 text-xs cursor-pointer">
							<input
								type="checkbox"
								checked={workbench.lithoParams.minerals.includes(minName)}
								onchange={() => toggleMineral(minName)}
								class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
							/>
							<span>{minName}</span>
						</label>
					{/each}
				</div>
			{/if}

			<NumericScrubber
				label="NPHI Clean Sand"
				bind:value={workbench.lithoParams.drySandNphi}
				unit="v/v"
				min={-0.1}
				max={0.3}
				step={0.005}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="RHOB Clean Sand"
				bind:value={workbench.lithoParams.drySandRhob}
				unit="g/cc"
				min={2.4}
				max={2.9}
				step={0.01}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="NPHI Dry Silt"
				bind:value={workbench.lithoParams.drySiltNphi}
				unit="v/v"
				min={-0.05}
				max={0.45}
				step={0.005}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="RHOB Dry Silt"
				bind:value={workbench.lithoParams.drySiltRhob}
				unit="g/cc"
				min={2.4}
				max={2.9}
				step={0.01}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="NPHI Dry Clay"
				bind:value={workbench.lithoParams.dryClayNphi}
				unit="v/v"
				min={0.1}
				max={0.65}
				step={0.005}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="RHOB Dry Clay"
				bind:value={workbench.lithoParams.dryClayRhob}
				unit="g/cc"
				min={2.4}
				max={2.95}
				step={0.01}
				onrun={() => executeLithoPoro(false)}
			/>
			<NumericScrubber
				label="Silt Line Angle"
				bind:value={workbench.lithoParams.siltLineAngle}
				unit="deg"
				min={95}
				max={160}
				step={0.5}
				onrun={() => executeLithoPoro(false)}
			/>
		</ParameterSection>

		<ParameterSection title="Hydrocarbon Correction">
			<label class="flex items-center gap-2 py-0.5 text-xs font-medium cursor-pointer">
				<input
					type="checkbox"
					bind:checked={workbench.lithoParams.enableHcCorrection}
					class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
				/>
				<span>Enable NPHI/RHOB HC Correction</span>
			</label>

			{#if workbench.lithoParams.enableHcCorrection}
				<NumericScrubber
					label="Water Density"
					bind:value={workbench.lithoParams.rhoWater}
					unit="g/cc"
					min={0.9}
					max={1.25}
					step={0.01}
					onrun={() => executeLithoPoro(false)}
				/>
				<NumericScrubber
					label="HC Fluid Density"
					bind:value={workbench.lithoParams.rhoHc}
					unit="g/cc"
					min={0.1}
					max={0.95}
					step={0.01}
					onrun={() => executeLithoPoro(false)}
				/>
				<NumericScrubber
					label="HC Hydrogen Index"
					bind:value={workbench.lithoParams.hiHc}
					unit="v/v"
					min={0.1}
					max={1.1}
					step={0.02}
					onrun={() => executeLithoPoro(false)}
				/>
			{/if}
		</ParameterSection>

		{#if wellData.poroCoreMetrics && wellData.poroCoreMetrics.count > 0}
			<ParameterSection title="Core Calibration QC" badge={`n=${wellData.poroCoreMetrics.count}`}>
				<div class="grid grid-cols-3 gap-1.5 text-center">
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[10px] text-[var(--text-muted)]">RMSE</div>
						<div class="mono-num text-xs font-bold text-[var(--text-primary)]">
							{fmtNum(wellData.poroCoreMetrics.rmse, 4)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[10px] text-[var(--text-muted)]">MAE</div>
						<div class="mono-num text-xs font-bold text-[var(--text-primary)]">
							{fmtNum(wellData.poroCoreMetrics.mae, 4)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[10px] text-[var(--text-muted)]">R²</div>
						<div class="mono-num text-xs font-bold text-[var(--status-ok)]">
							{fmtNum(wellData.poroCoreMetrics.r2, 3)}
						</div>
					</div>
				</div>
			</ParameterSection>
		{/if}

		{#if wellData.lastError}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{wellData.lastError}
			</div>
		{/if}
	</div>

	<!-- Execution Bar -->
	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<button
			type="button"
			disabled={wellData.isCalculating}
			onclick={() => executeLithoPoro(false)}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{wellData.isCalculating ? 'Running...' : 'Run (⌘+Enter)'}
		</button>
		<button
			type="button"
			disabled={wellData.isCalculating}
			onclick={() => executeLithoPoro(true)}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
		>
			Save
		</button>
	</div>
</div>

