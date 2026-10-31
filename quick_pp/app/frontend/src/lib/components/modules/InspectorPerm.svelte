<script lang="ts">
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	// Model parameters live in shared workbench state so they survive module switches and can be
	// restored per zone / depth interval from saved well config.
	const pp = workbench.permParams;

	let hasTsAvailable = $derived.by(() => {
		const ps = wellData.curves['PHIT_SAND'];
		return Boolean(
			workbench.tsParams.enabled || (ps && ps.some((v) => Number.isFinite(v)))
		);
	});

	async function executePerm(persist: boolean = false) {
		if (!workbench.projectId || !workbench.wellId) return;
		wellData.isCalculating = true;
		wellData.lastError = null;
		const deptArr = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
		const phitArr = wellData.curves['PHIT'] ?? wellData.curves['PHIE'] ?? [];
		const vclayArr = wellData.curves['VCLAY'] ?? wellData.curves['VSHALE'] ?? [];
		const vsiltArr = wellData.curves['VSILT'] ?? [];
		const phitSandArr = wellData.curves['PHIT_SAND'] ?? [];
		const vshLamArr = wellData.curves['VSH_LAM'] ?? [];
		const vshDisArr = wellData.curves['VSH_DIS'] ?? [];
		const useTs = hasTsAvailable && workbench.tsParams.useInPerm;
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
						method: pp.method,
						m: pp.m,
						A: pp.A,
						B: pp.B,
						C: pp.C,
						s: pp.s,
						swirr: pp.swirr,
						use_thomas_stieber: useTs,
						depths: cleanArr(deptArr) as number[] | undefined,
						phit: cleanArr(phitArr),
						vclay: cleanArr(vclayArr),
						vsilt: cleanArr(vsiltArr),
						phit_sand: useTs ? cleanArr(phitSandArr) : undefined,
						vsh_lam: useTs ? cleanArr(vshLamArr) : undefined,
						vsh_dis: useTs ? cleanArr(vshDisArr) : undefined,
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
			wellData.markModuleComputed('perm', data.data);
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

	async function savePerm() {
		await wellData.saveModuleComputed('perm', 'Permeability Saved ✓');
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['perm']} />
		<ParameterSection title="Permeability Model" badge={pp.method.toUpperCase()}>
			<div class="grid grid-cols-[1fr_118px] items-center gap-x-2 py-0.5">
				<span class="text-xs font-medium text-[var(--text-primary)]">Empirical Equation</span>
				<select
					bind:value={pp.method}
					class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0 text-xs font-medium text-[var(--text-primary)]"
				>
					<option value="choo">Choo (Sand-Silt-Clay)</option>
					<option value="kozeny_carman">Kozeny-Carman</option>
					<option value="timur">Timur (Swirr)</option>
					<option value="tixier">Tixier</option>
					<option value="coates">Coates-Dumanoir</option>
				</select>
			</div>

			{#if hasTsAvailable}
				<div class="my-1.5 rounded-xs border border-teal-200 bg-teal-50/70 px-2 py-1.5 text-[10.5px]">
					<label class="flex items-center justify-between gap-2 font-medium text-teal-950 cursor-pointer">
						<span class="flex items-center gap-1.5">
							<input
								type="checkbox"
								bind:checked={workbench.tsParams.useInPerm}
								onchange={() => executePerm(false)}
								class="h-3.5 w-3.5 rounded-xs border-teal-400 text-teal-700"
							/>
							<span>Use Thomas-Stieber Thin-Bed Inputs</span>
						</span>
						<span class="rounded-2xs bg-teal-100 px-1 py-0.5 font-mono text-[9px] font-bold text-teal-800">
							{workbench.tsParams.useInPerm ? 'TS Active' : 'Bulk'}
						</span>
					</label>
					<div class="mt-0.5 text-[9.5px] leading-snug text-teal-800">
						{#if workbench.tsParams.useInPerm}
							Evaluates sand lamina <span class="mono-num">k_sand(PHIT_SAND, VSH_DIS)</span> and weights horizontal permeability by <span class="mono-num">(1 − VSH_LAM)</span> while keeping bulk <span class="mono-num">PHIT</span> &amp; <span class="mono-num">VCLAY</span> intact.
						{:else}
							Using standard bulk <span class="mono-num">PHIT</span> and total <span class="mono-num">VCLAY</span>.
						{/if}
					</div>
				</div>
			{/if}

			{#if pp.method === 'choo'}
				<NumericScrubber
					label="Cementation m"
					bind:value={pp.m}
					min={1.0}
					max={4.0}
					step={0.05}
					onrun={() => executePerm(false)}
				/>
				<NumericScrubber
					label="Constant A"
					bind:value={pp.A}
					min={100}
					max={1e8}
					step={10000}
					decimals={0}
					onrun={() => executePerm(false)}
				/>
				<NumericScrubber
					label="Clay Exponent B"
					bind:value={pp.B}
					min={0.5}
					max={8.0}
					step={0.1}
					onrun={() => executePerm(false)}
				/>
				<NumericScrubber
					label="Silt Exponent C"
					bind:value={pp.C}
					min={0.5}
					max={8.0}
					step={0.1}
					onrun={() => executePerm(false)}
				/>
			{:else if pp.method === 'kozeny_carman'}
				<NumericScrubber
					label="Surface Area Factor s"
					bind:value={pp.s}
					min={0.0001}
					max={1.0}
					step={0.001}
					decimals={4}
					onrun={() => executePerm(false)}
				/>
			{:else}
				<NumericScrubber
					label="Irreducible Sw (Swirr)"
					bind:value={pp.swirr}
					unit="v/v"
					min={0.02}
					max={0.8}
					step={0.01}
					onrun={() => executePerm(false)}
				/>
			{/if}
		</ParameterSection>

		{#if wellData.coreComparison.perm.count > 0 || wellData.permCoreMetrics}
			{@const kStat = wellData.coreComparison.perm}
			{@const kCount = kStat.count || wellData.permCoreMetrics?.count || 0}
			<ParameterSection
				title="Core Permeability Calibration (PERM × CPERM)"
				badge={`n=${kCount}`}
			>
				<div class="grid grid-cols-4 gap-1 text-center">
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">log₁₀ RMSE</div>
						<div class="mono-num text-[11px] font-bold text-[var(--text-primary)]">
							{fmtNum(kStat.log10_rmse ?? wellData.permCoreMetrics?.rmse, 3)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">log₁₀ MAE</div>
						<div class="mono-num text-[11px] font-bold text-[var(--text-primary)]">
							{fmtNum(kStat.log10_mae ?? wellData.permCoreMetrics?.mae, 3)}
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">Factor Err</div>
						<div class="mono-num text-[11px] font-bold text-[var(--text-primary)]">
							{fmtNum(kStat.factor_error, 2)}×
						</div>
					</div>
					<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9.5px] text-[var(--text-muted)]">R²</div>
						<div class="mono-num text-[11px] font-bold text-[var(--status-ok)]">
							{fmtNum(kStat.r2 ?? wellData.permCoreMetrics?.r2, 3)}
						</div>
					</div>
				</div>
				<div class="mt-1.5 grid grid-cols-2 gap-1">
					<button
						type="button"
						onclick={() => {
							workbench.crossplotTemplateOverride = 'core-qc';
							workbench.coreQcView = 'perm';
						}}
						class="h-6 rounded-xs border border-blue-200 bg-blue-50/80 px-1.5 text-[10px] font-semibold text-blue-700 hover:bg-blue-100"
					>
						PERM × CPERM + Depth →
					</button>
					<button
						type="button"
						onclick={() => {
							workbench.crossplotTemplateOverride = 'core-qc';
							workbench.coreQcView = 'dual';
						}}
						class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-1.5 text-[10px] font-semibold text-[var(--text-primary)] hover:bg-slate-100"
					>
						Dual Φ &amp; K Parity →
					</button>
				</div>
			</ParameterSection>
		{/if}
		<!--
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

		-->

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
			disabled={wellData.isCalculating || !wellData.hasPendingComputed('perm')}
			onclick={savePerm}
			title={wellData.hasPendingComputed('perm')
				? wellData.getPendingSaveTooltip('perm', 'Save computed Permeability curve directly to the well')
				: 'Run the calculation first to enable saving'}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
		>
			Save
		</button>
	</div>
</div>
