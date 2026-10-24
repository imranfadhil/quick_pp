<script lang="ts">
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	interface Props {
		compact?: boolean;
		onapply?: () => void;
	}

	let { compact = false, onapply }: Props = $props();

	function handleGrCleanChange(val: number) {
		if (!Number.isFinite(val)) return;
		const clamped = Math.max(0, Math.min(workbench.lithoParams.grClay - 10, Number(val.toFixed(1))));
		workbench.lithoParams.grClean = clamped;
		wellData.applyLiveVshGr();
	}

	function handleGrClayChange(val: number) {
		if (!Number.isFinite(val)) return;
		const clamped = Math.max(
			workbench.lithoParams.grClean + 10,
			Math.min(300, Number(val.toFixed(1)))
		);
		workbench.lithoParams.grClay = clamped;
		wellData.applyLiveVshGr();
	}

	function resetToP5P95() {
		const grSt = wellData.activeStats['GR'] ?? wellData.stats['GR'];
		if (grSt && grSt.p10 !== null && grSt.p90 !== null) {
			workbench.lithoParams.grClean = Number(Math.max(0, grSt.p10).toFixed(1));
			workbench.lithoParams.grClay = Number(Math.max(grSt.p10 + 25, grSt.p90).toFixed(1));
		} else {
			workbench.lithoParams.grClean = 15.0;
			workbench.lithoParams.grClay = 115.0;
		}
		wellData.applyLiveVshGr();
	}
</script>

<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2 text-xs">
	<!-- Scope Switcher -->
	<div class="mb-1.5 flex flex-wrap items-center justify-between gap-1">
		<div class="flex flex-wrap items-center gap-1">
			{#each [
				['shale_sand', 'Shale + Water Sand', 'Calibrate in Clean Shale and Clean Water-Bearing Sand intervals'],
				['shale_only', 'Clean Shale Only', 'Calibrate in Clean Shale interval only (when clean water sand is unavailable)'],
				['all', 'All Good-Hole', 'Calibrate across all valid non-badhole, non-coal depths'],
				...(workbench.brushedDepthIndices.size > 0
					? [['brushed', `Lasso (${workbench.brushedDepthIndices.size})`, 'Calibrate on lasso-selected points']]
					: [])
			] as [sKey, sLbl, sTip]}
				<button
					type="button"
					title={sTip}
					onclick={() => (workbench.vshCalibScope = sKey as any)}
					class="h-5 rounded-xs border px-1.5 text-[10px] font-semibold transition-colors {workbench.vshCalibScope ===
					sKey
						? 'border-blue-600 bg-blue-600 text-white'
						: 'border-[var(--border-hairline)] bg-white text-[var(--text-secondary)] hover:bg-slate-100'}"
				>
					{sLbl}
				</button>
			{/each}
		</div>
	</div>

	<!-- Anchor Regime Status Cards (Clean Water Sand vs Clean Shale) -->
	<div class="mb-2 grid grid-cols-2 gap-1.5 text-[10px]">
		<div class="rounded-xs border border-amber-200/90 bg-amber-50/50 px-1.5 py-1">
			<div class="flex items-center justify-between font-semibold text-amber-950">
				<span>Clean Water Sand</span>
				<span class="mono-num text-[9.5px] text-amber-800">n={wellData.vshComparison.sand.count}</span>
			</div>
			{#if wellData.vshComparison.sand.count > 0}
				<div class="mono-num mt-0.5 flex items-center justify-between text-[9.5px] text-slate-700">
					<span>Vsh<sub>DN</sub>: <strong>{fmtNum(wellData.vshComparison.sand.mean_vsh_dn, 2)}</strong></span>
					<span>Vsh<sub>GR</sub>: <strong>{fmtNum(wellData.vshComparison.sand.mean_vsh_gr, 2)}</strong></span>
					<span
						class={Math.abs(wellData.vshComparison.sand.bias ?? 0) <= 0.04
							? 'font-bold text-emerald-700'
							: 'font-bold text-amber-800'}
					>
						Δ{wellData.vshComparison.sand.bias! >= 0 ? '+' : ''}{fmtNum(wellData.vshComparison.sand.bias, 2)}
					</span>
				</div>
			{:else}
				<div class="mt-0.5 text-[9.5px] text-[var(--text-muted)]">
					No clean water sand (Vsh<sub>DN</sub> ≤ {workbench.vshSandCutoff.toFixed(2)})
				</div>
			{/if}
		</div>

		<div class="rounded-xs border border-emerald-200/90 bg-emerald-50/50 px-1.5 py-1">
			<div class="flex items-center justify-between font-semibold text-emerald-950">
				<span>Clean Shale</span>
				<span class="mono-num text-[9.5px] text-emerald-800">n={wellData.vshComparison.shale.count}</span>
			</div>
			{#if wellData.vshComparison.shale.count > 0}
				<div class="mono-num mt-0.5 flex items-center justify-between text-[9.5px] text-slate-700">
					<span>Vsh<sub>DN</sub>: <strong>{fmtNum(wellData.vshComparison.shale.mean_vsh_dn, 2)}</strong></span>
					<span>Vsh<sub>GR</sub>: <strong>{fmtNum(wellData.vshComparison.shale.mean_vsh_gr, 2)}</strong></span>
					<span
						class={Math.abs(wellData.vshComparison.shale.bias ?? 0) <= 0.04
							? 'font-bold text-emerald-700'
							: 'font-bold text-amber-800'}
					>
						Δ{wellData.vshComparison.shale.bias! >= 0 ? '+' : ''}{fmtNum(wellData.vshComparison.shale.bias, 2)}
					</span>
				</div>
			{:else}
				<div class="mt-0.5 text-[9.5px] text-[var(--text-muted)]">
					No clean shale (Vsh<sub>DN</sub> ≥ {workbench.vshShaleCutoff.toFixed(2)})
				</div>
			{/if}
		</div>
	</div>

	<!-- GR Clean (Min GR) & GR Shale (Max GR) Interactive Sliders -->
	<div class="space-y-1.5">
		<div>
			<div class="mb-0.5 flex items-center justify-between text-[10.5px]">
				<span class="font-medium text-[var(--text-primary)]">
					GR Clean <span class="text-[var(--text-muted)]">(Min GR, Vsh=0)</span>
				</span>
				<div class="flex items-center gap-0.5">
					{#each [-5, -1, 1, 5] as step}
						<button
							type="button"
							onclick={() => handleGrCleanChange(workbench.lithoParams.grClean + step)}
							class="mono-num h-4 rounded-2xs border border-[var(--border-hairline)] bg-white px-1 text-[9.5px] font-medium text-[var(--text-secondary)] hover:bg-slate-100"
						>
							{step > 0 ? `+${step}` : step}
						</button>
					{/each}
				</div>
			</div>
			<div class="flex items-center gap-1.5">
				<input
					type="range"
					min="0"
					max="90"
					step="0.5"
					value={workbench.lithoParams.grClean}
					oninput={(e) =>
						handleGrCleanChange(parseFloat((e.currentTarget as HTMLInputElement).value))}
					class="h-1.5 min-w-0 flex-1 cursor-pointer accent-amber-600"
				/>
				<input
					type="number"
					step="0.5"
					min="0"
					max="120"
					value={workbench.lithoParams.grClean}
					onchange={(e) =>
						handleGrCleanChange(parseFloat((e.currentTarget as HTMLInputElement).value))}
					class="mono-num h-6 w-16 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-right text-xs font-semibold"
				/>
				<span class="mono-num text-[10px] text-[var(--text-muted)]">gAPI</span>
			</div>
		</div>

		<div>
			<div class="mb-0.5 flex items-center justify-between text-[10.5px]">
				<span class="font-medium text-[var(--text-primary)]">
					GR Shale <span class="text-[var(--text-muted)]">(Max GR, Vsh=1)</span>
				</span>
				<div class="flex items-center gap-0.5">
					{#each [-5, -1, 1, 5] as step}
						<button
							type="button"
							onclick={() => handleGrClayChange(workbench.lithoParams.grClay + step)}
							class="mono-num h-4 rounded-2xs border border-[var(--border-hairline)] bg-white px-1 text-[9.5px] font-medium text-[var(--text-secondary)] hover:bg-slate-100"
						>
							{step > 0 ? `+${step}` : step}
						</button>
					{/each}
				</div>
			</div>
			<div class="flex items-center gap-1.5">
				<input
					type="range"
					min="50"
					max="240"
					step="0.5"
					value={workbench.lithoParams.grClay}
					oninput={(e) =>
						handleGrClayChange(parseFloat((e.currentTarget as HTMLInputElement).value))}
					class="h-1.5 min-w-0 flex-1 cursor-pointer accent-emerald-700"
				/>
				<input
					type="number"
					step="0.5"
					min="40"
					max="300"
					value={workbench.lithoParams.grClay}
					onchange={(e) =>
						handleGrClayChange(parseFloat((e.currentTarget as HTMLInputElement).value))}
					class="mono-num h-6 w-16 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-right text-xs font-semibold"
				/>
				<span class="mono-num text-[10px] text-[var(--text-muted)]">gAPI</span>
			</div>
		</div>

		{#if !compact}
			<div class="grid grid-cols-2 gap-2 pt-0.5 text-[10px]">
				<div class="flex items-center justify-between gap-1 rounded-xs border border-[var(--border-hairline)] bg-white px-1.5 py-0.5">
					<span class="text-[var(--text-secondary)]">Sand Vsh<sub>DN</sub> ≤</span>
					<input
						type="number"
						step="0.05"
						min="0.05"
						max="0.45"
						bind:value={workbench.vshSandCutoff}
						class="mono-num h-5 w-14 rounded-2xs border border-[var(--border-hairline)] px-1 text-right text-[10.5px]"
					/>
				</div>
				<div class="flex items-center justify-between gap-1 rounded-xs border border-[var(--border-hairline)] bg-white px-1.5 py-0.5">
					<span class="text-[var(--text-secondary)]">Shale Vsh<sub>DN</sub> ≥</span>
					<input
						type="number"
						step="0.05"
						min="0.40"
						max="0.90"
						bind:value={workbench.vshShaleCutoff}
						class="mono-num h-5 w-14 rounded-2xs border border-[var(--border-hairline)] px-1 text-right text-[10.5px]"
					/>
				</div>
			</div>
		{/if}

		<!-- 1-Click Auto-Calibrate Bar -->
		<div
			class="mt-1 flex flex-wrap items-center justify-between gap-1.5 rounded-xs border border-blue-200/80 bg-blue-50/60 px-2 py-1"
		>
			<span class="text-[10px] font-medium text-blue-950">
				Match <strong>VSH_GR → VSH_DN</strong> ({workbench.vshCalibScope === 'shale_only'
					? 'Shale Only'
					: workbench.vshCalibScope === 'shale_sand'
						? 'Shale + Water Sand'
						: workbench.vshCalibScope === 'brushed'
							? 'Lasso'
							: 'All Good-Hole'})
			</span>
			<button
				type="button"
				onclick={() => {
					wellData.autoCalibrateVshGr();
					onapply?.();
				}}
				class="h-5 rounded-xs bg-blue-600 px-2 text-[10px] font-semibold text-white hover:bg-blue-700"
			>
				Auto-Calibrate GR
			</button>
		</div>
	</div>

	<!-- Footer: Live Error Readout + P10/P90 Reset & Apply -->
	<div
		class="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-[var(--border-hairline)] pt-1.5 text-[10px]"
	>
		<div class="mono-num flex flex-wrap items-center gap-2 text-[var(--text-secondary)]">
			<span>
				RMSE: <strong class="text-[var(--text-primary)]">{fmtNum(wellData.vshComparison.active.rmse, 4)}</strong>
			</span>
			<span>
				Bias: <strong class="text-[var(--text-primary)]">{wellData.vshComparison.active.bias !== null && wellData.vshComparison.active.bias >= 0 ? '+' : ''}{fmtNum(wellData.vshComparison.active.bias, 3)}</strong>
			</span>
			<span>
				R²: <strong class="text-emerald-700">{fmtNum(wellData.vshComparison.active.r2, 3)}</strong>
			</span>
		</div>

		<div class="flex items-center gap-1">
			<button
				type="button"
				onclick={resetToP5P95}
				title="Seed GR Clean & GR Shale from GR P10 and P90"
				class="h-5 rounded-xs border border-[var(--border-hairline)] bg-white px-2 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-slate-100"
			>
				P10/P90
			</button>
			{#if onapply}
				<button
					type="button"
					disabled={wellData.isCalculating}
					onclick={() => onapply?.()}
					class="h-5 rounded-xs bg-emerald-600 px-2 text-[10px] font-semibold text-white hover:bg-emerald-700 disabled:opacity-40"
				>
					{wellData.isCalculating ? 'Running...' : 'Update Litho'}
				</button>
			{/if}
		</div>
	</div>
</div>

