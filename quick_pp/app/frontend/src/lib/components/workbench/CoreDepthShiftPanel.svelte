<script lang="ts">
	import { wellData } from '$lib/state/wellData.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	interface Props {
		compact?: boolean;
		defaultTarget?: 'poro' | 'perm' | 'combined';
	}

	let { compact = false, defaultTarget = 'poro' }: Props = $props();

	let autoTarget = $state<'poro' | 'perm' | 'combined'>('poro');
	let allowAutoStretch = $state<boolean>(false);

	$effect(() => {
		autoTarget = defaultTarget;
	});

	let stretchPctLabel = $derived.by(() => {
		const pct = (wellData.coreStretchFactor - 1.0) * 100;
		if (Math.abs(pct) < 0.05) return 'No stretch (0.0%)';
		return pct > 0 ? `+${pct.toFixed(1)}% Stretch` : `${pct.toFixed(1)}% Compress`;
	});

	function handleDeltaChange(newDelta: number) {
		if (!Number.isFinite(newDelta)) return;
		wellData.coreShiftMode = 'affine';
		wellData.coreShiftDelta = Number(newDelta.toFixed(3));
		wellData.applyCoreDepthTransform();
	}

	function nudgeDelta(step: number) {
		handleDeltaChange(wellData.coreShiftDelta + step);
	}

	function handleStretchChange(newStretch: number) {
		if (!Number.isFinite(newStretch) || newStretch <= 0.5 || newStretch >= 2.0) return;
		wellData.coreShiftMode = 'affine';
		wellData.coreStretchFactor = Number(newStretch.toFixed(4));
		wellData.applyCoreDepthTransform();
	}

	function handlePivotChange(val: string) {
		const num = parseFloat(val);
		wellData.corePivotDepth = Number.isFinite(num) ? num : null;
		if (wellData.coreShiftDirty || wellData.coreStretchFactor !== 1.0) {
			wellData.applyCoreDepthTransform();
		}
	}

	function seedTiePointsFromEnds() {
		const plugs = wellData.getBaseCorePlugs();
		if (plugs.length === 0) return;
		const depths = plugs.map((p) => p.raw_depth).sort((a, b) => a - b);
		const topD = Number(depths[0].toFixed(2));
		const botD = Number(depths[depths.length - 1].toFixed(2));
		const topShifted = Number(wellData.computeShiftedCoreDepth(topD).toFixed(2));
		const botShifted = Number(wellData.computeShiftedCoreDepth(botD).toFixed(2));
		wellData.coreTiePoints =
			topD === botD
				? [{ coreDepth: topD, logDepth: topShifted }]
				: [
						{ coreDepth: topD, logDepth: topShifted },
						{ coreDepth: botD, logDepth: botShifted }
					];
		wellData.coreShiftMode = 'tie-points';
		wellData.applyCoreDepthTransform();
	}

	function addTiePoint() {
		const plugs = wellData.getBaseCorePlugs();
		const defaultD =
			wellData.coreTiePoints.length > 0
				? Number(
						(
							wellData.coreTiePoints[wellData.coreTiePoints.length - 1].coreDepth + 5.0
						).toFixed(2)
					)
				: plugs.length > 0
					? Number(plugs[0].raw_depth.toFixed(2))
					: Number(wellData.depthRange.min.toFixed(2));
		wellData.coreTiePoints = [
			...wellData.coreTiePoints,
			{ coreDepth: defaultD, logDepth: defaultD }
		];
		wellData.coreShiftMode = 'tie-points';
		wellData.applyCoreDepthTransform();
	}

	function updateTiePoint(idx: number, key: 'coreDepth' | 'logDepth', rawVal: string) {
		const num = parseFloat(rawVal);
		if (!Number.isFinite(num)) return;
		const next = [...wellData.coreTiePoints];
		next[idx] = { ...next[idx], [key]: num };
		wellData.coreTiePoints = next;
		wellData.coreShiftMode = 'tie-points';
		wellData.applyCoreDepthTransform();
	}

	function removeTiePoint(idx: number) {
		wellData.coreTiePoints = wellData.coreTiePoints.filter((_, i) => i !== idx);
		wellData.applyCoreDepthTransform();
	}
</script>

<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2 text-xs">
	<!-- Mode Switcher & Live Error Summary -->
	<div class="mb-1.5 flex flex-wrap items-center justify-between gap-1.5">
		<div class="flex items-center gap-1">
			<button
				type="button"
				onclick={() => {
					wellData.coreShiftMode = 'affine';
					if (wellData.coreShiftDirty) wellData.applyCoreDepthTransform();
				}}
				class="h-5 rounded-xs border px-1.5 text-[10px] font-semibold transition-colors {wellData.coreShiftMode ===
				'affine'
					? 'border-blue-600 bg-blue-600 text-white'
					: 'border-[var(--border-hairline)] bg-white text-[var(--text-secondary)] hover:bg-slate-100'}"
			>
				Shift &amp; Stretch
			</button>
			<button
				type="button"
				onclick={() => {
					if (wellData.coreTiePoints.length === 0) {
						seedTiePointsFromEnds();
					} else {
						wellData.coreShiftMode = 'tie-points';
						wellData.applyCoreDepthTransform();
					}
				}}
				class="h-5 rounded-xs border px-1.5 text-[10px] font-semibold transition-colors {wellData.coreShiftMode ===
				'tie-points'
					? 'border-blue-600 bg-blue-600 text-white'
					: 'border-[var(--border-hairline)] bg-white text-[var(--text-secondary)] hover:bg-slate-100'}"
			>
				Tie-Point Table ({wellData.coreTiePoints.length})
			</button>
		</div>

		{#if wellData.coreShiftDirty}
			<span
				class="rounded-2xs border border-amber-300 bg-amber-50 px-1.5 py-0.5 font-mono text-[9.5px] font-bold text-amber-800"
			>
				Preview Active
			</span>
		{/if}
	</div>

	{#if wellData.coreShiftMode === 'affine'}
		<!-- Bulk Shift Δd Control -->
		<div class="space-y-1.5">
			<div>
				<div class="mb-0.5 flex items-center justify-between text-[10.5px]">
					<span class="font-medium text-[var(--text-primary)]">
						Depth Shift Δd <span class="text-[var(--text-muted)]">(+ deeper / − shallower)</span>
					</span>
					<div class="flex items-center gap-0.5">
						{#each [-0.5, -0.1, 0.1, 0.5] as step}
							<button
								type="button"
								onclick={() => nudgeDelta(step)}
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
						min="-10"
						max="10"
						step="0.05"
						value={wellData.coreShiftDelta}
						oninput={(e) => handleDeltaChange(parseFloat((e.currentTarget as HTMLInputElement).value))}
						class="h-1.5 min-w-0 flex-1 cursor-pointer accent-[var(--accent-primary)]"
					/>
					<input
						type="number"
						step="0.05"
						value={wellData.coreShiftDelta}
						onchange={(e) => handleDeltaChange(parseFloat((e.currentTarget as HTMLInputElement).value))}
						class="mono-num h-6 w-16 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-right text-xs font-semibold"
					/>
					<span class="mono-num text-[10px] text-[var(--text-muted)]">m</span>
				</div>
			</div>

			<!-- Stretch / Compress Factor Control -->
			<div>
				<div class="mb-0.5 flex items-center justify-between text-[10.5px]">
					<span class="font-medium text-[var(--text-primary)]">
						Stretch / Compress Factor
					</span>
					<span
						class="mono-num text-[10px] font-semibold {Math.abs(wellData.coreStretchFactor - 1.0) >
						1e-3
							? 'text-blue-700'
							: 'text-[var(--text-muted)]'}"
					>
						{stretchPctLabel}
					</span>
				</div>
				<div class="flex items-center gap-1.5">
					<input
						type="range"
						min="0.85"
						max="1.15"
						step="0.002"
						value={wellData.coreStretchFactor}
						oninput={(e) =>
							handleStretchChange(parseFloat((e.currentTarget as HTMLInputElement).value))}
						class="h-1.5 min-w-0 flex-1 cursor-pointer accent-[var(--accent-primary)]"
					/>
					<input
						type="number"
						step="0.005"
						min="0.5"
						max="1.5"
						value={wellData.coreStretchFactor}
						onchange={(e) =>
							handleStretchChange(parseFloat((e.currentTarget as HTMLInputElement).value))}
						class="mono-num h-6 w-16 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-right text-xs font-semibold"
					/>
					<span class="mono-num text-[10px] text-[var(--text-muted)]">×</span>
				</div>
			</div>

			{#if !compact}
				<div class="flex items-center justify-between gap-2 pt-0.5 text-[10.5px]">
					<span class="text-[var(--text-secondary)]" title="Anchor depth held fixed before Δd shift when stretching or compressing">
						Anchor Pivot Depth (d<sub>ref</sub>):
					</span>
					<div class="flex items-center gap-1">
						<input
							type="number"
							step="0.5"
							value={wellData.effectiveCorePivotDepth}
							onchange={(e) => handlePivotChange((e.currentTarget as HTMLInputElement).value)}
							class="mono-num h-5 w-20 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-right text-[11px]"
						/>
						<span class="mono-num text-[10px] text-[var(--text-muted)]">m</span>
						{#if wellData.corePivotDepth !== null}
							<button
								type="button"
								onclick={() => handlePivotChange('')}
								title="Reset pivot to top core plug depth"
								class="h-5 rounded-2xs border border-[var(--border-hairline)] bg-white px-1 text-[9.5px] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
							>
								Auto Top
							</button>
						{/if}
					</div>
				</div>
			{/if}

			<!-- 1-Click Auto-Optimize Bar -->
			<div
				class="mt-1 flex flex-wrap items-center justify-between gap-1.5 rounded-xs border border-blue-200/80 bg-blue-50/60 px-2 py-1"
			>
				<div class="flex items-center gap-1.5 text-[10px]">
					<span class="font-semibold text-blue-950">Auto-Match:</span>
					<select
						bind:value={autoTarget}
						class="h-5 rounded-2xs border border-blue-200 bg-white px-1 py-0 text-[10px] font-medium text-slate-800"
					>
						<option value="poro">PHIT × CPORE</option>
						<option value="perm">PERM × CPERM</option>
						<option value="combined">Combined Φ &amp; K</option>
					</select>
					<label class="flex cursor-pointer items-center gap-1 text-blue-900">
						<input
							type="checkbox"
							bind:checked={allowAutoStretch}
							class="h-3 w-3 rounded-2xs border-blue-300 text-blue-600"
						/>
						<span>+Stretch</span>
					</label>
				</div>
				<button
					type="button"
					onclick={() =>
						wellData.autoOptimizeCoreDepthShift({
							allowStretch: allowAutoStretch,
							target: autoTarget
						})}
					class="h-5 rounded-xs bg-blue-600 px-2 text-[10px] font-semibold text-white hover:bg-blue-700"
				>
					Auto-Optimize Shift
				</button>
			</div>
		</div>
	{:else}
		<!-- Piecewise-Linear Tie-Point Table -->
		<div class="space-y-1.5">
			<div class="flex items-center justify-between text-[10px] text-[var(--text-secondary)]">
				<span>Map Core Depth → Target Log Depth (interpolates shift &amp; stretch between rows):</span>
				<div class="flex items-center gap-1">
					<button
						type="button"
						onclick={seedTiePointsFromEnds}
						class="h-5 rounded-2xs border border-[var(--border-hairline)] bg-white px-1.5 text-[9.5px] font-medium hover:bg-slate-100"
					>
						Seed Top/Bot
					</button>
					<button
						type="button"
						onclick={addTiePoint}
						class="h-5 rounded-2xs bg-[var(--accent-primary)] px-1.5 text-[9.5px] font-semibold text-white hover:bg-blue-700"
					>
						+ Add Tie
					</button>
				</div>
			</div>

			{#if wellData.coreTiePoints.length === 0}
				<div class="rounded-xs border border-dashed border-[var(--border-hairline)] bg-white py-2 text-center text-[10.5px] text-[var(--text-muted)]">
					No tie points defined. Click <strong>Seed Top/Bot</strong> or <strong>+ Add Tie</strong>.
				</div>
			{:else}
				<div class="max-h-32 overflow-y-auto rounded-xs border border-[var(--border-hairline)] bg-white">
					<table class="w-full border-collapse text-[10.5px]">
						<thead class="bg-[var(--bg-subtle)] text-[9.5px] font-semibold text-[var(--text-secondary)]">
							<tr class="border-b border-[var(--border-hairline)]">
								<th class="px-1.5 py-0.5 text-left">#</th>
								<th class="px-1.5 py-0.5 text-right">Core Depth (m)</th>
								<th class="px-1.5 py-0.5 text-right">Log Depth (m)</th>
								<th class="px-1.5 py-0.5 text-right">ΔShift (m)</th>
								<th class="w-6 px-1 py-0.5"></th>
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each wellData.coreTiePoints as tp, idx}
								{@const shift = tp.logDepth - tp.coreDepth}
								<tr>
									<td class="mono-num px-1.5 py-0.5 text-[var(--text-muted)]">{idx + 1}</td>
									<td class="px-1 py-0.5 text-right">
										<input
											type="number"
											step="0.1"
											value={tp.coreDepth}
											onchange={(e) =>
												updateTiePoint(
													idx,
													'coreDepth',
													(e.currentTarget as HTMLInputElement).value
												)}
											class="mono-num h-5 w-20 rounded-2xs border border-[var(--border-hairline)] px-1 text-right text-[10.5px]"
										/>
									</td>
									<td class="px-1 py-0.5 text-right">
										<input
											type="number"
											step="0.1"
											value={tp.logDepth}
											onchange={(e) =>
												updateTiePoint(
													idx,
													'logDepth',
													(e.currentTarget as HTMLInputElement).value
												)}
											class="mono-num h-5 w-20 rounded-2xs border border-[var(--border-hairline)] px-1 text-right text-[10.5px]"
										/>
									</td>
									<td class="mono-num px-1.5 py-0.5 text-right font-semibold text-blue-700">
										{shift >= 0 ? '+' : ''}{fmtNum(shift, 2)}
									</td>
									<td class="px-1 py-0.5 text-center">
										<button
											type="button"
											onclick={() => removeTiePoint(idx)}
											class="text-[var(--text-muted)] hover:text-red-600"
										>
											×
										</button>
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</div>
	{/if}

	<!-- Footer: Live Error Readout + Reset & Save -->
	<div
		class="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-[var(--border-hairline)] pt-1.5 text-[10px]"
	>
		<div class="mono-num flex flex-wrap items-center gap-2 text-[var(--text-secondary)]">
			<span>
				Φ RMSE: <strong class="text-[var(--text-primary)]">{fmtNum(wellData.coreComparison.poro.rmse, 4)}</strong>
				(R² <strong class="text-emerald-700">{fmtNum(wellData.coreComparison.poro.r2, 3)}</strong>)
			</span>
			<span>
				K log₁₀RMSE: <strong class="text-[var(--text-primary)]">{fmtNum(wellData.coreComparison.perm.log10_rmse, 3)}</strong>
				(R² <strong class="text-emerald-700">{fmtNum(wellData.coreComparison.perm.r2, 3)}</strong>)
			</span>
		</div>

		<div class="flex items-center gap-1">
			<button
				type="button"
				disabled={!wellData.coreShiftDirty && wellData.coreShiftDelta === 0 && wellData.coreStretchFactor === 1.0}
				onclick={() => wellData.resetCoreDepthShift()}
				class="h-5 rounded-xs border border-[var(--border-hairline)] bg-white px-2 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-slate-100 disabled:opacity-40"
			>
				Reset
			</button>
			<button
				type="button"
				disabled={!wellData.coreShiftDirty || wellData.isCalculating}
				onclick={() => wellData.saveCoreDepthShift()}
				class="h-5 rounded-xs bg-emerald-600 px-2 text-[10px] font-semibold text-white hover:bg-emerald-700 disabled:opacity-40"
			>
				{wellData.isCalculating ? 'Saving...' : 'Save Core Shift'}
			</button>
		</div>
	</div>
</div>

