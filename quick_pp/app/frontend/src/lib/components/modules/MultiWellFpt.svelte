<script lang="ts">
	import { untrack } from 'svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import {
		densityToGradient,
		fpt,
		FPT_DENSITY_BOUNDS,
		FPT_FLUID_COLORS,
		FPT_PHASES,
		type FptOverride,
		type FptPhase
	} from '$lib/state/fpt.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { zoneSelectionLabel } from '$lib/state/projectParams.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	let owcName = $state<'FWL' | 'OWC'>('FWL');
	let confirmWrite = $state(false);

	const PHASE_LABEL: Record<FptPhase, string> = { GAS: 'Gas', OIL: 'Oil', WATER: 'Water' };

	// Reset module state when switching projects; restore saved settings for the zone selection
	let lastProjectId: string | null = null;
	$effect(() => {
		const pid = workbench.projectId;
		const _gen = projectData.projectModuleParams;
		void workbench.selectedZones;
		untrack(() => {
			if (pid !== lastProjectId) {
				lastProjectId = pid;
				fpt.reset();
			}
			fpt.restoreSettings();
		});
		void _gen;
	});

	// Debounced live re-analysis whenever scope or settings change
	let debounceId: ReturnType<typeof setTimeout> | null = null;
	$effect(() => {
		const sig = JSON.stringify({
			pid: workbench.projectId,
			pu: fpt.pressureUnit,
			du: fpt.depthUnit,
			m: fpt.method,
			n: fpt.nSimulations,
			zf: fpt.useZoneFilter,
			gz: fpt.groupZones,
			az: fpt.activeZone,
			zones: workbench.selectedZones,
			wells: projectData.selectedWellNames,
			ov: fpt.overrides,
			ovz: fpt.overridesByZone,
			ref: fpt.refDensity
		});
		if (!workbench.projectId) return;
		void sig;
		if (debounceId) clearTimeout(debounceId);
		debounceId = setTimeout(() => {
			untrack(() => void fpt.analyze());
		}, 350);
		return () => {
			if (debounceId) clearTimeout(debounceId);
		};
	});

	let res = $derived(fpt.result);
	let currentOverrides = $derived(fpt.getOverridesForZone());
	let availableZones = $derived(fpt.availableZones);

	let zoneCounts = $derived.by(() => {
		const map: Record<string, number> = {};
		for (const p of res?.points ?? []) {
			const z = (p.zone || 'Unzoned').trim();
			map[z] = (map[z] ?? 0) + 1;
		}
		return map;
	});

	let counts = $derived.by(() => {
		const allPts = res?.points ?? [];
		const pts = fpt.activeZone !== 'ALL' ? allPts.filter((p) => (p.zone || 'Unzoned').trim() === fpt.activeZone) : allPts;
		const by: Record<'GAS' | 'OIL' | 'WATER' | 'UNASSIGNED' | 'QC', number> = {
			GAS: 0,
			OIL: 0,
			WATER: 0,
			UNASSIGNED: 0,
			QC: 0
		};
		for (const p of pts) {
			if (p.test_quality !== 'VALID') by.QC++;
			else by[p.fluid_type] += 1;
		}
		return {
			...by,
			total: pts.length,
			wells: new Set(pts.map((p) => p.well_name)).size,
			missing: fpt.activeZone !== 'ALL' ? pts.filter((p) => p.tvdss === null).length : (res?.n_missing_tvdss ?? 0)
		};
	});

	function densityHint(ph: FptPhase, rho: number | null | undefined): string | null {
		if (rho === null || rho === undefined || !Number.isFinite(rho)) return null;
		const [lo, hi] = FPT_DENSITY_BOUNDS[ph];
		return rho < lo || rho > hi ? `outside typical ${lo}–${hi} g/cc` : null;
	}

	function eqGradient(rho: number | null): string {
		if (rho === null || !Number.isFinite(Number(rho))) return '—';
		return `${fmtNum(densityToGradient(Number(rho), fpt.pressureUnit, fpt.depthUnit), 4)} ${fpt.gradientUnit}`;
	}

	function setOverride(ph: FptPhase, patch: Partial<FptOverride>) {
		fpt.setOverride(ph, patch);
	}

	function numOrNull(v: string): number | null {
		if (v.trim() === '') return null;
		const n = Number(v);
		return Number.isFinite(n) ? n : null;
	}

	async function writeContacts(zone?: string) {
		confirmWrite = false;
		await fpt.writeContactsToWells(owcName, zone);
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<!-- Scope summary -->
		<div class="border-b border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5 py-2 text-[10.5px]">
			<div class="mb-1 flex items-center justify-between">
				<span class="font-semibold text-[var(--text-primary)]">Formation Pressure &amp; Contacts</span>
				<span class="mono-num rounded-xs bg-sky-100 px-1 text-[9px] font-bold text-sky-800">STANDALONE</span>
			</div>
			<div class="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[var(--text-secondary)]">
				<span>Wells in scope</span>
				<span class="mono-num text-right font-semibold text-[var(--text-primary)]">
					{fpt.scopeWells()?.length ?? projectData.wells.length}/{projectData.wells.length}
					<span class="font-normal text-[var(--text-muted)]">({counts.wells} w/ FPT)</span>
				</span>
				<span>Zone filter</span>
				<label class="flex items-center justify-end gap-1">
					<input type="checkbox" bind:checked={fpt.useZoneFilter} class="h-3 w-3" />
					<span class="truncate text-right" title={zoneSelectionLabel()}>
						{fpt.useZoneFilter && workbench.selectedZones.length ? zoneSelectionLabel() : 'All zones'}
					</span>
				</label>
				<span>Points</span>
				<span class="mono-num text-right">
					{#each FPT_PHASES as ph (ph)}
						<span class="ml-1" style="color: {FPT_FLUID_COLORS[ph]}" title="{PHASE_LABEL[ph]} (valid)">
							●{counts[ph]}
						</span>
					{/each}
					<span class="ml-1 text-slate-500" title="Unassigned (valid)">○{counts.UNASSIGNED}</span>
					<span class="ml-1 text-slate-400" title="Excluded / QC-flagged">⊘{counts.QC}</span>
				</span>
			</div>
			{#if counts.missing > 0}
				<div class="mt-1 rounded-xs border border-amber-200 bg-amber-50 px-1.5 py-0.5 text-[9.5px] text-amber-800">
					{counts.missing} point(s) lack TVDSS (no TVD/TVDSS curve, survey or TVDSS override) — ignored.
				</div>
			{/if}

			<!-- Zone / Compartment Selector -->
			{#if availableZones.length > 1}
				<div class="mt-1.5 flex flex-col gap-1 border-t border-[var(--border-hairline)] pt-1.5">
					<div class="flex items-center justify-between text-[10px]">
						<span class="font-semibold text-[var(--text-primary)]">Fitting Zone / Compartment:</span>
						<label class="flex items-center gap-1 text-[9.5px] text-[var(--text-muted)] cursor-pointer" title="Fit gradients and contacts independently per zone">
							<input type="checkbox" bind:checked={fpt.groupZones} class="h-3 w-3" />
							<span>Group by zone</span>
						</label>
					</div>
					<div class="flex flex-wrap gap-1">
						<button
							type="button"
							onclick={() => (fpt.activeZone = 'ALL')}
							class="rounded-xs px-1.5 py-0.5 text-[9.5px] font-semibold transition-colors {fpt.activeZone === 'ALL' ? 'bg-[var(--accent-primary)] text-white' : 'bg-slate-100 text-[var(--text-secondary)] hover:bg-slate-200'}"
						>
							All Zones ({res?.points?.length ?? 0})
						</button>
						{#each availableZones as z (z)}
							<button
								type="button"
								onclick={() => (fpt.activeZone = z)}
								class="rounded-xs px-1.5 py-0.5 text-[9.5px] font-medium transition-colors {fpt.activeZone === z ? 'bg-[var(--accent-primary)] text-white font-semibold' : 'bg-slate-100 text-[var(--text-secondary)] hover:bg-slate-200'}"
							>
								{z} ({zoneCounts[z] ?? 0})
							</button>
						{/each}
					</div>
				</div>
			{/if}

			<p class="mt-1 text-[9.5px] leading-snug text-[var(--text-muted)]">
				TVDSS is resolved per well from a TVDSS/TVD curve or directional survey (same reference as SHF). Points are grouped and fitted independently per reservoir zone.
			</p>
		</div>

		<ParameterSection title="Units & Regression" badge="{fpt.gradientUnit} · {fpt.method.toUpperCase()}" defaultOpen={true}>
			<div class="grid grid-cols-2 gap-1.5 text-[10.5px]">
				<label class="flex flex-col gap-0.5">
					<span class="text-[var(--text-muted)]">Pressure unit</span>
					<select bind:value={fpt.pressureUnit} class="h-6 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-[11px]">
						<option value="psi">psi</option>
						<option value="bar">bar</option>
						<option value="kPa">kPa</option>
					</select>
				</label>
				<label class="flex flex-col gap-0.5">
					<span class="text-[var(--text-muted)]">Depth unit (TVDSS)</span>
					<select bind:value={fpt.depthUnit} class="h-6 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-[11px]">
						<option value="m">m</option>
						<option value="ft">ft</option>
					</select>
				</label>
				<label class="flex flex-col gap-0.5">
					<span class="text-[var(--text-muted)]">Regression</span>
					<select bind:value={fpt.method} class="h-6 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-[11px]">
						<option value="ols">OLS</option>
						<option value="huber">Huber (robust)</option>
						<option value="ransac">RANSAC (robust)</option>
					</select>
				</label>
				<label class="flex flex-col gap-0.5">
					<span class="text-[var(--text-muted)]">Monte Carlo runs</span>
					<input
						type="number"
						min="0"
						max="20000"
						step="500"
						bind:value={fpt.nSimulations}
						class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-right text-[11px]"
					/>
				</label>
			</div>
		</ParameterSection>

		<ParameterSection
			title="Point QC & Phase Assignment"
			badge={fpt.pendingCount ? `${fpt.pendingCount} unsaved` : 'Saved'}
			defaultOpen={true}
		>
			<p class="text-[10px] leading-snug text-[var(--text-secondary)]">
				Lasso or click points on the plot to assign <strong>Gas / Oil / Water</strong> or flag
				<strong>Supercharged / Tight / Excluded</strong>. Edits re-fit live and are saved to each well's
				pressure tests with <em>Save Assignments</em>.
			</p>
			<div class="mt-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10.5px]">
				<div class="mb-1 font-semibold text-[var(--text-primary)]">Auto-classify (suggestions)</div>
				<label class="flex items-center justify-between gap-1">
					<span class="text-[var(--text-muted)]">Flag TIGHT if mobility &lt; (mD/cP)</span>
					<input
						type="number"
						step="0.1"
						min="0"
						value={fpt.autoMinMobility ?? ''}
						oninput={(e) => (fpt.autoMinMobility = numOrNull((e.currentTarget as HTMLInputElement).value))}
						class="mono-num h-5.5 w-16 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-right text-[11px]"
					/>
				</label>
				<label class="mt-0.5 flex items-center gap-1.5">
					<input type="checkbox" bind:checked={fpt.autoSupercharge} class="h-3 w-3" />
					<span>Flag supercharged outliers (RANSAC, 2.5×MAD)</span>
				</label>
				<label class="mt-0.5 flex items-center gap-1.5">
					<input type="checkbox" bind:checked={fpt.autoAssignFluids} class="h-3 w-3" />
					<span>Suggest phase for unassigned points (local gradient density)</span>
				</label>
				<button
					type="button"
					disabled={fpt.isLoading || !res}
					onclick={() => fpt.analyze({ autoClassify: true })}
					class="mt-1 h-6 w-full rounded-xs border border-[var(--accent-primary)] bg-blue-50 text-[10.5px] font-semibold text-[var(--accent-primary)] hover:bg-blue-100 disabled:opacity-50"
				>
					Run Auto-Classify
				</button>
			</div>
			{#if fpt.pendingCount > 0}
				<div class="mt-1 flex items-center gap-1">
					<button
						type="button"
						disabled={fpt.isSaving}
						onclick={() => fpt.saveAssignments()}
						class="h-6 flex-1 rounded-xs bg-[var(--accent-primary)] text-[10.5px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
					>
						{fpt.isSaving ? 'Saving…' : `Save Assignments (${fpt.pendingCount})`}
					</button>
					<button
						type="button"
						disabled={fpt.isSaving}
						onclick={() => fpt.discardEdits()}
						class="h-6 rounded-xs border border-[var(--border-hairline)] bg-white px-2 text-[10.5px] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
					>
						Discard
					</button>
				</div>
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Fluid Gradients"
			badge={fpt.activeZone !== 'ALL' ? fpt.activeZone : (res?.by_zone && Object.keys(res.by_zone).length > 1 ? `${Object.keys(res.by_zone).length} zones` : (res ? `${Object.keys(res.gradients).length} phase(s)` : '—'))}
			defaultOpen={true}
		>
			{#if fpt.activeZone === 'ALL' && res?.by_zone && Object.keys(res.by_zone).length > 1}
				<!-- Multi-zone summary list -->
				<div class="space-y-2">
					{#each Object.entries(res.by_zone) as [zName, zRes] (zName)}
						<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10.5px]">
							<div class="flex items-center justify-between border-b border-[var(--border-hairline)] pb-1 mb-1">
								<span class="font-semibold text-[var(--text-primary)]">Zone: {zName}</span>
								<div class="flex items-center gap-1.5">
									<span class="mono-num text-[9.5px] text-[var(--text-muted)]">{zRes.n_points} pts</span>
									<button
										type="button"
										onclick={() => (fpt.activeZone = zName)}
										class="rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.5 text-[9.5px] font-semibold text-blue-700 hover:bg-blue-100"
									>
										Focus / Override →
									</button>
								</div>
							</div>
							{#if Object.keys(zRes.gradients).length === 0}
								<div class="text-[9.5px] text-[var(--text-muted)]">No fitted gradients in this zone</div>
							{:else}
								<div class="space-y-1">
									{#each FPT_PHASES as ph (ph)}
										{@const g = zRes.gradients[ph]}
										{#if g}
											<div class="flex items-center justify-between mono-num text-[10px] pl-1.5 border-l-2" style="border-color: {FPT_FLUID_COLORS[ph]}">
												<span><strong style="color: {FPT_FLUID_COLORS[ph]}">{PHASE_LABEL[ph]}</strong> dP/dz {fmtNum(g.gradient, 4)} <span class="text-[var(--text-muted)]">{g.gradient_unit}</span></span>
												<span>ρ <strong>{fmtNum(g.density_gcc, 3)}</strong> g/cc {g.r_squared !== null ? `· R² ${fmtNum(g.r_squared, 3)}` : ''}</span>
											</div>
										{/if}
									{/each}
								</div>
							{/if}
						</div>
					{/each}
				</div>
			{:else}
				{#if fpt.activeZone !== 'ALL'}
					<div class="mb-1.5 flex items-center justify-between rounded-xs bg-sky-50 px-2 py-1 text-[10px] text-sky-900 border border-sky-200">
						<span>Zone: <strong>{fpt.activeZone}</strong></span>
						<button
							type="button"
							onclick={() => (fpt.activeZone = 'ALL')}
							class="text-[9.5px] font-medium text-sky-700 hover:underline"
						>
							← View All Zones
						</button>
					</div>
				{/if}
				{@const activeGrads = fpt.activeZone !== 'ALL' ? (res?.by_zone?.[fpt.activeZone]?.gradients ?? res?.gradients) : res?.gradients}
				<div class="space-y-1.5">
					{#each FPT_PHASES as ph (ph)}
						{@const g = activeGrads?.[ph]}
						{@const ov = currentOverrides[ph]}
						{@const hint = densityHint(ph, g?.density_gcc)}
						<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10.5px]" style="border-left: 3px solid {FPT_FLUID_COLORS[ph]}">
							<div class="flex items-center justify-between">
								<span class="font-semibold" style="color: {FPT_FLUID_COLORS[ph]}">{PHASE_LABEL[ph]}</span>
								{#if g}
									<span class="mono-num text-[9.5px] text-[var(--text-muted)]">
										{g.method} · n={g.n_points}{g.r_squared !== null ? ` · R² ${fmtNum(g.r_squared, 4)}` : ''}
									</span>
								{:else}
									<span class="text-[9.5px] text-[var(--text-muted)]">no fit</span>
								{/if}
							</div>
							{#if g}
								<div class="mono-num mt-0.5 grid grid-cols-2 gap-x-2 text-[10.5px]">
									<span>dP/dz <strong>{fmtNum(g.gradient, 4)}</strong> <span class="text-[var(--text-muted)]">{g.gradient_unit}</span></span>
									<span class="text-right">ρ <strong>{fmtNum(g.density_gcc, 3)}</strong> g/cc</span>
									<span class="text-[9.5px] text-[var(--text-muted)]">± {fmtNum(g.std_err_gradient, 5)}</span>
									<span class="text-right text-[9.5px] text-[var(--text-muted)]">P₀ {fmtNum(g.intercept, 1)}</span>
								</div>
								{#if hint}
									<div class="mt-0.5 text-[9.5px] text-amber-700">⚠ density {hint}</div>
								{/if}
							{/if}
							<label class="mt-1 flex items-center gap-1.5 text-[10px]">
								<input
									type="checkbox"
									checked={ov.enabled}
									onchange={(e) => setOverride(ph, { enabled: (e.currentTarget as HTMLInputElement).checked })}
									class="h-3 w-3"
								/>
								<span>Manual gradient override {fpt.activeZone !== 'ALL' ? `(${fpt.activeZone})` : ''}</span>
							</label>
							{#if ov.enabled}
								<div class="mt-1 space-y-1 border-t border-dashed border-[var(--border-hairline)] pt-1">
									<div class="grid grid-cols-2 gap-1 rounded-xs bg-[var(--bg-subtle)] p-0.5 text-[9.5px]">
										{#each [['fixed', 'Fixed slope'], ['anchor', 'Anchor line']] as [k, lbl] (k)}
											<button
												type="button"
												onclick={() => setOverride(ph, { mode: k as 'fixed' | 'anchor' })}
												class="h-5 rounded-xs {ov.mode === k ? 'bg-white font-semibold text-[var(--accent-primary)] shadow-2xs' : 'text-[var(--text-secondary)]'}"
												title={k === 'fixed' ? 'Fix the slope; intercept fitted to the phase points (works with 1 point)' : 'Regional line through an anchor point (no points needed)'}
											>
												{lbl}
											</button>
										{/each}
									</div>
									<label class="flex items-center justify-between gap-1 text-[10px]">
										<span class="text-[var(--text-muted)]">Density (g/cc)</span>
										<span class="flex items-center gap-1">
											<span class="mono-num text-[9px] text-[var(--text-muted)]">= {eqGradient(ov.density)}</span>
											<input
												type="number"
												step="0.01"
												value={ov.density ?? ''}
												oninput={(e) => setOverride(ph, { density: numOrNull((e.currentTarget as HTMLInputElement).value) })}
												class="mono-num h-5.5 w-16 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-right text-[11px]"
											/>
										</span>
									</label>
									{#if ov.mode === 'anchor'}
										<div class="grid grid-cols-2 gap-1 text-[10px]">
											<label class="flex flex-col">
												<span class="text-[var(--text-muted)]">Anchor TVDSS ({fpt.depthUnit})</span>
												<input
													type="number"
													step="0.1"
													value={ov.anchorTvdss ?? ''}
													oninput={(e) => setOverride(ph, { anchorTvdss: numOrNull((e.currentTarget as HTMLInputElement).value) })}
													class="mono-num h-5.5 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-right text-[11px]"
												/>
											</label>
											<label class="flex flex-col">
												<span class="text-[var(--text-muted)]">Anchor P ({fpt.pressureUnit})</span>
												<input
													type="number"
													step="0.1"
													value={ov.anchorPressure ?? ''}
													oninput={(e) => setOverride(ph, { anchorPressure: numOrNull((e.currentTarget as HTMLInputElement).value) })}
													class="mono-num h-5.5 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-right text-[11px]"
												/>
											</label>
										</div>
									{/if}
								</div>
							{/if}
						</div>
					{/each}
					<button type="button" onclick={() => fpt.resetOverrides()} class="text-[10px] font-medium text-blue-600 hover:underline">
						Reset overrides {fpt.activeZone !== 'ALL' ? `for ${fpt.activeZone}` : ''}
					</button>
				</div>
			{/if}
		</ParameterSection>

		<ParameterSection
			title="Fluid Contacts (P10 / P50 / P90)"
			badge={fpt.activeZone !== 'ALL' ? fpt.activeZone : (res ? (Object.keys(res.contacts).join(' · ') || 'none') : '—')}
			defaultOpen={true}
		>
			{@const activeContacts = fpt.activeZone !== 'ALL' ? (res?.by_zone?.[fpt.activeZone]?.contacts ?? res?.contacts) : null}
			{@const allZoneContacts = fpt.activeZone === 'ALL' && res?.by_zone ? Object.entries(res.by_zone).flatMap(([zn, zr]) => Object.values(zr.contacts || {}).map((c) => ({ zone: zn, contact: c }))) : []}
			{@const showList = fpt.activeZone !== 'ALL' ? Object.values(activeContacts || {}).map((c) => ({ zone: fpt.activeZone, contact: c })) : allZoneContacts}

			{#if showList.length > 0}
				<div class="space-y-1.5">
					{#each showList as item (item.zone + '::' + item.contact?.contact_type)}
						{@const c = item.contact}
						{#if c}
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10.5px]" style="border-left: 3px solid {FPT_FLUID_COLORS[c.lower_fluid]}">
								<div class="flex items-center justify-between">
									<span class="font-semibold text-[var(--text-primary)]">
										{#if fpt.activeZone === 'ALL'}<span class="mr-1 text-slate-500 font-bold">[{item.zone}]</span>{/if}
										{c.label}
									</span>
									<span class="mono-num text-[9.5px] text-[var(--text-muted)]">
										{PHASE_LABEL[c.upper_fluid]} ρ {fmtNum(c.upper_density_gcc, 2)} / {PHASE_LABEL[c.lower_fluid]} ρ {fmtNum(c.lower_density_gcc, 2)}
									</span>
								</div>
								<div class="mono-num mt-0.5 grid grid-cols-3 gap-1 text-center">
									<div class="rounded-xs bg-white px-1 py-0.5">
										<div class="text-[8.5px] text-[var(--text-muted)]">P10 (shallow)</div>
										<div>{fmtNum(c.tvdss_p10, 1)}</div>
									</div>
									<div class="rounded-xs border border-[var(--accent-primary)] bg-blue-50 px-1 py-0.5">
										<div class="text-[8.5px] text-[var(--text-muted)]">P50</div>
										<div class="font-bold text-[var(--accent-primary)]">{fmtNum(c.tvdss_p50, 1)}</div>
									</div>
									<div class="rounded-xs bg-white px-1 py-0.5">
										<div class="text-[8.5px] text-[var(--text-muted)]">P90 (deep)</div>
										<div>{fmtNum(c.tvdss_p90, 1)}</div>
									</div>
								</div>
								<div class="mono-num mt-0.5 flex justify-between text-[9.5px] text-[var(--text-muted)]">
									<span>{res?.units?.depth ?? 'm'} TVDSS · σ {fmtNum(c.std_tvdss, 2)}</span>
									<span>@ {fmtNum(c.pressure_at_contact, 1)} {res?.units?.pressure ?? 'psi'}</span>
								</div>
								{#each c.warnings as w (w)}
									<div class="mt-0.5 text-[9.5px] text-amber-700">⚠ {w}</div>
								{/each}
								{#if c.contact_type === 'OWC' || c.contact_type === 'GWC'}
									<button
										type="button"
										onclick={() => fpt.applyFwlToShf(c.contact_type as 'OWC' | 'GWC', item.zone !== 'ALL' ? item.zone : undefined)}
										class="mt-1 h-5.5 w-full rounded-xs border border-emerald-300 bg-emerald-50 text-[10px] font-semibold text-emerald-800 hover:bg-emerald-100"
										title="Set the multi-well SHF free water level to this contact (P50, m TVDSS)"
									>
										Apply as FWL to SHF ({item.zone !== 'ALL' ? `Zone: ${item.zone} · ` : ''}{fmtNum(c.tvdss_p50_m, 1)} m TVDSS)
									</button>
								{/if}
							</div>
						{/if}
					{/each}
					<div class="rounded-xs border border-[var(--border-hairline)] p-1.5 text-[10px]">
						<div class="flex items-center justify-between gap-1">
							<span class="text-[var(--text-muted)]">Write P50 contacts into well fluid contacts as</span>
							<select bind:value={owcName} class="h-5 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-[10px]">
								<option value="FWL">FWL</option>
								<option value="OWC">OWC</option>
							</select>
						</div>
						{#if !confirmWrite}
							<button
								type="button"
								disabled={fpt.isSaving}
								onclick={() => (confirmWrite = true)}
								class="mt-1 h-5.5 w-full rounded-xs border border-[var(--border-hairline)] bg-white text-[10px] font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-50"
							>
								Write Contacts to Wells ({fpt.scopeWells()?.length ?? projectData.wells.length})…
							</button>
						{:else}
							<div class="mt-1 flex items-center gap-1">
								<span class="flex-1 text-[9.5px] text-amber-800">Overwrites same-named contacts (TVDSS→MD per well).</span>
								<button type="button" onclick={() => (confirmWrite = false)} class="h-5 rounded-xs border border-[var(--border-hairline)] px-1.5 text-[10px]">Cancel</button>
								<button type="button" onclick={() => writeContacts()} class="h-5 rounded-xs bg-[var(--accent-primary)] px-1.5 text-[10px] font-semibold text-white">Confirm</button>
							</div>
						{/if}
					</div>
					{#if fpt.lastApplied}
						<div class="text-[9.5px] text-emerald-700">✓ {fpt.lastApplied}</div>
					{/if}
				</div>
			{:else}
				<p class="text-[10px] text-[var(--text-muted)]">
					Contacts need at least two phases with fitted (or overridden) gradients, e.g. Oil + Water
					(OWC / FWL) or Gas + Oil (GOC). For a hydrocarbon-only well, enable a Water
					<em>Anchor line</em> override with a regional aquifer point.
				</p>
			{/if}
		</ParameterSection>

		<ParameterSection title="Excess Pressure Reference" badge={res ? `ρ ${fmtNum(res.reference.density_gcc, 3)} g/cc` : '—'}>
			<label class="flex items-center justify-between gap-1 text-[10.5px]">
				<span class="text-[var(--text-muted)]">Reference density (blank = water fit / 1.0)</span>
				<input
					type="number"
					step="0.01"
					value={fpt.refDensity ?? ''}
					oninput={(e) => (fpt.refDensity = numOrNull((e.currentTarget as HTMLInputElement).value))}
					class="mono-num h-5.5 w-16 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-right text-[11px]"
				/>
			</label>
			{#if res}
				<p class="mono-num mt-1 text-[9.5px] text-[var(--text-muted)]">
					Datum {fmtNum(res.reference.datum_tvdss, 1)} {res.units.depth} @ {fmtNum(res.reference.datum_pressure, 1)}
					{res.units.pressure} · {fmtNum(res.reference.gradient, 4)} {res.units.gradient}
				</p>
			{/if}
		</ParameterSection>

		{#if res?.warnings?.length}
			<div class="m-2 space-y-0.5 rounded-sm border border-amber-200 bg-amber-50 p-1.5 text-[10px] text-amber-800">
				{#each res.warnings as w, i (i)}
					<div>⚠ {w}</div>
				{/each}
			</div>
		{/if}
		{#if fpt.error}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{fpt.error}
			</div>
		{/if}
		{#if fpt.statusMsg}
			<div class="mx-2 mb-2 text-[10px] text-[var(--text-secondary)]">{fpt.statusMsg}</div>
		{/if}
	</div>

	<div class="flex h-9 shrink-0 items-center gap-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5">
		<button
			type="button"
			disabled={fpt.isLoading || !workbench.projectId}
			onclick={() => fpt.analyze()}
			class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{fpt.isLoading ? 'Analysing…' : 'Analyze'}
		</button>
		<button
			type="button"
			disabled={!workbench.projectId}
			onclick={() => fpt.saveSettings()}
			title="Save units, regression and manual overrides for the current zone selection"
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-45"
		>
			Save Settings
		</button>
	</div>
</div>
