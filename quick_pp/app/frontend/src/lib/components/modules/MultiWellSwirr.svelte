<script lang="ts">
	import { postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import {
		projectData,
		type SwirrApplyResult,
		type SwirrEquationType,
		type SwirrExtractionMode,
		type SwirrFitResult,
		type SwirrFitScope
	} from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { getRockTypeColor } from '$lib/utils/petrophysicsColors';

	let isApplying = $state(false);
	let autoLoadedProjectKey = $state<string | null>(null);

	function applyActiveWellSwirrCurve(res: SwirrApplyResult | null) {
		const activeWell = workbench.wellId;
		if (!activeWell || !res?.data?.well_names?.length) return;
		const deptArr = wellData.curves['DEPT'] || wellData.curves['DEPTH'] || [];
		if (deptArr.length === 0) return;

		const mapByDepth = new Map<number, number | null>();
		const wNames = res.data.well_names;
		const dArr = res.data.depths;
		const swArr = res.data.swirr;
		for (let i = 0; i < wNames.length; i++) {
			if (wNames[i] === activeWell && dArr[i] !== null && dArr[i] !== undefined) {
				mapByDepth.set(Math.round(Number(dArr[i]) * 10000), swArr[i] ?? null);
			}
		}
		if (mapByDepth.size === 0) return;
		const newCurve = deptArr.map((d) => {
			const hit = mapByDepth.get(Math.round(d * 10000));
			return hit !== undefined && hit !== null ? hit : Number.NaN;
		});
		wellData.curves = {
			...wellData.curves,
			SWIRR: newCurve
		};
	}

	async function runAutoFit(autoFit: boolean = true) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const cfg = projectData.swirrConfig;
			const res = await postJson<SwirrFitResult>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/swirr/fit`,
				{
					well_names:
						projectData.selectedWellNames.length > 0 ? projectData.selectedWellNames : null,
					extraction_mode: cfg.extractionMode,
					reference_pc_psi: cfg.referencePcPsi,
					equation_type: cfg.equationType,
					fit_scope: cfg.fitScope,
					swirr_min: cfg.swirrMin,
					swirr_max: cfg.swirrMax,
					excluded_plug_ids: cfg.excludedPlugIds,
					rules: cfg.rules,
					auto_fit: autoFit,
					drt_method: projectData.drtMethod,
					cutoffs: projectData.activeDrtCutoffs,
					ift: projectData.shfParams.ift,
					theta: projectData.shfParams.theta
				}
			);
			projectData.swirrConfig.rules = res.rules || {};
			projectData.swirrResult = res;
		} catch (e: any) {
			projectData.error = e?.message || 'Swirr correlation fit failed';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function applySwirrAndUpdateShf(save: boolean = false) {
		if (!workbench.projectId) return;
		isApplying = true;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const applyRes = await postJson<SwirrApplyResult>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/swirr/apply`,
				{
					well_names:
						projectData.selectedWellNames.length > 0 ? projectData.selectedWellNames : null,
					config: projectData.swirrConfig,
					phie_col: projectData.swirrInputCols.phieCol,
					perm_col: projectData.swirrInputCols.permCol,
					rt_col: projectData.swirrInputCols.rtCol,
					drt_method: projectData.drtMethod,
					cutoffs: projectData.activeDrtCutoffs,
					save
				}
			);
			projectData.swirrApplyResult = applyRes;
			if (!save) {
				projectData.markMultiWellComputed('swirr-corr');
			} else {
				projectData.pendingMultiWellModules = {
					...projectData.pendingMultiWellModules,
					'swirr-corr': false
				};
			}
			applyActiveWellSwirrCurve(applyRes);

			// Automatically switch SHF Swirr normalization source to 'poro_perm' and update SHF
			projectData.shfParams.swirrSource = 'poro_perm';
			try {
				const shfRes = await postJson<any>(
					`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/shf/compute`,
					{
						well_names:
							projectData.selectedWellNames.length > 0 ? projectData.selectedWellNames : null,
						model: projectData.shfParams.model,
						j_sat_mode: projectData.shfParams.jSatMode,
						swirr_source: 'poro_perm',
						swirr_config: projectData.swirrConfig,
						drt_method: projectData.drtMethod,
						rock_flag_source: projectData.drtPropagationMode,
						unit: projectData.shfParams.unit,
						cutoffs: projectData.activeDrtCutoffs,
						fwl: projectData.shfParams.fwl,
						fwl_by_zone:
							Object.keys(projectData.shfParams.fwlByZone || {}).length > 0
								? projectData.shfParams.fwlByZone
								: null,
						ift: projectData.shfParams.ift,
						theta: projectData.shfParams.theta,
						gw: projectData.shfParams.gw,
						ghc: projectData.shfParams.ghc,
						save: false
					}
				);
				projectData.shfResult = {
					model: shfRes.model,
					j_sat_mode: shfRes.j_sat_mode ?? projectData.shfParams.jSatMode,
					fits: shfRes.fits,
					fits_swn: shfRes.fits_swn,
					fits_sw: shfRes.fits_sw,
					fwl_by_zone: shfRes.fwl_by_zone ?? projectData.shfParams.fwlByZone,
					scal_data: shfRes.scal_data,
					...shfRes.data
				};
				projectData.markMultiWellComputed('shf');
			} catch {
				// If TVD is not present yet for full SHF log evaluation, still update client-side Swn if shfResult exists
				projectData.setShfSwirrSource('poro_perm');
			}

			if (save) {
				wellData.lastSavedMessage = `Saved SWIRR (${applyRes.saved_wells} wells) ✓`;
				void projectData.refreshSummary(workbench.projectId);
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to apply SWIRR to wells';
		} finally {
			isApplying = false;
			projectData.isComputingMultiWell = false;
		}
	}

	async function saveAllWells() {
		if (projectData.swirrApplyResult) {
			await projectData.saveMultiWellComputed('swirr-corr');
		} else {
			await applySwirrAndUpdateShf(true);
		}
	}

	function setExtractionMode(mode: SwirrExtractionMode) {
		if (projectData.swirrConfig.extractionMode === mode) return;
		projectData.swirrConfig.extractionMode = mode;
		projectData.recomputeSwirrClientSide(true);
	}

	function setEquationType(eq: SwirrEquationType) {
		if (projectData.swirrConfig.equationType === eq) return;
		projectData.swirrConfig.equationType = eq;
		projectData.recomputeSwirrClientSide(true);
	}

	function setFitScope(scope: SwirrFitScope) {
		if (projectData.swirrConfig.fitScope === scope) return;
		projectData.swirrConfig.fitScope = scope;
		projectData.recomputeSwirrClientSide(false);
	}

	function updateRuleCoeff(scopeKey: string, field: 'a' | 'b' | 'c', rawVal: string) {
		const num = parseFloat(rawVal);
		if (!Number.isFinite(num)) return;
		const existing = projectData.swirrConfig.rules[scopeKey];
		if (!existing) return;
		projectData.swirrConfig.rules = {
			...projectData.swirrConfig.rules,
			[scopeKey]: {
				...existing,
				[field]: num
			}
		};
		projectData.recomputeSwirrClientSide(false);
	}

	function clearExcludedPlugs() {
		projectData.swirrConfig.excludedPlugIds = [];
		projectData.recomputeSwirrClientSide(true);
	}

	let displayedRules = $derived.by(() => {
		const rules = projectData.swirrConfig.rules || {};
		if (projectData.swirrConfig.fitScope === 'global') {
			return rules['global'] ? [['global', rules['global']] as const] : [];
		}
		const rtEntries = Object.entries(rules)
			.filter(([k]) => k !== 'global')
			.sort(([a], [b]) => Number(a) - Number(b));
		if (rtEntries.length > 0) return rtEntries;
		return rules['global'] ? [['global', rules['global']] as const] : [];
	});

	let availablePoroCurves = $derived.by(() => {
		const all = projectData.allCurves || [];
		const cands = ['PHIT', 'PHIE', 'POROSITY', 'POR', 'CPORE'].filter((c) => all.includes(c));
		return cands.length > 0 ? cands : ['PHIT', 'PHIE'];
	});

	let availablePermCurves = $derived.by(() => {
		const all = projectData.allCurves || [];
		const cands = ['PERM_TRANS', 'PERM', 'KLOGH', 'CPERM'].filter((c) => all.includes(c));
		return cands.length > 0 ? cands : ['PERM_TRANS', 'PERM'];
	});

	let availableRtCurves = $derived.by(() => {
		const all = projectData.allCurves || [];
		const cands = ['ROCK_FLAG', 'ROCK_FLAG_ML', 'ROCK_FLAG_CUTOFF', 'DRT'].filter((c) =>
			all.includes(c)
		);
		return cands.length > 0 ? cands : ['ROCK_FLAG', 'ROCK_FLAG_ML'];
	});

	// Auto-load SCAL plugs and initial fit when entering module
	$effect(() => {
		const pid = workbench.projectId;
		if (!pid) return;
		if (!projectData.swirrResult && autoLoadedProjectKey !== pid && !projectData.isComputingMultiWell) {
			autoLoadedProjectKey = pid;
			void runAutoFit(true);
		}
	});
</script>

<div class="flex h-full flex-col overflow-hidden">
	<div class="flex-1 overflow-y-auto">
		<!-- Step 1: SCAL Reference Swirr Extraction -->
		<ParameterSection title="1. SCAL Reference Swirr Extraction" defaultOpen={true}>
			<p class="mb-2 text-[11px] leading-relaxed text-[var(--text-muted)]">
				Define how plug-level irreducible water saturation (<span class="font-mono text-[var(--text-primary)]">Swirr</span>) is extracted from SCAL drainage curves at a reference capillary pressure.
			</p>

			<div class="mb-2.5">
				<span class="mb-1 block text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase">
					Extraction Method
				</span>
				<div class="grid grid-cols-2 gap-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-canvas)] p-0.5">
					<button
						type="button"
						class="rounded py-1 text-[10px] font-medium transition-colors {projectData.swirrConfig
							.extractionMode === 'cutoff_pc'
							? 'bg-[var(--accent-primary)] font-semibold text-black'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => setExtractionMode('cutoff_pc')}
						title="Use terminal Sw from plugs reaching at least Reference Pc"
					>
						Max Pc Cutoff
					</button>
					<button
						type="button"
						class="rounded py-1 text-[10px] font-medium transition-colors {projectData.swirrConfig
							.extractionMode === 'interp_pc'
							? 'bg-[var(--accent-primary)] font-semibold text-black'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => setExtractionMode('interp_pc')}
						title="Interpolate or extrapolate log-linear Pc vs Sw trend at Reference Pc"
					>
						Interpolate @ Ref Pc
					</button>
				</div>
			</div>

			<NumericScrubber
				label="Reference / Cutoff Pc"
				unit="psi"
				min={5}
				max={500}
				step={5}
				decimals={1}
				bind:value={projectData.swirrConfig.referencePcPsi}
				onchange={() => projectData.recomputeSwirrClientSide(true)}
			/>

			{#if projectData.swirrResult}
				<div class="mt-2 flex items-center justify-between rounded border border-[var(--border-subtle)] bg-[var(--bg-canvas)] px-2.5 py-1.5 text-[10px]">
					<div class="flex items-center gap-2">
						<span class="text-[var(--text-muted)]">Valid Plugs:</span>
						<span class="font-mono font-semibold text-[var(--accent-primary)]">
							{projectData.swirrResult.valid_plug_count} / {projectData.swirrResult.total_plug_count}
						</span>
					</div>
					{#if projectData.swirrConfig.excludedPlugIds.length > 0}
						<div class="flex items-center gap-1.5">
							<span class="rounded bg-amber-500/15 px-1.5 py-0.5 font-mono text-[9px] text-amber-300">
								{projectData.swirrConfig.excludedPlugIds.length} excluded
							</span>
							<button
								type="button"
								class="text-[9px] text-[var(--accent-primary)] underline hover:opacity-80"
								onclick={clearExcludedPlugs}
							>
								Reset
							</button>
						</div>
					{:else}
						<span class="text-[9px] text-[var(--text-muted)]">Click crossplot points to exclude outliers</span>
					{/if}
				</div>
			{/if}
		</ParameterSection>

		<!-- Step 2: Correlation Model & Grouping -->
		<ParameterSection title="2. Correlation Model & Grouping" defaultOpen={true}>
			<div class="mb-2.5">
				<label
					for="swirr-eq-select"
					class="mb-1 block text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase"
				>
					Correlation Equation
				</label>
				<select
					id="swirr-eq-select"
					class="h-7 w-full rounded border border-[var(--border-subtle)] bg-[var(--bg-canvas)] px-2 font-mono text-[11px] text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-none"
					value={projectData.swirrConfig.equationType}
					onchange={(e) => setEquationType(e.currentTarget.value as SwirrEquationType)}
				>
					<option value="rqi_power">RQI Power-Law: a · √(k/φ)^(-b)</option>
					<option value="semilog_k">Semi-Log Perm: a - b · log10(k)</option>
					<option value="timur">Timur: a · (φ^b / k^c)</option>
				</select>
			</div>

			<div class="mb-2.5">
				<span class="mb-1 block text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase">
					Fit Scope
				</span>
				<div class="grid grid-cols-2 gap-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-canvas)] p-0.5">
					<button
						type="button"
						class="rounded py-1 text-[10px] font-medium transition-colors {projectData.swirrConfig
							.fitScope === 'global'
							? 'bg-[var(--accent-primary)] font-semibold text-black'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => setFitScope('global')}
					>
						Global Fit (All RTs)
					</button>
					<button
						type="button"
						class="rounded py-1 text-[10px] font-medium transition-colors {projectData.swirrConfig
							.fitScope === 'per_rt'
							? 'bg-[var(--accent-primary)] font-semibold text-black'
							: 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'}"
						onclick={() => setFitScope('per_rt')}
					>
						Per Rock Type (RT 1..N)
					</button>
				</div>
			</div>

			<div class="grid grid-cols-2 gap-2">
				<NumericScrubber
					label="Swirr Min"
					unit="v/v"
					min={0.005}
					max={0.4}
					step={0.005}
					decimals={3}
					bind:value={projectData.swirrConfig.swirrMin}
					onchange={() => projectData.recomputeSwirrClientSide(false)}
				/>
				<NumericScrubber
					label="Swirr Max"
					unit="v/v"
					min={0.3}
					max={0.99}
					step={0.01}
					decimals={2}
					bind:value={projectData.swirrConfig.swirrMax}
					onchange={() => projectData.recomputeSwirrClientSide(false)}
				/>
			</div>

			<button
				type="button"
				class="mt-2 h-7 w-full rounded border border-[var(--accent-primary)]/50 bg-[var(--accent-primary)]/15 text-[11px] font-semibold text-[var(--accent-primary)] transition-colors hover:bg-[var(--accent-primary)]/25 disabled:opacity-40"
				disabled={projectData.isComputingMultiWell}
				onclick={() => runAutoFit(true)}
			>
				{projectData.isComputingMultiWell ? 'Fitting...' : 'Auto-Fit SCAL Swirr'}
			</button>
		</ParameterSection>

		<!-- Step 3: Fitted Coefficients Table -->
		<ParameterSection title="3. Fitted Coefficients Table" defaultOpen={true}>
			<div class="mb-1.5 flex items-center justify-between text-[10px] text-[var(--text-muted)]">
				<span>Editable regression coefficients</span>
				<span class="font-mono text-[9px] text-[var(--text-secondary)]">
					{#if projectData.swirrConfig.equationType === 'rqi_power'}
						Swirr = a · √(k/φ)^(-b)
					{:else if projectData.swirrConfig.equationType === 'semilog_k'}
						Swirr = a - b · log10(k)
					{:else}
						Swirr = a · φ^b / k^c
					{/if}
				</span>
			</div>

			{#if displayedRules.length > 0}
				<div class="overflow-x-auto rounded border border-[var(--border-subtle)] bg-[var(--bg-canvas)]">
					<table class="w-full border-collapse text-left font-mono text-[10px]">
						<thead>
							<tr class="border-b border-[var(--border-hairline)] text-[var(--text-muted)]">
								<th class="px-2 py-1">Scope</th>
								<th class="px-1.5 py-1">a</th>
								<th class="px-1.5 py-1">b</th>
								{#if projectData.swirrConfig.equationType === 'timur'}
									<th class="px-1.5 py-1">c</th>
								{/if}
								<th class="px-1.5 py-1 text-right">R²</th>
								<th class="px-2 py-1 text-right">Plugs</th>
							</tr>
						</thead>
						<tbody>
							{#each displayedRules as [scopeKey, rule] (scopeKey)}
								{@const isGlobal = scopeKey === 'global'}
								{@const rtColor = isGlobal ? '#38bdf8' : getRockTypeColor(Number(scopeKey))}
								<tr class="border-b border-[var(--border-hairline)]/50">
									<td class="px-2 py-1 font-semibold whitespace-nowrap">
										<span
											class="mr-1 inline-block h-2 w-2 rounded-full"
											style="background:{rtColor}"
										></span>
										{isGlobal ? 'Global' : `RT ${scopeKey}`}
									</td>
									<td class="px-1 py-0.5">
										<input
											type="number"
											step="0.01"
											value={rule.a}
											onchange={(e) => updateRuleCoeff(scopeKey, 'a', e.currentTarget.value)}
											class="w-14 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-1 py-0.5 font-mono text-[10px] text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-none"
										/>
									</td>
									<td class="px-1 py-0.5">
										<input
											type="number"
											step="0.01"
											value={rule.b}
											onchange={(e) => updateRuleCoeff(scopeKey, 'b', e.currentTarget.value)}
											class="w-14 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-1 py-0.5 font-mono text-[10px] text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-none"
										/>
									</td>
									{#if projectData.swirrConfig.equationType === 'timur'}
										<td class="px-1 py-0.5">
											<input
												type="number"
												step="0.01"
												value={rule.c ?? 0.5}
												onchange={(e) => updateRuleCoeff(scopeKey, 'c', e.currentTarget.value)}
												class="w-14 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)] px-1 py-0.5 font-mono text-[10px] text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-none"
											/>
										</td>
									{/if}
									<td class="px-1.5 py-1 text-right text-[var(--accent-primary)]">
										{rule.r2 !== undefined ? rule.r2.toFixed(3) : '—'}
									</td>
									<td class="px-2 py-1 text-right text-[var(--text-secondary)]">
										{rule.count ?? 0}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{:else}
				<div class="rounded border border-dashed border-[var(--border-subtle)] p-2.5 text-center text-[10px] text-[var(--text-muted)]">
					Click "Auto-Fit SCAL Swirr" to fit correlation parameters.
				</div>
			{/if}
		</ParameterSection>

		<!-- Step 4: Apply & Propagate -->
		<ParameterSection title="4. Apply & Propagate" defaultOpen={true}>
			<div class="space-y-2">
				<div class="grid grid-cols-3 gap-1.5">
					<div>
						<label
							for="swirr-phie-col"
							class="mb-0.5 block text-[9px] font-semibold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Porosity
						</label>
						<select
							id="swirr-phie-col"
							bind:value={projectData.swirrInputCols.phieCol}
							class="h-6 w-full rounded border border-[var(--border-subtle)] bg-[var(--bg-canvas)] px-1.5 font-mono text-[10px] text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-none"
						>
							{#each availablePoroCurves as c (c)}
								<option value={c}>{c}</option>
							{/each}
						</select>
					</div>
					<div>
						<label
							for="swirr-perm-col"
							class="mb-0.5 block text-[9px] font-semibold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Permeability
						</label>
						<select
							id="swirr-perm-col"
							bind:value={projectData.swirrInputCols.permCol}
							class="h-6 w-full rounded border border-[var(--border-subtle)] bg-[var(--bg-canvas)] px-1.5 font-mono text-[10px] text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-none"
						>
							{#each availablePermCurves as c (c)}
								<option value={c}>{c}</option>
							{/each}
						</select>
					</div>
					<div>
						<label
							for="swirr-rt-col"
							class="mb-0.5 block text-[9px] font-semibold tracking-wider text-[var(--text-muted)] uppercase"
						>
							Rock Type
						</label>
						<select
							id="swirr-rt-col"
							bind:value={projectData.swirrInputCols.rtCol}
							class="h-6 w-full rounded border border-[var(--border-subtle)] bg-[var(--bg-canvas)] px-1.5 font-mono text-[10px] text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-none"
						>
							{#each availableRtCurves as c (c)}
								<option value={c}>{c}</option>
							{/each}
						</select>
					</div>
				</div>

				<button
					type="button"
					class="h-8 w-full rounded bg-[var(--accent-primary)] text-xs font-semibold text-black shadow-sm transition-opacity hover:opacity-90 disabled:opacity-40"
					disabled={projectData.isComputingMultiWell || isApplying}
					onclick={() => applySwirrAndUpdateShf(false)}
				>
					{isApplying ? 'Applying SWIRR & Updating SHF...' : 'Apply SWIRR to Wells & Update SHF'}
				</button>

				{#if projectData.swirrApplyResult}
					<div class="rounded border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1.5 text-[10px]">
						<div class="flex items-center justify-between font-semibold text-emerald-300">
							<span>SWIRR Applied ({projectData.swirrApplyResult.well_summaries.length} wells)</span>
							<span class="font-mono">
								Mean: {projectData.swirrApplyResult.mean_swirr?.toFixed(3) ?? '—'}
							</span>
						</div>
						<div class="mt-0.5 flex items-center justify-between font-mono text-[9px] text-[var(--text-secondary)]">
							<span>Min: {projectData.swirrApplyResult.min_swirr?.toFixed(3) ?? '—'}</span>
							<span>Max: {projectData.swirrApplyResult.max_swirr?.toFixed(3) ?? '—'}</span>
							<span class="text-[var(--accent-primary)]">SHF Swn Linked ✓</span>
						</div>
					</div>
				{/if}
			</div>
		</ParameterSection>
	</div>

	<div class="shrink-0 space-y-1.5 border-t border-[var(--border-subtle)] bg-[var(--bg-surface)] p-3">
		{#if projectData.error}
			<div class="rounded border border-red-500/40 bg-red-500/10 px-2 py-1 text-[10px] text-red-400">
				{projectData.error}
			</div>
		{/if}
		<div class="flex gap-2">
			<button
				type="button"
				class="h-8 flex-1 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] text-xs font-semibold text-[var(--text-primary)] hover:border-[var(--accent-primary)] disabled:opacity-40"
				disabled={projectData.isComputingMultiWell}
				onclick={() => runAutoFit(true)}
			>
				Auto-Fit SCAL
			</button>
			<button
				type="button"
				class="relative h-8 flex-1 rounded bg-[var(--accent-primary)] text-xs font-semibold text-black hover:opacity-90 disabled:opacity-40"
				disabled={projectData.isComputingMultiWell}
				onclick={saveAllWells}
			>
				Save All Wells
				{#if projectData.hasPendingMultiWell('swirr-corr')}
					<span
						class="ml-1 inline-block h-1.5 w-1.5 rounded-full bg-amber-950 align-middle"
						title="Unsaved multi-well curves ready to commit"
					></span>
				{/if}
			</button>
		</div>
	</div>
</div>
