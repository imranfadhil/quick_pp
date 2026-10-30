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
	import { fmtNum } from '$lib/utils/formatters';
	import { getRockTypeColor } from '$lib/utils/petrophysicsColors';

	let isApplying = $state(false);
	let copiedSwof = $state(false);
	let relpermPendingSave = $state(false);
	let hasManualRuleOverride = $state(false);
	let hasManualShfFitOverride = $state(false);

	function switchScalSubMode(mode: 'static' | 'relperm') {
		workbench.scalSubMode = mode;
		if (mode === 'relperm') {
			workbench.swirrCrossplotTab = 'relperm';
		} else if (workbench.swirrCrossplotTab === 'relperm') {
			workbench.swirrCrossplotTab = 'swirr-corr';
		}
	}

	// Per-RT & Global mean Swirr presets from SCAL plugs for quick Rel-Perm / SWOF initialization
	const scalSwirrPresets = $derived.by(() => {
		const plugs = (projectData.swirrResult?.plugs ?? []).filter(
			(p) => p.included && p.extracted_swirr !== null && Number.isFinite(p.extracted_swirr)
		);
		if (plugs.length === 0) return [] as Array<{ key: string; label: string; value: number; rt?: number }>;
		const items: Array<{ key: string; label: string; value: number; rt?: number }> = [];
		const meanAll =
			plugs.reduce((acc, p) => acc + (p.extracted_swirr as number), 0) / plugs.length;
		items.push({
			key: 'global',
			label: `Field Mean (${fmtNum(meanAll, 3)})`,
			value: Number(meanAll.toFixed(3))
		});
		const byRt = new Map<number, number[]>();
		for (const p of plugs) {
			const rt = p.rt ?? 1;
			if (!byRt.has(rt)) byRt.set(rt, []);
			byRt.get(rt)!.push(p.extracted_swirr as number);
		}
		for (const [rt, vals] of [...byRt.entries()].sort((a, b) => a[0] - b[0])) {
			const m = vals.reduce((a, b) => a + b, 0) / vals.length;
			items.push({
				key: `rt-${rt}`,
				label: `RT ${rt} (${fmtNum(m, 3)})`,
				value: Number(m.toFixed(3)),
				rt
			});
		}
		return items;
	});

	async function computeScal() {
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		const sp = workbench.scalParams;
		try {
			const [relpermRes, swofRes] = await Promise.all([
				postJson<{
					model: string;
					data: Array<{ sw: number; swn: number; krw: number; kro: number; fw: number }>;
				}>('/quick_pp/scal/relperm', {
					model: sp.model,
					swirr: sp.swirr,
					sor: sp.sor,
					krw_max: sp.krw_max,
					kro_max: sp.kro_max,
					nw: sp.nw,
					no: sp.no,
					lam: sp.lam,
					lw: sp.lw,
					ew: sp.ew,
					tw: sp.tw,
					lo: sp.lo,
					eo: sp.eo,
					to: sp.to
				}),
				postJson<{
					table: Array<{ SW: number; KRW: number; KROW: number; PCOW: number }>;
					swof_keyword: string;
				}>('/quick_pp/scal/swof', {
					model: sp.model,
					swirr: sp.swirr,
					sor: sp.sor,
					krw_max: sp.krw_max,
					kro_max: sp.kro_max,
					nw: sp.nw,
					no: sp.no,
					lam: sp.lam,
					lw: sp.lw,
					ew: sp.ew,
					tw: sp.tw,
					lo: sp.lo,
					eo: sp.eo,
					to: sp.to,
					pc_entry: sp.pc_entry,
					pc_exponent: sp.pc_exponent,
					pc_max: sp.pc_max,
					n_points: sp.n_points
				})
			]);

			const rows = relpermRes.data || [];
			const swArr = rows.map((r) => r.sw);
			const seArr = rows.map((r) => r.swn);
			const krwArr = rows.map((r) => r.krw);
			const kroArr = rows.map((r) => r.kro);
			const fwArr = rows.map((r, i) => {
				const kw = krwArr[i] / Math.max(1e-6, sp.mu_w);
				const ko = kroArr[i] / Math.max(1e-6, sp.mu_o);
				return kw + ko > 0 ? kw / (kw + ko) : (r.fw ?? 0);
			});

			workbench.scalResult = {
				sw: swArr,
				se: seArr,
				krw: krwArr,
				kro: kroArr,
				fw: fwArr,
				swofTable: swofRes.table,
				swofKeyword: swofRes.swof_keyword
			};
			relpermPendingSave = true;
		} catch (e: any) {
			projectData.error = e?.message || 'SCAL Rel-Perm calculation error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function saveRelpermParams() {
		const pid = workbench.projectId;
		const wid = workbench.wellId;
		if (!pid || !wid) {
			relpermPendingSave = false;
			return;
		}
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const extraConfig: Record<string, any> = {
				scal_params: { ...workbench.scalParams }
			};
			const cfgRes = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(pid)}/wells/${encodeURIComponent(wid)}/config`,
				{ extra: extraConfig }
			);
			if (cfgRes?.config) {
				wellData.config = { ...wellData.config, ...cfgRes.config };
			} else {
				wellData.config = { ...wellData.config, ...extraConfig };
			}
			relpermPendingSave = false;
			wellData.lastSavedMessage = 'Saved Rel-Perm & SWOF Parameters ✓';
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to save Rel-Perm parameters';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function copySwofKeyword() {
		const kw = workbench.scalResult?.swofKeyword;
		if (!kw) return;
		await navigator.clipboard.writeText(kw);
		copiedSwof = true;
		setTimeout(() => (copiedSwof = false), 2000);
	}

	function downloadSwofInc() {
		const kw = workbench.scalResult?.swofKeyword;
		if (!kw) return;
		const blob = new Blob([kw], { type: 'text/plain;charset=utf-8;' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `${workbench.projectId || 'FIELD_SCAL'}_SWOF.INC`;
		a.click();
		URL.revokeObjectURL(url);
	}

	const SHF_MODEL_META: Record<
		'leverett_j' | 'skelt_harrison' | 'cuddy',
		{ label: string; fitTitle: string; formula: string }
	> = {
		leverett_j: {
			label: 'Leverett-J',
			fitTitle: 'SCAL Leverett-J Curve Fit',
			formula: 'J = a · Swn^(-b)'
		},
		skelt_harrison: {
			label: 'Skelt-Harrison',
			fitTitle: 'SCAL Skelt-Harrison Pc Fit',
			formula: 'Sw = 1 - a·exp(-(b/(h+d))^c)'
		},
		cuddy: {
			label: 'Cuddy Bvw',
			fitTitle: 'SCAL Cuddy BVW Fit',
			formula: 'BVW = φ·Sw = a · Pc^(-b)'
		}
	};

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
			if (autoFit) {
				hasManualRuleOverride = false;
			}
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
			projectData.recomputeScalFitsFromPlugs(!hasManualShfFitOverride);
		} catch (e: any) {
			projectData.error = e?.message || 'Swirr correlation fit failed';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function computeSwirrAndPc() {
		if (!workbench.projectId) return;
		isApplying = true;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const cfg = projectData.swirrConfig;
			const fitRes = await postJson<SwirrFitResult>(
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
					auto_fit: !hasManualRuleOverride,
					drt_method: projectData.drtMethod,
					cutoffs: projectData.activeDrtCutoffs,
					ift: projectData.shfParams.ift,
					theta: projectData.shfParams.theta
				}
			);
			projectData.swirrConfig.rules = fitRes.rules || {};
			projectData.swirrResult = fitRes;
			projectData.recomputeScalFitsFromPlugs(!hasManualShfFitOverride);

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
					save: false
				}
			);
			projectData.swirrApplyResult = applyRes;
			projectData.markMultiWellComputed('swirr-corr');
			applyActiveWellSwirrCurve(applyRes);
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to compute Swirr & Pc fit';
		} finally {
			isApplying = false;
			projectData.isComputingMultiWell = false;
		}
	}

	async function saveAllWells() {
		if (!workbench.projectId || !projectData.hasPendingMultiWell('swirr-corr')) return;
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
					save: true
				}
			);
			projectData.swirrApplyResult = applyRes;
			projectData.pendingMultiWellModules = {
				...projectData.pendingMultiWellModules,
				'swirr-corr': false
			};
			applyActiveWellSwirrCurve(applyRes);
			wellData.lastSavedMessage = `Saved SWIRR (${applyRes.saved_wells} wells) ✓`;
			void projectData.refreshSummary(workbench.projectId);
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to save SWIRR to wells';
		} finally {
			isApplying = false;
			projectData.isComputingMultiWell = false;
		}
	}

	function setExtractionMode(mode: SwirrExtractionMode) {
		if (projectData.swirrConfig.extractionMode === mode) return;
		projectData.swirrConfig.extractionMode = mode;
		hasManualRuleOverride = false;
	}

	function setEquationType(eq: SwirrEquationType) {
		if (projectData.swirrConfig.equationType === eq) return;
		projectData.swirrConfig.equationType = eq;
		hasManualRuleOverride = false;
		workbench.swirrCrossplotTab = 'swirr-corr';
	}

	function setFitScope(scope: SwirrFitScope) {
		if (projectData.swirrConfig.fitScope === scope) return;
		projectData.swirrConfig.fitScope = scope;
		workbench.swirrCrossplotTab = 'swirr-corr';
	}

	function updateRuleCoeff(scopeKey: string, field: 'a' | 'b' | 'c', rawVal: string) {
		const num = parseFloat(rawVal);
		if (!Number.isFinite(num)) return;
		const existing = projectData.swirrConfig.rules[scopeKey];
		if (!existing) return;
		hasManualRuleOverride = true;
		projectData.swirrConfig.rules = {
			...projectData.swirrConfig.rules,
			[scopeKey]: {
				...existing,
				[field]: num
			}
		};
		workbench.swirrCrossplotTab = 'swirr-corr';
		projectData.recomputeSwirrClientSide(false);
	}

	function selectShfModel(m: 'leverett_j' | 'skelt_harrison' | 'cuddy') {
		projectData.shfParams.model = m;
		hasManualShfFitOverride = false;
		workbench.swirrCrossplotTab = 'scal-fit';
	}

	function updateShfFitParam(rf: string, paramKey: 'a' | 'b' | 'c' | 'd', rawVal: string) {
		const num = Number(rawVal);
		if (!Number.isFinite(num) || !projectData.shfResult?.fits?.[rf]) return;
		hasManualShfFitOverride = true;
		const nextEntry = {
			...projectData.shfResult.fits[rf],
			[paramKey]: num
		};
		const nextFits = {
			...projectData.shfResult.fits,
			[rf]: nextEntry
		};
		const isSwMode = (projectData.shfParams.jSatMode ?? 'swn') === 'sw';
		projectData.shfResult = {
			...projectData.shfResult,
			fits: nextFits,
			fits_swn: isSwMode ? projectData.shfResult.fits_swn : nextFits,
			fits_sw: isSwMode ? nextFits : projectData.shfResult.fits_sw
		};
		workbench.swirrCrossplotTab = 'scal-fit';
	}

	function clearExcludedPlugs() {
		projectData.swirrConfig.excludedPlugIds = [];
		hasManualRuleOverride = false;
		hasManualShfFitOverride = false;
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
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<!-- Top Mode Switcher: Static SCAL (Swirr & Pc) vs Dynamic SCAL (Rel-Perm & SWOF) -->
	<div
		class="shrink-0 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 py-1.5"
	>
		<div class="grid grid-cols-2 gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-0.5">
			{#each [['static', '1. Swirr & Pc Fit (Static)'], ['relperm', '2. Rel-Perm & SWOF (Dynamic)']] as [mKey, mLbl]}
				<button
					type="button"
					onclick={() => switchScalSubMode(mKey as 'static' | 'relperm')}
					class="h-6 rounded-xs px-2 text-[10.5px] font-semibold transition-colors {workbench.scalSubMode ===
					mKey
						? 'bg-[var(--accent-primary)] text-white shadow-2xs'
						: 'text-[var(--text-secondary)] hover:bg-slate-100 hover:text-[var(--text-primary)]'}"
				>
					{mLbl}
				</button>
			{/each}
		</div>
	</div>

	<div class="flex-1 overflow-y-auto">
		{#if workbench.scalSubMode === 'static'}
		<!-- Section 1: SCAL Reference Swirr Extraction -->
		<ParameterSection
			title="1. SCAL Reference Swirr Extraction"
			badge="{projectData.swirrConfig.referencePcPsi} psi"
			defaultOpen={true}
		>
			<p class="mb-2 text-[10.5px] leading-relaxed text-[var(--text-muted)]">
				Extract plug-level irreducible water saturation (<span
					class="mono-num font-semibold text-[var(--text-primary)]">Swirr</span
				>) from SCAL drainage curves at a reference capillary pressure.
			</p>

			<div class="mb-2">
				<span
					class="mb-1 block text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase"
				>
					Extraction Method
				</span>
				<div class="grid grid-cols-2 gap-1">
					{#each [['cutoff_pc', 'Max Pc Cutoff'], ['interp_pc', 'Interpolate @ Ref Pc']] as [mKey, mLbl]}
						<button
							type="button"
							onclick={() => setExtractionMode(mKey as SwirrExtractionMode)}
							class="h-6 rounded-sm border text-[10px] font-medium transition-colors {projectData
								.swirrConfig.extractionMode === mKey
								? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-50'}"
						>
							{mLbl}
						</button>
					{/each}
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
				onchange={() => {
					hasManualRuleOverride = false;
				}}
			/>

			{#if projectData.swirrResult}
				<div
					class="mt-2 flex items-center justify-between rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 py-1 text-[10px]"
				>
					<div class="flex items-center gap-1.5">
						<span class="text-[var(--text-secondary)]">Valid Plugs:</span>
						<span class="mono-num font-bold text-[var(--accent-primary)]">
							{projectData.swirrResult.valid_plug_count} / {projectData.swirrResult.total_plug_count}
						</span>
					</div>
					<div class="flex items-center gap-1.5">
						{#if projectData.swirrConfig.excludedPlugIds.length > 0}
							<span
								class="rounded-xs bg-amber-100 px-1 py-0.5 font-mono text-[9px] font-semibold text-amber-800"
							>
								{projectData.swirrConfig.excludedPlugIds.length} excl
							</span>
							<button
								type="button"
								class="text-[9.5px] font-medium text-blue-700 hover:underline"
								onclick={clearExcludedPlugs}
							>
								Reset
							</button>
						{:else}
							<button
								type="button"
								onclick={() => (workbench.swirrCrossplotTab = 'pc-qc')}
								class="text-[9.5px] font-medium text-blue-700 hover:underline"
							>
								Inspect Pc vs Sw →
							</button>
						{/if}
					</div>
				</div>
			{/if}
		</ParameterSection>

		<!-- Section 2: Swirr Correlation Model & Fit -->
		<ParameterSection
			title="2. Swirr Correlation Fit"
			badge={projectData.swirrConfig.equationType === 'rqi_power'
				? 'a·√(k/φ)^(-b)'
				: projectData.swirrConfig.equationType === 'semilog_k'
					? 'a - b·log10(k)'
					: 'a·φ^b / k^c'}
			defaultOpen={true}
		>
			<div class="mb-2">
				<label
					for="swirr-eq-select"
					class="mb-1 block text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase"
				>
					Correlation Equation
				</label>
				<select
					id="swirr-eq-select"
					class="h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 font-mono text-[10.5px] text-[var(--text-primary)] focus:border-[var(--accent-primary)] focus:outline-none"
					value={projectData.swirrConfig.equationType}
					onchange={(e) => setEquationType(e.currentTarget.value as SwirrEquationType)}
				>
					<option value="rqi_power">RQI Power-Law: a · √(k/φ)^(-b)</option>
					<option value="semilog_k">Semi-Log Perm: a - b · log10(k)</option>
					<option value="timur">Timur: a · (φ^b / k^c)</option>
				</select>
			</div>

			<div class="mb-2">
				<span
					class="mb-1 block text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase"
				>
					Fit Scope
				</span>
				<div class="grid grid-cols-2 gap-1">
					{#each [['global', 'Global Fit (All RTs)'], ['per_rt', 'Per Rock Type (RT 1..N)']] as [sKey, sLbl]}
						<button
							type="button"
							onclick={() => setFitScope(sKey as SwirrFitScope)}
							class="h-6 rounded-sm border text-[10px] font-medium transition-colors {projectData
								.swirrConfig.fitScope === sKey
								? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-50'}"
						>
							{sLbl}
						</button>
					{/each}
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
				/>
				<NumericScrubber
					label="Swirr Max"
					unit="v/v"
					min={0.3}
					max={0.99}
					step={0.01}
					decimals={2}
					bind:value={projectData.swirrConfig.swirrMax}
				/>
			</div>

			<div class="mt-2 mb-1 flex items-center justify-between text-[10px] text-[var(--text-secondary)]">
				<span class="font-semibold text-[var(--text-primary)]">Swirr Fit Coefficients</span>
				<button
					type="button"
					disabled={projectData.isComputingMultiWell}
					onclick={() => {
						workbench.swirrCrossplotTab = 'swirr-corr';
						void runAutoFit(true);
					}}
					class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-primary)] hover:bg-blue-50 hover:text-[var(--accent-primary)] disabled:opacity-50"
				>
					{projectData.isComputingMultiWell ? 'Fitting...' : 'Reset Auto-Fit'}
				</button>
			</div>

			{#if displayedRules.length > 0}
				<div class="overflow-x-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[11px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-6 border-b border-[var(--border-hairline)]">
								<th class="px-1.5 font-semibold">Scope</th>
								<th class="px-1 text-right font-semibold">a</th>
								<th class="px-1 text-right font-semibold">b</th>
								{#if projectData.swirrConfig.equationType === 'timur'}
									<th class="px-1 text-right font-semibold">c</th>
								{/if}
								<th class="px-1.5 text-right font-semibold">RMSE</th>
								<th class="px-1.5 text-right font-semibold">R²</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each displayedRules as [scopeKey, rule] (scopeKey)}
								{@const isGlobal = scopeKey === 'global'}
								{@const rtColor = isGlobal ? '#0F172A' : getRockTypeColor(Number(scopeKey))}
								<tr class="h-6">
									<td class="px-1.5 font-semibold whitespace-nowrap">
										<span
											class="mr-1 inline-block h-2 w-2 rounded-full"
											style="background:{rtColor}"
										></span>{isGlobal ? 'Global' : `RT ${scopeKey}`}
									</td>
									<td class="px-1 text-right">
										<input
											type="number"
											step="0.01"
											value={rule.a}
											onchange={(e) => updateRuleCoeff(scopeKey, 'a', e.currentTarget.value)}
											class="mono-num h-5 w-14 rounded-xs border border-[var(--border-hairline)] px-1 text-right text-[10px]"
										/>
									</td>
									<td class="px-1 text-right">
										<input
											type="number"
											step="0.01"
											value={rule.b}
											onchange={(e) => updateRuleCoeff(scopeKey, 'b', e.currentTarget.value)}
											class="mono-num h-5 w-14 rounded-xs border border-[var(--border-hairline)] px-1 text-right text-[10px]"
										/>
									</td>
									{#if projectData.swirrConfig.equationType === 'timur'}
										<td class="px-1 text-right">
											<input
												type="number"
												step="0.01"
												value={rule.c ?? 0.5}
												onchange={(e) => updateRuleCoeff(scopeKey, 'c', e.currentTarget.value)}
												class="mono-num h-5 w-12 rounded-xs border border-[var(--border-hairline)] px-1 text-right text-[10px]"
											/>
										</td>
									{/if}
									<td class="mono-num px-1.5 text-right text-[10px]">{fmtNum(rule.rmse, 3)}</td>
									<td class="mono-num px-1.5 text-right text-[10px]">{fmtNum(rule.r2, 3)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</ParameterSection>

		<!-- Section 3: SCAL J / Pc Curve Fit -->
		{#if true}
		{@const activeShfModel = projectData.shfParams.model}
		{@const isSwMode = (projectData.shfParams.jSatMode ?? 'swn') === 'sw'}
		{@const activeShfFormula =
			activeShfModel === 'leverett_j'
				? isSwMode
					? 'J = a · Sw^(-b)'
					: 'J = a · Swn^(-b)'
				: SHF_MODEL_META[activeShfModel].formula}
		<ParameterSection
			title="3. SCAL J / Pc Curve Fit"
			badge={activeShfFormula}
			defaultOpen={true}
		>
			<div class="grid grid-cols-3 gap-1">
				{#each [['leverett_j', 'Leverett-J'], ['skelt_harrison', 'Skelt-Harrison'], ['cuddy', 'Cuddy Bvw']] as [m, lbl]}
					<button
						type="button"
						onclick={() => selectShfModel(m as any)}
						class="h-6 rounded-sm border text-[10px] font-medium {projectData.shfParams.model === m
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-50'}"
					>
						{lbl}
					</button>
				{/each}
			</div>

			{#if activeShfModel === 'leverett_j'}
				<div class="mt-2 grid grid-cols-2 gap-2">
					<NumericScrubber
						label="IFT"
						bind:value={projectData.shfParams.ift}
						unit="dyn/cm"
						min={1.0}
						max={80.0}
						step={0.5}
						onchange={() => {
							hasManualShfFitOverride = false;
						}}
					/>
					<NumericScrubber
						label="Contact Angle (θ)"
						bind:value={projectData.shfParams.theta}
						unit="deg"
						min={0}
						max={90}
						step={1.0}
						onchange={() => {
							hasManualShfFitOverride = false;
						}}
					/>
				</div>

				{#if !isSwMode}
					<div class="mt-1.5">
						<div class="mb-1 flex items-center justify-between text-[10px]">
							<span class="font-semibold text-[var(--text-primary)]">
								Swn Swirr Normalization
							</span>
							<span class="mono-num text-[9.5px] text-[var(--text-muted)]">
								{(projectData.shfParams.swirrSource ?? 'poro_perm') === 'poro_perm'
									? 'Modeled Swirr(k, φ)'
									: 'Plug min(Sw)'}
							</span>
						</div>
						<div class="grid grid-cols-2 gap-1">
							{#each [['poro_perm', 'Poro-Perm Correlation'], ['lab_min', 'Lab Terminal Sw']] as [srcKey, srcLabel]}
								<button
									type="button"
									onclick={() => {
										workbench.swirrCrossplotTab = 'scal-fit';
										projectData.shfParams.swirrSource = srcKey as 'lab_min' | 'poro_perm';
										hasManualShfFitOverride = false;
									}}
									class="h-6 rounded-sm border px-1.5 text-[10px] font-medium {(projectData
										.shfParams.swirrSource ?? 'poro_perm') === srcKey
										? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
										: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-50'}"
								>
									{srcLabel}
								</button>
							{/each}
						</div>
					</div>
				{/if}
			{/if}

			{@const totalScalPts = projectData.shfResult?.scal_data?.pc?.length ?? 0}
			{@const excludedScalIds = projectData.swirrConfig.excludedScalPointIds ?? []}
			{@const excludedScalCount = (projectData.shfResult?.scal_data?.excluded ?? []).filter(Boolean).length || excludedScalIds.length}
			{@const activeScalPts = Math.max(0, totalScalPts - excludedScalCount)}
			<div class="mt-2 mb-1.5 flex items-center justify-between text-[10px] text-[var(--text-secondary)]">
				<span class="flex items-center gap-1">
					SCAL points:
					<strong class="mono-num text-[var(--text-primary)]">
						{activeScalPts} / {totalScalPts}
					</strong>
					{#if excludedScalCount > 0}
						<span
							class="rounded-xs bg-amber-100 px-1 py-0.5 font-mono text-[9px] font-semibold text-amber-800"
						>
							{excludedScalCount} excl
						</span>
					{/if}
				</span>
				<div class="flex items-center gap-1">
					{#if activeShfModel === 'leverett_j'}
						<div
							class="inline-flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)]"
						>
							{#each [['swn', 'J vs Swn'], ['sw', 'J vs Sw']] as [sKey, sLbl]}
								<button
									type="button"
									onclick={() => {
										workbench.swirrCrossplotTab = 'scal-fit';
										projectData.setShfJSatMode(sKey as 'swn' | 'sw');
										hasManualShfFitOverride = false;
									}}
									class="px-1.5 py-0.5 text-[10px] font-medium first:border-r first:border-[var(--border-hairline)] {(projectData
										.shfParams.jSatMode ?? 'swn') === sKey
										? 'bg-blue-600 font-semibold text-white'
										: 'text-[var(--text-secondary)] hover:bg-slate-100'}"
								>
									{sLbl}
								</button>
							{/each}
						</div>
					{/if}
					<button
						type="button"
						onclick={() => {
							workbench.swirrCrossplotTab = 'scal-fit';
							hasManualShfFitOverride = false;
							projectData.recomputeScalFitsFromPlugs(true);
						}}
						class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-primary)] hover:bg-blue-50 hover:text-[var(--accent-primary)]"
					>
						Reset Auto-Fit
					</button>
				</div>
			</div>

			<!-- SCAL Data Point & Core Sample QC & Outlier Filtering Controls -->
			{@const allPlugs = projectData.swirrResult?.plugs ?? []}
			{@const excludedPlugIds = projectData.swirrConfig.excludedPlugIds ?? []}
			{@const activePlugCount = Math.max(0, allPlugs.length - excludedPlugIds.length)}
			<div
				class="mb-2 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-1.5 text-[10px]"
			>
				<div class="mb-1.5 flex items-center justify-between gap-1">
					<span class="font-semibold text-[var(--text-primary)]">
						SCAL Point & Core Sample Filter
					</span>
					<div
						class="inline-flex rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)]"
						title="Choose whether clicking a point on the SCAL crossplot toggles that individual point or its entire core sample (or hold Shift+Click for core sample)"
					>
						{#each [['point', 'Click: Point'], ['plug', 'Click: Sample']] as [tKey, tLbl]}
							<button
								type="button"
								onclick={() => {
									projectData.scalClickFilterTarget = tKey as 'point' | 'plug';
								}}
								class="px-1.5 py-0.5 text-[9px] font-medium first:border-r first:border-[var(--border-hairline)] {projectData.scalClickFilterTarget ===
								tKey
									? 'bg-slate-800 font-semibold text-white'
									: 'text-[var(--text-secondary)] hover:bg-slate-100'}"
							>
								{tLbl}
							</button>
						{/each}
					</div>
				</div>
				<div class="flex flex-wrap items-center gap-1">
					<button
						type="button"
						onclick={() => {
							workbench.swirrCrossplotTab = 'scal-fit';
							hasManualShfFitOverride = false;
							projectData.autoFilterCorruptedScalPoints('all');
						}}
						class="rounded-xs border border-amber-300 bg-amber-50 px-1.5 py-0.5 text-[9.5px] font-semibold text-amber-900 hover:bg-amber-100"
						title="Automatically detect and exclude non-monotonic Pc steps, high log-J outliers, and boundary asymptotes"
					>
						Auto-Filter Corrupted / Outliers
					</button>
					<button
						type="button"
						onclick={() => {
							workbench.swirrCrossplotTab = 'scal-fit';
							hasManualShfFitOverride = false;
							projectData.autoFilterCorruptedScalPoints('asymptotes');
						}}
						class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[9.5px] font-medium text-[var(--text-secondary)] hover:bg-slate-100"
						title="Exclude extreme saturation asymptote pile-ups (Swn ≤ 0.035 or Swn ≥ 0.94)"
					>
						Trim Asymptotes
					</button>
					{#if excludedScalIds.length > 0}
						<button
							type="button"
							onclick={() => {
								hasManualShfFitOverride = false;
								projectData.clearExcludedScalPoints();
							}}
							class="ml-auto text-[9.5px] font-semibold text-blue-700 hover:underline"
						>
							Restore Pts ({excludedScalIds.length})
						</button>
					{/if}
					{#if excludedPlugIds.length > 0}
						<button
							type="button"
							onclick={clearExcludedPlugs}
							class="text-[9.5px] font-semibold text-blue-700 hover:underline"
						>
							Restore Samples ({excludedPlugIds.length})
						</button>
					{/if}
				</div>

				{#if projectData.selectedScalPointIds.length > 0}
					<div
						class="mt-1.5 flex items-center justify-between rounded-xs border border-amber-300 bg-amber-100/70 px-1.5 py-1 text-[9.5px] text-amber-950"
					>
						<span class="font-semibold">
							{projectData.selectedScalPointIds.length} pts selected
						</span>
						<div class="flex items-center gap-1">
							<button
								type="button"
								onclick={() => {
									hasManualShfFitOverride = false;
									projectData.excludeSelectedScalPoints();
								}}
								class="rounded-xs bg-rose-600 px-1.5 py-0.5 font-semibold text-white hover:bg-rose-700"
							>
								Exclude Pts
							</button>
							<button
								type="button"
								onclick={() => {
									hasManualShfFitOverride = false;
									projectData.excludeSelectedScalPlugs();
								}}
								class="rounded-xs border border-amber-400 bg-white px-1.5 py-0.5 font-semibold text-amber-900 hover:bg-amber-50"
							>
								Exclude Samples
							</button>
							<button
								type="button"
								onclick={() => projectData.setSelectedScalPoints([])}
								class="px-1 text-slate-600 hover:underline"
							>
								×
							</button>
						</div>
					</div>
				{/if}

				{#if allPlugs.length > 0}
					<details class="mt-1.5 border-t border-[var(--border-hairline)] pt-1">
						<summary
							class="flex cursor-pointer select-none items-center justify-between text-[9.5px] font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
						>
							<span>
								Filter Core Samples / Plugs (<strong class="mono-num text-[var(--text-primary)]"
									>{activePlugCount} / {allPlugs.length}</strong
								> active)
							</span>
							{#if excludedPlugIds.length > 0}
								<span class="rounded-xs bg-amber-100 px-1 py-0.2 font-mono text-[8.5px] font-semibold text-amber-800">
									{excludedPlugIds.length} sample{excludedPlugIds.length > 1 ? 's' : ''} excl
								</span>
							{/if}
						</summary>
						<div class="mt-1 max-h-36 overflow-y-auto rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)]">
							<table class="w-full border-collapse text-left text-[9.5px]">
								<thead class="sticky top-0 bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
									<tr class="h-5 border-b border-[var(--border-hairline)]">
										<th class="w-5 px-1 text-center font-semibold">Use</th>
										<th class="px-1 font-semibold">Sample</th>
										<th class="px-1 font-semibold">Well</th>
										<th class="px-1 text-center font-semibold">RT</th>
										<th class="px-1 text-right font-semibold">φ</th>
										<th class="px-1 text-right font-semibold">k (mD)</th>
										<th class="px-1 text-right font-semibold">Pts</th>
									</tr>
								</thead>
								<tbody class="divide-y divide-[var(--border-hairline)]">
									{#each allPlugs as plug (plug.plug_id)}
										{@const isExcl =
											Boolean(plug.excluded) || excludedPlugIds.includes(plug.plug_id)}
										{@const rtColor = getRockTypeColor(plug.rt ?? 1)}
										<tr
											class="h-5 cursor-pointer hover:bg-slate-50 {isExcl
												? 'bg-slate-50/70 text-slate-400 line-through'
												: ''}"
											onclick={() => {
												hasManualRuleOverride = false;
												hasManualShfFitOverride = false;
												projectData.toggleSwirrPlugExclusion(plug.plug_id);
											}}
										>
											<td class="px-1 text-center">
												<input
													type="checkbox"
													checked={!isExcl}
													onclick={(e) => e.stopPropagation()}
													onchange={() => {
														hasManualRuleOverride = false;
														hasManualShfFitOverride = false;
														projectData.toggleSwirrPlugExclusion(plug.plug_id);
													}}
													class="h-3 w-3 accent-blue-600"
												/>
											</td>
											<td class="mono-num px-1 font-semibold">{plug.sample}</td>
											<td class="truncate px-1 max-w-[70px]" title={plug.well}>{plug.well}</td>
											<td class="px-1 text-center whitespace-nowrap">
												<span
													class="mr-0.5 inline-block h-1.5 w-1.5 rounded-full"
													style="background:{rtColor}"
												></span>{plug.rt ?? 1}
											</td>
											<td class="mono-num px-1 text-right">{fmtNum(plug.cpore, 3)}</td>
											<td class="mono-num px-1 text-right">{fmtNum(plug.cperm, 1)}</td>
											<td class="mono-num px-1 text-right">{plug.pc_curve?.length ?? 0}</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					</details>
				{/if}

				{#if excludedScalIds.length > 0}
					<details class="mt-1">
						<summary
							class="cursor-pointer select-none text-[9.5px] font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
						>
							Excluded Individual Points ({excludedScalIds.length}) — click chip to restore
						</summary>
						<div class="mt-1 flex max-h-24 flex-wrap gap-1 overflow-y-auto pt-0.5">
							{#each excludedScalIds as ptId (ptId)}
								{@const parts = ptId.split('::')}
								{@const shortLabel =
									parts.length >= 3
										? `${parts[1]}#${Number(parts[2]) + 1}`
										: ptId}
								<button
									type="button"
									onclick={() => {
										hasManualShfFitOverride = false;
										projectData.toggleScalPointExclusion(ptId);
									}}
									class="inline-flex items-center gap-0.5 rounded-xs border border-slate-300 bg-white px-1 py-0.5 font-mono text-[9px] text-slate-700 hover:border-blue-400 hover:bg-blue-50 hover:text-blue-800"
									title={`Restore ${ptId}`}
								>
									<span>{shortLabel}</span>
									<span class="text-slate-400">×</span>
								</button>
							{/each}
						</div>
					</details>
				{/if}
			</div>

			{#if projectData.shfResult?.fits && Object.keys(projectData.shfResult.fits).length > 0}
				<div class="overflow-x-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[11px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-6 border-b border-[var(--border-hairline)]">
								<th class="px-1.5 font-semibold">RT</th>
								<th class="px-1 text-right font-semibold">a</th>
								<th class="px-1 text-right font-semibold">b</th>
								{#if activeShfModel === 'skelt_harrison'}
									<th class="px-1 text-right font-semibold">c</th>
									<th class="px-1 text-right font-semibold">d</th>
								{/if}
								<th class="px-1.5 text-right font-semibold">RMSE</th>
								{#if activeShfModel !== 'skelt_harrison'}
									<th class="px-1.5 text-right font-semibold">R²</th>
								{/if}
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each Object.entries(projectData.shfResult.fits) as [rf, fit]}
								{@const rtColor = getRockTypeColor(Number(rf))}
								<tr class="h-6">
									<td class="px-1.5 font-semibold whitespace-nowrap">
										<span
											class="mr-1 inline-block h-2 w-2 rounded-full"
											style="background:{rtColor}"
										></span>RT {rf}
									</td>
									<td class="px-1 text-right">
										<input
											type="number"
											step="0.005"
											value={fit.a}
											onchange={(e) =>
												updateShfFitParam(
													rf,
													'a',
													(e.currentTarget as HTMLInputElement).value
												)}
											class="mono-num h-5 w-14 rounded-xs border border-[var(--border-hairline)] px-1 text-right text-[10px]"
										/>
									</td>
									<td class="px-1 text-right">
										<input
											type="number"
											step="0.01"
											value={fit.b}
											onchange={(e) =>
												updateShfFitParam(
													rf,
													'b',
													(e.currentTarget as HTMLInputElement).value
												)}
											class="mono-num h-5 w-14 rounded-xs border border-[var(--border-hairline)] px-1 text-right text-[10px]"
										/>
									</td>
									{#if activeShfModel === 'skelt_harrison'}
										<td class="px-1 text-right">
											<input
												type="number"
												step="0.05"
												value={fit.c ?? 1.5}
												onchange={(e) =>
													updateShfFitParam(
														rf,
														'c',
														(e.currentTarget as HTMLInputElement).value
													)}
												class="mono-num h-5 w-12 rounded-xs border border-[var(--border-hairline)] px-1 text-right text-[10px]"
											/>
										</td>
										<td class="px-1 text-right">
											<input
												type="number"
												step="0.1"
												value={fit.d ?? 0.5}
												onchange={(e) =>
													updateShfFitParam(
														rf,
														'd',
														(e.currentTarget as HTMLInputElement).value
													)}
												class="mono-num h-5 w-12 rounded-xs border border-[var(--border-hairline)] px-1 text-right text-[10px]"
											/>
										</td>
									{/if}
									<td class="mono-num px-1.5 text-right text-[10px]">{fmtNum(fit.rmse, 3)}</td>
									{#if activeShfModel !== 'skelt_harrison'}
										<td class="mono-num px-1.5 text-right text-[10px]">{fmtNum(fit.r2, 3)}</td>
									{/if}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</ParameterSection>
		{/if}

		<!-- Section 4: Apply & Propagate SWIRR to Logs -->
		<ParameterSection title="4. Apply & Propagate SWIRR to Logs" badge="Track 7 SWIRR" defaultOpen={true}>
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
							class="h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 font-mono text-[10px] text-[var(--text-primary)]"
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
							class="h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 font-mono text-[10px] text-[var(--text-primary)]"
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
							class="h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 font-mono text-[10px] text-[var(--text-primary)]"
						>
							{#each availableRtCurves as c (c)}
								<option value={c}>{c}</option>
							{/each}
						</select>
					</div>
				</div>

				{#if projectData.swirrApplyResult}
					<div
						class="rounded-sm border border-emerald-300 bg-emerald-50/80 px-2 py-1.5 text-[10px] text-emerald-950"
					>
						<div class="flex items-center justify-between font-semibold">
							<span>SWIRR Applied ({projectData.swirrApplyResult.well_summaries.length} wells)</span>
							<span class="mono-num">
								Mean: {projectData.swirrApplyResult.mean_swirr?.toFixed(3) ?? '—'}
							</span>
						</div>
						<div class="mt-0.5 flex items-center justify-between text-[9.5px] text-emerald-800">
							<span class="mono-num">
								Range: {projectData.swirrApplyResult.min_swirr?.toFixed(3) ?? '—'} – {projectData.swirrApplyResult.max_swirr?.toFixed(3) ?? '—'}
							</span>
						</div>
					</div>
				{/if}
			</div>
		</ParameterSection>
		{:else}
		<!-- Mode 2: Rel-Perm & SWOF (Dynamic) -->
		<ParameterSection
			title="1. Relative Permeability Model"
			badge={workbench.scalParams.model.toUpperCase()}
			defaultOpen={true}
		>
			<div class="grid grid-cols-3 gap-1">
				{#each [['let', 'LET (3-Param)'], ['corey', 'Corey'], ['brooks_corey', 'Brooks-Corey']] as [m, lbl]}
					<button
						type="button"
						onclick={() => {
							workbench.scalParams.model = m as any;
						}}
						class="h-6 rounded-sm border text-[10.5px] font-medium {workbench.scalParams.model ===
						m
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-50'}"
					>
						{lbl}
					</button>
				{/each}
			</div>

			{#if scalSwirrPresets.length > 0}
				<div class="mt-2 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-1.5">
					<div class="mb-1 flex items-center justify-between text-[9.5px] text-[var(--text-secondary)]">
						<span class="font-semibold">Sync Sw,irr from SCAL Plugs:</span>
						<span class="mono-num text-[9px] text-[var(--text-muted)]">Click to apply</span>
					</div>
					<div class="flex flex-wrap gap-1">
						{#each scalSwirrPresets as preset (preset.key)}
							<button
								type="button"
								onclick={() => {
									workbench.scalParams.swirr = preset.value;
								}}
								class="mono-num inline-flex h-5 items-center gap-1 rounded-xs border px-1.5 text-[9.5px] font-medium transition-colors {Math.abs(
									workbench.scalParams.swirr - preset.value
								) < 0.001
									? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
									: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100'}"
							>
								{#if preset.rt !== undefined}
									<span
										class="inline-block h-1.5 w-1.5 rounded-full"
										style="background:{getRockTypeColor(preset.rt)}"
									></span>
								{/if}
								{preset.label}
							</button>
						{/each}
					</div>
				</div>
			{/if}

			<NumericScrubber
				label="Connate Water (Sw,irr)"
				bind:value={workbench.scalParams.swirr}
				unit="v/v"
				min={0.01}
				max={0.5}
				step={0.01}
			/>
			<NumericScrubber
				label="Residual Oil (Sor)"
				bind:value={workbench.scalParams.sor}
				unit="v/v"
				min={0.01}
				max={0.5}
				step={0.01}
			/>
			<NumericScrubber
				label="Max Water Rel-Perm (krw,max)"
				bind:value={workbench.scalParams.krw_max}
				unit="frac"
				min={0.05}
				max={1.0}
				step={0.02}
			/>
			<NumericScrubber
				label="Max Oil Rel-Perm (kro,max)"
				bind:value={workbench.scalParams.kro_max}
				unit="frac"
				min={0.1}
				max={1.0}
				step={0.02}
			/>

			{#if workbench.scalParams.model === 'corey'}
				<NumericScrubber
					label="Water Corey Exponent (nw)"
					bind:value={workbench.scalParams.nw}
					min={1.0}
					max={8.0}
					step={0.1}
				/>
				<NumericScrubber
					label="Oil Corey Exponent (no)"
					bind:value={workbench.scalParams.no}
					min={1.0}
					max={8.0}
					step={0.1}
				/>
			{:else if workbench.scalParams.model === 'brooks_corey'}
				<NumericScrubber
					label="Pore-Size Distribution (λ)"
					bind:value={workbench.scalParams.lam}
					min={0.3}
					max={6.0}
					step={0.1}
				/>
			{:else}
				<div class="grid grid-cols-2 gap-x-2">
					<NumericScrubber
						label="Water L (Lw)"
						bind:value={workbench.scalParams.lw}
						min={0.5}
						max={6.0}
						step={0.1}
					/>
					<NumericScrubber
						label="Oil L (Lo)"
						bind:value={workbench.scalParams.lo}
						min={0.5}
						max={6.0}
						step={0.1}
					/>
					<NumericScrubber
						label="Water E (Ew)"
						bind:value={workbench.scalParams.ew}
						min={0.1}
						max={10.0}
						step={0.2}
					/>
					<NumericScrubber
						label="Oil E (Eo)"
						bind:value={workbench.scalParams.eo}
						min={0.1}
						max={10.0}
						step={0.2}
					/>
					<NumericScrubber
						label="Water T (Tw)"
						bind:value={workbench.scalParams.tw}
						min={0.5}
						max={6.0}
						step={0.1}
					/>
					<NumericScrubber
						label="Oil T (To)"
						bind:value={workbench.scalParams.to}
						min={0.5}
						max={6.0}
						step={0.1}
					/>
				</div>
			{/if}

			<div class="grid grid-cols-2 gap-x-2">
				<NumericScrubber
					label="Water Viscosity (μw)"
					bind:value={workbench.scalParams.mu_w}
					unit="cP"
					min={0.1}
					max={5.0}
					step={0.1}
				/>
				<NumericScrubber
					label="Oil Viscosity (μo)"
					bind:value={workbench.scalParams.mu_o}
					unit="cP"
					min={0.1}
					max={50.0}
					step={0.2}
				/>
			</div>
		</ParameterSection>

		<ParameterSection
			title="2. Eclipse SWOF Simulation Deck"
			badge={`${workbench.scalResult?.swofTable.length ?? 0} rows`}
			defaultOpen={true}
		>
			<NumericScrubber
				label="Entry Capillary Pressure (Pd)"
				bind:value={workbench.scalParams.pc_entry}
				unit="psi"
				min={0.1}
				max={25.0}
				step={0.2}
			/>
			<NumericScrubber
				label="Capillary Exponent (1/λ)"
				bind:value={workbench.scalParams.pc_exponent}
				min={0.5}
				max={6.0}
				step={0.1}
			/>
			<NumericScrubber
				label="Max Capillary Pressure"
				bind:value={workbench.scalParams.pc_max}
				unit="psi"
				min={5.0}
				max={200.0}
				step={5.0}
			/>

			<div class="mt-1.5 flex gap-1.5">
				<button
					type="button"
					disabled={!workbench.scalResult?.swofKeyword}
					onclick={copySwofKeyword}
					class="h-7 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-xs font-semibold text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-40"
				>
					{copiedSwof ? 'Copied SWOF ✓' : 'Copy SWOF Keyword'}
				</button>
				<button
					type="button"
					disabled={!workbench.scalResult?.swofKeyword}
					onclick={downloadSwofInc}
					class="h-7 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs font-semibold text-[var(--accent-primary)] hover:bg-blue-50 disabled:opacity-40"
				>
					Download SWOF.INC
				</button>
			</div>
		</ParameterSection>
		{/if}

		{#if projectData.error}
			<div
				class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]"
			>
				{projectData.error}
			</div>
		{/if}
	</div>

	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		{#if workbench.scalSubMode === 'static'}
			<button
				type="button"
				disabled={projectData.isComputingMultiWell || isApplying}
				onclick={computeSwirrAndPc}
				class="mr-1.5 h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
			>
				{isApplying ? 'Computing...' : 'Compute Swirr & Pc'}
			</button>
			<button
				type="button"
				disabled={projectData.isComputingMultiWell || !projectData.hasPendingMultiWell('swirr-corr')}
				onclick={saveAllWells}
				class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
			>
				Save All Wells
			</button>
		{:else}
			<button
				type="button"
				disabled={projectData.isComputingMultiWell}
				onclick={computeScal}
				class="mr-1.5 h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
			>
				{projectData.isComputingMultiWell ? 'Computing...' : 'Compute Rel-Perm'}
			</button>
			<button
				type="button"
				disabled={projectData.isComputingMultiWell || !relpermPendingSave}
				onclick={saveRelpermParams}
				class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
			>
				Save Params
			</button>
		{/if}
	</div>
</div>
