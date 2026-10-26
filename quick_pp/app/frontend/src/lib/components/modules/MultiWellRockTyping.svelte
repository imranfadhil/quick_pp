<script lang="ts">
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import {
		projectData,
		type DrtCutoffMethod,
		type DrtMethod,
		type RockTypingPlotView
	} from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { getRockTypeColor } from '$lib/utils/petrophysicsColors';

	const DRT_METHOD_META: Record<
		DrtMethod,
		{
			label: string;
			shortName: string;
			badge: string;
			cutoffTitle: string;
			cutoffLabel: string;
			addLabel: string;
			computeLabel: string;
			unit: string;
			min: number;
			max: number;
			step: number;
			formula: string;
			description: string;
			higherIsBetter: boolean;
		}
	> = {
		fzi: {
			label: 'FZI / RQI',
			shortName: 'FZI',
			badge: 'Amaefule (1993)',
			cutoffTitle: 'FZI / RQI Boundary Cutoffs',
			cutoffLabel: 'FZI Cutoff',
			addLabel: '+ Add FZI Boundary',
			computeLabel: 'Compute FZI (⌘+Enter)',
			unit: '',
			min: 0.01,
			max: 50.0,
			step: 0.05,
			formula: 'FZI = 0.0314·√(K/Φ) / [Φ/(1−Φ)]',
			description:
				'Kozeny-Carman Hydraulic Flow Units (HFU). Higher FZI indicates larger hydraulic radius and better reservoir quality (DRT 1 = highest FZI).',
			higherIsBetter: true
		},
		winland: {
			label: 'Winland R35',
			shortName: 'R35',
			badge: 'Winland (1972)',
			cutoffTitle: 'Winland R35 Boundary Cutoffs',
			cutoffLabel: 'R35 Cutoff',
			addLabel: '+ Add R35 Boundary',
			computeLabel: 'Compute Winland R35 (⌘+Enter)',
			unit: 'μm',
			min: 0.02,
			max: 100.0,
			step: 0.1,
			formula: 'log₁₀(R35) = 0.732 + 0.588·log₁₀(K) − 0.864·log₁₀(100·Φ)',
			description:
				'Pore throat radius R35 (μm) at 35% mercury saturation. Classifies pore throats from nanoporous (<0.5 μm) to megaporous (>10 μm) (DRT 1 = largest R35).',
			higherIsBetter: true
		},
		lucia: {
			label: 'Lucia RFN',
			shortName: 'RFN',
			badge: 'Jennings-Lucia (2003)',
			cutoffTitle: 'Lucia RFN Boundary Cutoffs',
			cutoffLabel: 'RFN Cutoff',
			addLabel: '+ Add RFN Boundary',
			computeLabel: 'Compute Lucia RFN (⌘+Enter)',
			unit: 'RFN',
			min: 0.5,
			max: 4.0,
			step: 0.1,
			formula: 'log₁₀(K) = (9.789−12.084·log RFN) + (8.671−8.297·log RFN)·log₁₀(Φᵢₚ)',
			description:
				'Carbonate Rock Fabric Number (RFN 0.5–4.0): Class 1 Grainstone (0.5–1.5), Class 2 Packstone (1.5–2.5), Class 3 Mud-Dominated (2.5–4.0) (DRT 1 = lowest RFN).',
			higherIsBetter: false
		}
	};

	let activeMeta = $derived(DRT_METHOD_META[projectData.drtMethod]);
	let diagnostics = $derived(projectData.rockTypingResult?.diagnostics ?? null);
	let carbonateDiag = $derived(diagnostics?.carbonate ?? null);
	let lorenzDiag = $derived(diagnostics?.lorenz ?? null);
	let gmmDiag = $derived(diagnostics?.gmm ?? null);
	let classifierDiag = $derived(diagnostics?.classifier ?? null);
	let activeKScores = $derived<Array<Record<string, any>>>(
		projectData.drtCutoffMethod === 'lorenz'
			? (lorenzDiag?.k_scores ?? diagnostics?.k_scores ?? [])
			: (gmmDiag?.k_scores ?? diagnostics?.k_scores ?? [])
	);

	let hasSonicCurvesInProject = $derived(
		Boolean(
			carbonateDiag?.has_sonic_data ||
				projectData.allCurves.some((c) =>
					['DT', 'DTC', 'AC', 'DTCO', 'PHIS', 'PHI_SONIC', 'PHI_SEC', 'PHIS_SEC', 'SPI'].includes(
						c.toUpperCase()
					)
				)
		)
	);

	async function selectDrtMethod(m: DrtMethod) {
		if (projectData.drtMethod === m && projectData.rockTypingResult?.method === m) return;
		projectData.drtMethod = m;
		projectData.drtClassifierPropagated = false;
		if (workbench.projectId) {
			await executeRockTyping(false, Boolean(classifierDiag?.trained), false);
		}
	}

	function selectCutoffMethod(cm: DrtCutoffMethod) {
		projectData.drtCutoffMethod = cm;
		if (cm === 'lorenz' && projectData.rockTypingPlotView === 'gmm') {
			projectData.rockTypingPlotView = 'lorenz';
		} else if (cm === 'gmm' && projectData.rockTypingPlotView === 'lorenz') {
			projectData.rockTypingPlotView = 'gmm';
		}
	}

	async function selectCandidateK(k: number) {
		if (!Number.isFinite(k) || k < 2) return;
		projectData.drtAutoK = false;
		const targetCutoffsCount = k - 1;
		const cur = [...projectData.activeDrtCutoffs];
		while (cur.length < targetCutoffsCount) {
			const last = cur[cur.length - 1] ?? 1.0;
			cur.push(Number((last * 1.5).toFixed(2)));
		}
		if (cur.length > targetCutoffsCount) {
			cur.length = targetCutoffsCount;
		}
		projectData.activeDrtCutoffs = cur;
		await executeRockTyping(true, Boolean(classifierDiag?.trained), false);
	}

	function addCutoff() {
		const cur = projectData.activeDrtCutoffs;
		const last = cur[cur.length - 1] ?? 1.0;
		const nextVal =
			projectData.drtMethod === 'lucia'
				? Math.min(4.0, Number((last + 0.5).toFixed(2)))
				: Number((last * 1.8).toFixed(2));
		projectData.activeDrtCutoffs = [...cur, nextVal];
	}

	function removeCutoff(idx: number) {
		const cur = projectData.activeDrtCutoffs;
		if (cur.length <= 1) return;
		projectData.activeDrtCutoffs = cur.filter((_, i) => i !== idx);
	}

	function formatDrtRange(drtStr: string): string {
		const drtNum = Number(drtStr);
		const sorted = [...projectData.activeDrtCutoffs].sort((a, b) => a - b);
		const n = sorted.length;
		if (!Number.isFinite(drtNum) || n === 0) return '';
		const meta = activeMeta;
		const u = meta.unit ? ` ${meta.unit}` : '';
		if (meta.higherIsBetter) {
			const upperIdx = n - (drtNum - 1);
			const lowerIdx = n - drtNum;
			if (drtNum === 1) return `${meta.shortName} ≥ ${sorted[n - 1]}${u}`;
			if (drtNum === n + 1) return `${meta.shortName} < ${sorted[0]}${u}`;
			if (lowerIdx >= 0 && upperIdx < n) {
				return `${sorted[lowerIdx]}–${sorted[upperIdx]}${u}`;
			}
		} else {
			if (drtNum === 1) return `${meta.shortName} < ${sorted[0]}`;
			if (drtNum === n + 1) return `${meta.shortName} ≥ ${sorted[n - 1]}`;
			if (drtNum - 2 >= 0 && drtNum - 1 < n) {
				return `${sorted[drtNum - 2]}–${sorted[drtNum - 1]}`;
			}
		}
		return '';
	}

	async function executeRockTyping(
		autoFindCutoffs: boolean = false,
		trainClassifier: boolean = false,
		applyPropagation: boolean = false
	) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const method = projectData.drtMethod;
			const meta = DRT_METHOD_META[method];
			const sortedCutoffs = [...projectData.activeDrtCutoffs].sort((a, b) => a - b);
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/rock_typing/compute',
				{
					params: { path: { project_id: workbench.projectId } },
					body: {
						well_names:
							projectData.selectedWellNames.length > 0 &&
							projectData.selectedWellNames.length < projectData.wells.length
								? projectData.selectedWellNames
								: null,
						method,
						cutoffs: sortedCutoffs,
						higher_is_better: meta.higherIsBetter,
						auto_find_cutoffs: autoFindCutoffs,
						cutoff_method: projectData.drtCutoffMethod,
						auto_k: projectData.drtAutoK,
						gmm_criterion: projectData.drtGmmCriterion,
						max_clusters: projectData.drtMaxClusters,
						no_of_cutoffs: sortedCutoffs.length,
						use_interparticle_porosity: projectData.useInterparticlePorosity,
						lucia_vuggy_override: projectData.luciaVuggyOverride,
						phi_sec_cutoff: projectData.phiSecCutoff,
						lucia_cutoffs: [...projectData.luciaCutoffs].sort((a, b) => a - b),
						dt_matrix: projectData.dtMatrix,
						dt_fluid: projectData.dtFluid,
						train_classifier: trainClassifier || applyPropagation,
						apply_classifier_propagation: applyPropagation,
						propagation_mode: projectData.drtPropagationMode,
						classifier_model: projectData.drtClassifierModel,
						include_pef: projectData.drtIncludePef,
						include_vdl: projectData.drtIncludeVdl,
						test_size: projectData.drtClassifierTestSize,
						save: false
					} as any
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Rock typing calculation failed'));
			}
			if (autoFindCutoffs && Array.isArray(data.cutoffs) && data.cutoffs.length > 0) {
				projectData.activeDrtCutoffs = data.cutoffs.map((c: number) =>
					Number(Number(c).toFixed(3))
				);
			}
			projectData.drtClassifierPropagated = Boolean(
				(data as any).diagnostics?.classifier?.propagated
			);
			const counts = (data as any).rock_type_counts ?? (data as any).flag_counts ?? {};
			projectData.rockTypingResult = {
				method: (data as any).method ?? method,
				optimal_k: (data as any).optimal_k,
				selected_k: (data as any).selected_k,
				cutoff_method: (data as any).cutoff_method ?? projectData.drtCutoffMethod,
				diagnostics: (data as any).diagnostics ?? null,
				...(data.data as any),
				rock_type_counts: counts
			};
			projectData.markMultiWellComputed('rock-typing');

			if (trainClassifier && !applyPropagation) {
				setPlotView('classifier');
			}

			// Update active well's ROCK_FLAG & indicator curves in wellData so LogTrackCanvas updates in real time
			if (workbench.wellId && data.data?.well_names) {
				const wnArr = data.data.well_names;
				const dArr = data.data.depths;
				const rfArr = data.data.rock_flags;
				const rfMlArr = (data.data as any).rock_flags_ml;
				const vdlArr = (data.data as any).vdl;
				const fziArr = data.data.fzi;
				const r35Arr = (data.data as any).r35;
				const rfnArr = (data.data as any).rfn;
				const phiSecArr = (data.data as any).phi_sec;
				const wDepths: number[] = [];
				const wRf: (number | null)[] = [];
				const wRfMl: (number | null)[] = [];
				const wVdl: (number | null)[] = [];
				const wFzi: (number | null)[] = [];
				const wR35: (number | null)[] = [];
				const wRfn: (number | null)[] = [];
				const wPhiSec: (number | null)[] = [];
				for (let i = 0; i < wnArr.length; i++) {
					if (wnArr[i] === workbench.wellId) {
						wDepths.push(dArr[i]);
						wRf.push(rfArr[i]);
						wRfMl.push(rfMlArr?.[i] ?? null);
						wVdl.push(vdlArr?.[i] ?? null);
						wFzi.push(fziArr?.[i] ?? null);
						wR35.push(r35Arr?.[i] ?? null);
						wRfn.push(rfnArr?.[i] ?? null);
						wPhiSec.push(phiSecArr?.[i] ?? null);
					}
				}
				if (wDepths.length > 0) {
					const mergedPayload: Record<string, (number | null)[]> = {
						DEPTH: wDepths,
						ROCK_FLAG: wRf,
						FZI: wFzi,
						R35: wR35,
						RFN: wRfn
					};
					if (wPhiSec.some((v) => v !== null)) {
						mergedPayload.PHI_SEC = wPhiSec;
					}
					if (wRfMl.some((v) => v !== null)) {
						mergedPayload.ROCK_FLAG_ML = wRfMl;
					}
					if (wVdl.some((v) => v !== null)) {
						mergedPayload.VDL = wVdl;
					}
					wellData.mergeComputedCurves(mergedPayload);
				}
				const logWnArr = (data.data as any).log_well_names as string[] | undefined;
				const logDArr = (data.data as any).log_depths as number[] | undefined;
				const logRfArr = (data.data as any).log_rock_flags as (number | null)[] | undefined;
				const logRfMlArr = (data.data as any).log_rock_flags_ml as (number | null)[] | undefined;
				const logVdlArr = (data.data as any).log_vdl as (number | null)[] | undefined;
				if (Array.isArray(logWnArr) && Array.isArray(logDArr) && logWnArr.length > 0) {
					const lDepths: number[] = [];
					const lRf: (number | null)[] = [];
					const lRfMl: (number | null)[] = [];
					const lVdl: (number | null)[] = [];
					for (let i = 0; i < logWnArr.length; i++) {
						if (logWnArr[i] === workbench.wellId) {
							lDepths.push(logDArr[i]);
							lRf.push(logRfArr?.[i] ?? null);
							lRfMl.push(logRfMlArr?.[i] ?? null);
							lVdl.push(logVdlArr?.[i] ?? null);
						}
					}
					if (lDepths.length > 0) {
						const logPayload: Record<string, (number | null)[]> = {
							DEPTH: lDepths
						};
						if (lRf.some((v) => v !== null)) {
							logPayload.ROCK_FLAG = lRf;
						}
						if (lRfMl.some((v) => v !== null)) {
							logPayload.ROCK_FLAG_ML = lRfMl;
						}
						if (lVdl.some((v) => v !== null)) {
							logPayload.VDL = lVdl;
						}
						wellData.mergeComputedCurves(logPayload);
					}
				}
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Rock typing error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	function setPlotView(view: RockTypingPlotView) {
		workbench.crossplotTemplateOverride = 'phi-k';
		projectData.rockTypingPlotView = view;
	}

	async function saveRockTyping() {
		await projectData.saveMultiWellComputed('rock-typing');
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection title="Rock Typing Method (DRT)" badge={activeMeta.badge}>
			<div class="grid grid-cols-3 gap-1">
				{#each [['fzi', 'FZI / RQI'], ['winland', 'Winland R35'], ['lucia', 'Lucia RFN']] as [m, lbl]}
					<button
						type="button"
						onclick={() => selectDrtMethod(m as DrtMethod)}
						class="h-6 rounded-sm border text-[10px] font-medium transition-colors {projectData.drtMethod ===
						m
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
					>
						{lbl}
					</button>
				{/each}
			</div>

			<div
				class="mt-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10px]"
			>
				<div class="mono-num font-semibold text-[var(--text-primary)]">
					{activeMeta.formula}
				</div>
				<p class="mt-0.5 leading-snug text-[var(--text-secondary)]">
					{activeMeta.description}
				</p>
			</div>

			<!-- Quick Crossplot View Switcher -->
			<div class="mt-1.5 grid grid-cols-5 gap-1">
				{#each [['phi-k', 'Φ–K'], ['lorenz', 'Lorenz'], ['gmm', 'GMM'], ['carbonate', 'Φ_sec'], ['classifier', 'ML Matrix']] as [vKey, vLbl]}
					<button
						type="button"
						onclick={() => setPlotView(vKey as RockTypingPlotView)}
						class="h-5 rounded-xs border text-[9.5px] font-medium transition-colors {projectData.rockTypingPlotView ===
						vKey
							? 'border-slate-800 bg-slate-800 text-white'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
					>
						{vLbl}
					</button>
				{/each}
			</div>
		</ParameterSection>

		<ParameterSection
			title="Automated Optimal Group & Cutoff Selection"
			badge={diagnostics?.optimal_k ? `Optimal k=${diagnostics.optimal_k}` : 'GMM / Lorenz'}
		>
			<div class="space-y-1.5">
				<div class="grid grid-cols-3 gap-1">
					{#each [['gmm', 'GMM (BIC/Sil)'], ['lorenz', 'Lorenz Inflect'], ['kmeans', 'k-Means']] as [cm, cmLbl]}
						<button
							type="button"
							onclick={() => selectCutoffMethod(cm as DrtCutoffMethod)}
							class="h-6 rounded-sm border text-[10px] font-medium transition-colors {projectData.drtCutoffMethod ===
							cm
								? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
								: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
						>
							{cmLbl}
						</button>
					{/each}
				</div>

				<div class="flex items-center justify-between gap-2">
					<label
						class="flex cursor-pointer items-center gap-1.5 text-[10.5px] font-medium text-[var(--text-primary)]"
					>
						<input
							type="checkbox"
							bind:checked={projectData.drtAutoK}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] accent-[var(--accent-primary)]"
						/>
						<span>Auto-Find Optimal Group Count (k)</span>
					</label>
					{#if projectData.drtCutoffMethod === 'gmm'}
						<select
							bind:value={projectData.drtGmmCriterion}
							title="GMM Optimal k Scoring Criterion"
							class="h-5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1 text-[10px] font-medium text-[var(--text-primary)]"
						>
							<option value="combined">BIC + Silhouette</option>
							<option value="bic">BIC Criterion</option>
							<option value="silhouette">Silhouette Only</option>
						</select>
					{/if}
				</div>

				{#if projectData.drtAutoK}
					<NumericScrubber
						label="Max Rock Type Groups (k_max)"
						bind:value={projectData.drtMaxClusters}
						unit="DRTs"
						min={3}
						max={8}
						step={1}
					/>
				{/if}

				<button
					type="button"
					disabled={projectData.isComputingMultiWell}
					onclick={() => executeRockTyping(true, Boolean(classifierDiag?.trained), false)}
					class="h-7 w-full rounded-sm border border-[var(--accent-primary)] bg-blue-50/80 text-[11px] font-semibold text-[var(--accent-primary)] transition-colors hover:bg-blue-100 disabled:opacity-50"
				>
					{projectData.drtCutoffMethod === 'lorenz'
						? `Auto-Detect Cutoffs via Lorenz Inflection${projectData.drtAutoK ? ' (Optimal k)' : ''}`
						: projectData.drtCutoffMethod === 'gmm'
							? `Auto-Detect Cutoffs via GMM (${projectData.drtGmmCriterion.toUpperCase()})`
							: `Auto-Detect Cutoffs via k-Means`}
				</button>

				{#if activeKScores.length > 0}
					<div
						class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]"
					>
						<div class="mb-1 flex items-center justify-between">
							<span class="font-semibold text-[var(--text-primary)]">
								{projectData.drtCutoffMethod === 'lorenz'
									? `Lorenz Inflection Scores (Lc = ${lorenzDiag?.lorenz_coeff ?? '—'})`
									: 'GMM Silhouette & BIC Scores by Group Count (k)'}
							</span>
							<span class="mono-num text-[9.5px] text-[var(--text-muted)]">Click k to apply</span>
						</div>
						<div class="grid grid-cols-6 gap-1">
							{#each activeKScores as row}
								{@const isOptimal = row.k === diagnostics?.optimal_k}
								{@const isSelected = row.k === projectData.activeDrtCutoffs.length + 1}
								<button
									type="button"
									onclick={() => selectCandidateK(row.k)}
									title={`k=${row.k} DRTs | Silhouette=${row.silhouette} | BIC=${row.bic}${row.r2 !== undefined ? ` | R²=${row.r2}` : ''} | Score=${row.combined_score}`}
									class="flex flex-col items-center rounded-xs border px-1 py-0.5 transition-colors {isSelected
										? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
										: isOptimal
											? 'border-emerald-500 bg-emerald-50/70 text-emerald-800'
											: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
								>
									<span class="mono-num text-[10px] font-bold">
										k={row.k}{isOptimal ? '★' : ''}
									</span>
									<span class="mono-num text-[9px] opacity-85">
										S:{Number(row.silhouette).toFixed(2)}
									</span>
								</button>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		</ParameterSection>

		<ParameterSection
			title="Carbonate Interparticle vs. Vuggy (Lucia RFN)"
			badge={hasSonicCurvesInProject ? 'Φ_sec = Φ_T − Φ_sonic' : 'Optional Sonic'}
		>
			<div class="space-y-1.5">
				<div
					class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10px] text-[var(--text-secondary)]"
				>
					When sonic porosity (<span class="mono-num font-semibold">Φ_sonic</span> or
					<span class="mono-num font-semibold">DT/DTC</span>) is available, secondary vuggy
					porosity <span class="mono-num font-semibold">Φ_sec = Φ_T − Φ_sonic</span> is separated
					from matrix interparticle porosity
					<span class="mono-num font-semibold">Φ_ip = Φ_T − Φ_sec</span> for Lucia
					<span class="mono-num font-semibold">RFN</span> rock typing.
				</div>

				<label
					class="flex cursor-pointer items-center justify-between gap-2 text-[10.5px] font-medium text-[var(--text-primary)]"
				>
					<span>Use Interparticle Φ_ip (Φ_T − Φ_sonic) for RFN</span>
					<input
						type="checkbox"
						bind:checked={projectData.useInterparticlePorosity}
						onchange={() => executeRockTyping(false, Boolean(classifierDiag?.trained), false)}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] accent-[var(--accent-primary)]"
					/>
				</label>

				{#if projectData.drtMethod !== 'lucia'}
					<label
						class="flex cursor-pointer items-center justify-between gap-2 text-[10.5px] font-medium text-[var(--text-primary)]"
					>
						<span>Override Vuggy Zones (Φ_sec ≥ Cutoff) with Lucia RFN</span>
						<input
							type="checkbox"
							bind:checked={projectData.luciaVuggyOverride}
							onchange={() => executeRockTyping(false, Boolean(classifierDiag?.trained), false)}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] accent-[var(--accent-primary)]"
						/>
					</label>
				{/if}

				<NumericScrubber
					label="Vuggy Φ_sec Override Cutoff"
					bind:value={projectData.phiSecCutoff}
					unit="v/v"
					min={0.005}
					max={0.15}
					step={0.005}
					onrun={() => executeRockTyping(false, Boolean(classifierDiag?.trained), false)}
				/>

				<NumericScrubber
					label="Carbonate Matrix Δt_ma (Wyllie)"
					bind:value={projectData.dtMatrix}
					unit="μs/ft"
					min={40.0}
					max={65.0}
					step={0.5}
					onrun={() => executeRockTyping(false, Boolean(classifierDiag?.trained), false)}
				/>

				{#if carbonateDiag}
					<div
						class="flex items-center justify-between rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10px]"
					>
						<span class="text-[var(--text-secondary)]">
							{#if carbonateDiag.has_sonic_data}
								Vuggy pts (<span class="mono-num">Φ_sec ≥ {projectData.phiSecCutoff}</span>):
								<strong class="mono-num text-[var(--text-primary)]"
									>{carbonateDiag.vuggy_count}</strong
								>
								{#if projectData.luciaVuggyOverride && projectData.drtMethod !== 'lucia'}
									· Overridden:
									<strong class="mono-num text-amber-700"
										>{carbonateDiag.override_count}</strong
									>
								{/if}
							{:else}
								No DT/PHIS/PHI_SEC curves found in active wells
							{/if}
						</span>
						{#if carbonateDiag.mean_phi_sec !== null && carbonateDiag.mean_phi_sec !== undefined}
							<button
								type="button"
								onclick={() => setPlotView('carbonate')}
								class="mono-num font-semibold text-[var(--accent-primary)] hover:underline"
							>
								Mean Φ_sec: {Number(carbonateDiag.mean_phi_sec).toFixed(3)} →
							</button>
						{/if}
					</div>
				{/if}
			</div>
		</ParameterSection>

		<ParameterSection
			title={activeMeta.cutoffTitle}
			badge={`${projectData.activeDrtCutoffs.length + 1} DRTs`}
		>
			<div class="space-y-1">
				{#each projectData.activeDrtCutoffs as _, i}
					<div class="flex items-center gap-1">
						<div class="flex-1">
							{#if projectData.drtMethod === 'winland'}
								<NumericScrubber
									label={`${activeMeta.cutoffLabel} ${i + 1}`}
									bind:value={projectData.winlandCutoffs[i]}
									unit={activeMeta.unit}
									min={activeMeta.min}
									max={activeMeta.max}
									step={activeMeta.step}
									onrun={() => executeRockTyping(false, Boolean(classifierDiag?.trained), false)}
								/>
							{:else if projectData.drtMethod === 'lucia'}
								<NumericScrubber
									label={`${activeMeta.cutoffLabel} ${i + 1}`}
									bind:value={projectData.luciaCutoffs[i]}
									unit={activeMeta.unit}
									min={activeMeta.min}
									max={activeMeta.max}
									step={activeMeta.step}
									onrun={() => executeRockTyping(false, Boolean(classifierDiag?.trained), false)}
								/>
							{:else}
								<NumericScrubber
									label={`${activeMeta.cutoffLabel} ${i + 1}`}
									bind:value={projectData.fziCutoffs[i]}
									unit={activeMeta.unit}
									min={activeMeta.min}
									max={activeMeta.max}
									step={activeMeta.step}
									onrun={() => executeRockTyping(false, Boolean(classifierDiag?.trained), false)}
								/>
							{/if}
						</div>
						<button
							type="button"
							title="Remove cutoff"
							onclick={() => removeCutoff(i)}
							class="flex h-7 w-6 shrink-0 items-center justify-center rounded-sm border border-[var(--border-hairline)] text-xs text-[var(--text-muted)] hover:bg-red-50 hover:text-red-600"
						>
							×
						</button>
					</div>
				{/each}
			</div>

			<div class="mt-1.5">
				<button
					type="button"
					onclick={addCutoff}
					class="h-6 w-full rounded-sm border border-dashed border-[var(--border-hairline)] bg-[var(--bg-app)] text-[10.5px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
				>
					{activeMeta.addLabel}
				</button>
			</div>
		</ParameterSection>

		<ParameterSection
			title="ML Classifier & Contingency Validator"
			badge={classifierDiag?.propagated
				? 'Propagated ✓'
				: classifierDiag?.metrics?.accuracy !== undefined
					? `Acc: ${(Number(classifierDiag.metrics.accuracy) * 100).toFixed(1)}%`
					: 'GR·NPHI·RHOB·log(RT)'}
		>
			<div class="space-y-1.5">
				<div
					class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10px] text-[var(--text-secondary)]"
				>
					Trains a supervised classifier on
					<span class="mono-num font-semibold text-[var(--text-primary)]"
						>GR, NPHI, RHOB, log₁₀(RT)</span
					>
					+ optional <span class="mono-num font-semibold">PEF</span> &amp;
					<span class="mono-num font-semibold">VDL</span> (Velocity Deviation Log) against
					cutoff-binned <span class="mono-num font-semibold">ROCK_FLAG</span>.
				</div>

				<div class="grid grid-cols-2 gap-1.5">
					<label
						class="flex cursor-pointer items-center gap-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10px] font-medium text-[var(--text-primary)]"
					>
						<input
							type="checkbox"
							bind:checked={projectData.drtIncludePef}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] accent-[var(--accent-primary)]"
						/>
						<span>PEF (Optional)</span>
					</label>
					<label
						class="flex cursor-pointer items-center gap-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10px] font-medium text-[var(--text-primary)]"
					>
						<input
							type="checkbox"
							bind:checked={projectData.drtIncludeVdl}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] accent-[var(--accent-primary)]"
						/>
						<span>VDL (Vel. Dev. Log)</span>
					</label>
				</div>

				<div class="flex items-center justify-between gap-1.5">
					<select
						bind:value={projectData.drtClassifierModel}
						title="Supervised Classification Model"
						class="h-6 flex-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 text-[10px] font-medium text-[var(--text-primary)]"
					>
						<option value="random_forest">Random Forest (Balanced)</option>
						<option value="extra_trees">Extra Trees Ensemble</option>
						<option value="gradient_boosting">Gradient Boosting</option>
					</select>
					<select
						bind:value={projectData.drtClassifierTestSize}
						title="Validation Holdout Split"
						class="h-6 w-24 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1 text-[10px] font-medium text-[var(--text-primary)]"
					>
						<option value={0.2}>20% Val</option>
						<option value={0.25}>25% Val</option>
						<option value={0.3}>30% Val</option>
					</select>
				</div>

				<button
					type="button"
					disabled={projectData.isComputingMultiWell}
					onclick={() => executeRockTyping(false, true, false)}
					class="h-7 w-full rounded-sm border border-[var(--accent-primary)] bg-blue-50/80 text-[11px] font-semibold text-[var(--accent-primary)] transition-colors hover:bg-blue-100 disabled:opacity-50"
				>
					Train &amp; Validate (Contingency Matrix)
				</button>

				{#if classifierDiag}
					{@const ct = classifierDiag.contingency_table}
					{@const m = classifierDiag.metrics}
					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]"
					>
						<!-- Validation KPI Banner -->
						<div class="grid grid-cols-3 gap-1 text-center">
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0.5">
								<div class="text-[9px] text-[var(--text-muted)]">Accuracy</div>
								<div class="mono-num font-bold text-[var(--text-primary)]">
									{(Number(m.accuracy) * 100).toFixed(1)}%
								</div>
							</div>
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0.5">
								<div class="text-[9px] text-[var(--text-muted)]">Weighted F1</div>
								<div class="mono-num font-bold text-[var(--text-primary)]">
									{Number(m.f1_weighted).toFixed(3)}
								</div>
							</div>
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0.5">
								<div class="text-[9px] text-[var(--text-muted)]">Cohen's κ</div>
								<div class="mono-num font-bold text-[var(--text-primary)]">
									{Number(m.cohen_kappa).toFixed(3)}
								</div>
							</div>
						</div>

						<!-- Features & Sample Counts -->
						<div class="flex flex-wrap items-center justify-between gap-1 text-[9.5px] text-[var(--text-secondary)]">
							<span>
								Features:
								<strong class="mono-num text-[var(--text-primary)]"
									>{(classifierDiag.features_used ?? []).join(', ')}</strong
								>
							</span>
							<button
								type="button"
								onclick={() => setPlotView('classifier')}
								class="font-semibold text-[var(--accent-primary)] hover:underline"
							>
								Plot Matrix →
							</button>
						</div>

						<!-- Compact Contingency Table (Confusion Matrix) -->
						{#if ct?.labels?.length && ct?.counts?.length}
							<div class="overflow-x-auto">
								<table class="w-full border-collapse text-center text-[9.5px]">
									<thead>
										<tr class="border-b border-[var(--border-hairline)] text-[var(--text-muted)]">
											<th class="px-1 py-0.5 text-left font-medium">True \ Pred</th>
											{#each ct.labels as colLbl}
												<th class="mono-num px-1 py-0.5 font-semibold">DRT {colLbl}</th>
											{/each}
											<th class="px-1 py-0.5 font-medium">Recall</th>
										</tr>
									</thead>
									<tbody>
										{#each ct.labels as rowLbl, rIdx}
											{@const rowCounts = ct.counts[rIdx] ?? []}
											{@const rowNorm = ct.norm_recall?.[rIdx] ?? []}
											{@const diagRec = Number(rowNorm[rIdx] ?? 0)}
											<tr class="border-b border-[var(--border-hairline)] last:border-b-0">
												<td class="mono-num px-1 py-0.5 text-left font-semibold">
													<span
														style="background-color: {getRockTypeColor(rowLbl)}"
														class="mr-1 inline-block h-2 w-2 rounded-full"
													></span>DRT {rowLbl}
												</td>
												{#each rowCounts as cellVal, cIdx}
													{@const isDiag = rIdx === cIdx}
													<td
														title={`True DRT ${rowLbl} → Pred DRT ${ct.labels[cIdx]}: ${cellVal} samples (${((rowNorm[cIdx] ?? 0) * 100).toFixed(1)}%)`}
														class="mono-num px-1 py-0.5 {isDiag
															? 'bg-emerald-50 font-bold text-emerald-800'
															: cellVal > 0
																? 'bg-amber-50/60 text-amber-900'
																: 'text-[var(--text-muted)]'}"
													>
														{cellVal}
													</td>
												{/each}
												<td class="mono-num px-1 py-0.5 font-semibold text-[var(--text-primary)]">
													{(diagRec * 100).toFixed(0)}%
												</td>
											</tr>
										{/each}
									</tbody>
								</table>
							</div>
						{/if}

						<!-- Feature Importance Bars -->
						{#if classifierDiag.feature_importances?.length}
							<div class="space-y-0.5 pt-0.5">
								<div class="text-[9px] font-semibold text-[var(--text-muted)]">
									Feature Importance (MDI)
								</div>
								{#each classifierDiag.feature_importances as fi}
									<div class="flex items-center gap-1.5 text-[9.5px]">
										<span class="mono-num w-12 font-medium text-[var(--text-primary)]"
											>{fi.feature}</span
										>
										<div class="h-2 flex-1 overflow-hidden rounded-xs bg-slate-200">
											<div
												class="h-full bg-[var(--accent-primary)]"
												style="width: {Math.min(100, Math.round(Number(fi.importance) * 100))}%"
											></div>
										</div>
										<span class="mono-num w-8 text-right text-[var(--text-secondary)]">
											{(Number(fi.importance) * 100).toFixed(0)}%
										</span>
									</div>
								{/each}
							</div>
						{/if}

						<!-- Propagation Mode & Accept Action -->
						<div class="space-y-1 border-t border-[var(--border-hairline)] pt-1.5">
							<div class="flex items-center justify-between gap-1">
								<span class="text-[9.5px] font-medium text-[var(--text-secondary)]"
									>Propagation:</span
								>
								<select
									bind:value={projectData.drtPropagationMode}
									class="h-5 flex-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[9.5px] font-medium text-[var(--text-primary)]"
								>
									<option value="fill_missing"
										>Keep Binned Cutoffs + Fill Uncored Depths</option
									>
									<option value="full_ml"
										>Propagate ML Predicted DRT Across Full Log</option
									>
								</select>
							</div>
							<button
								type="button"
								disabled={projectData.isComputingMultiWell}
								onclick={() => executeRockTyping(false, true, true)}
								class="h-7 w-full rounded-sm border border-emerald-600 bg-emerald-600 text-[11px] font-semibold text-white transition-opacity hover:opacity-95 disabled:opacity-50"
							>
								{classifierDiag.propagated
									? `✓ Propagated (${m.total_predicted} log pts) — Re-Apply`
									: `✓ Accept Matrix & Propagate DRTs (${m.total_predicted} pts)`}
							</button>
						</div>
					</div>
				{/if}
			</div>
		</ParameterSection>

		{#if projectData.rockTypingResult?.rock_type_counts}
			<ParameterSection
				title="Discrete Rock Type (DRT) Counts"
				badge={classifierDiag?.propagated
					? `${activeMeta.shortName} + ML Propagated`
					: projectData.luciaVuggyOverride && projectData.drtMethod !== 'lucia'
						? `${activeMeta.shortName} + RFN Override`
						: activeMeta.shortName}
			>
				<div class="space-y-1">
					{#each Object.entries(projectData.rockTypingResult.rock_type_counts) as [drt, count]}
						{@const rangeLbl = formatDrtRange(drt)}
						<div
							class="flex h-6 items-center justify-between rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 text-xs"
						>
							<span class="flex items-center gap-1.5 font-medium">
								<span
									style="background-color: {getRockTypeColor(drt)}"
									class="inline-block h-2.5 w-2.5 rounded-full"
								></span>
								<span>DRT {drt}</span>
								{#if rangeLbl}
									<span class="mono-num text-[10px] font-normal text-[var(--text-muted)]">
										({rangeLbl})
									</span>
								{/if}
							</span>
							<span class="mono-num font-semibold text-[var(--text-primary)]">{count} pts</span>
						</div>
					{/each}
				</div>
			</ParameterSection>
		{/if}
	</div>

	<div
		class="space-y-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2"
	>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell}
			onclick={() => executeRockTyping(false, Boolean(classifierDiag?.trained), projectData.drtClassifierPropagated)}
			class="h-7 w-full rounded-sm bg-[var(--accent-primary)] text-xs font-medium text-white transition-opacity hover:opacity-95 disabled:opacity-50"
		>
			{projectData.isComputingMultiWell ? 'Computing...' : activeMeta.computeLabel}
		</button>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell ||
				!projectData.hasPendingMultiWell('rock-typing')}
			onclick={saveRockTyping}
			class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-xs font-medium text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-40"
		>
			Save ROCK_FLAG{classifierDiag?.propagated ? ', ROCK_FLAG_ML & VDL' : ` & ${activeMeta.shortName}`} to Project Wells
		</button>
	</div>
</div>
