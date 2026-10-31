<script lang="ts">
	import { onMount } from 'svelte';
	import { API_BASE, api, authHeaders } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import {
		projectData,
		type DrtCutoffMethod,
		type DrtFeatureKey,
		type DrtMethod,
		type DrtPropagationMode,
		type PhiKDataSource,
		type RockTypingPlotView
	} from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { getRockTypeColor } from '$lib/utils/petrophysicsColors';

	const ALL_RT_FEATURES: Array<{ key: DrtFeatureKey; label: string; short: string; title: string }> = [
		{ key: 'GR', label: 'GR', short: 'GR', title: 'Gamma Ray (API)' },
		{ key: 'NPHI', label: 'NPHI', short: 'NPHI', title: 'Neutron Porosity (v/v)' },
		{ key: 'RHOB', label: 'RHOB', short: 'RHOB', title: 'Bulk Density (g/cc)' },
		{ key: 'LOG_RT', label: 'log₁₀(RT)', short: 'log(RT)', title: 'Log10 Deep Resistivity (ohm.m)' },
		{ key: 'PEF', label: 'PEF', short: 'PEF', title: 'Photoelectric Factor (b/e)' },
		{ key: 'VDL', label: 'VDL', short: 'VDL', title: 'Velocity Deviation Log (Φ_T − Φ_sonic)' },
		{ key: 'PHIT', label: 'PHIT', short: 'PHIT', title: 'Total Porosity (v/v)' },
		{ key: 'VSAND', label: 'VSAND', short: 'VSAND', title: 'Sand Volume Fraction (v/v)' },
		{ key: 'VSHALE', label: 'VSHALE', short: 'VSHALE', title: 'Shale Volume Fraction (v/v)' },
		{ key: 'VSILT', label: 'VSILT', short: 'VSILT', title: 'Silt Volume Fraction (v/v)' },
		{ key: 'VCLAY', label: 'VCLAY', short: 'VCLAY', title: 'Clay Volume Fraction (v/v)' }
	];

	let customModelName = $state<string>('');
	let isSavingModel = $state<boolean>(false);
	let previewedSavedModel = $state<Record<string, any> | null>(null);

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

	function selectDrtMethod(m: DrtMethod) {
		if (projectData.drtMethod === m && projectData.rockTypingResult?.method === m) return;
		projectData.drtMethod = m;
		projectData.drtClassifierPropagated = false;
	}

	function selectDrtDataSource(src: PhiKDataSource) {
		if (projectData.drtDataSource === src) return;
		projectData.drtDataSource = src;
		projectData.drtClassifierPropagated = false;
	}

	function selectCutoffMethod(cm: DrtCutoffMethod) {
		projectData.drtCutoffMethod = cm;
		if (cm === 'lorenz' && projectData.rockTypingPlotView === 'gmm') {
			projectData.rockTypingPlotView = 'lorenz';
		} else if (cm === 'gmm' && projectData.rockTypingPlotView === 'lorenz') {
			projectData.rockTypingPlotView = 'gmm';
		}
	}

	function selectCandidateK(k: number) {
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

	let savedModelsList = $derived<Array<Record<string, any>>>(
		projectData.rtSavedModelsStatus?.models ?? diagnostics?.models_status?.models ?? []
	);
	let activeDisplayedClassifier = $derived<Record<string, any> | null>(
		previewedSavedModel ?? classifierDiag
	);

	function toggleFeature(feat: DrtFeatureKey) {
		const cur = projectData.drtSelectedFeatures;
		if (cur.includes(feat)) {
			if (cur.length <= 1) return;
			projectData.drtSelectedFeatures = cur.filter((f) => f !== feat);
		} else {
			projectData.drtSelectedFeatures = [...cur, feat];
		}
		projectData.drtIncludePef = projectData.drtSelectedFeatures.includes('PEF');
		projectData.drtIncludeVdl = projectData.drtSelectedFeatures.includes('VDL');
	}

	function setFeaturePreset(features: DrtFeatureKey[]) {
		if (!features.length) return;
		projectData.drtSelectedFeatures = [...features];
		projectData.drtIncludePef = features.includes('PEF');
		projectData.drtIncludeVdl = features.includes('VDL');
	}

	function toggleZoneFilter(zone: string) {
		if (workbench.selectedZones.includes(zone)) {
			workbench.selectedZones = workbench.selectedZones.filter((z) => z !== zone);
		} else {
			workbench.selectedZones = [...workbench.selectedZones, zone];
		}
	}

	async function loadSavedClassifierModels() {
		if (!workbench.projectId) return;
		try {
			const res = await fetch(
				`${API_BASE}/quick_pp/workspace/projects/${workbench.projectId}/rock_typing/models`,
				{ headers: authHeaders() }
			);
			if (res.ok) {
				const payload = await res.json();
				projectData.rtSavedModelsStatus = payload;
			}
		} catch {
			// Ignore background status check errors
		}
	}

	onMount(() => {
		loadSavedClassifierModels();
	});

	async function saveTrainedClassifierModel() {
		if (!workbench.projectId || !classifierDiag) return;
		isSavingModel = true;
		projectData.error = null;
		try {
			const accPct =
				classifierDiag.metrics?.accuracy !== undefined
					? `${(Number(classifierDiag.metrics.accuracy) * 100).toFixed(0)}%`
					: '';
			const algoShort =
				classifierDiag.model_type === 'extra_trees'
					? 'ET'
					: classifierDiag.model_type === 'gradient_boosting'
						? 'GB'
						: 'RF';
			const defaultName = `${algoShort} (${(classifierDiag.features_used ?? []).join('+')}) ${activeMeta.shortName} ${accPct}`.trim();
			const nameToUse = customModelName.trim() || defaultName;
			const keySlug = nameToUse
				.toLowerCase()
				.replace(/[^a-z0-9]+/g, '_')
				.replace(/^_+|_+$/g, '')
				.slice(0, 48);

			const res = await fetch(
				`${API_BASE}/quick_pp/workspace/projects/${workbench.projectId}/rock_typing/models/save`,
				{
					method: 'POST',
					headers: authHeaders({ 'Content-Type': 'application/json' }),
					body: JSON.stringify({
						model_key: keySlug || `rt_${Date.now()}`,
						model_name: nameToUse,
						drt_method: projectData.drtMethod,
						data_source: projectData.drtDataSource,
						cutoffs: [...projectData.activeDrtCutoffs].sort((a, b) => a - b)
					})
				}
			);
			if (!res.ok) {
				const errBody = await res.json().catch(() => ({}));
				throw new Error(String(errBody?.detail ?? 'Failed to save classifier model'));
			}
			const savedPayload = await res.json();
			if (savedPayload.models_status) {
				projectData.rtSavedModelsStatus = savedPayload.models_status;
			} else {
				await loadSavedClassifierModels();
			}
			if (savedPayload.model_key) {
				projectData.drtSelectedSavedModelKey = savedPayload.model_key;
			}
			customModelName = '';
			// Persist classifier settings keyed by the active zone selection
			await projectData.persistProjectParams([], ['rt-classifier']);
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to save classifier model';
		} finally {
			isSavingModel = false;
		}
	}

	async function deleteSavedClassifierModel(modelKey: string) {
		if (!workbench.projectId || !modelKey) return;
		try {
			const res = await fetch(
				`${API_BASE}/quick_pp/workspace/projects/${workbench.projectId}/rock_typing/models/${encodeURIComponent(modelKey)}`,
				{
					method: 'DELETE',
					headers: authHeaders()
				}
			);
			if (res.ok) {
				const payload = await res.json();
				projectData.rtSavedModelsStatus = payload;
				if (projectData.drtSelectedSavedModelKey === modelKey) {
					projectData.drtSelectedSavedModelKey = '';
				}
				if (previewedSavedModel?.model_key === modelKey) {
					previewedSavedModel = null;
				}
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to delete saved classifier model';
		}
	}

	function inspectSavedClassifierModel(model: Record<string, any>) {
		previewedSavedModel = model;
		projectData.drtSelectedSavedModelKey = model.model_key;
		if (projectData.rockTypingResult) {
			projectData.rockTypingResult = {
				...projectData.rockTypingResult,
				diagnostics: {
					...(projectData.rockTypingResult.diagnostics ?? {}),
					classifier: {
						...(projectData.rockTypingResult.diagnostics?.classifier ?? {}),
						trained: true,
						model_type: model.model_type,
						saved_model_key: model.model_key,
						saved_model_name: model.model_name,
						features_used: model.features_used,
						labels: model.labels,
						contingency_table: model.contingency_table,
						metrics: model.metrics,
						class_metrics: model.class_metrics,
						feature_importances: model.feature_importances
					}
				}
			};
		}
		setPlotView('classifier');
	}

	async function applySavedClassifierModel(modelKey: string) {
		if (!modelKey) return;
		previewedSavedModel = null;
		projectData.drtSelectedSavedModelKey = modelKey;
		if (projectData.drtPropagationMode === 'cutoff_only') {
			projectData.drtPropagationMode = 'fill_missing';
		}
		await executeRockTyping(false, false, true, modelKey);
	}

	async function selectRockFlagSourceForWorkflow(mode: DrtPropagationMode) {
		projectData.drtPropagationMode = mode;
		const rt = projectData.rockTypingResult;
		if (rt) {
			const nextCoreRf = (rt.rock_flags_binned ?? rt.rock_flags ?? []).map((cb, idx) => {
				const ml = rt.rock_flags_ml?.[idx] ?? null;
				if (mode === 'cutoff_only') return cb ?? null;
				if (mode === 'full_ml') return ml ?? cb ?? null;
				return cb ?? ml ?? null;
			});
			const nextLogRf = (rt.log_rock_flags_cutoff ?? rt.log_rock_flags ?? []).map((cb, idx) => {
				const ml = rt.log_rock_flags_ml?.[idx] ?? null;
				if (mode === 'cutoff_only') return cb ?? null;
				if (mode === 'full_ml') return ml ?? cb ?? null;
				return cb ?? ml ?? null;
			});
			const nextCounts: Record<string, number> = {};
			const countSource = nextLogRf.length > 0 ? nextLogRf : nextCoreRf;
			for (const v of countSource) {
				if (v !== null && v !== undefined && Number.isFinite(v)) {
					const k = String(Math.round(v));
					nextCounts[k] = (nextCounts[k] ?? 0) + 1;
				}
			}
			projectData.drtClassifierPropagated = mode !== 'cutoff_only';
			projectData.rockTypingResult = {
				...rt,
				rock_flags: nextCoreRf,
				log_rock_flags: nextLogRf,
				rock_type_counts: nextCounts,
				diagnostics: rt.diagnostics
					? {
							...rt.diagnostics,
							classifier: rt.diagnostics.classifier
								? {
										...rt.diagnostics.classifier,
										propagated: mode !== 'cutoff_only',
										propagation_mode: mode
									}
								: null
						}
					: undefined
			};
			projectData.markMultiWellComputed('rock-typing');

			if (workbench.wellId && Array.isArray(rt.log_well_names) && Array.isArray(rt.log_depths)) {
				const lDepths: number[] = [];
				const lRf: (number | null)[] = [];
				for (let i = 0; i < rt.log_well_names.length; i++) {
					if (rt.log_well_names[i] === workbench.wellId) {
						lDepths.push(rt.log_depths[i]);
						lRf.push(nextLogRf[i] ?? null);
					}
				}
				if (lDepths.length > 0) {
					wellData.mergeComputedCurves({
						DEPTH: lDepths,
						ROCK_FLAG: lRf
					});
				}
			}
		}





	}

	async function executeRockTyping(
		autoFindCutoffs: boolean = false,
		trainClassifier: boolean = false,
		applyPropagation: boolean = false,
		savedModelKey: string | null = null
	) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		previewedSavedModel = null;
		try {
			const method = projectData.drtMethod;
			const meta = DRT_METHOD_META[method];
			const sortedCutoffs = [...projectData.activeDrtCutoffs].sort((a, b) => a - b);
			const activeMinDepth =
				projectData.drtPropagateFocusedOnly && workbench.depthFilter?.min != null
					? workbench.depthFilter.min
					: null;
			const activeMaxDepth =
				projectData.drtPropagateFocusedOnly && workbench.depthFilter?.max != null
					? workbench.depthFilter.max
					: null;
			const activeZones =
				projectData.drtPropagateFocusedOnly && workbench.selectedZones.length > 0
					? workbench.selectedZones
					: null;

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
						data_source: projectData.drtDataSource,
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
						train_classifier: trainClassifier,
						apply_classifier_propagation: applyPropagation,
						propagation_mode: projectData.drtPropagationMode,
						propagate_focused_only: projectData.drtPropagateFocusedOnly,
						zones: activeZones,
						min_depth: activeMinDepth,
						max_depth: activeMaxDepth,
						classifier_model: projectData.drtClassifierModel,
						selected_features: projectData.drtSelectedFeatures,
						include_pef: projectData.drtSelectedFeatures.includes('PEF'),
						include_vdl: projectData.drtSelectedFeatures.includes('VDL'),
						test_size: projectData.drtClassifierTestSize,
						saved_model_key: savedModelKey,
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
			if ((data as any).diagnostics?.models_status) {
				projectData.rtSavedModelsStatus = (data as any).diagnostics.models_status;
			}
			projectData.drtClassifierPropagated = Boolean(
				(data as any).diagnostics?.classifier?.propagated
			);
			const counts = (data as any).rock_type_counts ?? (data as any).flag_counts ?? {};
			projectData.rockTypingResult = {
				method: (data as any).method ?? method,
				data_source: (data as any).data_source ?? projectData.drtDataSource,
				poro_label:
					(data as any).poro_label ??
					((data as any).data_source === 'log' || projectData.drtDataSource === 'log'
						? 'PHIT'
						: 'CPORE'),
				perm_label:
					(data as any).perm_label ??
					((data as any).data_source === 'log' || projectData.drtDataSource === 'log'
						? 'PERM'
						: 'CPERM'),
				optimal_k: (data as any).optimal_k,
				selected_k: (data as any).selected_k,
				cutoff_method: (data as any).cutoff_method ?? projectData.drtCutoffMethod,
				diagnostics: (data as any).diagnostics ?? null,
				...(data.data as any),
				rock_type_counts: counts
			};
			projectData.markMultiWellComputed('rock-typing');

			if ((trainClassifier || savedModelKey) && !applyPropagation) {
				setPlotView('classifier');
			}

			// Update active well's ROCK_FLAG & indicator curves in wellData so LogTrackCanvas updates in real time
			if (workbench.wellId && data.data?.well_names) {
				const wnArr = data.data.well_names;
				const dArr = data.data.depths;
				const rfArr = data.data.rock_flags;
				const rfCutArr = (data.data as any).rock_flags_cutoff ?? (data.data as any).rock_flags_binned;
				const rfMlArr = (data.data as any).rock_flags_ml;
				const vdlArr = (data.data as any).vdl;
				const fziArr = data.data.fzi;
				const r35Arr = (data.data as any).r35;
				const rfnArr = (data.data as any).rfn;
				const phiSecArr = (data.data as any).phi_sec;
				const wDepths: number[] = [];
				const wRf: (number | null)[] = [];
				const wRfCut: (number | null)[] = [];
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
						wRfCut.push(rfCutArr?.[i] ?? null);
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
						ROCK_FLAG_CUTOFF: wRfCut,
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
				const logRfCutArr = (data.data as any).log_rock_flags_cutoff as (number | null)[] | undefined;
				const logRfMlArr = (data.data as any).log_rock_flags_ml as (number | null)[] | undefined;
				const logVdlArr = (data.data as any).log_vdl as (number | null)[] | undefined;
				if (Array.isArray(logWnArr) && Array.isArray(logDArr) && logWnArr.length > 0) {
					const lDepths: number[] = [];
					const lRf: (number | null)[] = [];
					const lRfCut: (number | null)[] = [];
					const lRfMl: (number | null)[] = [];
					const lVdl: (number | null)[] = [];
					for (let i = 0; i < logWnArr.length; i++) {
						if (logWnArr[i] === workbench.wellId) {
							lDepths.push(logDArr[i]);
							lRf.push(logRfArr?.[i] ?? null);
							lRfCut.push(logRfCutArr?.[i] ?? null);
							lRfMl.push(logRfMlArr?.[i] ?? null);
							lVdl.push(logVdlArr?.[i] ?? null);
						}
					}
					if (lDepths.length > 0) {
						const logPayload: Record<string, (number | null)[]> = {
							DEPTH: lDepths,
							ROCK_FLAG: lRf
						};
						if (lRfCut.some((v) => v !== null)) {
							logPayload.ROCK_FLAG_CUTOFF = lRfCut;
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
		<SavedParamsIndicator modules={['rock-typing', 'rt-classifier']} />
		<ParameterSection
			title="Rock Typing Method (DRT)"
			badge="{projectData.drtDataSource === 'core' ? 'Core (CPORE & CPERM)' : 'Log (PHIT & PERM)'} · {activeMeta.badge}"
			defaultOpen={true}
		>
			<!-- Input Data Source Selector (Core CPORE/CPERM vs Calculated Log PHIT/PERM) -->
			<div class="mb-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
				<div class="mb-1 flex items-center justify-between text-[9.5px]">
					<span class="font-semibold text-[var(--text-primary)]">
						Input Porosity &amp; Permeability Source (Φ &amp; K):
					</span>
					<span class="mono-num text-[9px] text-[var(--text-muted)]">
						{#if projectData.rockTypingResult?.data_source}
							Computed on:
							<strong class="text-[var(--text-primary)]">
								{projectData.rockTypingResult.data_source === 'log'
									? 'Log (PHIT & PERM)'
									: 'Core (CPORE & CPERM)'}
							</strong>
						{:else}
							Select source
						{/if}
					</span>
				</div>
				<div class="grid grid-cols-2 gap-1">
					<button
						type="button"
						onclick={() => selectDrtDataSource('core')}
						class="flex flex-col items-start rounded-xs border px-2 py-1 text-left transition-colors {projectData.drtDataSource ===
						'core'
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
					>
						<div class="flex w-full items-center justify-between gap-1 text-[10px]">
							<span>Core Data</span>
							<span class="mono-num rounded-xs bg-amber-100/80 px-1 py-0.2 text-[8.5px] font-semibold text-amber-800">
								CPORE &amp; CPERM
							</span>
						</div>
						<span class="mono-num mt-0.5 text-[8.5px] opacity-80">
							{projectData.wellsWithCorePhiK.length}/{projectData.activeWells.length} wells · {projectData.totalCorePlugCount} RCA plugs
						</span>
					</button>
					<button
						type="button"
						onclick={() => selectDrtDataSource('log')}
						class="flex flex-col items-start rounded-xs border px-2 py-1 text-left transition-colors {projectData.drtDataSource ===
						'log'
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
					>
						<div class="flex w-full items-center justify-between gap-1 text-[10px]">
							<span>Calculated Logs</span>
							<span class="mono-num rounded-xs bg-indigo-100/80 px-1 py-0.2 text-[8.5px] font-semibold text-indigo-800">
								PHIT &amp; PERM
							</span>
						</div>
						<span class="mono-num mt-0.5 text-[8.5px] opacity-80">
							{projectData.wellsWithLogPhiK.length}/{projectData.activeWells.length} wells with PHIT &amp; PERM
						</span>
					</button>
				</div>
				<div class="mt-1 text-[9px] leading-snug text-[var(--text-secondary)]">
					{#if projectData.drtDataSource === 'core'}
						Modelling <span class="mono-num font-semibold">{activeMeta.shortName}</span> and cutoff DRT labels strictly from
						<strong class="mono-num text-[var(--text-primary)]">Core Plugs (CPORE &amp; CPERM)</strong>.
						{#if projectData.wellsWithCorePhiK.length === 0}
							<span class="font-semibold text-amber-700">
								No selected wells have RCA core data — switch to Calculated Logs (PHIT &amp; PERM).
							</span>
						{/if}
					{:else}
						Modelling <span class="mono-num font-semibold">{activeMeta.shortName}</span> and cutoff DRT labels strictly from
						<strong class="mono-num text-[var(--text-primary)]">Calculated Logs (PHIT &amp; PERM)</strong>.
						{#if projectData.wellsWithLogPhiK.length === 0}
							<span class="font-semibold text-amber-700">
								No selected wells have both PHIT and PERM curves — switch to Core Data (CPORE &amp; CPERM).
							</span>
						{/if}
					{/if}
				</div>
			</div>

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

				/>

				<NumericScrubber
					label="Carbonate Matrix Δt_ma (Wyllie)"
					bind:value={projectData.dtMatrix}
					unit="μs/ft"
					min={40.0}
					max={65.0}
					step={0.5}

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
				
								/>
							{:else if projectData.drtMethod === 'lucia'}
								<NumericScrubber
									label={`${activeMeta.cutoffLabel} ${i + 1}`}
									bind:value={projectData.luciaCutoffs[i]}
									unit={activeMeta.unit}
									min={activeMeta.min}
									max={activeMeta.max}
									step={activeMeta.step}
				
								/>
							{:else}
								<NumericScrubber
									label={`${activeMeta.cutoffLabel} ${i + 1}`}
									bind:value={projectData.fziCutoffs[i]}
									unit={activeMeta.unit}
									min={activeMeta.min}
									max={activeMeta.max}
									step={activeMeta.step}
				
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
			title="ML Model Selection &amp; Propagation"
			badge={classifierDiag?.propagated
				? `Propagated (${projectData.drtPropagationMode === 'full_ml' ? 'Full ML' : 'ML Fill'}) ✓`
				: savedModelsList.length > 0
					? `${savedModelsList.length} Saved`
					: 'Select Model'}
		>
			<div class="space-y-2">
				<!-- 1) Select Saved Classifier Model or Jump to Standalone RT Classifier Trainer -->
				<div
					class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]"
				>
					<div class="flex items-center justify-between gap-1">
						<span class="font-semibold text-[var(--text-primary)]">
							1. Select Classifier Model ({savedModelsList.length} Saved)
						</span>
					</div>

					{#if savedModelsList.length > 0}
						<div class="max-h-40 space-y-1 overflow-y-auto">
							{#each savedModelsList as sm}
								{@const isActiveModel =
									projectData.drtSelectedSavedModelKey === sm.model_key ||
									classifierDiag?.saved_model_key === sm.model_key}
								{@const smAcc =
									sm.metrics?.accuracy !== undefined
										? `${(Number(sm.metrics.accuracy) * 100).toFixed(1)}%`
										: '—'}
								{@const smF1 =
									sm.metrics?.f1_weighted !== undefined
										? Number(sm.metrics.f1_weighted).toFixed(2)
										: '—'}
								{@const smKappa =
									sm.metrics?.cohen_kappa !== undefined
										? Number(sm.metrics.cohen_kappa).toFixed(2)
										: '—'}
								<div
									class="rounded-xs border p-1 transition-colors {isActiveModel
										? 'border-[var(--accent-primary)] bg-blue-50/60'
										: 'border-[var(--border-hairline)] bg-[var(--bg-surface)]'}"
								>
									<div class="flex items-center justify-between gap-1">
										<span class="truncate font-semibold text-[var(--text-primary)]" title={sm.model_name}>
											{sm.model_name}
										</span>
										<div class="flex shrink-0 items-center gap-1">
											<button
												type="button"
												onclick={() => inspectSavedClassifierModel(sm)}
												class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1 py-0.5 text-[9px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
												title="Plot saved Contingency Matrix on crossplot"
											>
												CM
											</button>
											<button
												type="button"
												disabled={projectData.isComputingMultiWell}
												onclick={() => applySavedClassifierModel(sm.model_key)}
												class="rounded-xs border border-emerald-600 bg-emerald-600 px-1.5 py-0.5 text-[9px] font-semibold text-white hover:opacity-95 disabled:opacity-50"
												title="Propagate DRTs using this saved classifier"
											>
												{isActiveModel && classifierDiag?.propagated ? 'Active ✓' : 'Propagate'}
											</button>
										</div>
									</div>
									<div class="mt-0.5 flex flex-wrap items-center justify-between gap-1 text-[9px] text-[var(--text-secondary)]">
										<span class="mono-num flex items-center gap-1">
											<span
												class="rounded-xs px-1 py-0.2 text-[8px] font-semibold {sm.data_source ===
												'log'
													? 'bg-indigo-100/80 text-indigo-800'
													: 'bg-amber-100/80 text-amber-800'}"
											>
												{sm.data_source === 'log' ? 'Log: PHIT/PERM' : 'Core: CPORE/CPERM'}
											</span>
											<span>{(sm.features_used ?? []).join('·')}</span>
										</span>
										<span class="mono-num font-medium text-[var(--text-primary)]">
											Acc:{smAcc} · F1:{smF1} · κ:{smKappa}
										</span>
									</div>
								</div>
							{/each}
						</div>
					{:else if classifierDiag?.trained}
						<div
							class="flex items-center justify-between rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 py-1 text-[9.5px]"
						>
							<span>
								In-memory model ({(classifierDiag.features_used ?? []).join('+')}) · Acc:
								<strong class="mono-num"
									>{(Number(classifierDiag.metrics?.accuracy ?? 0) * 100).toFixed(1)}%</strong
								>
							</span>
							<button
								type="button"
								disabled={projectData.isComputingMultiWell}
								onclick={() => executeRockTyping(false, true, true)}
								class="rounded-xs border border-emerald-600 bg-emerald-600 px-1.5 py-0.5 text-[9px] font-semibold text-white hover:opacity-95 disabled:opacity-50"
							>
								Propagate
							</button>
						</div>
					{:else}
						<div class="rounded-xs border border-dashed border-[var(--border-hairline)] bg-[var(--bg-surface)] p-2 text-[9.5px] text-[var(--text-secondary)]">
							No saved Rock Type classifier models yet. Use the <strong>[DRT] RT Classifier</strong> standalone module to train, validate with a Contingency Matrix, and save a model.
						</div>
					{/if}
				</div>

				<!-- 2) Focused / Filtered Interval Propagation Scope -->
				<div
					class="space-y-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]"
				>
					<label
						class="flex cursor-pointer items-center justify-between gap-2 font-semibold text-[var(--text-primary)]"
					>
						<span>2. Propagate at Focused / Filtered Intervals Only</span>
						<input
							type="checkbox"
							bind:checked={projectData.drtPropagateFocusedOnly}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] accent-[var(--accent-primary)]"
						/>
					</label>
					<div class="text-[9.5px] text-[var(--text-secondary)]">
						Restricts ML DRT propagation to active zone filters, depth filter bounds, and evaluated
						reservoir intervals.
					</div>
					{#if projectData.allZones.length > 0}
						<div class="flex flex-wrap items-center gap-1 pt-0.5">
							<span class="text-[9px] font-medium text-[var(--text-muted)]">Zones:</span>
							<button
								type="button"
								onclick={() => (workbench.selectedZones = [])}
								class="rounded-xs border px-1 py-0.5 text-[9px] {workbench.selectedZones.length ===
								0
									? 'border-slate-700 bg-slate-800 font-semibold text-white'
									: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
							>
								All Zones
							</button>
							{#each projectData.allZones as z}
								{@const zActive = workbench.selectedZones.includes(z)}
								<button
									type="button"
									onclick={() => toggleZoneFilter(z)}
									class="rounded-xs border px-1 py-0.5 text-[9px] {zActive
										? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
										: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
								>
									{z}
								</button>
							{/each}
						</div>
					{/if}
					{#if classifierDiag?.focused_interval}
						{@const fi = classifierDiag.focused_interval}
						<div
							class="flex items-center justify-between rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[9px] text-[var(--text-secondary)]"
						>
							<span>
								Focused samples:
								<strong class="mono-num text-[var(--text-primary)]"
									>{fi.focused_samples} / {fi.total_well_samples}</strong
								>
								{#if fi.excluded_samples > 0}
									<span class="text-[var(--text-muted)]">({fi.excluded_samples} excluded)</span>
								{/if}
							</span>
							{#if fi.min_depth !== null && fi.max_depth !== null}
								<span class="mono-num font-medium text-[var(--text-primary)]">
									{fi.min_depth}–{fi.max_depth} m
								</span>
							{/if}
						</div>
					{/if}
				</div>

				<!-- 3) Comparison of Cutoff ROCK_FLAG vs ML Predicted ROCK_FLAG & Workflow Selection -->
				{#if classifierDiag?.comparison}
					{@const comp = classifierDiag.comparison}
					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]"
					>
						<div class="flex items-center justify-between">
							<span class="font-semibold text-[var(--text-primary)]">
								3. Cutoff vs. ML ROCK_FLAG Comparison
							</span>
							<span class="mono-num text-[9px] font-semibold text-emerald-700">
								Agree: {(Number(comp.exact_agreement ?? 0) * 100).toFixed(1)}% (±1: {(Number(comp.within_one_agreement ?? 0) * 100).toFixed(1)}%)
							</span>
						</div>

						<div class="grid grid-cols-3 gap-1 text-center text-[9px]">
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0.5">
								<div class="text-[8.5px] text-[var(--text-muted)]">Pure Cutoffs</div>
								<div class="mono-num font-bold text-[var(--text-primary)]">
									{comp.cutoff_total} pts
								</div>
							</div>
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0.5">
								<div class="text-[8.5px] text-[var(--text-muted)]">ML Focused</div>
								<div class="mono-num font-bold text-[var(--text-primary)]">
									{comp.ml_total} pts
								</div>
							</div>
							<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] py-0.5">
								<div class="text-[8.5px] text-[var(--text-muted)]">Uncored Filled</div>
								<div class="mono-num font-bold text-emerald-700">
									+{comp.uncored_filled_count} pts
								</div>
							</div>
						</div>

						{#if comp.per_drt?.length}
							<div class="overflow-x-auto">
								<table class="w-full border-collapse text-center text-[9px]">
									<thead>
										<tr class="border-b border-[var(--border-hairline)] text-[var(--text-muted)]">
											<th class="px-1 py-0.5 text-left font-medium">Group</th>
											<th class="px-1 py-0.5 font-medium">Cutoffs</th>
											<th class="px-1 py-0.5 font-medium">ML Pred</th>
											<th class="px-1 py-0.5 font-medium">Cut+ML</th>
											<th class="px-1 py-0.5 font-medium">Δ(ML−Cut)</th>
										</tr>
									</thead>
									<tbody>
										{#each comp.per_drt as row}
											<tr class="border-b border-[var(--border-hairline)] last:border-b-0">
												<td class="mono-num px-1 py-0.5 text-left font-semibold">
													<span
														style="background-color: {getRockTypeColor(row.drt)}"
														class="mr-1 inline-block h-2 w-2 rounded-full"
													></span>DRT {row.drt}
												</td>
												<td class="mono-num px-1 py-0.5">
													{row.cutoff_count}
													<span class="text-[8.5px] text-[var(--text-muted)]"
														>({row.cutoff_pct}%)</span
													>
												</td>
												<td class="mono-num px-1 py-0.5">
													{row.ml_count}
													<span class="text-[8.5px] text-[var(--text-muted)]"
														>({row.ml_pct}%)</span
													>
												</td>
												<td class="mono-num px-1 py-0.5 font-medium">
													{row.hybrid_count}
												</td>
												<td
													class="mono-num px-1 py-0.5 font-semibold {row.delta_count > 0
														? 'text-emerald-700'
														: row.delta_count < 0
															? 'text-amber-700'
															: 'text-[var(--text-muted)]'}"
												>
													{row.delta_count > 0 ? `+${row.delta_count}` : row.delta_count}
												</td>
											</tr>
										{/each}
									</tbody>
								</table>
							</div>
						{/if}

						<!-- Active ROCK_FLAG Selector for Next Workflow -->
						<div class="space-y-1 pt-0.5">
							<div class="text-[9.5px] font-semibold text-[var(--text-primary)]">
								4. Select Active ROCK_FLAG for Next Workflow (Perm Transform &amp; SHF):
							</div>
							<div class="grid grid-cols-3 gap-1">
								<button
									type="button"
									disabled={projectData.isComputingMultiWell}
									onclick={() => selectRockFlagSourceForWorkflow('cutoff_only')}
									class="flex flex-col items-center rounded-xs border px-1 py-1 text-center transition-colors {projectData.drtPropagationMode ===
										'cutoff_only' || !projectData.drtClassifierPropagated
										? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
										: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
									title="Use purely cutoff-binned ROCK_FLAG_CUTOFF for downstream Perm Transform and SHF"
								>
									<span class="text-[9.5px]">Pure Cutoffs</span>
									<span class="mono-num text-[8.5px] opacity-80">{comp.cutoff_total} pts</span>
								</button>
								<button
									type="button"
									disabled={projectData.isComputingMultiWell}
									onclick={() => selectRockFlagSourceForWorkflow('fill_missing')}
									class="flex flex-col items-center rounded-xs border px-1 py-1 text-center transition-colors {projectData.drtClassifierPropagated &&
									projectData.drtPropagationMode === 'fill_missing'
										? 'border-emerald-600 bg-emerald-50 font-semibold text-emerald-800'
										: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
									title="Keep cutoff DRTs where available and fill uncored focused intervals with ML predictions"
								>
									<span class="text-[9.5px]">Cutoffs + ML Fill</span>
									<span class="mono-num text-[8.5px] opacity-80">
										{comp.cutoff_total + comp.uncored_filled_count} pts
									</span>
								</button>
								<button
									type="button"
									disabled={projectData.isComputingMultiWell}
									onclick={() => selectRockFlagSourceForWorkflow('full_ml')}
									class="flex flex-col items-center rounded-xs border px-1 py-1 text-center transition-colors {projectData.drtClassifierPropagated &&
									projectData.drtPropagationMode === 'full_ml'
										? 'border-emerald-600 bg-emerald-50 font-semibold text-emerald-800'
										: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
									title="Use ML predicted ROCK_FLAG_ML across the entire focused interval"
								>
									<span class="text-[9.5px]">Full ML Pred</span>
									<span class="mono-num text-[8.5px] opacity-80">{comp.ml_total} pts</span>
								</button>
							</div>
						</div>
					</div>
				{:else}
					<!-- Active ROCK_FLAG Selector even before ML propagation -->
					<div
						class="space-y-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]"
					>
						<div class="text-[9.5px] font-semibold text-[var(--text-primary)]">
							3. Active ROCK_FLAG for Next Workflow (Perm Transform &amp; SHF):
						</div>
						<div class="grid grid-cols-3 gap-1">
							<button
								type="button"
								disabled={projectData.isComputingMultiWell}
								onclick={() => selectRockFlagSourceForWorkflow('cutoff_only')}
								class="flex flex-col items-center rounded-xs border px-1 py-1 text-center transition-colors {projectData.drtPropagationMode ===
									'cutoff_only' || !projectData.drtClassifierPropagated
									? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
									: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
							>
								<span class="text-[9.5px]">Pure Cutoffs</span>
								<span class="text-[8.5px] opacity-80">Active</span>
							</button>
							<button
								type="button"
								disabled={
									projectData.isComputingMultiWell ||
									(!projectData.drtSelectedSavedModelKey &&
										!savedModelsList[0]?.model_key &&
										!classifierDiag?.trained)
								}
								onclick={() => {
									const keyToUse =
										projectData.drtSelectedSavedModelKey ||
										savedModelsList[0]?.model_key ||
										'';
									if (keyToUse) {
										projectData.drtPropagationMode = 'fill_missing';
										void applySavedClassifierModel(keyToUse);
									} else {
										void selectRockFlagSourceForWorkflow('fill_missing');
									}
								}}
								class="flex flex-col items-center rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 py-1 text-center text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] disabled:opacity-45"
							>
								<span class="text-[9.5px]">Cutoffs + ML Fill</span>
								<span class="text-[8.5px] opacity-80">Propagate</span>
							</button>
							<button
								type="button"
								disabled={
									projectData.isComputingMultiWell ||
									(!projectData.drtSelectedSavedModelKey &&
										!savedModelsList[0]?.model_key &&
										!classifierDiag?.trained)
								}
								onclick={() => {
									const keyToUse =
										projectData.drtSelectedSavedModelKey ||
										savedModelsList[0]?.model_key ||
										'';
									if (keyToUse) {
										projectData.drtPropagationMode = 'full_ml';
										void applySavedClassifierModel(keyToUse);
									} else {
										void selectRockFlagSourceForWorkflow('full_ml');
									}
								}}
								class="flex flex-col items-center rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 py-1 text-center text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] disabled:opacity-45"
							>
								<span class="text-[9.5px]">Full ML Pred</span>
								<span class="text-[8.5px] opacity-80">Propagate</span>
							</button>
						</div>
					</div>
				{/if}
			</div>
		</ParameterSection>

		{#if projectData.rockTypingResult?.rock_type_counts}
			<ParameterSection
				title="Active Discrete Rock Type (DRT) Counts"
				badge="{(projectData.rockTypingResult.data_source ?? projectData.drtDataSource) === 'log' ? 'Log: PHIT & PERM' : 'Core: CPORE & CPERM'} · {projectData.drtClassifierPropagated
					? projectData.drtPropagationMode === 'full_ml'
						? 'Full ML Predicted'
						: `${activeMeta.shortName} + ML Fill`
					: projectData.luciaVuggyOverride && projectData.drtMethod !== 'lucia'
						? `${activeMeta.shortName} + RFN Override`
						: `${activeMeta.shortName} Cutoffs Only`}"
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

		{#if projectData.error}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{projectData.error}
			</div>
		{/if}
	</div>

	<div
		class="grid grid-cols-2 gap-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2"
	>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell}
			onclick={() =>
				executeRockTyping(
					false,
					Boolean(classifierDiag?.trained && !projectData.drtSelectedSavedModelKey),
					projectData.drtClassifierPropagated,
					projectData.drtSelectedSavedModelKey || null
				)}
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
			Save All Wells
		</button>
	</div>
</div>
