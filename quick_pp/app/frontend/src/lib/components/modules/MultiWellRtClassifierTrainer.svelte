<script lang="ts">
	import { onMount } from 'svelte';
	import { API_BASE, api, authHeaders } from '$lib/api/client';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import {
		projectData,
		type DrtFeatureKey,
		type DrtMethod,
		type PhiKDataSource
	} from '$lib/state/projectData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { getRockTypeColor } from '$lib/utils/petrophysicsColors';

	const ALL_RT_FEATURES: Array<{ key: DrtFeatureKey; label: string; short: string; title: string }> = [
		{ key: 'GR', label: 'GR', short: 'GR', title: 'Gamma Ray (API)' },
		{ key: 'NPHI', label: 'NPHI', short: 'NPHI', title: 'Neutron Porosity (v/v)' },
		{ key: 'RHOB', label: 'RHOB', short: 'RHOB', title: 'Bulk Density (g/cc)' },
		{ key: 'LOG_RT', label: 'log₁₀(RT)', short: 'log(RT)', title: 'Log10 Deep Resistivity (ohm.m)' },
		{ key: 'PEF', label: 'PEF', short: 'PEF', title: 'Photoelectric Factor (b/e)' },
		{ key: 'VDL', label: 'VDL', short: 'VDL', title: 'Velocity Deviation Log (Φ_T − Φ_sonic)' }
	];

	const DRT_METHOD_META: Record<
		DrtMethod,
		{ label: string; shortName: string; higherIsBetter: boolean; unit: string }
	> = {
		fzi: { label: 'FZI / RQI', shortName: 'FZI', higherIsBetter: true, unit: '' },
		winland: { label: 'Winland R35', shortName: 'R35', higherIsBetter: true, unit: 'μm' },
		lucia: { label: 'Lucia RFN', shortName: 'RFN', higherIsBetter: false, unit: 'RFN' }
	};

	let customModelName = $state<string>('');
	let isSavingModel = $state<boolean>(false);
	let previewedSavedModel = $state<Record<string, any> | null>(null);

	let activeMeta = $derived(DRT_METHOD_META[projectData.drtMethod]);
	let diagnostics = $derived(projectData.rockTypingResult?.diagnostics ?? null);
	let classifierDiag = $derived(diagnostics?.classifier ?? null);
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
		workbench.crossplotTemplateOverride = 'phi-k';
		projectData.rockTypingPlotView = 'classifier';
		loadSavedClassifierModels();
	});

	async function trainAndValidateClassifier() {
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
						auto_find_cutoffs: false,
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
						train_classifier: true,
						apply_classifier_propagation: projectData.drtClassifierPropagated,
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
						saved_model_key: null,
						save: false
					} as any
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Rock type classifier training failed'));
			}
			if ((data as any).diagnostics?.models_status) {
				projectData.rtSavedModelsStatus = (data as any).diagnostics.models_status;
			}
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
			workbench.crossplotTemplateOverride = 'phi-k';
			projectData.rockTypingPlotView = 'classifier';
		} catch (e: any) {
			projectData.error = e?.message || 'Rock type classifier training error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

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
			const srcTag = projectData.drtDataSource === 'log' ? 'Log' : 'Core';
			const defaultName = `${algoShort} (${(classifierDiag.features_used ?? []).join('+')}) ${activeMeta.shortName}-${srcTag} ${accPct}`.trim();
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
		const nextClf = {
			...(projectData.rockTypingResult?.diagnostics?.classifier ?? {}),
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
		};
		if (projectData.rockTypingResult) {
			projectData.rockTypingResult = {
				...projectData.rockTypingResult,
				diagnostics: {
					...(projectData.rockTypingResult.diagnostics ?? {}),
					classifier: nextClf
				}
			};
		} else {
			projectData.rockTypingResult = {
				method: projectData.drtMethod,
				well_names: [],
				depths: [],
				phit: [],
				perm: [],
				fzi: [],
				rock_flags: [],
				rock_type_counts: {},
				diagnostics: {
					classifier: nextClf
				}
			};
		}
		workbench.crossplotTemplateOverride = 'phi-k';
		projectData.rockTypingPlotView = 'classifier';
	}

	function useModelInRockTyping(modelKey: string) {
		const matched = savedModelsList.find((m) => m.model_key === modelKey);
		if (matched?.data_source === 'core' || matched?.data_source === 'log') {
			projectData.drtDataSource = matched.data_source as PhiKDataSource;
		}
		projectData.drtSelectedSavedModelKey = modelKey;
		if (projectData.drtPropagationMode === 'cutoff_only') {
			projectData.drtPropagationMode = 'fill_missing';
		}
		workbench.multiWellStep = 'rock-typing';
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection
			title="Target DRT Labels & Log Features"
			badge="{projectData.drtDataSource === 'core' ? 'Core (CPORE & CPERM)' : 'Log (PHIT & PERM)'} · {projectData.drtSelectedFeatures.length}/6 Inputs"
			defaultOpen={true}
		>
			<div class="space-y-2">
				<!-- Target DRT Label Data Source Selector (Core vs Log) -->
				<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]">
					<div class="mb-1 flex items-center justify-between text-[9.5px]">
						<span class="font-semibold text-[var(--text-primary)]">
							Target DRT Ground-Truth Label Source:
						</span>
						<span class="mono-num text-[9px] text-[var(--text-muted)]">
							{projectData.drtDataSource === 'core' ? 'CPORE & CPERM' : 'PHIT & PERM'}
						</span>
					</div>
					<div class="grid grid-cols-2 gap-1">
						<button
							type="button"
							onclick={() => (projectData.drtDataSource = 'core')}
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
								{projectData.wellsWithCorePhiK.length}/{projectData.activeWells.length} wells · {projectData.totalCorePlugCount} plugs
							</span>
						</button>
						<button
							type="button"
							onclick={() => (projectData.drtDataSource = 'log')}
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
							Classifier trains against <span class="mono-num font-semibold">{activeMeta.shortName}</span> DRT labels derived from
							<strong class="mono-num text-[var(--text-primary)]">Core Data (CPORE &amp; CPERM)</strong>.
						{:else}
							Classifier trains against <span class="mono-num font-semibold">{activeMeta.shortName}</span> DRT labels derived from
							<strong class="mono-num text-[var(--text-primary)]">Calculated Logs (PHIT &amp; PERM)</strong>.
						{/if}
					</div>
				</div>

				<!-- Target DRT summary + shortcut back to Step 3 -->
				<div
					class="flex items-center justify-between gap-2 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1.5 text-[10px]"
				>
					<div>
						<div class="font-semibold text-[var(--text-primary)]">
							Target DRT: {activeMeta.label} ({projectData.activeDrtCutoffs.length + 1} Groups)
						</div>
						<div class="mono-num text-[9.5px] text-[var(--text-secondary)]">
							Cutoffs: {[...projectData.activeDrtCutoffs]
								.sort((a, b) => a - b)
								.join(', ')}{activeMeta.unit ? ` ${activeMeta.unit}` : ''}
						</div>
					</div>
					<button
						type="button"
						onclick={() => (workbench.multiWellStep = 'rock-typing')}
						class="shrink-0 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[9.5px] font-semibold text-[var(--accent-primary)] hover:bg-blue-50"
					>
						← Edit Cutoffs
					</button>
				</div>

				<!-- 1. Selectable Log Features -->
				<div
					class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]"
				>
					<div class="flex items-center justify-between gap-1">
						<span class="font-semibold text-[var(--text-primary)]">
							1. Selectable Log Features ({projectData.drtSelectedFeatures.length}/6)
						</span>
						<div class="flex items-center gap-1">
							<button
								type="button"
								onclick={() =>
									setFeaturePreset(['GR', 'NPHI', 'RHOB', 'LOG_RT', 'PEF', 'VDL'])}
								class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 py-0.5 text-[9px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
								title="Select all 6 candidate log features"
							>
								All
							</button>
							<button
								type="button"
								onclick={() => setFeaturePreset(['NPHI', 'RHOB', 'PEF', 'VDL'])}
								class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 py-0.5 text-[9px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
								title="Exclude GR and log10(RT) when they do not contribute to rock typing"
							>
								No GR/RT
							</button>
							<button
								type="button"
								onclick={() => setFeaturePreset(['NPHI', 'RHOB', 'VDL'])}
								class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 py-0.5 text-[9px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
								title="Porosity & Velocity Deviation only"
							>
								Φ+VDL
							</button>
						</div>
					</div>

					<div class="grid grid-cols-3 gap-1">
						{#each ALL_RT_FEATURES as feat}
							{@const isSelected = projectData.drtSelectedFeatures.includes(feat.key)}
							<button
								type="button"
								title={feat.title}
								onclick={() => toggleFeature(feat.key)}
								class="flex h-6 items-center justify-between rounded-xs border px-1.5 text-[9.5px] font-medium transition-colors {isSelected
									? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
									: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-muted)] hover:bg-[var(--bg-subtle)]'}"
							>
								<span class="mono-num">{feat.label}</span>
								<span class="text-[9px]">{isSelected ? '✓' : '○'}</span>
							</button>
						{/each}
					</div>
				</div>

				<!-- 2. Focused / Filtered Interval Scope -->
				<div
					class="space-y-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]"
				>
					<label
						class="flex cursor-pointer items-center justify-between gap-2 font-semibold text-[var(--text-primary)]"
					>
						<span>2. Train on Focused / Filtered Intervals Only</span>
						<input
							type="checkbox"
							bind:checked={projectData.drtPropagateFocusedOnly}
							class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] accent-[var(--accent-primary)]"
						/>
					</label>
					<div class="text-[9.5px] text-[var(--text-secondary)]">
						Restricts training &amp; validation samples to active zone filters, depth filter bounds,
						and evaluated reservoir intervals.
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

				<!-- 3. Algorithm & Holdout Split -->
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
			</div>
		</ParameterSection>

		{#if activeDisplayedClassifier}
			{@const ct = activeDisplayedClassifier.contingency_table}
			{@const m = activeDisplayedClassifier.metrics}
			<ParameterSection
				title="Contingency Matrix & Validation KPIs"
				badge={m?.accuracy !== undefined
					? `Acc: ${(Number(m.accuracy) * 100).toFixed(1)}%`
					: 'Validated'}
				defaultOpen={true}
			>
				<div class="space-y-2 text-[10px]">
					{#if activeDisplayedClassifier.saved_model_name}
						<div
							class="flex items-center justify-between rounded-xs bg-blue-50 px-1.5 py-1 text-[9.5px] font-semibold text-[var(--accent-primary)]"
						>
							<span class="truncate">Inspecting: {activeDisplayedClassifier.saved_model_name}</span>
							{#if previewedSavedModel}
								<button
									type="button"
									onclick={() => useModelInRockTyping(previewedSavedModel?.model_key)}
									class="underline"
								>
									Use in Rock Typing →
								</button>
							{/if}
						</div>
					{/if}

					<!-- Validation KPI Banner -->
					<div class="grid grid-cols-3 gap-1 text-center">
						<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] py-1">
							<div class="text-[9px] text-[var(--text-muted)]">Accuracy</div>
							<div class="mono-num font-bold text-[var(--text-primary)]">
								{(Number(m?.accuracy ?? 0) * 100).toFixed(1)}%
							</div>
						</div>
						<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] py-1">
							<div class="text-[9px] text-[var(--text-muted)]">Weighted F1</div>
							<div class="mono-num font-bold text-[var(--text-primary)]">
								{Number(m?.f1_weighted ?? 0).toFixed(3)}
							</div>
						</div>
						<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] py-1">
							<div class="text-[9px] text-[var(--text-muted)]">Cohen's κ</div>
							<div class="mono-num font-bold text-[var(--text-primary)]">
								{Number(m?.cohen_kappa ?? 0).toFixed(3)}
							</div>
						</div>
					</div>

					<!-- Compact Contingency Table (Confusion Matrix) -->
					{#if ct?.labels?.length && ct?.counts?.length}
						<div class="overflow-x-auto rounded-xs border border-[var(--border-hairline)]">
							<table class="w-full border-collapse text-center text-[9.5px]">
								<thead>
									<tr class="border-b border-[var(--border-hairline)] bg-[var(--bg-app)] text-[var(--text-muted)]">
										<th class="px-1.5 py-1 text-left font-medium">True \ Pred</th>
										{#each ct.labels as colLbl}
											<th class="mono-num px-1 py-1 font-semibold">DRT {colLbl}</th>
										{/each}
										<th class="px-1.5 py-1 font-medium">Recall</th>
									</tr>
								</thead>
								<tbody>
									{#each ct.labels as rowLbl, rIdx}
										{@const rowCounts = ct.counts[rIdx] ?? []}
										{@const rowNorm = ct.norm_recall?.[rIdx] ?? []}
										{@const diagRec = Number(rowNorm[rIdx] ?? 0)}
										<tr class="border-b border-[var(--border-hairline)] last:border-b-0">
											<td class="mono-num px-1.5 py-1 text-left font-semibold">
												<span
													style="background-color: {getRockTypeColor(rowLbl)}"
													class="mr-1 inline-block h-2 w-2 rounded-full"
												></span>DRT {rowLbl}
											</td>
											{#each rowCounts as cellVal, cIdx}
												{@const isDiag = rIdx === cIdx}
												<td
													title={`True DRT ${rowLbl} → Pred DRT ${ct.labels[cIdx]}: ${cellVal} samples (${((rowNorm[cIdx] ?? 0) * 100).toFixed(1)}%)`}
													class="mono-num px-1 py-1 {isDiag
														? 'bg-emerald-50 font-bold text-emerald-800'
														: cellVal > 0
															? 'bg-amber-50/60 text-amber-900'
															: 'text-[var(--text-muted)]'}"
												>
													{cellVal}
												</td>
											{/each}
											<td class="mono-num px-1.5 py-1 font-semibold text-[var(--text-primary)]">
												{(diagRec * 100).toFixed(0)}%
											</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					{/if}

					<!-- Feature Importance Bars -->
					{#if activeDisplayedClassifier.feature_importances?.length}
						<div class="space-y-1 pt-0.5">
							<div class="text-[9.5px] font-semibold text-[var(--text-muted)]">
								Feature Importance (MDI)
							</div>
							{#each activeDisplayedClassifier.feature_importances as fi}
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

					<!-- Save & Name Trained Classifier Model -->
					{#if classifierDiag && !previewedSavedModel}
						<div class="space-y-1 border-t border-[var(--border-hairline)] pt-2">
							<div class="text-[9.5px] font-semibold text-[var(--text-primary)]">
								Save &amp; Name Trained Classifier (with Performance &amp; CM)
							</div>
							<div class="flex items-center gap-1">
								<input
									type="text"
									bind:value={customModelName}
									placeholder={`e.g. ${classifierDiag.model_type === 'extra_trees' ? 'ET' : classifierDiag.model_type === 'gradient_boosting' ? 'GB' : 'RF'} (${(classifierDiag.features_used ?? []).join('+')}) ${activeMeta.shortName}`}
									class="h-6.5 flex-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[10px] text-[var(--text-primary)]"
								/>
								<button
									type="button"
									disabled={isSavingModel || projectData.isComputingMultiWell}
									onclick={saveTrainedClassifierModel}
									class="h-6.5 shrink-0 rounded-xs border border-[var(--accent-primary)] bg-blue-50 px-2.5 text-[10px] font-semibold text-[var(--accent-primary)] hover:bg-blue-100 disabled:opacity-50"
								>
									{isSavingModel ? 'Saving...' : 'Save Model & CM'}
								</button>
							</div>
						</div>
					{/if}
				</div>
			</ParameterSection>
		{/if}

		<ParameterSection
			title="Saved Classifier Models Registry"
			badge={`${savedModelsList.length} Saved`}
			defaultOpen={true}
		>
			{#if savedModelsList.length === 0}
				<p class="text-[10.5px] text-[var(--text-secondary)]">
					No saved Rock Type classifiers yet. Train a model above and click <strong>Save Model &amp; CM</strong> to register it for selection in Step 3 (Rock Typing).
				</p>
			{:else}
				<div class="space-y-1.5 text-[10px]">
					{#each savedModelsList as sm}
						{@const isActiveModel =
							projectData.drtSelectedSavedModelKey === sm.model_key ||
							previewedSavedModel?.model_key === sm.model_key}
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
							class="rounded-xs border p-1.5 transition-colors {isActiveModel
								? 'border-[var(--accent-primary)] bg-blue-50/60'
								: 'border-[var(--border-hairline)] bg-[var(--bg-app)]'}"
						>
							<div class="flex items-center justify-between gap-1">
								<span class="truncate font-semibold text-[var(--text-primary)]" title={sm.model_name}>
									{sm.model_name}
								</span>
								<div class="flex shrink-0 items-center gap-1">
									<button
										type="button"
										onclick={() => inspectSavedClassifierModel(sm)}
										class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[9px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
										title="Inspect saved performance metrics & Contingency Matrix"
									>
										Inspect CM
									</button>
									<button
										type="button"
										onclick={() => useModelInRockTyping(sm.model_key)}
										class="rounded-xs border border-emerald-600 bg-emerald-600 px-1.5 py-0.5 text-[9px] font-semibold text-white hover:opacity-95"
										title="Select this model for Step 3 Rock Typing propagation"
									>
										Use in Rock Typing →
									</button>
									<button
										type="button"
										onclick={() => deleteSavedClassifierModel(sm.model_key)}
										class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 py-0.5 text-[9px] text-[var(--text-muted)] hover:bg-red-50 hover:text-red-600"
										title="Delete saved classifier"
									>
										×
									</button>
								</div>
							</div>
							<div
								class="mt-1 flex flex-wrap items-center justify-between gap-1 text-[9px] text-[var(--text-secondary)]"
							>
								<span class="mono-num flex items-center gap-1">
									<span
										class="rounded-xs px-1 py-0.2 text-[8px] font-semibold {sm.data_source ===
										'log'
											? 'bg-indigo-100/80 text-indigo-800'
											: 'bg-amber-100/80 text-amber-800'}"
									>
										{sm.data_source === 'log' ? 'Log: PHIT/PERM' : 'Core: CPORE/CPERM'}
									</span>
									<span>
										{(sm.features_used ?? []).join('·')}
										{#if sm.drt_method}
											· {String(sm.drt_method).toUpperCase()}
										{/if}
									</span>
								</span>
								<span class="mono-num font-medium text-[var(--text-primary)]">
									Acc:{smAcc} · F1:{smF1} · κ:{smKappa}
								</span>
							</div>
						</div>
					{/each}
				</div>
			{/if}
		</ParameterSection>

		{#if projectData.error}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{projectData.error}
			</div>
		{/if}
	</div>

	<div class="space-y-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2">
		<div class="grid grid-cols-2 gap-1.5">
			<button
				type="button"
				disabled={projectData.isComputingMultiWell}
				onclick={trainAndValidateClassifier}
				class="h-7 rounded-sm bg-[var(--accent-primary)] text-xs font-medium text-white transition-opacity hover:opacity-95 disabled:opacity-50"
			>
				{projectData.isComputingMultiWell
					? 'Training...'
					: 'Train & Validate (⌘+Enter)'}
			</button>
			<button
				type="button"
				disabled={!classifierDiag || Boolean(previewedSavedModel) || isSavingModel || projectData.isComputingMultiWell}
				onclick={saveTrainedClassifierModel}
				class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-xs font-medium text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-40"
			>
				{isSavingModel ? 'Saving...' : 'Save Model & CM'}
			</button>
		</div>
		<button
			type="button"
			onclick={() => (workbench.multiWellStep = 'rock-typing')}
			class="h-6.5 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[11px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
		>
			← Back to Rock Typing &amp; Propagation
		</button>
	</div>
</div>

