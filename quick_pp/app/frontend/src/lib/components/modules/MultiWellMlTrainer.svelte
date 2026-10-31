<script lang="ts">
	import { deleteJson, getJson, postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import SavedParamsIndicator from '$lib/components/workbench/SavedParamsIndicator.svelte';
	import { authState } from '$lib/state/auth.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	const RESTRICTED_RAW_INPUTS = ['GR', 'RT', 'NPHI', 'RHOB'];
	const RESTRICTED_FE_INPUTS = [
		'RHOB_INT',
		'DPHI',
		'GAS_XOVER',
		'RHOMA_APP',
		'AI',
		'M_LITH',
		'N_LITH'
	];
	const CLASTIC_OUTPUTS = ['PHIT', 'SWT', 'LOG_PERM', 'VSAND', 'VSILT', 'VCLAY'];
	const CARBONATE_OUTPUTS = ['PHIT', 'SWT', 'LOG_PERM', 'VCALC', 'VCLAY', 'VDOLO'];

	// Training configuration state
	// Training configuration lives in shared project state so it is saved/restored per zone selection;
	// zones use the shared workbench zone selection (same key as other multi-well modules).
	const mp = projectData.mlTrainerParams;

	// Well, Zone & Flag Filter state

	// Hyperparameters & Conformal calibration

	// Global bundle status
	let bundleStatus = $state<{
		pretrained_available: boolean;
		fallback_active: boolean;
		model_path: string | null;
		expected_path: string;
		suites: Array<{
			key: string;
			label: string;
			architecture?: string;
			pretrained: boolean;
			sub_models?: string[];
			targets?: string[];
			meta?: Record<string, any>;
		}>;
		notice: string;
	} | null>(null);

	let activeTargets = $derived(
		mp.architecture === 'carbonate' ? CARBONATE_OUTPUTS : CLASTIC_OUTPUTS
	);

	let lastInitProjectId = '';

	// Refresh bundle status once per project; keep training wells valid for the active project without
	// overwriting a restored / user-edited well selection when the inspector remounts.
	$effect(() => {
		const pid = workbench.projectId ? String(workbench.projectId) : '';
		const allNames = projectData.wells.map((w) => w.name);
		if (pid && pid !== lastInitProjectId) {
			lastInitProjectId = pid;
			refreshBundleStatus(pid);
		}
		if (allNames.length > 0) {
			const valid = mp.selectedWells.filter((w) => allNames.includes(w));
			if (valid.length === 0) {
				mp.selectedWells = [...allNames];
			} else if (valid.length !== mp.selectedWells.length) {
				mp.selectedWells = valid;
			}
		}
	});

	function handleArchitectureChange(nextArch: 'clastic' | 'carbonate') {
		mp.architecture = nextArch;
		if (mp.suiteKey === 'clastic_custom' || mp.suiteKey === 'carbonate_custom') {
			mp.suiteKey = `${nextArch}_custom`;
		}
		if (
			mp.suiteLabel === 'Custom Clastic PIML' ||
			mp.suiteLabel === 'Custom Carbonate PIML'
		) {
			mp.suiteLabel = `Custom ${nextArch === 'carbonate' ? 'Carbonate' : 'Clastic'} PIML`;
		}
	}

	function toggleWell(wellName: string) {
		if (mp.selectedWells.includes(wellName)) {
			if (mp.selectedWells.length <= 1) return;
			mp.selectedWells = mp.selectedWells.filter((w) => w !== wellName);
		} else {
			mp.selectedWells = [...mp.selectedWells, wellName];
		}
	}

	function toggleZone(zoneName: string) {
		if (workbench.selectedZones.includes(zoneName)) {
			workbench.selectedZones = workbench.selectedZones.filter((z) => z !== zoneName);
		} else {
			workbench.selectedZones = [...workbench.selectedZones, zoneName];
		}
	}

	function toggleRockFlag(rf: number) {
		if (mp.selectedRockFlags.includes(rf)) {
			mp.selectedRockFlags = mp.selectedRockFlags.filter((r) => r !== rf);
		} else {
			mp.selectedRockFlags = [...mp.selectedRockFlags, rf].sort((a, b) => a - b);
		}
	}

	function checkWellReady(w: any): { hasRaw: boolean; hasTargets: boolean } {
		const curves: string[] = w.curves ?? w.available_curves ?? [];
		const hasRaw = RESTRICTED_RAW_INPUTS.every((c) => curves.includes(c));
		const hasPoro = curves.includes('PHIT') || curves.includes('PHIE');
		const hasSw = curves.includes('SWT') || curves.includes('SW') || curves.includes('SWE');
		const hasPerm =
			curves.includes('PERM') || curves.includes('LOG_PERM') || curves.includes('PERM_TRANS');
		return { hasRaw, hasTargets: hasPoro && hasSw && hasPerm };
	}

	async function refreshBundleStatus(projectId: string | number) {
		try {
			const res = await getJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(projectId)}/ml_models`
			);
			if (res) {
				bundleStatus = res;
			}
		} catch {
			// Ignore status probe error
		}
	}

	let isSavingModel = $state<boolean>(false);

	async function executeTrainModel() {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/ml_models/train`,
				{
					well_names: mp.selectedWells.length > 0 ? mp.selectedWells : null,
					zones: workbench.selectedZones.length > 0 ? workbench.selectedZones : null,
					architecture: mp.architecture,
					suite_key: mp.suiteKey.trim() || `${mp.architecture}_custom`,
					suite_label: mp.suiteLabel.trim() || null,
					exclude_coal: mp.excludeCoal,
					exclude_tight: mp.excludeTight,
					exclude_bad_hole: mp.excludeBadHole,
					rock_flags: mp.selectedRockFlags.length > 0 ? mp.selectedRockFlags : null,
					test_size: mp.testSize,
					alpha: mp.conformalAlpha,
					n_estimators: Math.round(mp.nEstimators),
					max_depth: Math.round(mp.maxDepth),
					save: false
				}
			);
			projectData.mlTrainingResult = res;
			projectData.pendingProjectParamSnapshots['ml-training'] =
				projectData.snapshotProjectParams('ml-training');
			if (res.models_status) {
				bundleStatus = res.models_status;
			}
			wellData.lastSavedMessage = `Evaluated '${res.suite_key}' (R²=${fmtNum(res.overall_metrics?.r2_score, 3)})`;
		} catch (e: any) {
			projectData.error = e?.message || 'Multi-well ML training failed';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function executeSaveModel() {
		if (!workbench.projectId || !projectData.mlTrainingResult) return;
		isSavingModel = true;
		projectData.error = null;
		try {
			const res = await postJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/ml_models/save`,
				{
					suite_key: mp.suiteKey.trim() || `${mp.architecture}_custom`,
					suite_label: mp.suiteLabel.trim() || null
				}
			);
			projectData.mlTrainingResult = {
				...projectData.mlTrainingResult,
				saved: true,
				suite_key: res.suite_key ?? projectData.mlTrainingResult.suite_key,
				suite_label: res.suite_label ?? projectData.mlTrainingResult.suite_label,
				model_path: res.model_path ?? projectData.mlTrainingResult.model_path
			};
			if (res.models_status) {
				bundleStatus = res.models_status;
			}
			wellData.lastSavedMessage = `Saved ML Suite '${res.suite_key}' to Global Bundle ✓`;
			// Persist training configuration (captured at train time) keyed by zone selection
			await projectData.persistProjectParams(['ml-training'], ['ml-training']);
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to save trained model';
		} finally {
			isSavingModel = false;
		}
	}

	async function handleDeleteSuite(key: string) {
		if (!workbench.projectId) return;
		projectData.error = null;
		try {
			const res = await deleteJson<any>(
				`/quick_pp/workspace/projects/${encodeURIComponent(workbench.projectId)}/ml_models/${encodeURIComponent(key)}`
			);
			if (res?.models_status) {
				bundleStatus = res.models_status;
			}
			wellData.lastSavedMessage = `Removed suite '${key}' ✓`;
		} catch (e: any) {
			projectData.error = e?.message || `Failed to delete suite '${key}'`;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<SavedParamsIndicator modules={['ml-training']} />
		<!-- 1. Restricted Architecture & I/O Specification -->
		<ParameterSection
			title="Conformal PIML Architecture (Restricted I/O)"
			badge={mp.architecture.toUpperCase()}
		>
			{#if authState.isDemoMode}
				<div class="mb-2 rounded-xs border border-amber-300 bg-amber-50 px-2 py-1 text-[10px] text-amber-900">
					<strong>Read-Only Demo Guardrail:</strong> Heavy model training is disabled when <code>QPP_DEMO_MODE=true</code>.
				</div>
			{/if}

			<div class="grid grid-cols-2 gap-1 mb-2">
				{#each [['clastic', 'Clastic PIML Preset'], ['carbonate', 'Carbonate PIML Preset']] as [archKey, archLbl]}
					<button
						type="button"
						onclick={() => handleArchitectureChange(archKey as 'clastic' | 'carbonate')}
						class="h-6.5 rounded-sm border text-[11px] font-medium transition-colors {mp.architecture ===
						archKey
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
					>
						{archLbl}
					</button>
				{/each}
			</div>

			<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px] space-y-1">
				<div>
					<span class="font-semibold text-[var(--text-secondary)]">Fixed Raw Inputs (4):</span>
					<div class="mt-0.5 flex flex-wrap gap-1">
						{#each RESTRICTED_RAW_INPUTS as inp}
							<span class="mono-num rounded-xs border border-slate-300 bg-white px-1 py-0.2 font-semibold text-slate-800">
								{inp}
							</span>
						{/each}
					</div>
				</div>
				<div>
					<span class="font-semibold text-[var(--text-secondary)]">Auto Physics Features (7):</span>
					<div class="mt-0.5 flex flex-wrap gap-1">
						{#each RESTRICTED_FE_INPUTS as fe}
							<span class="mono-num rounded-xs bg-slate-200/80 px-1 py-0.2 text-[9px] text-slate-700">
								{fe}
							</span>
						{/each}
					</div>
				</div>
				<div>
					<span class="font-semibold text-[var(--text-secondary)]">Fixed Conformal Targets (6 + Fluid Flags):</span>
					<div class="mt-0.5 flex flex-wrap gap-1">
						{#each activeTargets as tgt}
							<span class="mono-num rounded-xs border border-indigo-200 bg-indigo-50 px-1 py-0.2 font-semibold text-indigo-900">
								{tgt} (P10/P90)
							</span>
						{/each}
						<span class="mono-num rounded-xs border border-emerald-200 bg-emerald-50 px-1 py-0.2 text-[9px] font-semibold text-emerald-800">
							OIL_FLAG / GAS_FLAG
						</span>
					</div>
				</div>
			</div>
		</ParameterSection>

		<!-- 2. Well Selection -->
		<ParameterSection
			title="1. Select Training Wells"
			badge={`${mp.selectedWells.length}/${projectData.wells.length} Wells`}
		>
			<div class="mb-1 flex items-center justify-between text-[10px]">
				<span class="text-[var(--text-muted)]">Select evaluated wells to pool for training</span>
				<div class="flex gap-1.5">
					<button
						type="button"
						onclick={() => (mp.selectedWells = projectData.wells.map((w) => w.name))}
						class="font-semibold text-[var(--accent-primary)] hover:underline"
					>
						All
					</button>
					<span class="text-[var(--border-hairline)]">|</span>
					<button
						type="button"
						onclick={() => {
							if (projectData.wells.length > 0) mp.selectedWells = [projectData.wells[0].name];
						}}
						class="font-medium text-[var(--text-secondary)] hover:underline"
					>
						First Only
					</button>
				</div>
			</div>

			<div class="max-h-36 overflow-y-auto rounded-xs border border-[var(--border-hairline)] divide-y divide-[var(--border-hairline)]">
				{#each projectData.wells as w (w.well_id)}
					{@const isChecked = mp.selectedWells.includes(w.name)}
					{@const status = checkWellReady(w)}
					<label
						class="flex items-center justify-between px-2 py-1 text-xs cursor-pointer hover:bg-[var(--bg-subtle)] {isChecked
							? 'bg-blue-50/40 font-medium'
							: 'opacity-70'}"
					>
						<div class="flex items-center gap-1.5 min-w-0">
							<input
								type="checkbox"
								checked={isChecked}
								disabled={isChecked && mp.selectedWells.length <= 1}
								onchange={() => toggleWell(w.name)}
								class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
							/>
							<span class="mono-num truncate font-semibold text-[var(--text-primary)]">{w.name}</span>
						</div>
						<div class="flex items-center gap-1.5 shrink-0">
							<span class="mono-num text-[10px] text-[var(--text-muted)]">
								{w.depth_count ?? (w as any).row_count ?? 0} rows
							</span>
							{#if status.hasRaw && status.hasTargets}
								<span class="rounded-xs bg-emerald-100 px-1 py-0.2 text-[9px] font-semibold text-emerald-800">
									Ready
								</span>
							{:else if status.hasRaw}
								<span
									title="Has GR/RT/NPHI/RHOB; run Litho/Poro/Perm/Sw first"
									class="rounded-xs bg-amber-100 px-1 py-0.2 text-[9px] font-semibold text-amber-800"
								>
									Needs CPI
								</span>
							{:else}
								<span class="rounded-xs bg-red-100 px-1 py-0.2 text-[9px] font-semibold text-red-800">
									Missing Logs
								</span>
							{/if}
						</div>
					</label>
				{/each}
			</div>
		</ParameterSection>

		<!-- 3. Zone & Flag Filtering -->
		<ParameterSection
			title="2. Zone & Flag Filters"
			badge={workbench.selectedZones.length > 0 ? `${workbench.selectedZones.length} Zones` : 'All Zones'}
		>
			{#if projectData.allZones.length > 0}
				<div class="mb-1 flex items-center justify-between text-[10px]">
					<span class="font-medium text-[var(--text-secondary)]">Include Zones (empty = all):</span>
					{#if workbench.selectedZones.length > 0}
						<button
							type="button"
							onclick={() => (workbench.selectedZones = [])}
							class="text-[10px] font-semibold text-[var(--accent-primary)] hover:underline"
						>
							Clear ({workbench.selectedZones.length})
						</button>
					{/if}
				</div>
				<div class="mb-2 flex flex-wrap gap-1">
					{#each projectData.allZones as z}
						{@const active = workbench.selectedZones.includes(z)}
						<button
							type="button"
							onclick={() => toggleZone(z)}
							class="rounded-xs border px-1.5 py-0.5 text-[10px] font-medium transition-colors {active
								? 'border-[var(--accent-primary)] bg-blue-600 text-white font-semibold'
								: 'border-[var(--border-hairline)] bg-[var(--bg-app)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
						>
							{z}
						</button>
					{/each}
				</div>
			{/if}

			<div class="space-y-1 border-t border-[var(--border-hairline)] pt-1.5 text-xs">
				<div class="text-[10px] font-medium text-[var(--text-secondary)]">Flag Exclusion Rules:</div>
				<label class="flex items-center gap-2 py-0.5 text-[11px] cursor-pointer">
					<input
						type="checkbox"
						bind:checked={mp.excludeCoal}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Exclude Coal Intervals (<code>COAL_FLAG=1</code> or <code>VCOAL&gt;0.5</code>)</span>
				</label>
				<label class="flex items-center gap-2 py-0.5 text-[11px] cursor-pointer">
					<input
						type="checkbox"
						bind:checked={mp.excludeBadHole}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Exclude Bad Hole Washouts (<code>BAD_HOLE=1</code>)</span>
				</label>
				<label class="flex items-center gap-2 py-0.5 text-[11px] cursor-pointer">
					<input
						type="checkbox"
						bind:checked={mp.excludeTight}
						class="h-3.5 w-3.5 rounded-xs border-[var(--border-hairline)] text-[var(--accent-primary)]"
					/>
					<span>Exclude Tight Non-Reservoir (<code>TIGHT_FLAG=1</code>)</span>
				</label>
			</div>

			{#if projectData.allCurves.includes('ROCK_FLAG')}
				<div class="mt-1.5 border-t border-[var(--border-hairline)] pt-1.5">
					<div class="mb-1 flex items-center justify-between text-[10px]">
						<span class="font-medium text-[var(--text-secondary)]">Filter by Rock Type (<code>ROCK_FLAG</code>):</span>
						{#if mp.selectedRockFlags.length > 0}
							<button
								type="button"
								onclick={() => (mp.selectedRockFlags = [])}
								class="text-[10px] font-semibold text-[var(--accent-primary)] hover:underline"
							>
								All RTs
							</button>
						{/if}
					</div>
					<div class="flex gap-1">
						{#each [1, 2, 3, 4] as rf}
							{@const isRfActive = mp.selectedRockFlags.includes(rf)}
							<button
								type="button"
								onclick={() => toggleRockFlag(rf)}
								class="mono-num h-5 flex-1 rounded-xs border text-[10px] font-semibold {isRfActive
									? 'border-indigo-600 bg-indigo-600 text-white'
									: 'border-[var(--border-hairline)] bg-[var(--bg-app)] text-[var(--text-secondary)]'}"
							>
								RT {rf}
							</button>
						{/each}
					</div>
				</div>
			{/if}
		</ParameterSection>

		<!-- 4. Hyperparameters & Global Bundle Target -->
		<ParameterSection
			title="3. Calibration & Global Save Target"
			badge={`CI=${Math.round((1 - mp.conformalAlpha) * 100)}%`}
		>
			<NumericScrubber
				label="Conformal Alpha (α)"
				bind:value={mp.conformalAlpha}
				min={0.05}
				max={0.4}
				step={0.02}
				decimals={2}
			/>
			<NumericScrubber
				label="Validation Holdout Split"
				bind:value={mp.testSize}
				min={0.1}
				max={0.4}
				step={0.05}
				decimals={2}
			/>
			<NumericScrubber
				label="RF Trees (n_estimators)"
				bind:value={mp.nEstimators}
				min={10}
				max={200}
				step={10}
				decimals={0}
			/>
			<NumericScrubber
				label="Max Tree Depth"
				bind:value={mp.maxDepth}
				min={4}
				max={25}
				step={1}
				decimals={0}
			/>

			<div class="mt-2 space-y-1.5 border-t border-[var(--border-hairline)] pt-2">
				<div class="text-[10px] font-semibold text-[var(--text-secondary)]">
					Global Bundle Identifier (Available Across All Projects):
				</div>
				<div class="grid grid-cols-[95px_1fr] items-center gap-1.5 text-xs">
					<span class="text-[11px] text-[var(--text-secondary)]">Suite Key:</span>
					<input
						type="text"
						bind:value={mp.suiteKey}
						placeholder="e.g. clastic, carbonate, volve_v2"
						class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs font-medium text-[var(--text-primary)]"
					/>
				</div>
				<div class="grid grid-cols-[95px_1fr] items-center gap-1.5 text-xs">
					<span class="text-[11px] text-[var(--text-secondary)]">Dropdown Label:</span>
					<input
						type="text"
						bind:value={mp.suiteLabel}
						placeholder="Display name in Single Well > Litho"
						class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs font-medium text-[var(--text-primary)]"
					/>
				</div>
				<div class="flex gap-1 pt-0.5">
					<button
						type="button"
						onclick={() => {
							mp.suiteKey = mp.architecture;
							mp.suiteLabel = `${mp.architecture === 'carbonate' ? 'Carbonate' : 'Clastic'} (Pre-Trained PIML)`;
						}}
						class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-0.5 text-[10px] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
					>
						Use Default '{mp.architecture}' Key
					</button>
				</div>
			</div>
		</ParameterSection>

		<!-- 5. Model Evaluation & Performance Metrics -->
		{#if projectData.mlTrainingResult}
			{@const res = projectData.mlTrainingResult}
			<ParameterSection
				title="Model Evaluation & Conformal Performance"
				badge={res.saved ? 'Saved Globally ✓' : 'Evaluated'}
			>
				<div class="mb-1.5 flex items-center justify-between rounded-xs border border-indigo-200 bg-indigo-50/70 px-2 py-1 text-[10px] text-indigo-950">
					<span class="font-semibold truncate">{res.suite_label} (<code>{res.suite_key}</code>)</span>
					<span class="mono-num text-[9px] text-indigo-800">
						Train: {res.data_summary.train_rows} | Test: {res.data_summary.test_rows}
					</span>
				</div>

				<!-- Overall KPIs -->
				<div class="grid grid-cols-4 gap-1 text-center mb-2">
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9px] text-[var(--text-muted)]">Test R²</div>
						<div class="mono-num text-xs font-bold text-[var(--status-ok)]">
							{fmtNum(res.overall_metrics.r2_score, 3)}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9px] text-[var(--text-muted)]">Test MAE</div>
						<div class="mono-num text-xs font-bold text-[var(--text-primary)]">
							{fmtNum(res.overall_metrics.mae, 4)}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9px] text-[var(--text-muted)]">Physics Loss</div>
						<div class="mono-num text-xs font-bold text-indigo-900">
							{fmtNum(res.overall_metrics.physics_loss, 4)}
						</div>
					</div>
					<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1">
						<div class="text-[9px] text-[var(--text-muted)]">P10–P90 Cov</div>
						<div class="mono-num text-xs font-bold text-purple-900">
							{fmtNum(res.overall_metrics.p10_p90_coverage_pct, 1)}%
						</div>
					</div>
				</div>

				<!-- Per-Target Metrics Table -->
				<div class="overflow-x-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[10px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-5.5 border-b border-[var(--border-hairline)]">
								<th class="px-1.5 font-semibold">Target</th>
								<th class="px-1 text-right font-semibold">R²</th>
								<th class="px-1 text-right font-semibold">MAE</th>
								<th class="px-1 text-right font-semibold">RMSE</th>
								<th class="px-1 text-right font-semibold">Cov%</th>
								<th class="px-1.5 text-right font-semibold">Width</th>
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each res.target_metrics as tm (tm.Target)}
								<tr class="h-5.5 hover:bg-[var(--bg-subtle)]/40">
									<td class="mono-num px-1.5 font-semibold text-[var(--text-primary)]">{tm.Target}</td>
									<td class="mono-num px-1 text-right font-semibold text-[var(--status-ok)]">
										{fmtNum(tm.R2, 3)}
									</td>
									<td class="mono-num px-1 text-right">{fmtNum(tm.MAE, 3)}</td>
									<td class="mono-num px-1 text-right">{fmtNum(tm.RMSE, 3)}</td>
									<td class="mono-num px-1 text-right font-medium text-indigo-800">
										{fmtNum(tm.CoveragePct, 1)}%
									</td>
									<td class="mono-num px-1.5 text-right text-[var(--text-secondary)]">
										±{fmtNum(tm.MeanWidth / 2, 3)}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>

				<!-- Fluid Classifier Accuracy -->
				{#if res.classifier_metrics}
					<div class="mt-1.5 flex items-center justify-between rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10px]">
						<span class="text-[var(--text-secondary)]">Fluid Classifiers:</span>
						<span class="mono-num font-medium">
							OIL Acc: <strong>{fmtNum((res.classifier_metrics.OIL?.accuracy ?? 0) * 100, 1)}%</strong>
							| GAS Acc: <strong>{fmtNum((res.classifier_metrics.GAS?.accuracy ?? 0) * 100, 1)}%</strong>
						</span>
					</div>
				{/if}
			</ParameterSection>
		{/if}

		<!-- 6. Global Pre-Trained Bundle Registry -->
		<ParameterSection
			title="Global Pre-Trained Model Registry"
			badge={bundleStatus?.pretrained_available
				? `${bundleStatus.suites.length} Saved`
				: 'No File Yet'}
		>
			{#if bundleStatus && !bundleStatus.pretrained_available}
				<div class="rounded-xs border border-amber-300 bg-amber-50/80 p-1.5 text-[10px] text-amber-900">
					No <code>.joblib</code> file exists yet at <code>{bundleStatus.expected_path}</code>. Evaluate a model and click <strong>Save Model</strong> below to create it and make your model available across all projects.
				</div>
			{:else if bundleStatus?.pretrained_available}
				<div class="mb-1 truncate text-[10px] text-[var(--text-muted)]" title={bundleStatus.model_path || ''}>
					File: <code class="text-[var(--text-secondary)]">{bundleStatus.model_path}</code>
				</div>
				<div class="divide-y divide-[var(--border-hairline)] border border-[var(--border-hairline)] rounded-xs">
					{#each bundleStatus.suites as s (s.key)}
						<div class="flex items-center justify-between px-2 py-1 text-[11px]">
							<div class="min-w-0">
								<div class="truncate font-semibold text-[var(--text-primary)]">{s.label}</div>
								<div class="mono-num text-[9.5px] text-[var(--text-muted)]">
									key: <code>{s.key}</code>
									{#if s.meta?.r2_score !== undefined}
										| R²={fmtNum(s.meta.r2_score, 3)}
									{/if}
									{#if s.meta?.source_project}
										| {s.meta.source_project}
									{/if}
								</div>
							</div>
							{#if !authState.isDemoMode}
								<button
									type="button"
									onclick={() => handleDeleteSuite(s.key)}
									title="Remove suite '{s.key}' from global bundle"
									class="ml-2 h-5 shrink-0 rounded-xs border border-red-200 bg-red-50 px-1.5 text-[9.5px] font-medium text-red-700 hover:bg-red-100"
								>
									Delete
								</button>
							{/if}
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

	<!-- Execution Bar -->
	<div
		class="flex h-9 shrink-0 items-center justify-between gap-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell || isSavingModel || authState.isDemoMode}
			onclick={executeTrainModel}
			class="h-7 flex-1 rounded-sm bg-[var(--accent-primary)] px-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{projectData.isComputingMultiWell ? 'Training...' : 'Evaluate Model'}
		</button>
		<button
			type="button"
			disabled={!projectData.mlTrainingResult || projectData.isComputingMultiWell || isSavingModel || authState.isDemoMode}
			onclick={executeSaveModel}
			title={projectData.mlTrainingResult
				? 'Save the evaluated model directly to the global .joblib bundle'
				: 'Evaluate a model first before saving'}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-40"
		>
			{isSavingModel ? 'Saving...' : 'Save Model'}
		</button>
	</div>
</div>

