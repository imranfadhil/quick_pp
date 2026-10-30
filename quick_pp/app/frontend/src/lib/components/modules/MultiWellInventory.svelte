<script lang="ts">
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { downloadHandoverPackage } from '$lib/api/client';
	import { projectData } from '$lib/state/projectData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtDepth } from '$lib/utils/formatters';

	let isExportingHandover = $state(false);
	async function handleExportHandover() {
		if (!workbench.projectId || projectData.wells.length === 0) return;
		isExportingHandover = true;
		projectData.error = null;
		try {
			const subset =
				projectData.selectedWellNames.length > 0 &&
				projectData.selectedWellNames.length < projectData.wells.length
					? projectData.selectedWellNames
					: undefined;
			await downloadHandoverPackage(workbench.projectId, { wells: subset });
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to export handover package';
		} finally {
			isExportingHandover = false;
		}
	}

	const KEY_CURVES = [
		'TVD',
		'TVDSS',
		'GR',
		'RT',
		'NPHI',
		'RHOB',
		'VCLAY',
		'PHIT',
		'PERM',
		'SWT',
		'ROCK_FLAG',
		'PERM_TRANS',
		'SHF'
	];

	const COMPUTED_CURVES = [
		'VCLAY',
		'PHIT',
		'PERM',
		'SWT',
		'ROCK_FLAG',
		'PERM_TRANS',
		'SHF'
	] as const;

	const CORE_PETRO_CURVES = ['VCLAY', 'PHIT', 'PERM', 'SWT'] as const;

	const CURVE_HEADER_TITLES: Record<string, string> = {
		TVD: 'True Vertical Depth (from TVD curve or computed from directional survey — never assumed equal to MD)',
		TVDSS: 'True Vertical Depth Sub-Sea (TVDSS = TVD − DF elevation from well header)',
		GR: 'Gamma Ray',
		RT: 'Deep Resistivity',
		NPHI: 'Neutron Porosity',
		RHOB: 'Bulk Density',
		VCLAY: 'Shale / Clay Volume (VCLAY / VSH)',
		PHIT: 'Total Porosity',
		PERM: 'Permeability',
		SWT: 'Total Water Saturation',
		ROCK_FLAG: 'Discrete Rock Type Flag (DRT)',
		PERM_TRANS: 'Rock-Type Permeability Transform',
		SHF: 'Saturation Height Function (Sw from Pc / HAFWL)'
	};

	type StratMatrixFilter = 'all' | 'evaluated' | 'missing';
	type StratCurveMode =
		| 'all_computed'
		| 'core_petro'
		| 'VCLAY'
		| 'PHIT'
		| 'PERM'
		| 'SWT'
		| 'ROCK_FLAG'
		| 'PERM_TRANS'
		| 'SHF';

	let stratMatrixFilter = $state<StratMatrixFilter>('all');
	let stratCurveMode = $state<StratCurveMode>('all_computed');

	// Project & Well Management state
	let isEditingProject = $state(false);
	let projNameDraft = $state('');
	let projDescDraft = $state('');
	let confirmDeleteProj = $state(false);

	let editingWellName = $state<string | null>(null);
	let wellNameDraft = $state('');
	let confirmDeleteWellName = $state<string | null>(null);

	function startEditProject() {
		projNameDraft = projectData.summary?.name ?? '';
		projDescDraft = projectData.summary?.description ?? '';
		confirmDeleteProj = false;
		isEditingProject = true;
	}

	async function saveProjectEdit() {
		if (!workbench.projectId || !projNameDraft.trim()) return;
		projectData.error = null;
		try {
			await projectData.renameProject(workbench.projectId, projNameDraft, projDescDraft);
			isEditingProject = false;
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to rename project';
		}
	}

	async function executeDeleteProject() {
		if (!workbench.projectId) return;
		projectData.error = null;
		try {
			await projectData.deleteProject(workbench.projectId);
			confirmDeleteProj = false;
			isEditingProject = false;
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to delete project';
		}
	}

	function startRenameWell(wellName: string) {
		confirmDeleteWellName = null;
		editingWellName = wellName;
		wellNameDraft = wellName;
	}

	async function saveWellRename(oldName: string) {
		if (!wellNameDraft.trim() || wellNameDraft.trim() === oldName) {
			editingWellName = null;
			return;
		}
		projectData.error = null;
		try {
			await projectData.renameWell(oldName, wellNameDraft);
			editingWellName = null;
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to rename well';
		}
	}

	async function executeDeleteWell(wellName: string) {
		projectData.error = null;
		try {
			await projectData.deleteWell(wellName);
			confirmDeleteWellName = null;
		} catch (e: any) {
			projectData.error = e?.message || 'Failed to delete well';
		}
	}

	async function openWellInSingleMode(wellName: string, step?: 'qaqc' | 'ancillary') {
		await projectData.selectWell(wellName);
		workbench.scope = 'single-well';
		if (step) {
			workbench.singleWellStep = step;
		}
	}

	let ancillaryTotals = $derived.by(() => {
		const wells = projectData.wells;
		let rcaSamples = 0;
		let rcaWells = 0;
		let scalPoints = 0;
		let scalPlugs = 0;
		let scalWells = 0;
		let relpermPoints = 0;
		let topsTotal = 0;
		let topsWells = 0;
		let contactsTotal = 0;
		let contactsWells = 0;
		let perfsTotal = 0;
		let perfsWells = 0;
		let pressureTotal = 0;
		let pressureWells = 0;
		let surveyTotal = 0;
		let surveyWells = 0;

		for (const w of wells) {
			const rca = w.core_sample_count ?? w.core_samples_count ?? 0;
			if (rca > 0) {
				rcaSamples += rca;
				rcaWells++;
			}
			const scPts = w.scal_pc_count ?? 0;
			const scPlugs = w.scal_sample_count ?? 0;
			const rpPts = w.relperm_count ?? 0;
			if (scPts > 0 || rpPts > 0) {
				scalPoints += scPts;
				scalPlugs += scPlugs;
				relpermPoints += rpPts;
				scalWells++;
			}
			const tops = w.formation_top_count ?? w.formation_tops_count ?? (w.zones?.length ?? 0);
			if (tops > 0) {
				topsTotal += tops;
				topsWells++;
			}
			const contacts = w.fluid_contact_count ?? w.fluid_contacts_count ?? 0;
			if (contacts > 0) {
				contactsTotal += contacts;
				contactsWells++;
			}
			const perfs = w.perforation_count ?? (w.perforations?.length ?? 0);
			if (perfs > 0) {
				perfsTotal += perfs;
				perfsWells++;
			}
			const press = w.pressure_test_count ?? w.pressure_tests_count ?? 0;
			if (press > 0) {
				pressureTotal += press;
				pressureWells++;
			}
			const surv = w.survey_count ?? 0;
			if (surv > 0) {
				surveyTotal += surv;
				surveyWells++;
			}
		}

		return {
			totalWells: wells.length,
			rcaSamples,
			rcaWells,
			scalPoints,
			scalPlugs,
			scalWells,
			relpermPoints,
			topsTotal,
			topsWells,
			contactsTotal,
			contactsWells,
			perfsTotal,
			perfsWells,
			pressureTotal,
			pressureWells,
			surveyTotal,
			surveyWells
		};
	});

	function getWellTopEntry(w: (typeof projectData.wells)[number], zoneName: string) {
		return (
			(w.formation_tops ?? []).find(
				(t) => t.name.trim().toLowerCase() === zoneName.trim().toLowerCase()
			) ?? null
		);
	}

	function getWellTopDepth(w: (typeof projectData.wells)[number], zoneName: string): number | null {
		return getWellTopEntry(w, zoneName)?.depth ?? null;
	}

	function hasWellZone(w: (typeof projectData.wells)[number], zoneName: string): boolean {
		return (
			getWellTopDepth(w, zoneName) !== null ||
			(w.zones ?? []).some((z) => z.trim().toLowerCase() === zoneName.trim().toLowerCase())
		);
	}

	function getTargetCurvesForMode(mode: StratCurveMode): readonly string[] {
		if (mode === 'all_computed') return COMPUTED_CURVES;
		if (mode === 'core_petro') return CORE_PETRO_CURVES;
		return [mode];
	}

	function evaluateWellZoneStatus(
		w: (typeof projectData.wells)[number],
		zoneName: string,
		mode: StratCurveMode = stratCurveMode
	): {
		penetrated: boolean;
		topDepth: number | null;
		baseDepth: number | null;
		computedCurves: string[];
		rawCurves: string[];
		targetHitCount: number;
		targetTotalCount: number;
		status: 'absent' | 'complete' | 'partial' | 'none';
	} {
		const topEntry = getWellTopEntry(w, zoneName);
		const topDepth = topEntry?.depth ?? null;
		const baseDepth = topEntry?.base_depth ?? null;
		const penetrated = hasWellZone(w, zoneName);
		if (!penetrated) {
			return {
				penetrated: false,
				topDepth: null,
				baseDepth: null,
				computedCurves: [],
				rawCurves: [],
				targetHitCount: 0,
				targetTotalCount: 0,
				status: 'absent'
			};
		}
		const allZoneCurves = projectData.getWellZoneCurves(w, zoneName) ?? [];
		const computedCurves = COMPUTED_CURVES.filter((c) => allZoneCurves.includes(c));
		const rawCurves = ['TVD', 'TVDSS', 'GR', 'RT', 'NPHI', 'RHOB'].filter((c) =>
			allZoneCurves.includes(c)
		);
		const targets = getTargetCurvesForMode(mode);
		const targetHits = targets.filter((c) => allZoneCurves.includes(c));

		let status: 'complete' | 'partial' | 'none' = 'none';
		if (mode === 'all_computed') {
			const coreHits = CORE_PETRO_CURVES.filter((c) => allZoneCurves.includes(c)).length;
			if (coreHits === CORE_PETRO_CURVES.length) {
				status = 'complete';
			} else if (computedCurves.length > 0) {
				status = 'partial';
			} else {
				status = 'none';
			}
		} else if (mode === 'core_petro') {
			if (targetHits.length === CORE_PETRO_CURVES.length) {
				status = 'complete';
			} else if (targetHits.length > 0) {
				status = 'partial';
			} else {
				status = 'none';
			}
		} else {
			status = targetHits.length > 0 ? 'complete' : 'none';
		}

		return {
			penetrated: true,
			topDepth,
			baseDepth,
			computedCurves,
			rawCurves,
			targetHitCount: targetHits.length,
			targetTotalCount: targets.length,
			status
		};
	}

	let zoneCoverageSummary = $derived.by(() => {
		const out: Record<
			string,
			{
				penetratedWells: number;
				completeWells: number;
				partialWells: number;
				evaluatedWells: number;
				anyComputedWells: number;
			}
		> = {};
		for (const zone of projectData.allZones) {
			let penetratedWells = 0;
			let completeWells = 0;
			let partialWells = 0;
			let anyComputedWells = 0;
			for (const w of projectData.wells) {
				const st = evaluateWellZoneStatus(w, zone, stratCurveMode);
				if (!st.penetrated) continue;
				penetratedWells++;
				if (st.computedCurves.length > 0) anyComputedWells++;
				if (st.status === 'complete') completeWells++;
				else if (st.status === 'partial') partialWells++;
			}
			out[zone] = {
				penetratedWells,
				completeWells,
				partialWells,
				evaluatedWells: completeWells + partialWells,
				anyComputedWells
			};
		}
		return out;
	});

	let totalEvaluatedFormationsCount = $derived(
		projectData.allZones.filter((z) => (zoneCoverageSummary[z]?.anyComputedWells ?? 0) > 0).length
	);

	let filteredStratZones = $derived(
		projectData.allZones.filter((z) => {
			const s = zoneCoverageSummary[z];
			if (!s) return true;
			if (stratMatrixFilter === 'evaluated') {
				return s.evaluatedWells > 0;
			}
			if (stratMatrixFilter === 'missing') {
				return s.penetratedWells > 0 && s.completeWells < s.penetratedWells;
			}
			return true;
		})
	);

	function toggleInventoryZoneFilter(zoneName: string) {
		if (projectData.inventorySelectedZone === zoneName) {
			projectData.inventorySelectedZone = '';
		} else {
			projectData.inventorySelectedZone = zoneName;
		}
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection
			title="Field Overview"
			badge={`${projectData.wells.length} Wells`}
		>
			<div class="mb-2 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2 text-xs">
				{#if !isEditingProject}
					<div class="flex items-start justify-between gap-2">
						<div class="min-w-0">
							<div class="text-[10px] uppercase text-[var(--text-muted)]">Active Project</div>
							<div class="truncate font-semibold text-[var(--text-primary)]">
								{projectData.summary?.name || 'No Project Selected'}
							</div>
							{#if projectData.summary?.description}
								<div class="mt-0.5 truncate text-[10px] text-[var(--text-secondary)]">
									{projectData.summary.description}
								</div>
							{/if}
						</div>
						{#if workbench.projectId}
							<div class="flex shrink-0 items-center gap-1">
								<button
									type="button"
									title="Rename Project or Edit Description"
									onclick={startEditProject}
									class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
								>
									✎ Rename
								</button>
								{#if !confirmDeleteProj}
									<button
										type="button"
										title="Delete Project"
										onclick={() => (confirmDeleteProj = true)}
										class="h-6 rounded-xs border border-red-200 bg-red-50 px-1.5 text-[10px] font-medium text-red-700 hover:bg-red-100"
									>
										Delete
									</button>
								{:else}
									<button
										type="button"
										onclick={executeDeleteProject}
										class="h-6 rounded-xs bg-red-600 px-1.5 text-[10px] font-bold text-white hover:bg-red-700"
									>
										Confirm?
									</button>
									<button
										type="button"
										onclick={() => (confirmDeleteProj = false)}
										class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[10px]"
									>
										×
									</button>
								{/if}
							</div>
						{/if}
					</div>
				{:else}
					<div class="space-y-1.5">
						<div class="text-[10px] font-semibold text-[var(--text-secondary)]">
							Edit Project Details
						</div>
						<input
							type="text"
							placeholder="Project Name"
							bind:value={projNameDraft}
							onkeydown={(e) => e.key === 'Enter' && saveProjectEdit()}
							class="h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-xs font-medium"
						/>
						<input
							type="text"
							placeholder="Description (optional)"
							bind:value={projDescDraft}
							onkeydown={(e) => e.key === 'Enter' && saveProjectEdit()}
							class="h-6 w-full rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
						/>
						<div class="flex justify-end gap-1">
							<button
								type="button"
								onclick={() => (isEditingProject = false)}
								class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[10px]"
							>
								Cancel
							</button>
							<button
								type="button"
								onclick={saveProjectEdit}
								class="h-6 rounded-xs bg-[var(--accent-primary)] px-2 text-[10px] font-semibold text-white"
							>
								Save
							</button>
						</div>
					</div>
				{/if}
			</div>

			<div class="grid grid-cols-2 gap-1.5 text-xs">
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
					<div class="text-[10px] text-[var(--text-muted)]">Field Min Depth</div>
					<div class="mono-num font-semibold text-[var(--text-primary)]">
						{fmtDepth(projectData.summary?.min_depth)}
					</div>
				</div>
				<div class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5">
					<div class="text-[10px] text-[var(--text-muted)]">Field Max Depth</div>
					<div class="mono-num font-semibold text-[var(--text-primary)]">
						{fmtDepth(projectData.summary?.max_depth)}
					</div>
				</div>
			</div>

			<!-- Field-Wide Core (RCA & SCAL) and Ancillary KPI Summary -->
			<div class="mt-1.5 grid grid-cols-3 gap-1 text-[10.5px]">
				<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-1">
					<div class="text-[9.5px] text-[var(--text-muted)]">RCA Core Plugs</div>
					<div class="mono-num font-bold {ancillaryTotals.rcaSamples > 0 ? 'text-amber-700' : 'text-[var(--text-muted)]'}">
						{ancillaryTotals.rcaSamples}
						<span class="text-[9px] font-normal text-[var(--text-secondary)]">
							({ancillaryTotals.rcaWells}/{ancillaryTotals.totalWells}w)
						</span>
					</div>
				</div>
				<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-1">
					<div class="text-[9.5px] text-[var(--text-muted)]">SCAL Pc Pairs</div>
					<div class="mono-num font-bold {ancillaryTotals.scalPoints > 0 ? 'text-indigo-700' : 'text-[var(--text-muted)]'}">
						{ancillaryTotals.scalPoints}
						<span class="text-[9px] font-normal text-[var(--text-secondary)]">
							({ancillaryTotals.scalWells}/{ancillaryTotals.totalWells}w)
						</span>
					</div>
				</div>
				<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-1">
					<div class="text-[9.5px] text-[var(--text-muted)]">Formation Tops</div>
					<div class="mono-num font-bold {ancillaryTotals.topsTotal > 0 ? 'text-emerald-700' : 'text-[var(--text-muted)]'}">
						{ancillaryTotals.topsTotal}
						<span class="text-[9px] font-normal text-[var(--text-secondary)]">
							({projectData.allZones.length}z)
						</span>
					</div>
				</div>
				<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-1">
					<div class="text-[9.5px] text-[var(--text-muted)]">Fluid Contacts</div>
					<div class="mono-num font-semibold {ancillaryTotals.contactsTotal > 0 ? 'text-sky-700' : 'text-[var(--text-muted)]'}">
						{ancillaryTotals.contactsTotal}
						<span class="text-[9px] font-normal text-[var(--text-secondary)]">
							({ancillaryTotals.contactsWells}/{ancillaryTotals.totalWells}w)
						</span>
					</div>
				</div>
				<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-1">
					<div class="text-[9.5px] text-[var(--text-muted)]">Perforations</div>
					<div class="mono-num font-semibold {ancillaryTotals.perfsTotal > 0 ? 'text-orange-700' : 'text-[var(--text-muted)]'}">
						{ancillaryTotals.perfsTotal}
						<span class="text-[9px] font-normal text-[var(--text-secondary)]">
							({ancillaryTotals.perfsWells}/{ancillaryTotals.totalWells}w)
						</span>
					</div>
				</div>
				<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-1">
					<div class="text-[9.5px] text-[var(--text-muted)]">Pressure Tests</div>
					<div class="mono-num font-semibold {ancillaryTotals.pressureTotal > 0 ? 'text-teal-700' : 'text-[var(--text-muted)]'}">
						{ancillaryTotals.pressureTotal}
						<span class="text-[9px] font-normal text-[var(--text-secondary)]">
							({ancillaryTotals.pressureWells}/{ancillaryTotals.totalWells}w)
						</span>
					</div>
				</div>
				<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-1.5 py-1">
					<div class="text-[9.5px] text-[var(--text-muted)]">Well Surveys</div>
					<div class="mono-num font-semibold {ancillaryTotals.surveyTotal > 0 ? 'text-[var(--text-primary)]' : 'text-[var(--text-muted)]'}">
						{ancillaryTotals.surveyTotal}
						<span class="text-[9px] font-normal text-[var(--text-secondary)]">
							({ancillaryTotals.surveyWells}/{ancillaryTotals.totalWells}w)
						</span>
					</div>
				</div>
			</div>

			<div class="mt-2 flex flex-col gap-1.5">
				<button
					type="button"
					onclick={() => (workbench.importerOpen = true)}
					class="h-7 w-full rounded-sm bg-[var(--accent-primary)] px-2.5 text-xs font-semibold text-white hover:bg-blue-700"
				>
					+ Batch Import LAS / Ancillary Files
				</button>
				<button
					type="button"
					disabled={!workbench.projectId || projectData.wells.length === 0 || isExportingHandover}
					onclick={handleExportHandover}
					title="Export Handover Package (.zip) with LAS 2.0, Tops, Surveys, Core RCA, SCAL, Excel Workbook & Loader Scripts"
					class="h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:opacity-45"
				>
					{isExportingHandover ? 'Building Handover (.zip)...' : '⬇ Export Handover Package (.zip)'}
				</button>
			</div>
		</ParameterSection>

		<ParameterSection
			title="Well & Curve Availability Matrix"
			badge={projectData.inventorySelectedZone
				? `Zone: ${projectData.inventorySelectedZone}`
				: 'Whole Well'}
		>
			{#if projectData.allZones.length > 0}
				<div
					class="mb-1.5 flex items-center justify-between gap-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10px]"
				>
					<label
						for="inv-zone-filter-select"
						class="shrink-0 font-semibold text-[var(--text-secondary)]"
					>
						Formation Scope:
					</label>
					<select
						id="inv-zone-filter-select"
						bind:value={projectData.inventorySelectedZone}
						class="h-5.5 min-w-0 flex-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1 text-[10px] font-medium text-[var(--text-primary)]"
					>
						<option value="">All Formations (Whole Well)</option>
						{#each projectData.allZones as z}
							{@const zStat = zoneCoverageSummary[z]}
							<option value={z}>
								{z} ({zStat?.anyComputedWells ?? 0}/{zStat?.penetratedWells ?? 0} eval)
							</option>
						{/each}
					</select>
					{#if projectData.inventorySelectedZone}
						<button
							type="button"
							title="Clear formation filter and show whole-well curve availability"
							onclick={() => (projectData.inventorySelectedZone = '')}
							class="h-5.5 shrink-0 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[9.5px] font-semibold text-[var(--accent-primary)] hover:bg-blue-50"
						>
							× Clear
						</button>
					{/if}
				</div>
			{/if}

			<div class="overflow-x-auto border border-[var(--border-hairline)]">
				<table class="w-full border-collapse text-left text-[11px]">
					<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
						<tr class="h-6 border-b border-[var(--border-hairline)]">
							<th class="px-1.5 font-semibold">Well</th>
							<th class="px-1 text-right font-semibold">
								{projectData.inventorySelectedZone ? 'Top' : 'Rows'}
							</th>
							{#each KEY_CURVES as kc}
								<th
									class="px-1 text-center font-semibold text-[9px]"
									title={CURVE_HEADER_TITLES[kc] ?? kc}
								>
									{kc}
								</th>
							{/each}
						</tr>
					</thead>
					<tbody class="divide-y divide-[var(--border-hairline)]">
						{#if projectData.wells.length === 0}
							<tr>
								<td colspan={KEY_CURVES.length + 2} class="p-3 text-center text-xs text-[var(--text-muted)]">
									No wells in project. Click Batch Import above.
								</td>
							</tr>
						{:else}
							{#each projectData.wells as w (w.well_id)}
								{@const activeZone = projectData.inventorySelectedZone}
								{@const zonePenetrated = activeZone ? hasWellZone(w, activeZone) : true}
								{@const zoneTopD = activeZone ? getWellTopDepth(w, activeZone) : null}
								<tr
									onclick={() => openWellInSingleMode(w.name)}
									title="Click to inspect well in Single-Well mode"
									class="h-6 cursor-pointer hover:bg-[var(--bg-subtle)] {workbench.wellId ===
									w.name
										? 'bg-blue-50/70'
										: ''} {!zonePenetrated ? 'opacity-45' : ''}"
								>
									<td class="mono-num px-1.5 font-semibold text-[var(--accent-primary)]">
										{#if editingWellName === w.name}
											<!-- svelte-ignore a11y_click_events_have_key_events -->
											<!-- svelte-ignore a11y_no_static_element_interactions -->
											<div
												onclick={(e) => e.stopPropagation()}
												class="flex items-center gap-0.5 py-0.5"
											>
												<input
													type="text"
													bind:value={wellNameDraft}
													onkeydown={(e) => {
														if (e.key === 'Enter') saveWellRename(w.name);
														if (e.key === 'Escape') editingWellName = null;
													}}
													class="mono-num h-5 w-24 rounded-xs border border-[var(--accent-primary)] bg-white px-1 text-[11px] font-semibold text-[var(--text-primary)]"
												/>
												<button
													type="button"
													title="Save Well Name"
													onclick={() => saveWellRename(w.name)}
													class="flex h-5 w-5 items-center justify-center rounded-xs bg-[var(--accent-primary)] text-[10px] font-bold text-white"
												>
													✓
												</button>
												<button
													type="button"
													title="Cancel Rename"
													onclick={() => (editingWellName = null)}
													class="flex h-5 w-5 items-center justify-center rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[10px]"
												>
													×
												</button>
											</div>
										{:else}
											<div class="group flex items-center justify-between gap-1">
												<span class="truncate">{w.name}</span>
												<!-- svelte-ignore a11y_click_events_have_key_events -->
												<!-- svelte-ignore a11y_no_static_element_interactions -->
												<span
													onclick={(e) => e.stopPropagation()}
													class="flex shrink-0 items-center gap-0.5 opacity-75 group-hover:opacity-100"
												>
													{#if confirmDeleteWellName === w.name}
														<button
															type="button"
															title="Confirm Delete Well"
															onclick={() => executeDeleteWell(w.name)}
															class="h-4 rounded-xs bg-red-600 px-1 text-[9px] font-bold text-white hover:bg-red-700"
														>
															Del?
														</button>
														<button
															type="button"
															title="Cancel"
															onclick={() => (confirmDeleteWellName = null)}
															class="h-4 rounded-xs border border-[var(--border-hairline)] bg-white px-1 text-[9px] text-[var(--text-secondary)]"
														>
															×
														</button>
													{:else}
														<button
															type="button"
															title="Rename Well {w.name}"
															onclick={() => startRenameWell(w.name)}
															class="flex h-4 w-4 items-center justify-center rounded-xs text-[10px] text-[var(--text-muted)] hover:bg-blue-100 hover:text-[var(--accent-primary)]"
														>
															✎
														</button>
														<button
															type="button"
															title="Delete Well {w.name}"
															onclick={() => (confirmDeleteWellName = w.name)}
															class="flex h-4 w-4 items-center justify-center rounded-xs text-[10px] text-[var(--text-muted)] hover:bg-red-100 hover:text-red-700"
														>
															×
														</button>
													{/if}
												</span>
											</div>
										{/if}
									</td>
									<td class="mono-num px-1 text-right text-[var(--text-secondary)]">
										{#if activeZone}
											{#if zoneTopD !== null}
												{zoneTopD.toFixed(1)}
											{:else if zonePenetrated}
												✓
											{:else}
												—
											{/if}
										{:else}
											{w.depth_count ?? (w as any).row_count ?? 0}
										{/if}
									</td>
									{#each KEY_CURVES as kc}
										{@const has = projectData.wellHasCurveInZone(w, kc, activeZone)}
										{@const isTvdFromSurvey =
											(kc === 'TVD' || kc === 'TVDSS') && (w.survey_count ?? 0) >= 2}
										<td
											class="px-1 text-center"
											title={!zonePenetrated
												? `${w.name}: Formation '${activeZone}' not present in this well`
												: kc === 'TVD'
													? has
														? isTvdFromSurvey
															? `${w.name}${activeZone ? ` [${activeZone}]` : ''}: TVD available (computed from ${w.survey_count} directional survey stations)`
															: `${w.name}${activeZone ? ` [${activeZone}]` : ''}: TVD curve available`
														: `${w.name}${activeZone ? ` [${activeZone}]` : ''}: No TVD curve or directional survey — SHF & Stick Plot disabled for this well`
													: kc === 'TVDSS'
														? has
															? `${w.name}${activeZone ? ` [${activeZone}]` : ''}: TVDSS available (TVD − DF elevation${w.df_elev != null ? ` [${w.df_elev}m]` : ''})`
															: `${w.name}${activeZone ? ` [${activeZone}]` : ''}: No TVDSS / TVD curve or directional survey`
														: `${w.name}${activeZone ? ` [${activeZone}]` : ''}: ${kc} ${has ? 'available' : 'missing'}`}
										>
											{#if !zonePenetrated}
												<span class="text-[9px] text-[var(--text-muted)]">—</span>
											{:else}
												<span
													class="inline-block h-2 w-2 rounded-full {has
														? 'bg-emerald-500'
														: 'bg-slate-200'}"
												></span>
											{/if}
										</td>
									{/each}
								</tr>
							{/each}
						{/if}
					</tbody>
				</table>
			</div>
		</ParameterSection>

		<ParameterSection
			title="RCA, SCAL & Ancillary Data Matrix"
			badge={`${ancillaryTotals.rcaSamples} RCA · ${ancillaryTotals.scalPoints} SCAL`}
		>
			<div class="overflow-x-auto border border-[var(--border-hairline)]">
				<table class="w-full border-collapse text-left text-[10.5px]">
					<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
						<tr class="h-6 border-b border-[var(--border-hairline)]">
							<th class="px-1.5 font-semibold">Well</th>
							<th class="px-1 text-right font-semibold" title="Routine Core Analysis (CPORE / CPERM plugs)">RCA</th>
							<th class="px-1 text-right font-semibold" title="SCAL Capillary Pressure (Pc-Sw pairs & plugs)">SCAL Pc</th>
							<th class="px-1 text-right font-semibold" title="Picked Formation Tops / Stratigraphic Zones">Tops</th>
							<th class="px-1 text-center font-semibold" title="Fluid Contacts (OWC / GOC / FWL)">Contacts</th>
							<th class="px-1 text-right font-semibold" title="MDT / RFT Formation Pressure Tests">Press</th>
							<th class="px-1 text-right font-semibold" title="Directional Survey Stations (MD / Inc / Azim)">Survey</th>
						</tr>
					</thead>
					<tbody class="divide-y divide-[var(--border-hairline)]">
						{#if projectData.wells.length === 0}
							<tr>
								<td colspan="7" class="p-3 text-center text-xs text-[var(--text-muted)]">
									No wells in project.
								</td>
							</tr>
						{:else}
							{#each projectData.wells as w (w.well_id)}
								{@const rca = w.core_sample_count ?? w.core_samples_count ?? 0}
								{@const scalPts = w.scal_pc_count ?? 0}
								{@const scalPlugs = w.scal_sample_count ?? 0}
								{@const tops = w.formation_top_count ?? w.formation_tops_count ?? (w.zones?.length ?? 0)}
								{@const contacts = w.fluid_contact_count ?? w.fluid_contacts_count ?? 0}
								{@const contactsList = w.fluid_contacts ?? []}
								{@const press = w.pressure_test_count ?? w.pressure_tests_count ?? 0}
								{@const surv = w.survey_count ?? 0}
								{@const topsTooltip = (w.formation_tops ?? []).length > 0
									? (w.formation_tops ?? []).map((t) => `${t.name}: ${t.depth ?? '—'}m`).join(' | ')
									: (w.zones ?? []).join(', ') || 'No formation tops'}
								{@const contactsTooltip = contactsList.length > 0
									? contactsList.map((c) => `${c.type} @ ${c.depth ?? '—'}m`).join(' | ')
									: 'No fluid contacts'}
								<tr
									onclick={() => openWellInSingleMode(w.name, 'ancillary')}
									title={`Click to open ${w.name} in Single-Well Ancillary editor`}
									class="h-6 cursor-pointer hover:bg-[var(--bg-subtle)] {workbench.wellId === w.name
										? 'bg-blue-50/70'
										: ''}"
								>
									<td class="mono-num truncate px-1.5 font-semibold text-[var(--accent-primary)]">
										{w.name}
									</td>
									<td
										class="mono-num px-1 text-right"
										title={rca > 0 ? `${rca} RCA core samples (${w.core_availability_pct ?? 0}% log coverage)` : 'No RCA core plugs'}
									>
										{#if rca > 0}
											<span class="rounded-xs bg-amber-50 px-1 py-0.5 font-semibold text-amber-800">
												{rca}
											</span>
										{:else}
											<span class="text-[var(--text-muted)]">—</span>
										{/if}
									</td>
									<td
										class="mono-num px-1 text-right"
										title={scalPts > 0 ? `${scalPts} Pc-Sw pairs across ${scalPlugs} core plug(s)` : 'No SCAL Pc data'}
									>
										{#if scalPts > 0}
											<span class="rounded-xs bg-indigo-50 px-1 py-0.5 font-semibold text-indigo-800">
												{scalPts}{#if scalPlugs > 0}<span class="text-[9px] font-normal text-indigo-600"> ({scalPlugs}p)</span>{/if}
											</span>
										{:else}
											<span class="text-[var(--text-muted)]">—</span>
										{/if}
									</td>
									<td class="mono-num px-1 text-right" title={topsTooltip}>
										{#if tops > 0}
											<span class="rounded-xs bg-emerald-50 px-1 py-0.5 font-semibold text-emerald-800">
												{tops}
											</span>
										{:else}
											<span class="text-[var(--text-muted)]">—</span>
										{/if}
									</td>
									<td class="mono-num px-1 text-center" title={contactsTooltip}>
										{#if contactsList.length > 0}
											<span class="rounded-xs bg-sky-50 px-1 py-0.5 text-[9.5px] font-semibold text-sky-800">
												{contactsList.map((c) => c.type).join(',')}
											</span>
										{:else if contacts > 0}
											<span class="rounded-xs bg-sky-50 px-1 py-0.5 font-semibold text-sky-800">
												{contacts}
											</span>
										{:else}
											<span class="text-[var(--text-muted)]">—</span>
										{/if}
									</td>
									<td class="mono-num px-1 text-right">
										{#if press > 0}
											<span class="font-semibold text-teal-700">{press}</span>
										{:else}
											<span class="text-[var(--text-muted)]">—</span>
										{/if}
									</td>
									<td class="mono-num px-1 text-right">
										{#if surv > 0}
											<span class="font-medium text-[var(--text-secondary)]">{surv}</span>
										{:else}
											<span class="text-[var(--text-muted)]">—</span>
										{/if}
									</td>
								</tr>
							{/each}
						{/if}
					</tbody>
				</table>
			</div>
		</ParameterSection>

		{#if projectData.allZones.length > 0 && projectData.wells.length > 0}
			<ParameterSection
				title="Formation Tops & Computed Curve Matrix"
				badge={`${totalEvaluatedFormationsCount}/${projectData.allZones.length} Eval · ${projectData.allZones.length} Zones`}
				defaultOpen={true}
			>
				<!-- Filter & Computed Curve Inspector Bar -->
				<div
					class="mb-1.5 space-y-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[9.5px]"
				>
					<div class="flex flex-wrap items-center justify-between gap-1">
						<div class="flex items-center gap-1">
							<span class="font-semibold text-[var(--text-secondary)]">Show:</span>
							{#each [
								['all', `All (${projectData.allZones.length})`],
								['evaluated', `Evaluated (${totalEvaluatedFormationsCount})`],
								['missing', 'Missing / Partial']
							] as [fKey, fLbl]}
								<button
									type="button"
									onclick={() => (stratMatrixFilter = fKey as StratMatrixFilter)}
									class="rounded-xs border px-1.5 py-0.5 text-[9px] font-medium transition-colors {stratMatrixFilter ===
									fKey
										? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
										: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
								>
									{fLbl}
								</button>
							{/each}
						</div>
						<div class="flex items-center gap-1.5 text-[8.5px] text-[var(--text-muted)]">
							<span class="flex items-center gap-0.5">
								<span class="inline-block h-2 w-2 rounded-full bg-emerald-500"></span>Computed
							</span>
							<span class="flex items-center gap-0.5">
								<span class="inline-block h-2 w-2 rounded-full bg-amber-500"></span>Partial
							</span>
							<span class="flex items-center gap-0.5">
								<span class="inline-block h-2 w-2 rounded-full bg-slate-300"></span>Raw Only
							</span>
						</div>
					</div>

					<div class="flex flex-wrap items-center gap-1 pt-0.5">
						<span class="font-semibold text-[var(--text-secondary)]">Inspect:</span>
						{#each [
							['all_computed', 'All Computed'],
							['core_petro', 'Core Petro'],
							['VCLAY', 'VCLAY'],
							['PHIT', 'PHIT'],
							['PERM', 'PERM'],
							['SWT', 'SWT'],
							['ROCK_FLAG', 'ROCK_FLAG'],
							['PERM_TRANS', 'PERM_TRANS'],
							['SHF', 'SHF']
						] as [cKey, cLbl]}
							<button
								type="button"
								onclick={() => (stratCurveMode = cKey as StratCurveMode)}
								class="mono-num rounded-xs border px-1 py-0.5 text-[8.5px] font-medium transition-colors {stratCurveMode ===
								cKey
									? 'border-slate-800 bg-slate-800 font-semibold text-white'
									: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
							>
								{cLbl}
							</button>
						{/each}
					</div>
				</div>

				<div class="overflow-x-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[10.5px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-6 border-b border-[var(--border-hairline)]">
								<th class="px-1.5 font-semibold" title="Click a formation row to filter the Well & Curve Matrix above">
									Zone / Top
								</th>
								<th
									class="px-1 text-center font-semibold text-[9px]"
									title="Wells with computed curves / Wells penetrating this formation"
								>
									Eval
								</th>
								{#each projectData.wells as w (w.well_id)}
									<th class="mono-num px-1 text-right font-semibold text-[9.5px]" title={w.name}>
										{w.name}
									</th>
								{/each}
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#if filteredStratZones.length === 0}
								<tr>
									<td
										colspan={projectData.wells.length + 2}
										class="p-3 text-center text-xs text-[var(--text-muted)]"
									>
										No formations match the selected filter.
									</td>
								</tr>
							{:else}
								{#each filteredStratZones as zone (zone)}
									{@const zSum = zoneCoverageSummary[zone]}
									{@const isSelectedZone = projectData.inventorySelectedZone === zone}
									<tr
										onclick={() => toggleInventoryZoneFilter(zone)}
										title={`Click to filter Well & Curve Matrix by '${zone}'`}
										class="h-6 cursor-pointer hover:bg-[var(--bg-subtle)] {isSelectedZone
											? 'bg-blue-50/70'
											: ''}"
									>
										<td
											class="truncate px-1.5 font-semibold {isSelectedZone
												? 'text-[var(--accent-primary)]'
												: 'text-[var(--text-primary)]'}"
											title={`${zone} — Click to filter Well & Curve Matrix`}
										>
											{zone}
										</td>
										<td class="mono-num px-1 text-center">
											{#if !zSum || zSum.penetratedWells === 0}
												<span class="text-[9px] text-[var(--text-muted)]">0w</span>
											{:else if zSum.completeWells === zSum.penetratedWells}
												<span
													class="rounded-xs bg-emerald-50 px-1 py-0.5 text-[9px] font-semibold text-emerald-800"
													title={`${zSum.completeWells}/${zSum.penetratedWells} wells evaluated in ${zone}`}
												>
													{zSum.completeWells}/{zSum.penetratedWells}
												</span>
											{:else if zSum.evaluatedWells > 0}
												<span
													class="rounded-xs bg-amber-50 px-1 py-0.5 text-[9px] font-semibold text-amber-800"
													title={`${zSum.evaluatedWells}/${zSum.penetratedWells} wells have computed curves (${zSum.completeWells} complete, ${zSum.partialWells} partial)`}
												>
													{zSum.evaluatedWells}/{zSum.penetratedWells}
												</span>
											{:else}
												<span
													class="rounded-xs bg-slate-100 px-1 py-0.5 text-[9px] font-medium text-slate-600"
													title={`0/${zSum.penetratedWells} wells have computed curves in ${zone} (raw logs only)`}
												>
													0/{zSum.penetratedWells}
												</span>
											{/if}
										</td>
										{#each projectData.wells as w (w.well_id)}
											{@const st = evaluateWellZoneStatus(w, zone, stratCurveMode)}
											{@const missingComp = COMPUTED_CURVES.filter(
												(c) => !st.computedCurves.includes(c)
											)}
											{@const cellTooltip = !st.penetrated
												? `${w.name} — ${zone}: Not penetrated`
												: `${w.name} — ${zone} (${st.topDepth !== null ? st.topDepth.toFixed(1) : '?'}–${st.baseDepth !== null ? st.baseDepth.toFixed(1) : 'TD'} m)\nComputed (${st.computedCurves.length}/7): ${st.computedCurves.length > 0 ? st.computedCurves.join(', ') : 'None'}\nMissing: ${missingComp.length > 0 ? missingComp.join(', ') : 'None'}\nRaw Logs: ${st.rawCurves.join(', ') || 'None'}`}
											<td class="mono-num px-1 text-right" title={cellTooltip}>
												{#if !st.penetrated}
													<span class="text-[var(--text-muted)]">—</span>
												{:else}
													<span
														class="inline-flex items-center justify-end gap-1 rounded-xs px-1 py-0.5 {st.status ===
														'complete'
															? 'bg-emerald-50/80 font-semibold text-emerald-900'
															: st.status === 'partial'
																? 'bg-amber-50/80 font-medium text-amber-900'
																: 'text-[var(--text-secondary)]'}"
													>
														<span
															class="inline-block h-1.5 w-1.5 shrink-0 rounded-full {st.status ===
															'complete'
																? 'bg-emerald-500'
																: st.status === 'partial'
																	? 'bg-amber-500'
																	: 'bg-slate-300'}"
														></span>
														<span>
															{st.topDepth !== null ? st.topDepth.toFixed(1) : '✓'}
														</span>
													</span>
												{/if}
											</td>
										{/each}
									</tr>
								{/each}
							{/if}
						</tbody>
					</table>
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
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<span class="text-[11px] text-[var(--text-muted)]">
			Zones: {projectData.allZones.length} | Curves: {projectData.allCurves.length}
		</span>
		<button
			type="button"
			onclick={() => {
				if (workbench.projectId) projectData.selectProject(workbench.projectId);
			}}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs font-medium text-[var(--text-primary)] hover:bg-[var(--bg-subtle)]"
		>
			Refresh Field
		</button>
	</div>
</div>


