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
		'GR',
		'RT',
		'NPHI',
		'RHOB',
		'PHIT',
		'PERM',
		'SWT',
		'ROCK_FLAG',
		'PERM_TRANS',
		'SHF'
	];

	const CURVE_HEADER_TITLES: Record<string, string> = {
		TVD: 'True Vertical Depth (from TVD/TVDSS curve or computed from directional survey — never assumed equal to MD)',
		GR: 'Gamma Ray',
		RT: 'Deep Resistivity',
		NPHI: 'Neutron Porosity',
		RHOB: 'Bulk Density',
		PHIT: 'Total Porosity',
		PERM: 'Permeability',
		SWT: 'Total Water Saturation',
		ROCK_FLAG: 'Discrete Rock Type Flag (DRT)',
		PERM_TRANS: 'Rock-Type Permeability Transform',
		SHF: 'Saturation Height Function (Sw from Pc / HAFWL)'
	};

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
			pressureTotal,
			pressureWells,
			surveyTotal,
			surveyWells
		};
	});

	function getWellTopDepth(w: (typeof projectData.wells)[number], zoneName: string): number | null {
		const found = (w.formation_tops ?? []).find(
			(t) => t.name.trim().toLowerCase() === zoneName.trim().toLowerCase()
		);
		return found?.depth ?? null;
	}

	function hasWellZone(w: (typeof projectData.wells)[number], zoneName: string): boolean {
		return (w.zones ?? []).some(
			(z) => z.trim().toLowerCase() === zoneName.trim().toLowerCase()
		);
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

		<ParameterSection title="Well & Curve Availability Matrix">
			<div class="overflow-x-auto border border-[var(--border-hairline)]">
				<table class="w-full border-collapse text-left text-[11px]">
					<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
						<tr class="h-6 border-b border-[var(--border-hairline)]">
							<th class="px-1.5 font-semibold">Well</th>
							<th class="px-1 text-right font-semibold">Rows</th>
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
								<tr
									onclick={() => openWellInSingleMode(w.name)}
									title="Click to inspect well in Single-Well mode"
									class="h-6 cursor-pointer hover:bg-[var(--bg-subtle)] {workbench.wellId ===
									w.name
										? 'bg-blue-50/70'
										: ''}"
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
										{w.depth_count ?? (w as any).row_count ?? 0}
									</td>
									{#each KEY_CURVES as kc}
										{@const rawCurves = w.curves ?? (w as any).available_curves ?? []}
										{@const has =
											kc === 'TVD'
												? projectData.wellHasTvd(w)
												: rawCurves.includes(kc)}
										{@const isTvdFromSurvey =
											kc === 'TVD' && (w.survey_count ?? 0) >= 2}
										<td
											class="px-1 text-center"
											title={kc === 'TVD'
												? has
													? isTvdFromSurvey
														? `${w.name}: TVD available (computed from ${w.survey_count} directional survey stations)`
														: `${w.name}: TVD curve available`
													: `${w.name}: No TVD curve or directional survey — SHF & Stick Plot disabled for this well`
												: `${w.name}: ${kc} ${has ? 'available' : 'missing'}`}
										>
											<span
												class="inline-block h-2 w-2 rounded-full {has
													? 'bg-emerald-500'
													: 'bg-slate-200'}"
											></span>
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
				title="Formation Tops Stratigraphic Matrix"
				badge={`${projectData.allZones.length} Zones`}
				defaultOpen={false}
			>
				<div class="overflow-x-auto border border-[var(--border-hairline)]">
					<table class="w-full border-collapse text-left text-[10.5px]">
						<thead class="bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
							<tr class="h-6 border-b border-[var(--border-hairline)]">
								<th class="px-1.5 font-semibold">Zone / Top</th>
								{#each projectData.wells as w (w.well_id)}
									<th class="mono-num px-1 text-right font-semibold text-[9.5px]" title={w.name}>
										{w.name}
									</th>
								{/each}
							</tr>
						</thead>
						<tbody class="divide-y divide-[var(--border-hairline)]">
							{#each projectData.allZones as zone (zone)}
								<tr class="h-6 hover:bg-[var(--bg-subtle)]">
									<td class="truncate px-1.5 font-semibold text-[var(--text-primary)]" title={zone}>
										{zone}
									</td>
									{#each projectData.wells as w (w.well_id)}
										{@const topDepth = getWellTopDepth(w, zone)}
										{@const present = hasWellZone(w, zone)}
										<td class="mono-num px-1 text-right">
											{#if topDepth !== null}
												<span class="font-medium text-[var(--text-primary)]">
													{topDepth.toFixed(1)}
												</span>
											{:else if present}
												<span
													class="rounded-2xs bg-emerald-50 px-1 py-0.5 text-[9px] font-semibold text-emerald-700"
													title="Present in ZONES curve"
												>
													✓
												</span>
											{:else}
												<span class="text-[var(--text-muted)]">—</span>
											{/if}
										</td>
									{/each}
								</tr>
							{/each}
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


