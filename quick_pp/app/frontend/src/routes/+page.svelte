<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { replaceState } from '$app/navigation';
	import { browser } from '$app/environment';
	import { PaneGroup, Pane, PaneResizer } from 'paneforge';

	import LogTrackCanvas from '$lib/components/workbench/LogTrackCanvas.svelte';
	import StickPlotCanvas from '$lib/components/workbench/StickPlotCanvas.svelte';
	import CrossplotCanvas from '$lib/components/workbench/CrossplotCanvas.svelte';
	import CompactDataGrid, {
		type GridColumn
	} from '$lib/components/workbench/CompactDataGrid.svelte';
	import ParametersManagementPanel from '$lib/components/workbench/ParametersManagementPanel.svelte';
	import ReportPreviewCanvas from '$lib/components/workbench/ReportPreviewCanvas.svelte';

	import InspectorQaqc from '$lib/components/modules/InspectorQaqc.svelte';
	import InspectorAncillary from '$lib/components/modules/InspectorAncillary.svelte';
	import InspectorLithoPoro from '$lib/components/modules/InspectorLithoPoro.svelte';
	import InspectorPerm from '$lib/components/modules/InspectorPerm.svelte';
	import InspectorSaturation from '$lib/components/modules/InspectorSaturation.svelte';
	import InspectorScal from '$lib/components/modules/InspectorScal.svelte';
	import InspectorResSum from '$lib/components/modules/InspectorResSum.svelte';
	import InspectorReport from '$lib/components/modules/InspectorReport.svelte';

	import MultiWellInventory from '$lib/components/modules/MultiWellInventory.svelte';
	import MultiWellQaqc from '$lib/components/modules/MultiWellQaqc.svelte';
	import MultiWellRockTyping from '$lib/components/modules/MultiWellRockTyping.svelte';
	import MultiWellPermTransform from '$lib/components/modules/MultiWellPermTransform.svelte';
	import MultiWellSwirr from '$lib/components/modules/MultiWellSwirr.svelte';
	import MultiWellShf from '$lib/components/modules/MultiWellShf.svelte';
	import MultiWellRtClassifierTrainer from '$lib/components/modules/MultiWellRtClassifierTrainer.svelte';
	import MultiWellMlTrainer from '$lib/components/modules/MultiWellMlTrainer.svelte';
	import MultiWellFpt from '$lib/components/modules/MultiWellFpt.svelte';
	import FptPlotCanvas from '$lib/components/workbench/FptPlotCanvas.svelte';
	import MultiWellVolumetrics from '$lib/components/modules/MultiWellVolumetrics.svelte';
	import VolumetricsCanvas from '$lib/components/workbench/VolumetricsCanvas.svelte';
	import VolumetricsRightPanel from '$lib/components/workbench/VolumetricsRightPanel.svelte';
	import { fpt } from '$lib/state/fpt.svelte';

	import {
		workbench,
		type MultiWellStep,
		type SingleWellStep
	} from '$lib/state/workbench.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import {
		currentSelection,
		moduleParams,
		selectionKey
	} from '$lib/state/moduleParams.svelte';
	import { projectParams, zoneSelectionKey } from '$lib/state/projectParams.svelte';
	import { isCrossplotAvailable } from '$lib/utils/crossplots';

	$effect(() => {
		const selected = workbench.crossplotTemplateOverride;
		if (selected !== 'auto' && !isCrossplotAvailable(selected, workbench.scope)) {
			workbench.crossplotTemplateOverride = 'auto';
		}
	});

	// Auto-restore saved single-well module parameters on exact zone / depth-interval match.
	// Re-applies only when the loaded well, the selection key, or the scope changes (not on every
	// config update) so unsaved edits are not overwritten by unrelated config writes.
	let lastParamsRestoreTrigger = '';
	$effect(() => {
		const gen = wellData.wellLoadGeneration;
		const loading = wellData.isLoading;
		const scope = workbench.scope;
		const key = selectionKey(currentSelection());
		if (loading || gen === 0 || scope !== 'single-well' || !workbench.wellId) return;
		const trigger = `${workbench.projectId}|${workbench.wellId}|${gen}|${key}`;
		if (trigger === lastParamsRestoreTrigger) return;
		lastParamsRestoreTrigger = trigger;
		untrack(() => moduleParams.applyForCurrentSelection(wellData.config));
	});

	// Auto-restore saved multi-well (project-level) parameters on exact zone-selection match.
	let lastProjectParamsRestoreTrigger = '';
	$effect(() => {
		const loadGen = projectParams.loadGeneration;
		const scope = workbench.scope;
		const key = zoneSelectionKey(workbench.selectedZones);
		if (loadGen === 0 || scope !== 'multi-well' || !workbench.projectId) return;
		const trigger = `${workbench.projectId}|${loadGen}|${key}`;
		if (trigger === lastProjectParamsRestoreTrigger) return;
		lastProjectParamsRestoreTrigger = trigger;
		untrack(() => projectData.applyProjectParamsForCurrentSelection());
	});

	// Keep SCAL Swirr & Leverett-J / Pc curve fits synchronized with active Well, Zone, and Depth filters
	let lastScalFilterKey = '';
	$effect(() => {
		const wellsKey = Array.from(projectData.selectedWellNames).sort().join(',');
		const zonesKey = zoneSelectionKey(workbench.selectedZones);
		const depthKey = workbench.depthFilter
			? `${workbench.depthFilter.min}:${workbench.depthFilter.max}`
			: 'all';
		const scopeKey = `${workbench.scope}:${workbench.wellId ?? ''}`;
		const hasSwirr = Boolean(projectData.swirrResult?.plugs?.length);
		if (!hasSwirr) return;
		const nextKey = `${scopeKey}|${wellsKey}|${zonesKey}|${depthKey}`;
		if (nextKey === lastScalFilterKey) return;
		lastScalFilterKey = nextKey;
		untrack(() => {
			projectData.recomputeSwirrClientSide(!projectData.swirrManualRuleOverride);
		});
	});

	let initialized = $state(false);
	let isReportMode = $derived(
		(workbench.scope === 'single-well' && workbench.singleWellStep === 'report') ||
			(workbench.scope === 'multi-well' && workbench.multiWellStep === 'report')
	);
	let isFptMode = $derived(workbench.scope === 'multi-well' && workbench.multiWellStep === 'fpt');
	let isVolumetricsMode = $derived(workbench.scope === 'multi-well' && workbench.multiWellStep === 'volumetrics');

	/** FPT grid: clicking a row selects the point on the Pressure vs TVDSS plots. SCAL grids: clicking a row spotlights that sample plug. */
	function handleRightGridRowClick(row: Record<string, any>) {
		if (isFptMode && typeof row?._id === 'number') {
			fpt.selectedIds = [row._id];
			return;
		}
		if (typeof row?._scalPlugId === 'string' && row._scalPlugId) {
			projectData.toggleFocusedScalPlug(row._scalPlugId);
		}
	}

	onMount(async () => {
		const params = new URLSearchParams(window.location.search);
		const urlProj = params.get('project');
		const urlWell = params.get('well');
		const urlScope = params.get('scope');
		const urlStep = params.get('step');

		if (urlScope === 'multi-well' || urlScope === 'single-well') {
			workbench.scope = urlScope;
		}
		if (urlStep) {
			if (workbench.scope === 'multi-well') {
				workbench.multiWellStep = urlStep as MultiWellStep;
				if (urlStep === 'shf') {
					workbench.centerPaneMultiWellView = 'stick-plot';
				}
			} else {
				workbench.singleWellStep = urlStep as SingleWellStep;
			}
		}

		await projectData.loadProjects();
		if (urlProj) {
			await projectData.selectProject(urlProj, urlWell);
		}
		initialized = true;
	});

	// Keep URL search parameters synchronized with active project, well, scope, and step
	$effect(() => {
		if (!browser || !initialized) return;
		const url = new URL(window.location.href);
		if (workbench.projectId) {
			url.searchParams.set('project', workbench.projectId);
		} else {
			url.searchParams.delete('project');
		}
		if (workbench.wellId) {
			url.searchParams.set('well', workbench.wellId);
		} else {
			url.searchParams.delete('well');
		}
		url.searchParams.set('scope', workbench.scope);
		url.searchParams.set(
			'step',
			workbench.scope === 'single-well' ? workbench.singleWellStep : workbench.multiWellStep
		);
		replaceState(url, {});
	});

	let totalSavedParamsCount = $derived(
		workbench.scope === 'multi-well'
			? Object.values(projectParams.savedCounts).reduce((a, b) => a + (b ?? 0), 0)
			: Object.values(moduleParams.savedCounts).reduce((a, b) => a + (b ?? 0), 0)
	);

	// Dynamic columns & rows for the Bottom-Right Quick Zonal / Context Data Grid
	let rightGridConfig = $derived.by(
		(): { title: string; columns: GridColumn[]; rows: Record<string, any>[] } => {
			if (
				(workbench.effectiveCrossplot === 'relperm' ||
					(workbench.effectiveCrossplot === 'swirr-fit' &&
						workbench.swirrCrossplotTab === 'relperm')) &&
				workbench.scalResult?.swofTable?.length
			) {
				return {
					title: 'Eclipse SWOF Simulation Deck Table',
					columns: [
						{ key: 'SW', label: 'Sw', unit: 'v/v', numeric: true, decimals: 4 },
						{ key: 'KRW', label: 'krw', unit: 'frac', numeric: true, decimals: 5 },
						{ key: 'KROW', label: 'krow', unit: 'frac', numeric: true, decimals: 5 },
						{ key: 'PCOW', label: 'Pcow', unit: 'psi', numeric: true, decimals: 3 }
					],
					rows: workbench.scalResult.swofTable
				};
			}

			if (workbench.scope === 'multi-well' && workbench.multiWellStep === 'fpt') {
				const res = fpt.result;
				const du = res?.units.depth ?? fpt.depthUnit;
				const pu = res?.units.pressure ?? fpt.pressureUnit;
				const edits = fpt.pointEdits;
				return {
					title: res
						? `Formation Pressure Points (${res.n_points} pts${fpt.pendingCount ? ` · ${fpt.pendingCount} unsaved` : ''})`
						: 'Formation Pressure Points — run Analyze',
					columns: [
						{ key: 'WELL', label: 'Well' },
						{ key: 'MD', label: 'MD', unit: 'm', numeric: true, decimals: 1 },
						{ key: 'TVDSS', label: 'TVDSS', unit: du, numeric: true, decimals: 1 },
						{ key: 'PRESSURE', label: 'Pressure', unit: pu, numeric: true, decimals: 2 },
						{ key: 'EXCESS', label: 'Excess', unit: pu, numeric: true, decimals: 2 },
						{ key: 'MOBILITY', label: 'Mobility', unit: 'mD/cP', numeric: true, decimals: 2 },
						{ key: 'FLUID', label: 'Fluid' },
						{ key: 'QC', label: 'QC' },
						{ key: 'FIT', label: 'In Fit' },
						{ key: 'ZONE', label: 'Zone' },
						{ key: 'REMARKS', label: 'Remarks' }
					],
					rows: (res?.points ?? []).map((p) => ({
						_id: p.id,
						WELL: p.well_name,
						MD: p.md,
						TVDSS: p.tvdss_display,
						PRESSURE: p.pressure_display,
						EXCESS: p.excess_pressure,
						MOBILITY: p.mobility,
						FLUID: `${p.fluid_type}${edits[String(p.id)]?.fluid_type ? ' *' : ''}`,
						QC: `${p.test_quality}${edits[String(p.id)]?.test_quality ? ' *' : ''}`,
						FIT: p.used_in_fit ? '✓' : '',
						ZONE: p.zone || '—',
						REMARKS: p.tvdss_source === 'missing' ? 'No TVDSS' : p.remarks || ''
					}))
				};
			}

			if (
				workbench.scope === 'multi-well' &&
				workbench.multiWellStep === 'inventory' &&
				projectData.wells.length > 0
			) {
				const selZone = projectData.inventorySelectedZone;
				if (selZone) {
					return {
						title: `Formation Curve Coverage — ${selZone} (${projectData.wells.length} Wells)`,
						columns: [
							{ key: 'WELL', label: 'Well' },
							{ key: 'TOP_DEPTH', label: 'Top', unit: 'm', numeric: true, decimals: 1 },
							{ key: 'BASE_DEPTH', label: 'Base', unit: 'm', numeric: true, decimals: 1 },
							{ key: 'COMPUTED_STATUS', label: 'Eval Status' },
							{ key: 'COMPUTED_CURVES', label: 'Computed Curves Present' },
							{ key: 'MISSING_CURVES', label: 'Missing Computed' },
							{ key: 'RAW_CURVES', label: 'Base Logs' }
						],
						rows: projectData.wells.map((w) => {
							const topEntry = (w.formation_tops ?? []).find(
								(t) => t.name.trim().toLowerCase() === selZone.trim().toLowerCase()
							);
							const penetrated =
								topEntry !== undefined ||
								(w.zones ?? []).some(
									(z) => z.trim().toLowerCase() === selZone.trim().toLowerCase()
								);
							const zCurves = projectData.getWellZoneCurves(w, selZone) ?? [];
							const compAll = [
								'VCLAY',
								'PHIT',
								'PERM',
								'SWT',
								'ROCK_FLAG',
								'PERM_TRANS',
								'SHF'
							];
							const compPresent = compAll.filter((c) => zCurves.includes(c));
							const compMissing = compAll.filter((c) => !zCurves.includes(c));
							const rawPresent = ['TVD', 'GR', 'RT', 'NPHI', 'RHOB'].filter((c) =>
								zCurves.includes(c)
							);
							return {
								WELL: w.name,
								TOP_DEPTH: topEntry?.depth ?? null,
								BASE_DEPTH: topEntry?.base_depth ?? null,
								COMPUTED_STATUS: !penetrated
									? 'Not Penetrated'
									: compPresent.length >= 4
										? `Evaluated (${compPresent.length}/7)`
										: compPresent.length > 0
											? `Partial (${compPresent.length}/7)`
											: 'Raw Only (0/7)',
								COMPUTED_CURVES: penetrated ? compPresent.join(', ') || '—' : '—',
								MISSING_CURVES: penetrated ? compMissing.join(', ') || 'None' : '—',
								RAW_CURVES: penetrated ? rawPresent.join(', ') || '—' : '—'
							};
						})
					};
				}

				return {
					title: `Field Well, Core (RCA & SCAL), Tops & Ancillary Inventory (${projectData.wells.length} Wells)`,
					columns: [
						{ key: 'WELL', label: 'Well' },
						{ key: 'TVD', label: 'TVD' },
						{ key: 'MIN_DEPTH', label: 'Top', unit: 'm', numeric: true, decimals: 1 },
						{ key: 'MAX_DEPTH', label: 'Bottom', unit: 'm', numeric: true, decimals: 1 },
						{ key: 'ROWS', label: 'Rows', numeric: true, decimals: 0 },
						{ key: 'CURVES', label: 'Curves', numeric: true, decimals: 0 },
						{ key: 'RCA_PLUGS', label: 'RCA Plugs', numeric: true, decimals: 0 },
						{ key: 'CORE_PCT', label: 'Core %', unit: '%', numeric: true, decimals: 1 },
						{ key: 'SCAL_PC', label: 'SCAL Pc Pts', numeric: true, decimals: 0 },
						{ key: 'SCAL_PLUGS', label: 'SCAL Plugs', numeric: true, decimals: 0 },
						{ key: 'TOPS', label: 'Tops', numeric: true, decimals: 0 },
						{ key: 'EVAL_ZONES', label: 'Eval Zones' },
						{ key: 'ZONES', label: 'Formations / Zones' },
						{ key: 'CONTACTS', label: 'Fluid Contacts' },
						{ key: 'PERFS', label: 'Perforations' },
						{ key: 'PRESSURE', label: 'Pressure Pts', numeric: true, decimals: 0 },
						{ key: 'SURVEY', label: 'Survey Stns', numeric: true, decimals: 0 }
					],
					rows: projectData.wells.map((w) => {
						const wZones = w.zones ?? [];
						const evalCnt = wZones.filter(
							(z) => projectData.getWellZoneComputedCurves(w, z).length > 0
						).length;
						return {
							WELL: w.name,
							TVD:
								(w.survey_count ?? 0) >= 2
									? `Survey (${w.survey_count})`
									: projectData.wellHasTvd(w)
										? 'Curve'
										: 'No TVD',
							MIN_DEPTH: w.min_depth,
							MAX_DEPTH: w.max_depth,
							ROWS: w.depth_count ?? w.row_count ?? 0,
							CURVES: (w.curves ?? w.available_curves ?? []).length,
							RCA_PLUGS: w.core_sample_count ?? w.core_samples_count ?? 0,
							CORE_PCT: w.core_availability_pct ?? 0,
							SCAL_PC: w.scal_pc_count ?? 0,
							SCAL_PLUGS: w.scal_sample_count ?? 0,
							TOPS: w.formation_top_count ?? w.formation_tops_count ?? wZones.length,
							EVAL_ZONES: wZones.length > 0 ? `${evalCnt}/${wZones.length}` : '—',
							ZONES: wZones.join(', ') || '—',
							CONTACTS:
								(w.fluid_contacts ?? []).length > 0
									? (w.fluid_contacts ?? [])
											.map((c) => `${c.type}${c.depth !== null ? ` @ ${c.depth}m` : ''}`)
											.join(', ')
									: (w.fluid_contact_count ?? 0) > 0
										? String(w.fluid_contact_count)
										: '—',
							PERFS:
								(w.perforations ?? []).length > 0
									? (w.perforations ?? [])
											.map(
												(p) =>
													`${p.name}${p.top !== null && p.bottom !== null ? ` ${p.top}–${p.bottom}m` : ''}`
											)
											.join(', ')
									: (w.perforation_count ?? 0) > 0
										? String(w.perforation_count)
										: '—',
							PRESSURE: w.pressure_test_count ?? w.pressure_tests_count ?? 0,
							SURVEY: w.survey_count ?? 0
						};
					})
				};
			}

			if (
				workbench.scope === 'multi-well' &&
				workbench.multiWellStep === 'qaqc' &&
				projectData.multiWellQaqcResult?.group_stats?.length
			) {
				const qaqc = projectData.multiWellQaqcResult;
				const cols: GridColumn[] = [
					{ key: 'WELL_NAME', label: 'Well' },
					{ key: 'ZONE', label: 'Zone' },
					{ key: 'COUNT', label: 'Count', numeric: true, decimals: 0 },
					{ key: 'BADHOLE_PCT', label: 'Bad %', unit: '%', numeric: true, decimals: 1 },
					{ key: 'MIN', label: 'Min', numeric: true, decimals: 2 },
					{ key: 'P10', label: 'P10', numeric: true, decimals: 2 },
					{ key: 'P50', label: 'P50', numeric: true, decimals: 2 },
					{ key: 'MEAN', label: 'Mean', numeric: true, decimals: 2 },
					{ key: 'P90', label: 'P90', numeric: true, decimals: 2 },
					{ key: 'MAX', label: 'Max', numeric: true, decimals: 2 },
					...(qaqc.norm_curve
						? [{ key: 'NORM_P50', label: 'Norm P50', numeric: true, decimals: 2 }]
						: []),
					{ key: 'DRIFT', label: 'Drift Δ', numeric: true, decimals: 2 },
					{ key: 'STATUS', label: 'Status' }
				];
				return {
					title: `Multi-Well ${qaqc.active_curve} Distribution Summary (${qaqc.group_by.toUpperCase()})`,
					columns: cols,
					rows: qaqc.group_stats
				};
			}

			if (
				workbench.scope === 'multi-well' &&
				((workbench.multiWellStep === 'swirr-corr' &&
					workbench.swirrCrossplotTab === 'scal-fit') ||
					workbench.multiWellStep === 'shf') &&
				projectData.shfResult?.scal_data?.pc?.length
			) {
				const scal = projectData.shfResult.scal_data;
				const plugMeta = new Map<string, { depth: number; zone: string }>();
				for (const p of projectData.swirrResult?.plugs ?? []) {
					const pid = projectData.normalizeScalPlugId(p.plug_id, p.well, p.sample);
					plugMeta.set(pid, {
						depth: p.depth ?? 0,
						zone: projectData.resolveScalSampleZone(p.well, p.depth, p.zone)
					});
				}
				const focusedPid = projectData.focusedScalPlugId
					? projectData.normalizeScalPlugId(projectData.focusedScalPlugId)
					: null;
				const rtFilter = projectData.scalFilterRockType;
				const rows = scal.pc
					.map((pcVal, i) => {
						const wn = scal.well_names?.[i] ?? '';
						const sname = scal.samples?.[i] ?? '';
						const pid = projectData.normalizeScalPlugId(scal.plug_ids?.[i], wn, sname);
						const meta = plugMeta.get(pid);
						const d = scal.depths?.[i] ?? meta?.depth ?? null;
						const zn = projectData.resolveScalSampleZone(
							wn,
							d,
							scal.zones?.[i] ?? meta?.zone
						);
						const rf = scal.rock_flags?.[i] ?? null;
						const isExcl = Boolean(scal.excluded?.[i]);
						const isFocused = focusedPid === pid;
						return {
							_scalPlugId: pid,
							Well: wn,
							Sample: isFocused ? `★ ${sname}` : sname,
							ZONE: zn,
							DEPTH: d,
							ROCK_FLAG: rf,
							PC: pcVal,
							SW: scal.sw?.[i] ?? null,
							SWIRR_CALC: scal.swirr_calc?.[i] ?? null,
							SWN: scal.swn?.[i] ?? null,
							J: scal.j?.[i] ?? null,
							H: scal.h?.[i] ?? null,
							PHIT: scal.phit?.[i] ?? null,
							PERM: scal.perm?.[i] ?? null,
							STATUS: isExcl ? 'Excluded' : isFocused ? 'Focused' : 'Active'
						};
					})
					.filter((r) => {
						if (!projectData.isScalItemInActiveFilter(r.Well, r.DEPTH, r.ZONE)) return false;
						if (rtFilter !== 'all' && r.ROCK_FLAG !== rtFilter) return false;
						if (focusedPid && r._scalPlugId !== focusedPid) return false;
						return true;
					});
				return {
					title: focusedPid
						? `SCAL Capillary Pressure & Leverett-J — Sample ${focusedPid.replace('::', ' · ')} (click row to clear focus)`
						: 'Multi-Well SCAL Capillary Pressure & Leverett-J Table (click row to spotlight sample)',
					columns: [
						{ key: 'Well', label: 'Well' },
						{ key: 'Sample', label: 'Sample' },
						{ key: 'ZONE', label: 'Zone' },
						{ key: 'DEPTH', label: 'Depth', unit: 'm', numeric: true, decimals: 1 },
						{ key: 'ROCK_FLAG', label: 'RT', numeric: true, decimals: 0 },
						{ key: 'PC', label: 'Pc', unit: 'psi', numeric: true, decimals: 2 },
						{ key: 'SW', label: 'Sw', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'SWIRR_CALC', label: 'Swirr', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'SWN', label: 'Swn', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'J', label: 'J', numeric: true, decimals: 3 },
						{ key: 'H', label: 'Height', unit: 'm', numeric: true, decimals: 1 },
						{ key: 'PHIT', label: 'CPORE', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'PERM', label: 'CPERM', unit: 'mD', numeric: true, decimals: 2 },
						{ key: 'STATUS', label: 'Status' }
					],
					rows
				};
			}

			if (
				workbench.scope === 'multi-well' &&
				workbench.multiWellStep === 'swirr-corr' &&
				projectData.swirrResult?.plugs?.length
			) {
				const focusedPid = projectData.focusedScalPlugId
					? projectData.normalizeScalPlugId(projectData.focusedScalPlugId)
					: null;
				const rtFilter = projectData.scalFilterRockType;
				const plugs = projectData.swirrResult.plugs.filter((p) => {
					const zn = projectData.resolveScalSampleZone(p.well, p.depth, p.zone);
					if (!projectData.isScalItemInActiveFilter(p.well, p.depth, zn)) return false;
					if (rtFilter !== 'all' && (p.rt ?? 1) !== rtFilter) return false;
					return true;
				});
				return {
					title: `Multi-Well SCAL Plug Swirr Summary (Ref Pc=${projectData.swirrConfig.referencePcPsi} psi | n=${plugs.filter((p) => p.included).length}/${plugs.length} in filter — click row to spotlight)`,
					columns: [
						{ key: 'WELL', label: 'Well' },
						{ key: 'SAMPLE', label: 'Sample' },
						{ key: 'ZONE', label: 'Zone' },
						{ key: 'DEPTH', label: 'Depth', unit: 'm', numeric: true, decimals: 1 },
						{ key: 'RT', label: 'RT', numeric: true, decimals: 0 },
						{ key: 'CPORE', label: 'CPORE', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'CPERM', label: 'CPERM', unit: 'mD', numeric: true, decimals: 2 },
						{ key: 'RQI', label: 'RQI', unit: 'μm', numeric: true, decimals: 3 },
						{ key: 'MAX_PC', label: 'Max Pc [psi]', unit: 'psi', numeric: true, decimals: 1 },
						{ key: 'SW_AT_MAX_PC', label: 'Sw @ Max Pc', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'EXTRACTED_SWIRR', label: 'Swirr (Lab)', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'MODELED_SWIRR', label: 'Modeled Swirr', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'RESIDUAL', label: 'Residual', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'STATUS', label: 'Status' }
					],
					rows: plugs.map((p) => {
						const pid = projectData.normalizeScalPlugId(p.plug_id, p.well, p.sample);
						const isFocused = focusedPid === pid;
						return {
							_scalPlugId: pid,
							WELL: p.well,
							SAMPLE: isFocused ? `★ ${p.sample}` : p.sample,
							ZONE: projectData.resolveScalSampleZone(p.well, p.depth, p.zone),
							DEPTH: p.depth,
							RT: p.rt,
							CPORE: p.cpore,
							CPERM: p.cperm,
							RQI: p.rqi,
							MAX_PC: p.max_pc,
							SW_AT_MAX_PC: p.sw_at_max_pc,
							EXTRACTED_SWIRR: p.extracted_swirr,
							MODELED_SWIRR: p.modeled_swirr,
							RESIDUAL: p.residual,
							STATUS: isFocused
								? 'Focused'
								: p.included
									? 'Included'
									: !p.valid_pc
										? 'Low Pc (< Ref)'
										: 'Excluded'
						};
					})
				};
			}

			if (
				workbench.scope === 'multi-well' &&
				(workbench.multiWellStep === 'rt-classifier' ||
					workbench.multiWellStep === 'rock-typing') &&
				projectData.rockTypingResult?.diagnostics?.classifier?.class_metrics?.length
			) {
				const clf = projectData.rockTypingResult.diagnostics.classifier;
				const compRows: Array<Record<string, any>> = clf.comparison?.per_drt ?? [];
				const compByDrt = new Map<number, Record<string, any>>(
					compRows.map((r: any) => [Number(r.drt), r])
				);
				return {
					title: `Rock Type Classifier Per-DRT Validation & Propagation Metrics (Acc=${((Number(clf.metrics?.accuracy ?? 0)) * 100).toFixed(1)}%, F1=${Number(clf.metrics?.f1_weighted ?? 0).toFixed(3)}, κ=${Number(clf.metrics?.cohen_kappa ?? 0).toFixed(3)})`,
					columns: [
						{ key: 'DRT', label: 'Group' },
						{ key: 'PRECISION', label: 'Precision', numeric: true, decimals: 3 },
						{ key: 'RECALL', label: 'Recall', numeric: true, decimals: 3 },
						{ key: 'F1', label: 'F1 Score', numeric: true, decimals: 3 },
						{ key: 'SUPPORT', label: 'Val Support', numeric: true, decimals: 0 },
						{ key: 'CUTOFF_PTS', label: 'Cutoff Pts', numeric: true, decimals: 0 },
						{ key: 'ML_PTS', label: 'ML Pred Pts', numeric: true, decimals: 0 },
						{ key: 'HYBRID_PTS', label: 'Cut+ML Pts', numeric: true, decimals: 0 }
					],
					rows: (clf.class_metrics ?? []).map((cm: any) => {
						const cr = compByDrt.get(Number(cm.drt));
						return {
							DRT: `DRT ${cm.drt}`,
							PRECISION: cm.precision,
							RECALL: cm.recall,
							F1: cm.f1,
							SUPPORT: cm.support,
							CUTOFF_PTS: cr?.cutoff_count ?? null,
							ML_PTS: cr?.ml_count ?? null,
							HYBRID_PTS: cr?.hybrid_count ?? null
						};
					})
				};
			}

			if (
				workbench.scope === 'multi-well' &&
				workbench.multiWellStep === 'ml-training' &&
				projectData.mlTrainingResult?.parity_rows?.length
			) {
				return {
					title: 'Held-Out Validation Parity & Conformal P10–P90 Bounds',
					columns: [
						{ key: 'WELL_NAME', label: 'Well' },
						{ key: 'DEPTH', label: 'Depth', unit: 'm', numeric: true, decimals: 1 },
						{
							key: 'PHIT_TRUE',
							label: 'Φ True',
							unit: 'v/v',
							numeric: true,
							decimals: 3
						},
						{
							key: 'PHIT_PRED',
							label: 'Φ Pred',
							unit: 'v/v',
							numeric: true,
							decimals: 3
						},
						{
							key: 'PHIT_P10',
							label: 'Φ P10',
							unit: 'v/v',
							numeric: true,
							decimals: 3
						},
						{
							key: 'PHIT_P90',
							label: 'Φ P90',
							unit: 'v/v',
							numeric: true,
							decimals: 3
						},
						{
							key: 'SWT_TRUE',
							label: 'Sw True',
							unit: 'v/v',
							numeric: true,
							decimals: 3
						},
						{
							key: 'SWT_PRED',
							label: 'Sw Pred',
							unit: 'v/v',
							numeric: true,
							decimals: 3
						},
						{ key: 'LOG_PERM_TRUE', label: 'logK True', numeric: true, decimals: 2 },
						{ key: 'LOG_PERM_PRED', label: 'logK Pred', numeric: true, decimals: 2 }
					],
					rows: projectData.mlTrainingResult.parity_rows
				};
			}

			if (
				workbench.scope === 'multi-well' &&
				workbench.multiWellStep === 'volumetrics' &&
				projectData.volumetricResult?.zonal_summary?.length
			) {
				const vr = projectData.volumetricResult;
				const isOil = workbench.multiWellVolumetrics.fluidMetric === 'oil';
				return {
					title: `Zonal Volumetric Breakdown (${isOil ? 'STOIIP MMbbl' : 'GIIP Bcf'})`,
					columns: [
						{ key: 'zone', label: 'Formation' },
						{ key: 'well_count', label: 'Wells', numeric: true, decimals: 0 },
						{ key: 'mean_net_pay', label: 'Net Pay', unit: 'm', numeric: true, decimals: 1 },
						{ key: 'ntg_pay', label: 'NTG', unit: 'v/v', numeric: true, decimals: 2 },
						{ key: 'mean_phie', label: 'Φe', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'mean_sw', label: 'Sw', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'mean_hcpt', label: 'HCPT', unit: 'm', numeric: true, decimals: 2 },
						...(isOil
							? [
									{ key: 'p90_vol', label: '1P (P90)', unit: 'MMbbl', numeric: true, decimals: 2 },
									{ key: 'p50_vol', label: '2P (P50)', unit: 'MMbbl', numeric: true, decimals: 2 },
									{ key: 'p10_vol', label: '3P (P10)', unit: 'MMbbl', numeric: true, decimals: 2 },
									{ key: 'rec_vol', label: 'Rec Oil', unit: 'MMbbl', numeric: true, decimals: 2 }
								]
							: [
									{ key: 'p90_vol', label: '1P (P90)', unit: 'Bcf', numeric: true, decimals: 2 },
									{ key: 'p50_vol', label: '2P (P50)', unit: 'Bcf', numeric: true, decimals: 2 },
									{ key: 'p10_vol', label: '3P (P10)', unit: 'Bcf', numeric: true, decimals: 2 },
									{ key: 'rec_vol', label: 'Rec Gas', unit: 'Bcf', numeric: true, decimals: 2 }
								])
					],
					rows: vr.zonal_summary.map((z) => ({
						zone: z.zone,
						well_count: z.well_count,
						mean_net_pay: z.mean_net_pay,
						ntg_pay: z.ntg_pay,
						mean_phie: z.mean_phie,
						mean_sw: z.mean_sw,
						mean_hcpt: z.mean_hcpt,
						p90_vol: isOil ? z.stoiip?.p90 : z.giip?.p90,
						p50_vol: isOil ? z.stoiip?.p50 : z.giip?.p50,
						p10_vol: isOil ? z.stoiip?.p10 : z.giip?.p10,
						rec_vol: isOil ? z.rec_oil?.p50 : z.rec_gas?.p50
					}))
				};
			}

			if (
				workbench.effectiveCrossplot === 'vsh-qc' &&
				wellData.vshComparison.rows.length > 0
			) {
				const vc = wellData.vshComparison;
				return {
					title: `Vshale Calibration (VSH_GR × VSH_DN) Depth Error Comparison (n=${vc.active.count} | RMSE=${vc.active.rmse ?? '—'}, Bias=${vc.active.bias !== null ? `${vc.active.bias >= 0 ? '+' : ''}${vc.active.bias}` : '—'}, R²=${vc.active.r2 ?? '—'})`,
					columns: [
						{ key: 'DEPTH', label: 'Depth', unit: 'm', numeric: true, decimals: 2 },
						{ key: 'ZONE', label: 'Zone' },
						{ key: 'REGIME_LABEL', label: 'Regime' },
						{ key: 'GR', label: 'GR', unit: 'gAPI', numeric: true, decimals: 1 },
						{ key: 'VSH_DN', label: 'VSH_DN', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'VSH_GR', label: 'VSH_GR', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'ERR_VSH', label: 'ΔVsh (GR−DN)', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'ABS_ERR_VSH', label: '|ΔVsh|', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'NPHI', label: 'NPHI', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'RHOB', label: 'RHOB', unit: 'g/cc', numeric: true, decimals: 3 }
					],
					rows: vc.rows
				};
			}

			if (
				wellData.coreComparison.rows.length > 0 &&
				(workbench.effectiveCrossplot === 'core-qc' ||
					(workbench.scope === 'single-well' &&
						(workbench.singleWellStep === 'litho-poro' ||
							workbench.singleWellStep === 'perm' ||
							workbench.singleWellStep === 'scal')))
			) {
				const cc = wellData.coreComparison;
				return {
					title: `Core vs. Estimated Depth Error Comparison (n=${cc.rows.length} | Φ RMSE=${cc.poro.rmse ?? '—'}, K log₁₀RMSE=${cc.perm.log10_rmse ?? '—'})`,
					columns: [
						{ key: 'DEPTH', label: 'Depth', unit: 'm', numeric: true, decimals: 2 },
						{ key: 'SHIFT_M', label: 'ΔDepth', unit: 'm', numeric: true, decimals: 2 },
						{ key: 'SAMPLE', label: 'Sample' },
						{ key: 'ZONE', label: 'Zone' },
						{ key: 'CPORE', label: 'CPORE', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'PHIT', label: 'PHIT', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'ERR_PHIT', label: 'ΔPHIT', unit: 'v/v', numeric: true, decimals: 4 },
						{ key: 'REL_ERR_PHIT', label: 'Φ Err', unit: '%', numeric: true, decimals: 1 },
						{ key: 'CPERM', label: 'CPERM', unit: 'mD', numeric: true, decimals: 2 },
						{ key: 'PERM', label: 'PERM', unit: 'mD', numeric: true, decimals: 2 },
						{ key: 'ERR_PERM', label: 'ΔPERM', unit: 'mD', numeric: true, decimals: 2 },
						{ key: 'LOG10_ERR_PERM', label: 'Δlog₁₀K', unit: 'dec', numeric: true, decimals: 3 },
						{ key: 'FACTOR_PERM', label: 'K Ratio', unit: '×', numeric: true, decimals: 2 }
					],
					rows: cc.rows
				};
			}

			if (wellData.zonalSummary.length > 0) {
				const sample = wellData.zonalSummary[0];
				const cols: GridColumn[] = Object.keys(sample).map((k) => ({
					key: k,
					label: k,
					numeric: typeof sample[k] === 'number',
					decimals:
						k.toLowerCase().includes('thick') || k.toLowerCase().includes('depth')
							? 1
							: 3
				}));
				return {
					title: 'Zonal Reservoir Summary',
					columns: cols,
					rows: wellData.zonalSummary
				};
			}

			if (wellData.tops.length > 0) {
				const sorted = [...wellData.tops].sort((a, b) => a.depth - b.depth);
				const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'] ?? [];
				const phit = wellData.curves['PHIT'] ?? [];
				const phie = wellData.curves['PHIE'] ?? [];
				const vcl = wellData.curves['VCLAY'] ?? wellData.curves['GR'] ?? [];
				const sw = wellData.curves['SWT'] ?? wellData.curves['SWE'] ?? [];

				const zoneRows = sorted.map((t, idx) => {
					const topD = t.depth;
					const botD =
						idx + 1 < sorted.length ? sorted[idx + 1].depth : wellData.depthRange.max;
					const gross = Math.max(0, botD - topD);

					let phitSum = 0;
					let phitN = 0;
					let phieSum = 0;
					let phieN = 0;
					let vclSum = 0;
					let vclN = 0;
					let swSum = 0;
					let swN = 0;

					dept.forEach((d, i) => {
						if (d >= topD && d < botD) {
							if (Number.isFinite(phit[i])) {
								phitSum += phit[i];
								phitN++;
							}
							if (Number.isFinite(phie[i])) {
								phieSum += phie[i];
								phieN++;
							}
							if (Number.isFinite(vcl[i])) {
								vclSum += vcl[i];
								vclN++;
							}
							if (Number.isFinite(sw[i])) {
								swSum += sw[i];
								swN++;
							}
						}
					});

					return {
						Zone: t.name,
						depth: topD,
						Gross: gross,
						PHIT: phitN > 0 ? phitSum / phitN : null,
						PHIE_QC: phieN > 0 ? phieSum / phieN : null,
						VCLAY: vclN > 0 ? vclSum / vclN : null,
						SW: swN > 0 ? swSum / swN : null
					};
				});

				return {
					title: 'Quick Zonal Metrics (Click row to center log)',
					columns: [
						{ key: 'Zone', label: 'Zone' },
						{ key: 'depth', label: 'Top', unit: 'm', numeric: true, decimals: 1 },
						{ key: 'Gross', label: 'Gross', unit: 'm', numeric: true, decimals: 1 },
						{ key: 'PHIT', label: 'PHIT', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'PHIE_QC', label: 'PHIE (QC)', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'VCLAY', label: 'VCLAY', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'SW', label: 'SW', unit: 'v/v', numeric: true, decimals: 3 }
					],
					rows: zoneRows
				};
			}

			// Fallback: Curve summary statistics table
			const statRows = Object.entries(wellData.stats).map(([curve, s]) => ({
				Curve: curve,
				Top: s.top,
				Bottom: s.bottom,
				Min: s.min,
				Max: s.max,
				Mean: s.mean,
				Std: s.std
			}));

			return {
				title: 'Curve Summary Metrics',
				columns: [
					{ key: 'Curve', label: 'Curve' },
					{ key: 'Top', label: 'Top', unit: 'm', numeric: true, decimals: 1 },
					{ key: 'Bottom', label: 'Bottom', unit: 'm', numeric: true, decimals: 1 },
					{ key: 'Min', label: 'Min', numeric: true, decimals: 3 },
					{ key: 'Max', label: 'Max', numeric: true, decimals: 3 },
					{ key: 'Mean', label: 'Mean', numeric: true, decimals: 3 },
					{ key: 'Std', label: 'Std', numeric: true, decimals: 3 }
				],
				rows: statRows
			};
		}
	);
</script>

<div class="h-full w-full overflow-hidden">
	<PaneGroup direction="horizontal" autoSaveId="quick-pp-3pane-layout" class="h-full w-full">
		<!-- LEFT PANE (25% default width): Parameter Inspector -->
		<Pane defaultSize={25} minSize={18} maxSize={42} class="h-full overflow-hidden">
			<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
				{#if workbench.scope === 'multi-well' || workbench.singleWellStep !== 'ancillary'}
					<div
						class="flex h-7 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-[10px] select-none"
					>
						<span class="font-bold tracking-wider text-[var(--text-muted)] uppercase">
							Sections
						</span>
						<div class="flex items-center gap-1">
							<button
								type="button"
								onclick={() => workbench.expandAllSections()}
								title="Expand all parameter sections"
								class="h-5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 font-medium text-[var(--text-secondary)] transition-colors hover:bg-slate-100 hover:text-[var(--text-primary)]"
							>
								Show All
							</button>
							<button
								type="button"
								onclick={() => workbench.collapseAllSections()}
								title="Collapse all parameter sections"
								class="h-5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 font-medium text-[var(--text-secondary)] transition-colors hover:bg-slate-100 hover:text-[var(--text-primary)]"
							>
								Collapse All
							</button>
							<button
								type="button"
								onclick={() => workbench.toggleFocusMode()}
								title="Focus mode: only open the selected section and automatically close others"
								class="h-5 rounded-xs border px-1.5 font-medium transition-colors {workbench.sectionFocusMode
									? 'border-[var(--accent-primary)] bg-blue-600 font-semibold text-white'
									: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-slate-100 hover:text-[var(--text-primary)]'}"
							>
								Focus
							</button>
						</div>
					</div>
				{/if}
				<div class="min-h-0 flex-1 overflow-hidden">
					{#if workbench.scope === 'single-well'}
						{#if workbench.singleWellStep === 'qaqc'}
							<InspectorQaqc />
						{:else if workbench.singleWellStep === 'ancillary'}
							<InspectorAncillary />
						{:else if workbench.singleWellStep === 'litho-poro'}
							<InspectorLithoPoro />
						{:else if workbench.singleWellStep === 'perm'}
							<InspectorPerm />
						{:else if workbench.singleWellStep === 'saturation'}
							<InspectorSaturation />
						{:else if workbench.singleWellStep === 'scal'}
							<InspectorScal />
						{:else if workbench.singleWellStep === 'report'}
							<InspectorReport />
						{:else}
							<InspectorResSum />
						{/if}
					{:else if workbench.multiWellStep === 'inventory'}
						<MultiWellInventory />
					{:else if workbench.multiWellStep === 'qaqc'}
						<MultiWellQaqc />
					{:else if workbench.multiWellStep === 'rock-typing'}
						<MultiWellRockTyping />
					{:else if workbench.multiWellStep === 'perm-transform'}
						<MultiWellPermTransform />
					{:else if workbench.multiWellStep === 'swirr-corr'}
						<MultiWellSwirr />
					{:else if workbench.multiWellStep === 'rt-classifier'}
						<MultiWellRtClassifierTrainer />
					{:else if workbench.multiWellStep === 'ml-training'}
						<MultiWellMlTrainer />
					{:else if workbench.multiWellStep === 'fpt'}
						<MultiWellFpt />
					{:else if workbench.multiWellStep === 'volumetrics'}
						<MultiWellVolumetrics />
					{:else if workbench.multiWellStep === 'report'}
						<InspectorReport />
					{:else}
						<MultiWellShf />
					{/if}
				</div>
			</div>
		</Pane>

		<PaneResizer
			class="w-[3px] cursor-col-resize bg-[var(--border-hairline)] transition-colors hover:bg-[var(--accent-primary)] active:bg-[var(--accent-primary)]"
		/>

		{#if isReportMode}
			<!-- REPORT WORKBENCH (75% default width): Live A4 Preview & Markdown Editor -->
			<Pane defaultSize={75} minSize={45} class="h-full overflow-hidden">
				<ReportPreviewCanvas />
			</Pane>
		{:else}
			<!-- CENTER PANE (45% default width): Synchronized Well Log Canvas or Multi-Well Stick Plot -->
			<Pane defaultSize={45} minSize={28} class="h-full overflow-hidden">
				{#if isFptMode}
					<FptPlotCanvas initialMode="pressure" />
				{:else if isVolumetricsMode}
					<VolumetricsCanvas />
				{:else if workbench.scope === 'multi-well' && workbench.centerPaneMultiWellView === 'stick-plot'}
					<StickPlotCanvas />
				{:else}
					<LogTrackCanvas />
				{/if}
			</Pane>

			<PaneResizer
				class="w-[3px] cursor-col-resize bg-[var(--border-hairline)] transition-colors hover:bg-[var(--accent-primary)] active:bg-[var(--accent-primary)]"
			/>

			<!-- RIGHT PANE (30% default width): Crossplot & Zonal Data Grid -->
			<Pane defaultSize={30} minSize={20} class="h-full overflow-hidden">
				{#if isVolumetricsMode}
					<VolumetricsRightPanel />
				{:else if workbench.rightPaneTab === 'crossplot'}
					{#if isFptMode}
						<FptPlotCanvas initialMode="excess" />
					{:else}
						<CrossplotCanvas />
					{/if}
				{:else if workbench.rightPaneTab === 'grid'}
					<div class="flex h-full flex-col bg-[var(--bg-surface)]">
						<div
							class="flex h-7 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-[11px]"
						>
							<div class="flex items-center gap-1">
								<button
									type="button"
									onclick={() => (workbench.rightGridSubTab = 'data')}
									class="flex h-5.5 items-center gap-1 rounded-xs px-2 font-medium transition-colors {workbench.rightGridSubTab === 'data'
										? 'bg-[var(--bg-surface)] font-semibold text-[var(--text-primary)] shadow-2xs'
										: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
								>
									<span>📊</span>
									<span class="truncate max-w-[200px]">{rightGridConfig.title}</span>
									<span class="mono-num ml-0.5 text-[9.5px] text-[var(--text-muted)]">
										({rightGridConfig.rows.length})
									</span>
								</button>
								<button
									type="button"
									onclick={() => (workbench.rightGridSubTab = 'params')}
									class="flex h-5.5 items-center gap-1 rounded-xs px-2 font-medium transition-colors {workbench.rightGridSubTab === 'params'
										? 'bg-[var(--bg-surface)] font-semibold text-[var(--text-primary)] shadow-2xs'
										: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
								>
									<span>⚙</span>
									<span>Parameters & Fits</span>
									{#if totalSavedParamsCount > 0}
										<span class="ml-1 rounded-2xs bg-emerald-500/15 px-1 py-0.2 text-[9px] font-semibold text-emerald-700">
											{totalSavedParamsCount}
										</span>
									{/if}
								</button>
							</div>
						</div>
						<div class="min-h-0 flex-1">
							{#if workbench.rightGridSubTab === 'data'}
								<CompactDataGrid
									columns={rightGridConfig.columns}
									rows={rightGridConfig.rows}
									onrowclick={handleRightGridRowClick}
								/>
							{:else}
								<ParametersManagementPanel />
							{/if}
						</div>
					</div>
				{:else}
					<PaneGroup
						direction="vertical"
						autoSaveId="quick-pp-right-split"
						class="h-full w-full"
					>
						<Pane defaultSize={58} minSize={25} class="w-full overflow-hidden">
							{#if isFptMode}
								<FptPlotCanvas initialMode="excess" />
							{:else}
								<CrossplotCanvas />
							{/if}
						</Pane>

						<PaneResizer
							class="h-[3px] cursor-row-resize bg-[var(--border-hairline)] transition-colors hover:bg-[var(--accent-primary)] active:bg-[var(--accent-primary)]"
						/>

						<Pane defaultSize={42} minSize={20} class="w-full overflow-hidden">
							<div class="flex h-full flex-col bg-[var(--bg-surface)]">
								<div
									class="flex h-7 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-[11px]"
								>
									<div class="flex items-center gap-1">
										<button
											type="button"
											onclick={() => (workbench.rightGridSubTab = 'data')}
											class="flex h-5.5 items-center gap-1 rounded-xs px-2 font-medium transition-colors {workbench.rightGridSubTab === 'data'
												? 'bg-[var(--bg-surface)] font-semibold text-[var(--text-primary)] shadow-2xs'
												: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
										>
											<span>📊</span>
											<span class="truncate max-w-[200px]">{rightGridConfig.title}</span>
											<span class="mono-num ml-0.5 text-[9.5px] text-[var(--text-muted)]">
												({rightGridConfig.rows.length})
											</span>
										</button>
										<button
											type="button"
											onclick={() => (workbench.rightGridSubTab = 'params')}
											class="flex h-5.5 items-center gap-1 rounded-xs px-2 font-medium transition-colors {workbench.rightGridSubTab === 'params'
												? 'bg-[var(--bg-surface)] font-semibold text-[var(--text-primary)] shadow-2xs'
												: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
										>
											<span>⚙</span>
											<span>Parameters & Fits</span>
											{#if totalSavedParamsCount > 0}
												<span class="ml-1 rounded-2xs bg-emerald-500/15 px-1 py-0.2 text-[9px] font-semibold text-emerald-700">
													{totalSavedParamsCount}
												</span>
											{/if}
										</button>
									</div>
								</div>
								<div class="min-h-0 flex-1">
									{#if workbench.rightGridSubTab === 'data'}
										<CompactDataGrid
											columns={rightGridConfig.columns}
											rows={rightGridConfig.rows}
											onrowclick={handleRightGridRowClick}
										/>
									{:else}
										<ParametersManagementPanel />
									{/if}
								</div>
							</div>
						</Pane>
					</PaneGroup>
				{/if}
			</Pane>
		{/if}
	</PaneGroup>
</div>
