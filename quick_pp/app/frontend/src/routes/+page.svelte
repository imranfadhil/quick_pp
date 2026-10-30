<script lang="ts">
	import { onMount } from 'svelte';
	import { replaceState } from '$app/navigation';
	import { browser } from '$app/environment';
	import { PaneGroup, Pane, PaneResizer } from 'paneforge';

	import LogTrackCanvas from '$lib/components/workbench/LogTrackCanvas.svelte';
	import StickPlotCanvas from '$lib/components/workbench/StickPlotCanvas.svelte';
	import CrossplotCanvas from '$lib/components/workbench/CrossplotCanvas.svelte';
	import CompactDataGrid, {
		type GridColumn
	} from '$lib/components/workbench/CompactDataGrid.svelte';
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

	import {
		workbench,
		type MultiWellStep,
		type SingleWellStep
	} from '$lib/state/workbench.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';

	let initialized = $state(false);
	let isReportMode = $derived(
		(workbench.scope === 'single-well' && workbench.singleWellStep === 'report') ||
			(workbench.scope === 'multi-well' && workbench.multiWellStep === 'report')
	);

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

	// Dynamic columns & rows for the Bottom-Right Quick Zonal / Context Data Grid
	let rightGridConfig = $derived.by(
		(): { title: string; columns: GridColumn[]; rows: Record<string, any>[] } => {
			if (
				workbench.scope === 'single-well' &&
				workbench.singleWellStep === 'scal' &&
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
				workbench.multiWellStep === 'swirr-corr' &&
				projectData.swirrResult?.plugs?.length
			) {
				const plugs = projectData.swirrResult.plugs;
				return {
					title: `Multi-Well SCAL Plug Swirr Summary (Ref Pc=${projectData.swirrConfig.referencePcPsi} psi | n=${plugs.filter((p) => p.included).length}/${plugs.length} included)`,
					columns: [
						{ key: 'WELL', label: 'Well' },
						{ key: 'SAMPLE', label: 'Sample' },
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
					rows: plugs.map((p) => ({
						WELL: p.well,
						SAMPLE: p.sample,
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
						STATUS: p.included ? 'Included' : !p.valid_pc ? 'Low Pc (< Ref)' : 'Excluded'
					}))
				};
			}

			if (
				workbench.scope === 'multi-well' &&
				workbench.multiWellStep === 'shf' &&
				projectData.shfResult?.scal_data?.pc?.length
			) {
				const scal = projectData.shfResult.scal_data;
				return {
					title: 'Multi-Well SCAL Capillary Pressure & Leverett-J Table',
					columns: [
						{ key: 'Well', label: 'Well' },
						{ key: 'Sample', label: 'Sample' },
						{ key: 'ROCK_FLAG', label: 'RT', numeric: true, decimals: 0 },
						{ key: 'PC', label: 'Pc', unit: 'psi', numeric: true, decimals: 2 },
						{ key: 'SW', label: 'Sw', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'SWN', label: 'Swn', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'J', label: 'J', numeric: true, decimals: 3 },
						{ key: 'H', label: 'Height', unit: 'm', numeric: true, decimals: 1 },
						{ key: 'PHIT', label: 'CPORE', unit: 'v/v', numeric: true, decimals: 3 },
						{ key: 'PERM', label: 'CPERM', unit: 'mD', numeric: true, decimals: 2 }
					],
					rows: scal.pc.map((pcVal, i) => ({
						Well: scal.well_names?.[i] ?? '',
						Sample: scal.samples?.[i] ?? '',
						ROCK_FLAG: scal.rock_flags?.[i] ?? null,
						PC: pcVal,
						SW: scal.sw?.[i] ?? null,
						SWN: scal.swn?.[i] ?? null,
						J: scal.j?.[i] ?? null,
						H: scal.h?.[i] ?? null,
						PHIT: scal.phit?.[i] ?? null,
						PERM: scal.perm?.[i] ?? null
					}))
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
				workbench.crossplotTemplateOverride === 'vsh-qc' &&
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
				(workbench.crossplotTemplateOverride === 'core-qc' ||
					(workbench.scope === 'single-well' &&
						(workbench.singleWellStep === 'litho-poro' || workbench.singleWellStep === 'perm')))
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
				const phie = wellData.curves['PHIE'] ?? wellData.curves['PHIT'] ?? [];
				const vcl = wellData.curves['VCLAY'] ?? wellData.curves['GR'] ?? [];
				const sw = wellData.curves['SWT'] ?? wellData.curves['SWE'] ?? [];

				const zoneRows = sorted.map((t, idx) => {
					const topD = t.depth;
					const botD =
						idx + 1 < sorted.length ? sorted[idx + 1].depth : wellData.depthRange.max;
					const gross = Math.max(0, botD - topD);

					let phiSum = 0;
					let phiN = 0;
					let vclSum = 0;
					let vclN = 0;
					let swSum = 0;
					let swN = 0;

					dept.forEach((d, i) => {
						if (d >= topD && d < botD) {
							if (Number.isFinite(phie[i])) {
								phiSum += phie[i];
								phiN++;
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
						PHIE: phiN > 0 ? phiSum / phiN : null,
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
						{ key: 'PHIE', label: 'PHIE', unit: 'v/v', numeric: true, decimals: 3 },
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
				{#if workbench.scope === 'multi-well' && workbench.centerPaneMultiWellView === 'stick-plot'}
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
				{#if workbench.rightPaneTab === 'crossplot'}
					<CrossplotCanvas />
				{:else if workbench.rightPaneTab === 'grid'}
					<div class="flex h-full flex-col bg-[var(--bg-surface)]">
						<div
							class="flex h-7 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-[11px] font-semibold text-[var(--text-primary)]"
						>
							<span>{rightGridConfig.title}</span>
						</div>
						<div class="min-h-0 flex-1">
							<CompactDataGrid
								columns={rightGridConfig.columns}
								rows={rightGridConfig.rows}
							/>
						</div>
					</div>
				{:else}
					<PaneGroup
						direction="vertical"
						autoSaveId="quick-pp-right-split"
						class="h-full w-full"
					>
						<Pane defaultSize={58} minSize={25} class="w-full overflow-hidden">
							<CrossplotCanvas />
						</Pane>

						<PaneResizer
							class="h-[3px] cursor-row-resize bg-[var(--border-hairline)] transition-colors hover:bg-[var(--accent-primary)] active:bg-[var(--accent-primary)]"
						/>

						<Pane defaultSize={42} minSize={20} class="w-full overflow-hidden">
							<div class="flex h-full flex-col bg-[var(--bg-surface)]">
								<div
									class="flex h-7 shrink-0 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-[11px] font-semibold text-[var(--text-primary)]"
								>
									<span>{rightGridConfig.title}</span>
									<span
										class="mono-num text-[10px] font-normal text-[var(--text-muted)]"
									>
										{rightGridConfig.rows.length} rows
									</span>
								</div>
								<div class="min-h-0 flex-1">
									<CompactDataGrid
										columns={rightGridConfig.columns}
										rows={rightGridConfig.rows}
									/>
								</div>
							</div>
						</Pane>
					</PaneGroup>
				{/if}
			</Pane>
		{/if}
	</PaneGroup>
</div>
