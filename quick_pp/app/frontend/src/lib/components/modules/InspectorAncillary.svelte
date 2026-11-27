<script lang="ts">
	import { untrack } from 'svelte';
	import { calculateSurveyTvd, saveAncillaryItems } from '$lib/api/client';
	import CompactDataGrid, {
		type GridColumn
	} from '$lib/components/workbench/CompactDataGrid.svelte';
	import CoreDepthShiftPanel from '$lib/components/workbench/CoreDepthShiftPanel.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { depthDisplay } from '$lib/state/depthDisplay.svelte';
	import { depthFromMeters, depthToMeters } from '$lib/utils/depthUnits';
	import { salinityFromRmf } from '$lib/utils/fluidUnits';
	import { fmtNum } from '$lib/utils/formatters';
	import { computeMinimumCurvature, interpolateTrajectoryOnGrid } from '$lib/utils/trajectory';

	type AncillaryTab =
		| 'hole_sections'
		| 'tops'
		| 'contacts'
		| 'perfs'
		| 'core'
		| 'scal'
		| 'pressure'
		| 'survey';
	let { initialTab }: { initialTab?: AncillaryTab } = $props();
	// svelte-ignore state_referenced_locally
	if (initialTab) workbench.ancillarySubTab = initialTab;
	let activeTab = $derived(workbench.ancillarySubTab);
	function setActiveTab(tab: AncillaryTab) {
		workbench.ancillarySubTab = tab;
	}
	let isSaving = $state(false);
	let statusMsg = $state<string | null>(null);

	// Well Header & Metadata Editor state
	let showWellHeader = $state(true);
	let editingWellHeader = $state(false);
	let savingWellHeader = $state(false);
	let headerDraft = $state<Record<string, any>>({});

	const derivedFiltrateSalinity = $derived.by(() => {
		const r = Number(headerDraft.rmf);
		if (!Number.isFinite(r) || r <= 0) return null;
		const t = Number(headerDraft.rmf_temp);
		return salinityFromRmf(r, Number.isFinite(t) ? t : 20);
	});

	const filtrateQcWarning = $derived.by(() => {
		const r = Number(headerDraft.rmf);
		const t = Number(headerDraft.rmf_temp);
		const s = Number(headerDraft.mud_filtrate_salinity);

		if (Number.isFinite(r)) {
			if (r <= 0) return 'Rmf must be greater than 0.';
			if (r > 15)
				return `Rmf (${r} Ωm) is unusually high for mud filtrate (typically 0.02 – 5 Ωm).`;
		}
		if (Number.isFinite(t)) {
			if (t < 0 || t > 70)
				return `Rmf measurement temperature (${t}°C) is outside typical surface pit range (10 – 40°C).`;
		}
		if (Number.isFinite(s) && s > 350000) {
			return `Mud filtrate salinity (${s.toLocaleString()} ppm) exceeds NaCl saturation limit (~350,000 ppm).`;
		}
		if (
			Number.isFinite(r) &&
			r > 0 &&
			Number.isFinite(s) &&
			s > 0 &&
			derivedFiltrateSalinity != null
		) {
			const ratio = s / derivedFiltrateSalinity;
			if (ratio > 1.4 || ratio < 0.6) {
				const tempStr = Number.isFinite(t) ? `${t}°C` : '20°C';
				return `Input salinity (${s.toLocaleString()} ppm) differs from Rmf-derived NaCl equivalent (~${derivedFiltrateSalinity.toLocaleString()} ppm @ ${tempStr}). Verify if mud contains non-NaCl salts or check units.`;
			}
		}
		return null;
	});

	function handleRmfInput() {
		const derived = derivedFiltrateSalinity;
		if (
			derived != null &&
			(headerDraft.mud_filtrate_salinity == null || headerDraft.mud_filtrate_salinity === '')
		) {
			headerDraft.mud_filtrate_salinity = derived;
		}
	}

	let lastSyncedWellGen = -1;
	$effect(() => {
		const gen = wellData.wellLoadGeneration;
		const hdr = wellData.header || {};
		if (gen !== lastSyncedWellGen && Object.keys(hdr).length > 0) {
			lastSyncedWellGen = gen;
			editingWellHeader = false;
		}
	});

	function openHeaderEditor() {
		const h = wellData.header || {};
		headerDraft = {
			uwi: h.uwi ?? workbench.wellId ?? '',
			company: h.company ?? '',
			field: h.field ?? '',
			country: h.country ?? '',
			location: h.location ?? '',
			province: h.province ?? '',
			service_company: h.service_company ?? '',
			log_date: h.log_date ?? '',
			spud_date: h.spud_date ?? '',
			latitude: h.latitude ?? null,
			longitude: h.longitude ?? null,
			x_coord: h.x_coord ?? null,
			y_coord: h.y_coord ?? null,
			crs: h.crs ?? '',
			datum_unit: h.datum_unit || 'm',
			kb_elev: h.kb_elev ?? null,
			df_elev: h.df_elev ?? null,
			gl_elev: h.gl_elev ?? null,
			water_depth: h.water_depth ?? null,
			bht: h.bht ?? null,
			bht_unit: h.bht_unit || 'degC',
			bit_size: h.bit_size ?? null,
			rmf: h.rmf ?? null,
			rmf_temp: h.rmf_temp ?? h.surface_temp ?? null,
			mud_density: h.mud_density ?? null,
			mud_filtrate_salinity: h.mud_filtrate_salinity ?? null,
			mud_filtrate_density: h.mud_filtrate_density ?? null,
			mud_type: h.mud_type ?? '',
			obm_base_density: h.obm_base_density ?? null
		};
		showWellHeader = true;
		editingWellHeader = true;
	}

	async function saveHeaderDraft() {
		savingWellHeader = true;
		try {
			await wellData.saveWellHeader(headerDraft, true);
			editingWellHeader = false;
		} catch (e: any) {
			wellData.lastError = e?.message || 'Failed to save well header';
		} finally {
			savingWellHeader = false;
		}
	}

	// New item quick-entry drafts (expressed directly in depthDisplay.unit)
	let newName = $state('');
	let newDepth = $state<number | null>(null);
	let newPerfTop = $state<number | null>(null);
	let newPerfBottom = $state<number | null>(null);
	let newPerfDesc = $state('');
	let newHoleTop = $state<number | null>(null);
	let newHoleBottom = $state<number | null>(null);
	let newHoleBitSize = $state<number | null>(8.5);
	let newHoleMudType = $state<'WBM' | 'OBM' | 'SBM'>('WBM');
	let newHoleMudWeight = $state<number | null>(9.5);
	let newHoleSalinity = $state<number | null>(30000);
	let newHoleRmf = $state<number | null>(null);
	let newHoleRmfTemp = $state<number | null>(null);
	let newVal1 = $state<number | null>(0.2);
	let newVal2 = $state<number | null>(100);
	let newScalPc = $state<number | null>(15);
	let newScalSw = $state<number | null>(0.35);
	let newScalCpore = $state<number | null>(null);
	let newScalCperm = $state<number | null>(null);
	let newScalExpType = $state<string>('porous_plate');
	// Formation pressure test quick-entry drafts (optional fields may be left blank)
	let newPressMob = $state<number | null>(null);
	let newPressPerm = $state<number | null>(null);
	let newPressTemp = $state<number | null>(null);
	let newPressRemarks = $state('');

	// Keep quick-entry drafts aligned if user switches display unit (e.g. m <-> ft)
	let previousDepthUnit = depthDisplay.unit;
	$effect(() => {
		const unit = depthDisplay.unit;
		untrack(() => {
			const oldUnit = previousDepthUnit;
			previousDepthUnit = unit;
			if (oldUnit === unit) return;
			if (newDepth !== null && Number.isFinite(newDepth)) {
				newDepth = Number(
					depthFromMeters(depthToMeters(newDepth, oldUnit), unit).toFixed(2)
				);
			}
			if (newPerfTop !== null && Number.isFinite(newPerfTop)) {
				newPerfTop = Number(
					depthFromMeters(depthToMeters(newPerfTop, oldUnit), unit).toFixed(2)
				);
			}
			if (newPerfBottom !== null && Number.isFinite(newPerfBottom)) {
				newPerfBottom = Number(
					depthFromMeters(depthToMeters(newPerfBottom, oldUnit), unit).toFixed(2)
				);
			}
			if (newHoleTop !== null && Number.isFinite(newHoleTop)) {
				newHoleTop = Number(
					depthFromMeters(depthToMeters(newHoleTop, oldUnit), unit).toFixed(2)
				);
			}
			if (newHoleBottom !== null && Number.isFinite(newHoleBottom)) {
				newHoleBottom = Number(
					depthFromMeters(depthToMeters(newHoleBottom, oldUnit), unit).toFixed(2)
				);
			}
		});
	});

	const HOLE_SECTION_COLS: GridColumn[] = [
		{ key: 'name', label: 'Section', editable: true },
		{ key: 'top', label: 'Top', unit: 'm', numeric: true, decimals: 1, editable: true },
		{ key: 'bottom', label: 'Bottom', unit: 'm', numeric: true, decimals: 1, editable: true },
		{ key: 'bit_size', label: 'Bit', unit: 'in', numeric: true, decimals: 3, editable: true },
		{ key: 'mud_type', label: 'Mud', editable: true },
		{ key: 'mud_weight', label: 'MW', unit: 'ppg', numeric: true, decimals: 2, editable: true },
		{
			key: 'mud_salinity',
			label: 'Salinity',
			unit: 'ppm',
			numeric: true,
			decimals: 0,
			editable: true
		},
		{ key: 'rmf', label: 'Rmf', unit: 'Ωm', numeric: true, decimals: 3, editable: true },
		{ key: 'rmf_temp', label: 'Rmf T', unit: '°C', numeric: true, decimals: 1, editable: true }
	];

	const TOP_COLS: GridColumn[] = [
		{ key: 'name', label: 'Formation / Zone', editable: true },
		{ key: 'depth', label: 'Depth', unit: 'm', numeric: true, decimals: 1, editable: true }
	];

	const CONTACT_COLS: GridColumn[] = [
		{ key: 'type', label: 'Contact Type', editable: true },
		{ key: 'depth', label: 'Depth', unit: 'm', numeric: true, decimals: 1, editable: true }
	];

	const PERF_COLS: GridColumn[] = [
		{ key: 'name', label: 'Name', editable: true },
		{ key: 'top', label: 'Top', unit: 'm', numeric: true, decimals: 1, editable: true },
		{ key: 'bottom', label: 'Bottom', unit: 'm', numeric: true, decimals: 1, editable: true },
		{ key: 'description', label: 'Description', editable: true }
	];

	const CORE_COLS: GridColumn[] = [
		{ key: 'sample_name', label: 'Sample', editable: true },
		{ key: 'depth', label: 'Depth', unit: 'm', numeric: true, decimals: 2, editable: true },
		{
			key: 'porosity',
			label: 'CPORE',
			unit: 'v/v',
			numeric: true,
			decimals: 3,
			editable: true
		},
		{
			key: 'permeability',
			label: 'CPERM',
			unit: 'mD',
			numeric: true,
			decimals: 2,
			editable: true
		},
		{
			key: 'grain_density',
			label: 'CGD',
			unit: 'g/cc',
			numeric: true,
			decimals: 3,
			editable: true
		},
		{ key: 'qv', label: 'QV', unit: 'meq/mL', numeric: true, decimals: 3, editable: true },
		{ key: 'rock_type', label: 'Rock Type', editable: false }
	];

	const SCAL_PC_COLS: GridColumn[] = [
		{ key: 'sample_name', label: 'Sample', editable: true },
		{ key: 'depth', label: 'Depth', unit: 'm', numeric: true, decimals: 2, editable: true },
		{ key: 'experiment_type', label: 'Method', editable: true, width: '100px' },
		{
			key: 'porosity',
			label: 'CPORE',
			unit: 'v/v',
			numeric: true,
			decimals: 3,
			editable: true
		},
		{
			key: 'permeability',
			label: 'CPERM',
			unit: 'mD',
			numeric: true,
			decimals: 2,
			editable: true
		},
		{ key: 'rock_type', label: 'Rock Type', editable: false },
		{ key: 'pc', label: 'Pc', unit: 'psi', numeric: true, decimals: 2, editable: true },
		{ key: 'sw', label: 'Sw', unit: 'v/v', numeric: true, decimals: 3, editable: true }
	];

	// Reactively annotate rock_type whenever DRT cutoffs, method, or core/scal data change
	$effect(() => {
		const _m = projectData.drtMethod;
		const _c = projectData.activeDrtCutoffs;
		if (wellData.coreSamples && wellData.coreSamples.length > 0) {
			for (const r of wellData.coreSamples) {
				const rt = projectData.classifyRockType(r.porosity, r.permeability);
				if (r.rock_type !== rt) r.rock_type = rt;
			}
		}
		if (wellData.scalPc && wellData.scalPc.length > 0) {
			for (const r of wellData.scalPc) {
				const rt = projectData.classifyRockType(r.porosity, r.permeability);
				if (r.rock_type !== rt) r.rock_type = rt;
			}
		}
	});

	const PRESSURE_COLS: GridColumn[] = [
		{ key: 'depth', label: 'Depth', unit: 'm', numeric: true, decimals: 1, editable: true },
		{
			key: 'pressure',
			label: 'Form. Press.',
			unit: 'psi',
			numeric: true,
			decimals: 1,
			editable: true
		},
		{
			key: 'mobility',
			label: 'Mobility',
			unit: 'mD/cP',
			numeric: true,
			decimals: 2,
			editable: true
		},
		{
			key: 'permeability',
			label: 'Perm',
			unit: 'mD',
			numeric: true,
			decimals: 2,
			editable: true
		},
		{
			key: 'temperature',
			label: 'Temp',
			unit: 'degC',
			numeric: true,
			decimals: 1,
			editable: true
		},
		{ key: 'remarks', label: 'Remarks', editable: true },
		{ key: 'fluid_type', label: 'Fluid', editable: true },
		{ key: 'test_quality', label: 'QC', editable: true }
	];

	/** Parse an optional numeric input (blank / invalid -> null). */
	function optNum(v: unknown): number | null {
		if (v === null || v === undefined || (typeof v === 'string' && v.trim() === ''))
			return null;
		const n = Number(v);
		return Number.isFinite(n) ? n : null;
	}

	/** Normalize a formation pressure test row to the backend payload shape. */
	function normalizePressureRow(r: Record<string, any>): Record<string, any> {
		return {
			depth: Number(r.depth),
			pressure: Number(r.pressure),
			pressure_uom: r.pressure_uom || 'psi',
			mobility: optNum(r.mobility),
			permeability: optNum(r.permeability),
			temperature: optNum(r.temperature),
			temperature_uom: r.temperature_uom || 'degC',
			remarks: String(r.remarks ?? r.remark ?? '').trim(),
			// Preserve FPT interpretation (set in the multi-well Pressure & Contacts module)
			tvdss: optNum(r.tvdss),
			fluid_type: String(r.fluid_type || 'UNASSIGNED')
				.trim()
				.toUpperCase(),
			test_quality: String(r.test_quality || 'VALID')
				.trim()
				.toUpperCase()
				.replace(/[\s-]+/g, '_'),
			zone: r.zone ?? ''
		};
	}

	let surveyColumns = $derived<GridColumn[]>([
		{ key: 'md', label: 'MD', unit: 'm', numeric: true, decimals: 1, editable: true },
		{ key: 'inc', label: 'Inc', unit: 'deg', numeric: true, decimals: 2, editable: true },
		{ key: 'azim', label: 'Azim', unit: 'deg', numeric: true, decimals: 2, editable: true },
		{
			key: 'tvdss',
			label: wellData.datumElevationInfo.hasElevation ? 'TVDSS' : 'TVDDF',
			unit: 'm',
			numeric: true,
			decimals: 1,
			editable: false
		}
	]);

	let isRecalculating = $state(false);

	function recomputeSurveyTrajectoryLocally() {
		if (!wellData.survey || wellData.survey.length === 0) return;
		const elevMeters = wellData.datumElevationInfo.hasElevation
			? wellData.datumElevationInfo.elevationMeters
			: 0;
		wellData.survey = computeMinimumCurvature(wellData.survey, elevMeters);
	}

	async function recalculateSurvey() {
		if (!workbench.projectId || !workbench.wellId) return;
		isRecalculating = true;
		statusMsg = null;
		try {
			recomputeSurveyTrajectoryLocally();

			// Interpolate continuous TVD / TVDSS curves if depth grid is present
			const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'];
			const elevMeters = wellData.datumElevationInfo.hasElevation
				? wellData.datumElevationInfo.elevationMeters
				: 0;
			if (dept && dept.length > 0) {
				const trajCurves = interpolateTrajectoryOnGrid(dept, wellData.survey, elevMeters);
				if (trajCurves) {
					wellData.curves = {
						...wellData.curves,
						TVD: trajCurves.tvd,
						TVDSS: trajCurves.tvdss
					};
				}
			}

			// Persist surveys to backend
			await saveAncillaryItems(
				workbench.projectId,
				workbench.wellId,
				'well_surveys',
				'surveys',
				wellData.survey
			);

			// Call backend calculate_tvd endpoint if available
			try {
				await calculateSurveyTvd(workbench.projectId, workbench.wellId);
			} catch (err) {
				console.warn('Backend calculate_tvd notice:', err);
			}

			statusMsg = wellData.datumElevationInfo.hasElevation
				? `Survey & TVDSS recalculated (${wellData.datumElevationInfo.label}) ✓`
				: 'Survey & TVDDF recalculated (No DF/KB Datum) ✓';
		} catch (e: any) {
			statusMsg = `Recalculation error: ${e?.message || 'Failed'}`;
		} finally {
			isRecalculating = false;
		}
	}

	// Auto-compute TVDSS/TVDDF locally if rows are loaded or missing computed values
	$effect(() => {
		const s = wellData.survey;
		const _hasElev = wellData.datumElevationInfo.hasElevation;
		if (s && s.length > 0) {
			const needsCalc = s.some((r) => r.tvdss === undefined || r.tvdss === null);
			if (needsCalc) {
				recomputeSurveyTrajectoryLocally();
			}
		}
	});

	async function handleSurveyCellChange(_rowIndex: number, _key: string, _newValue: any) {
		recomputeSurveyTrajectoryLocally();
		await saveCurrentTab();
	}

	async function addQuickRow() {
		if (!workbench.projectId || !workbench.wellId) return;
		if (activeTab === 'hole_sections') {
			if (
				newHoleTop === null ||
				!Number.isFinite(newHoleTop) ||
				newHoleBottom === null ||
				!Number.isFinite(newHoleBottom)
			) {
				statusMsg = 'Enter valid top and bottom depths before adding a hole section.';
				return;
			}
			const t = depthToMeters(Number(newHoleTop), depthDisplay.unit);
			const b = depthToMeters(Number(newHoleBottom), depthDisplay.unit);
			if (Math.abs(b - t) < 1e-6) {
				statusMsg = 'Hole section bottom depth must differ from top depth.';
				return;
			}
			const bsVal = optNum(newHoleBitSize);
			wellData.holeSections = [
				...wellData.holeSections,
				{
					id: `hs-${Date.now()}`,
					name:
						newName.trim() ||
						(bsVal != null ? `${bsVal}" Section` : `Section-${wellData.holeSections.length + 1}`),
					top: Math.min(t, b),
					bottom: Math.max(t, b),
					bit_size: bsVal,
					mud_type: (newHoleMudType || 'WBM').toUpperCase(),
					mud_weight: optNum(newHoleMudWeight),
					mud_salinity: optNum(newHoleSalinity),
					rmf: optNum(newHoleRmf),
					rmf_temp: optNum(newHoleRmfTemp)
				}
			].sort((a, b2) => a.top - b2.top);
			newName = '';
			newHoleTop = newHoleBottom;
			newHoleBottom = null;
			await saveCurrentTab();
			return;
		}
		if (activeTab === 'perfs') {
			if (
				newPerfTop === null ||
				!Number.isFinite(newPerfTop) ||
				newPerfBottom === null ||
				!Number.isFinite(newPerfBottom)
			) {
				statusMsg = 'Enter valid top and bottom depths before adding a perforation.';
				return;
			}
		} else if (newDepth === null || !Number.isFinite(newDepth)) {
			statusMsg = 'Enter a valid depth before adding a row.';
			return;
		}

		const depthM = newDepth !== null ? depthToMeters(Number(newDepth), depthDisplay.unit) : 0;

		if (activeTab === 'tops') {
			const name = newName.trim() || `Zone-${wellData.tops.length + 1}`;
			wellData.tops = [
				...wellData.tops,
				{ id: `top-${Date.now()}`, name, depth: depthM }
			].sort((a, b) => a.depth - b.depth);
			newName = '';
			newDepth = null;
		} else if (activeTab === 'contacts') {
			const type = (newName.trim() || 'OWC').toUpperCase();
			wellData.contacts = [
				...wellData.contacts,
				{ id: `contact-${Date.now()}`, type, name: type, depth: depthM }
			];
			newName = '';
			newDepth = null;
		} else if (activeTab === 'perfs') {
			const t = depthToMeters(Number(newPerfTop), depthDisplay.unit);
			const b = depthToMeters(Number(newPerfBottom), depthDisplay.unit);
			wellData.perforations = [
				...wellData.perforations,
				{
					id: `perf-${Date.now()}`,
					name: newName.trim() || `PERF-${wellData.perforations.length + 1}`,
					top: Math.min(t, b),
					bottom: Math.max(t, b),
					description: newPerfDesc.trim()
				}
			].sort((a, b2) => a.top - b2.top);
			newName = '';
			newPerfTop = null;
			newPerfBottom = null;
			newPerfDesc = '';
		} else if (activeTab === 'core') {
			let por = newVal1 !== null ? Number(newVal1) : null;
			if (por !== null && Number.isFinite(por) && por > 1.0) por /= 100.0;
			const perm =
				newVal2 !== null && Number.isFinite(Number(newVal2)) ? Number(newVal2) : null;
			wellData.coreSamples = [
				...wellData.coreSamples,
				{
					sample_name: newName.trim() || `S-${wellData.coreSamples.length + 1}`,
					depth: depthM,
					porosity: Number.isFinite(por) ? por : null,
					permeability: Number.isFinite(perm) ? perm : null,
					rock_type: projectData.classifyRockType(por, perm)
				}
			];
			newName = '';
			newDepth = null;
		} else if (activeTab === 'scal') {
			// Keep sample_name, depth, cpore, cperm sticky so user can enter multiple (Pc, Sw) pairs for the same plug
			const lastScal =
				wellData.scalPc.length > 0 ? wellData.scalPc[wellData.scalPc.length - 1] : null;
			const sampleName = newName.trim() || (lastScal ? String(lastScal.sample_name) : 'S-1');
			newName = sampleName;
			let effPor: number | null = null;
			if (newScalCpore !== null && Number.isFinite(Number(newScalCpore))) {
				effPor = Number(newScalCpore);
			} else if (lastScal?.porosity !== null && lastScal?.porosity !== undefined) {
				effPor = Number(lastScal.porosity);
			}
			if (effPor !== null && Number.isFinite(effPor) && effPor > 1.0) {
				effPor /= 100.0;
			}
			let effPerm: number | null = null;
			if (newScalCperm !== null && Number.isFinite(Number(newScalCperm))) {
				effPerm = Number(newScalCperm);
			} else if (lastScal?.permeability !== null && lastScal?.permeability !== undefined) {
				effPerm = Number(lastScal.permeability);
			}
			const rt = projectData.classifyRockType(effPor, effPerm);
			wellData.scalPc = [
				...wellData.scalPc,
				{
					sample_name: sampleName,
					depth: depthM,
					pc: Number(newScalPc ?? 0),
					sw: Number(newScalSw ?? 0),
					porosity: effPor,
					permeability: effPerm,
					rock_type: rt,
					experiment_type: (newScalExpType || lastScal?.experiment_type || 'porous_plate')
						.trim()
						.toLowerCase(),
					cycle: 'drainage'
				}
			];
		} else if (activeTab === 'pressure') {
			wellData.pressureTests = [
				...wellData.pressureTests,
				normalizePressureRow({
					depth: depthM,
					pressure: Number(newVal2 ?? 0),
					mobility: newPressMob,
					permeability: newPressPerm,
					temperature: newPressTemp,
					remarks: newPressRemarks
				})
			].sort((a, b) => a.depth - b.depth);
			newDepth = null;
			newPressRemarks = '';
		} else if (activeTab === 'survey') {
			wellData.survey = [
				...wellData.survey,
				{ md: depthM, inc: Number(newVal1 ?? 0), azim: Number(newVal2 ?? 0) }
			].sort((a, b) => a.md - b.md);
			newDepth = null;
			recomputeSurveyTrajectoryLocally();
		}
		await saveCurrentTab();
	}

	async function handleDeleteRows(indices: number[], _deletedRows?: Record<string, any>[]) {
		if (!workbench.projectId || !workbench.wellId || !indices.length) return;
		const removeSet = new Set(indices);
		const removedCount = removeSet.size;
		try {
			if (activeTab === 'hole_sections') {
				wellData.holeSections = wellData.holeSections.filter((_, i) => !removeSet.has(i));
				wellData.syncBitSizeCurveFromHoleSections();
			} else if (activeTab === 'tops') {
				wellData.tops = wellData.tops.filter((_, i) => !removeSet.has(i));
				if (workbench.selectedZones.length > 0) {
					const remainingNames = new Set(wellData.tops.map((t) => t.name));
					workbench.selectedZones = workbench.selectedZones.filter((z) =>
						remainingNames.has(z)
					);
				}
			} else if (activeTab === 'contacts') {
				wellData.contacts = wellData.contacts.filter((_, i) => !removeSet.has(i));
			} else if (activeTab === 'perfs') {
				wellData.perforations = wellData.perforations.filter((_, i) => !removeSet.has(i));
			} else if (activeTab === 'core') {
				wellData.coreSamples = wellData.coreSamples.filter((_, i) => !removeSet.has(i));
				wellData.syncCoreCurvesFromSamples();
			} else if (activeTab === 'scal') {
				wellData.scalPc = wellData.scalPc.filter((_, i) => !removeSet.has(i));
				wellData.syncCoreCurvesFromSamples();
			} else if (activeTab === 'pressure') {
				wellData.pressureTests = wellData.pressureTests.filter((_, i) => !removeSet.has(i));
			} else if (activeTab === 'survey') {
				wellData.survey = wellData.survey.filter((_, i) => !removeSet.has(i));
			}
			await saveCurrentTab();
			statusMsg = `Deleted ${removedCount} ${removedCount === 1 ? 'row' : 'rows'} ✓`;
		} catch (e: any) {
			statusMsg = e?.message || 'Delete failed';
		}
	}

	async function handleDeleteRow(idx: number, row: Record<string, any>) {
		await handleDeleteRows([idx], [row]);
	}

	async function handlePasteRows(pastedRows: Record<string, any>[]) {
		if (!pastedRows.length) return;
		const now = Date.now();
		if (activeTab === 'hole_sections') {
			const mapped = pastedRows
				.map((r, i) => {
					const t = Number(r.top ?? r.top_md ?? r.depth);
					const b = Number(r.bottom ?? r.bottom_md ?? r.base);
					const bs = optNum(r.bit_size ?? r.bs ?? r.bit);
					const mtRaw = String(r.mud_type ?? r.mud ?? '').trim().toUpperCase();
					const mt = mtRaw.startsWith('OB')
						? 'OBM'
						: mtRaw.startsWith('SB')
							? 'SBM'
							: mtRaw.startsWith('WB')
								? 'WBM'
								: mtRaw;
					return {
						id: `hs-${now}-${i}`,
						name: String(
							r.name ||
								r.section_name ||
								(bs != null ? `${bs}" Section` : `Section-${wellData.holeSections.length + i + 1}`)
						).trim(),
						top: Math.min(t, b),
						bottom: Math.max(t, b),
						bit_size: bs,
						mud_type: mt,
						mud_weight: optNum(r.mud_weight ?? r.mw ?? r.mud_density),
						mud_salinity: optNum(r.mud_salinity ?? r.salinity),
						rmf: optNum(r.rmf),
						rmf_temp: optNum(r.rmf_temp ?? r.rmf_temp_c)
					};
				})
				.filter((s) => Number.isFinite(s.top) && Number.isFinite(s.bottom) && s.bottom > s.top);
			wellData.holeSections = [...wellData.holeSections, ...mapped].sort(
				(a, b) => a.top - b.top
			);
		} else if (activeTab === 'tops') {
			const mapped = pastedRows
				.map((r, i) => ({
					id: `top-${now}-${i}`,
					name: String(r.name || `Zone-${wellData.tops.length + i + 1}`).trim(),
					depth: Number(r.depth ?? 0)
				}))
				.filter((t) => Number.isFinite(t.depth));
			wellData.tops = [...wellData.tops, ...mapped].sort((a, b) => a.depth - b.depth);
		} else if (activeTab === 'contacts') {
			const mapped = pastedRows
				.map((r, i) => {
					const t = String(r.type || r.name || 'OWC')
						.trim()
						.toUpperCase();
					return {
						id: `contact-${now}-${i}`,
						type: t,
						name: t,
						depth: Number(r.depth ?? 0)
					};
				})
				.filter((c) => Number.isFinite(c.depth));
			wellData.contacts = [...wellData.contacts, ...mapped];
		} else if (activeTab === 'perfs') {
			const mapped = pastedRows
				.map((r, i) => {
					const t = Number(r.top ?? r.depth);
					const b = Number(r.bottom ?? r.top ?? r.depth);
					return {
						id: `perf-${now}-${i}`,
						name: String(
							r.name || `PERF-${wellData.perforations.length + i + 1}`
						).trim(),
						top: Math.min(t, b),
						bottom: Math.max(t, b),
						description: String(r.description ?? '').trim()
					};
				})
				.filter((p) => Number.isFinite(p.top) && Number.isFinite(p.bottom));
			wellData.perforations = [...wellData.perforations, ...mapped].sort(
				(a, b) => a.top - b.top
			);
		} else if (activeTab === 'core') {
			const mapped = pastedRows
				.map((r, i) => {
					let por =
						r.porosity !== null && r.porosity !== undefined ? Number(r.porosity) : null;
					if (por !== null && Number.isFinite(por) && por > 1.0) por /= 100.0;
					const perm =
						r.permeability !== null && r.permeability !== undefined
							? Number(r.permeability)
							: null;
					const gdRaw =
						r.grain_density ?? r.cgd ?? r.CGD ?? r.rhog ?? r.rhoma ?? r.RHOMA ?? null;
					const gd =
						gdRaw !== null && gdRaw !== undefined && Number.isFinite(Number(gdRaw))
							? Number(gdRaw)
							: null;
					return {
						sample_name: String(
							r.sample_name || `S-${wellData.coreSamples.length + i + 1}`
						).trim(),
						depth: Number(r.depth ?? 0),
						porosity: por,
						permeability: perm,
						grain_density: gd,
						rock_type: projectData.classifyRockType(por, perm)
					};
				})
				.filter((s) => Number.isFinite(s.depth));
			wellData.coreSamples = [...wellData.coreSamples, ...mapped];
		} else if (activeTab === 'scal') {
			const mapped = pastedRows
				.map((r, i) => {
					let por = r.porosity ?? r.cpore ?? r.CPORE ?? r.POR ?? r.por ?? null;
					if (por !== null && por !== undefined && Number.isFinite(Number(por))) {
						por = Number(por);
						if (por > 1.0) por /= 100.0;
					} else {
						por = null;
					}
					let perm = r.permeability ?? r.cperm ?? r.CPERM ?? r.PERM ?? r.perm ?? null;
					if (perm !== null && perm !== undefined && Number.isFinite(Number(perm))) {
						perm = Number(perm);
					} else {
						perm = null;
					}
					return {
						sample_name: String(
							r.sample_name || `S-${wellData.scalPc.length + i + 1}`
						).trim(),
						depth: Number(r.depth ?? 0),
						pc: Number(r.pc ?? 0),
						sw: Number(r.sw ?? 0),
						porosity: por,
						permeability: perm,
						rock_type: projectData.classifyRockType(por, perm),
						experiment_type: String(
							r.experiment_type || r.type || r.method || 'porous_plate'
						)
							.trim()
							.toLowerCase(),
						cycle: r.cycle || 'drainage'
					};
				})
				.filter((s) => Number.isFinite(s.pc) && Number.isFinite(s.sw));
			wellData.scalPc = [...wellData.scalPc, ...mapped];
		} else if (activeTab === 'pressure') {
			const mapped = pastedRows
				.map((r) => normalizePressureRow(r))
				.filter((p) => Number.isFinite(p.depth) && Number.isFinite(p.pressure));
			wellData.pressureTests = [...wellData.pressureTests, ...mapped].sort(
				(a, b) => a.depth - b.depth
			);
		} else if (activeTab === 'survey') {
			const mapped = pastedRows
				.map((r) => ({
					md: Number(r.md ?? r.depth ?? 0),
					inc: Number(r.inc ?? 0),
					azim: Number(r.azim ?? 0)
				}))
				.filter((s) => Number.isFinite(s.md));
			wellData.survey = [...wellData.survey, ...mapped].sort((a, b) => a.md - b.md);
			recomputeSurveyTrajectoryLocally();
		}
		await saveCurrentTab();
	}

	async function saveCurrentTab() {
		if (!workbench.projectId || !workbench.wellId) return;
		isSaving = true;
		statusMsg = null;
		try {
			if (activeTab === 'hole_sections') {
				wellData.holeSections = wellData.holeSections
					.filter(
						(s) => Number.isFinite(Number(s.top)) && Number.isFinite(Number(s.bottom))
					)
					.map((s, i) => {
						const t = Number(s.top);
						const b = Number(s.bottom);
						const bs = optNum(s.bit_size);
						const mtRaw = String(s.mud_type ?? '').trim().toUpperCase();
						const mt = mtRaw.startsWith('OB')
							? 'OBM'
							: mtRaw.startsWith('SB')
								? 'SBM'
								: mtRaw.startsWith('WB')
									? 'WBM'
									: mtRaw;
						return {
							id: s.id || `hs-${i}`,
							name:
								String(s.name || '').trim() ||
								(bs != null ? `${bs}" Section` : `Section-${i + 1}`),
							top: Math.min(t, b),
							bottom: Math.max(t, b),
							bit_size: bs,
							mud_type: mt,
							mud_weight: optNum(s.mud_weight),
							mud_salinity: optNum(s.mud_salinity),
							rmf: optNum(s.rmf),
							rmf_temp: optNum(s.rmf_temp)
						};
					})
					.filter((s) => s.bottom > s.top)
					.sort((a, b) => a.top - b.top);
				wellData.syncBitSizeCurveFromHoleSections();
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'hole_sections',
					'hole_sections',
					wellData.holeSections.map((s) => ({
						name: s.name,
						top: s.top,
						bottom: s.bottom,
						bit_size: s.bit_size ?? null,
						mud_type: s.mud_type || '',
						mud_weight: s.mud_weight ?? null,
						mud_salinity: s.mud_salinity ?? null,
						rmf: s.rmf ?? null,
						rmf_temp: s.rmf_temp ?? null
					}))
				);
			} else if (activeTab === 'tops') {
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'formation_tops',
					'tops',
					wellData.tops.map((t) => ({ name: t.name, depth: Number(t.depth) }))
				);
			} else if (activeTab === 'contacts') {
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'fluid_contacts',
					'contacts',
					wellData.contacts.map((c) => ({
						name: c.type || c.name,
						depth: Number(c.depth)
					}))
				);
			} else if (activeTab === 'perfs') {
				// Re-assign a normalized copy so plots (reference-tracked deps) refresh after inline cell edits
				wellData.perforations = wellData.perforations
					.filter(
						(p) => Number.isFinite(Number(p.top)) && Number.isFinite(Number(p.bottom))
					)
					.map((p) => ({
						id: p.id,
						name: String(p.name || '').trim() || 'PERF',
						top: Math.min(Number(p.top), Number(p.bottom)),
						bottom: Math.max(Number(p.top), Number(p.bottom)),
						description: p.description ?? ''
					}))
					.sort((a, b) => a.top - b.top);
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'perforations',
					'perforations',
					wellData.perforations.map((p) => ({
						name: p.name,
						top: p.top,
						bottom: p.bottom,
						description: p.description ?? ''
					}))
				);
				if (projectData.stickPlotResult) {
					void projectData.loadStickPlot(true);
				}
			} else if (activeTab === 'core') {
				for (const r of wellData.coreSamples) {
					if (
						r.porosity !== null &&
						r.porosity !== undefined &&
						Number(r.porosity) > 1.0
					) {
						r.porosity = Number(r.porosity) / 100.0;
					}
					r.rock_type = projectData.classifyRockType(r.porosity, r.permeability);
				}
				wellData.syncCoreCurvesFromSamples();
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'core_samples',
					'samples',
					wellData.coreSamples
				);
			} else if (activeTab === 'scal') {
				// Propagate shared depth, porosity, permeability, and experiment_type for same sample_name if user edited one row
				const depthBySample = new Map<string, number>();
				const cporeBySample = new Map<string, number | null>();
				const cpermBySample = new Map<string, number | null>();
				const expBySample = new Map<string, string>();
				for (const r of wellData.scalPc) {
					const sname = String(r.sample_name || '').trim();
					if (sname) {
						if (Number.isFinite(Number(r.depth)))
							depthBySample.set(sname, Number(r.depth));
						if (
							r.porosity !== null &&
							r.porosity !== undefined &&
							Number.isFinite(Number(r.porosity))
						) {
							let p = Number(r.porosity);
							if (p > 1.0) p /= 100.0;
							cporeBySample.set(sname, p);
						}
						if (
							r.permeability !== null &&
							r.permeability !== undefined &&
							Number.isFinite(Number(r.permeability))
						) {
							cpermBySample.set(sname, Number(r.permeability));
						}
						if (r.experiment_type) {
							expBySample.set(sname, String(r.experiment_type).trim().toLowerCase());
						}
					}
				}
				const normalizedScal = wellData.scalPc.map((r, i) => {
					const sname = String(r.sample_name || `S-${i + 1}`).trim();
					let por =
						r.porosity !== null &&
						r.porosity !== undefined &&
						Number.isFinite(Number(r.porosity))
							? Number(r.porosity)
							: (cporeBySample.get(sname) ?? null);
					if (por !== null && por > 1.0) por /= 100.0;
					const perm =
						r.permeability !== null &&
						r.permeability !== undefined &&
						Number.isFinite(Number(r.permeability))
							? Number(r.permeability)
							: (cpermBySample.get(sname) ?? null);
					return {
						sample_name: sname,
						depth: Number(r.depth ?? depthBySample.get(sname) ?? 0),
						pc: Number(r.pc ?? 0),
						sw: Number(r.sw ?? 0),
						porosity: por,
						permeability: perm,
						rock_type: projectData.classifyRockType(por, perm),
						experiment_type: String(
							expBySample.get(sname) || r.experiment_type || 'porous_plate'
						)
							.trim()
							.toLowerCase(),
						cycle: r.cycle || 'drainage'
					};
				});
				wellData.scalPc = normalizedScal;
				wellData.syncCoreCurvesFromSamples();
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'scal_pc',
					'pc_data',
					normalizedScal
				);
				void projectData.ensureSwirrDataLoaded(true);
			} else if (activeTab === 'pressure') {
				// Normalize (blank optional cells -> null) & drop rows missing depth/pressure
				wellData.pressureTests = wellData.pressureTests
					.map((r) => normalizePressureRow(r))
					.filter((p) => Number.isFinite(p.depth) && Number.isFinite(p.pressure))
					.sort((a, b) => a.depth - b.depth);
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'pressure_tests',
					'tests',
					wellData.pressureTests
				);
			} else if (activeTab === 'survey') {
				recomputeSurveyTrajectoryLocally();
				const dept = wellData.curves['DEPT'] ?? wellData.curves['DEPTH'];
				const elevMeters = wellData.datumElevationInfo.hasElevation
					? wellData.datumElevationInfo.elevationMeters
					: 0;
				if (dept && dept.length > 0) {
					const trajCurves = interpolateTrajectoryOnGrid(
						dept,
						wellData.survey,
						elevMeters
					);
					if (trajCurves) {
						wellData.curves = {
							...wellData.curves,
							TVD: trajCurves.tvd,
							TVDSS: trajCurves.tvdss
						};
					}
				}
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'well_surveys',
					'surveys',
					wellData.survey
				);
			}
			if (wellData.getPendingCascadingCurves('qaqc-env').length > 0) {
				await wellData.saveModuleComputed('qaqc-env', 'QAQC Curves Saved ✓');
			}
			statusMsg = 'Saved ✓';
			if (activeTab === 'scal' || activeTab === 'core') {
				projectData.shfResult = null;
			}
			void projectData.refreshSummary(workbench.projectId);
			wellData.lastSavedMessage = activeTab === 'scal' ? 'SCAL Pc Saved' : 'Ancillary Saved';
		} catch (e: any) {
			statusMsg = `Error: ${e?.message || 'Save failed'}`;
		} finally {
			isSaving = false;
		}
	}

	async function handleScalCellChange(rowIndex: number, key: string, _newValue: any) {
		const row = wellData.scalPc[rowIndex];
		if (
			row &&
			['porosity', 'permeability', 'depth', 'sample_name', 'experiment_type'].includes(key)
		) {
			const sname = String(row.sample_name || '').trim();
			if (sname) {
				const por =
					row.porosity !== null &&
					row.porosity !== undefined &&
					Number.isFinite(Number(row.porosity))
						? Number(row.porosity)
						: null;
				const normPor = por !== null ? (por > 1.0 ? por / 100.0 : por) : null;
				const perm =
					row.permeability !== null &&
					row.permeability !== undefined &&
					Number.isFinite(Number(row.permeability))
						? Number(row.permeability)
						: null;
				const depth = Number.isFinite(Number(row.depth)) ? Number(row.depth) : null;
				const expType = row.experiment_type
					? String(row.experiment_type).trim().toLowerCase()
					: null;

				for (const r of wellData.scalPc) {
					if (String(r.sample_name || '').trim() === sname) {
						if (key === 'porosity') r.porosity = normPor;
						if (key === 'permeability') r.permeability = perm;
						if (key === 'depth' && depth !== null) r.depth = depth;
						if (key === 'experiment_type' && expType) r.experiment_type = expType;
						r.rock_type = projectData.classifyRockType(r.porosity, r.permeability);
					}
				}
			}
		}
		await saveCurrentTab();
	}

	async function handleCoreCellChange(rowIndex: number, _key: string, _newValue: any) {
		const row = wellData.coreSamples[rowIndex];
		if (row) {
			if (row.porosity !== null && row.porosity !== undefined && Number(row.porosity) > 1.0) {
				row.porosity = Number(row.porosity) / 100.0;
			}
			row.rock_type = projectData.classifyRockType(row.porosity, row.permeability);
		}
		await saveCurrentTab();
	}

	let hdr = $derived(wellData.header || {});
	let dUnit = $derived(hdr.datum_unit || 'm');
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<!-- Collapsible Well Header & Metadata Panel -->
	<div class="shrink-0 border-b border-[var(--border-hairline)] bg-[var(--bg-surface)]">
		<div
			class="flex h-7 items-center justify-between bg-[var(--bg-subtle)] px-2.5 text-[11px] select-none"
		>
			<button
				type="button"
				onclick={() => (showWellHeader = !showWellHeader)}
				class="flex min-w-0 flex-1 items-center gap-1.5 text-left font-bold text-[var(--text-primary)] hover:text-[var(--accent-primary)]"
			>
				<span class="text-[9px] text-[var(--text-muted)]">{showWellHeader ? '▼' : '▶'}</span>
				<span class="truncate">Well Header &amp; Metadata</span>
				<span
					class="mono-num truncate rounded-xs bg-[var(--bg-surface)] px-1.5 py-0.2 text-[9.5px] font-medium text-[var(--text-secondary)]"
				>
					{hdr.uwi ? `UWI: ${hdr.uwi}` : (workbench.wellId ?? '—')}
				</span>
			</button>
			<div class="flex shrink-0 items-center gap-1">
				<button
					type="button"
					onclick={() =>
						editingWellHeader ? (editingWellHeader = false) : openHeaderEditor()}
					class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-secondary)] hover:bg-slate-200 hover:text-[var(--text-primary)]"
				>
					{editingWellHeader ? 'Cancel' : 'Edit Header'}
				</button>
			</div>
		</div>

		{#if showWellHeader}
			<div class="max-h-[42vh] overflow-y-auto px-2.5 py-2">
				{#if hdr.company || hdr.field || hdr.country || hdr.crs}
					<div class="mb-1.5 flex flex-wrap items-center gap-1 text-[10px]">
						{#if hdr.company}
							<span
								class="rounded-xs bg-blue-50 px-1.5 py-0.5 font-semibold text-blue-800"
							>
								{hdr.company}
							</span>
						{/if}
						{#if hdr.field}
							<span
								class="rounded-xs bg-indigo-50 px-1.5 py-0.5 font-semibold text-indigo-800"
							>
								{hdr.field}
							</span>
						{/if}
						{#if hdr.country}
							<span
								class="rounded-xs bg-emerald-50 px-1.5 py-0.5 font-semibold text-emerald-800"
							>
								{hdr.country}
							</span>
						{/if}
						{#if hdr.crs}
							<span
								class="rounded-xs bg-slate-100 px-1.5 py-0.5 font-mono text-[9.5px] text-[var(--text-secondary)]"
							>
								CRS: {hdr.crs}
							</span>
						{/if}
					</div>
				{/if}

				{#if !editingWellHeader}
					<div class="grid grid-cols-2 gap-1.5 text-[10.5px]">
						<div
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
						>
							<div class="text-[9.5px] text-[var(--text-muted)]">Well Name / UWI</div>
							<div
								class="truncate font-semibold text-[var(--text-primary)]"
								title={`${hdr.well_name || workbench.wellId || '—'} (${hdr.uwi || '—'})`}
							>
								{hdr.well_name || workbench.wellId || '—'}
								<span
									class="font-mono text-[10px] font-normal text-[var(--text-secondary)]"
								>
									({hdr.uwi || hdr.api || '—'})
								</span>
							</div>
						</div>
						<div
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
						>
							<div class="text-[9.5px] text-[var(--text-muted)]">Company / Operator</div>
							<div
								class="truncate font-semibold text-[var(--text-primary)]"
								title={hdr.company || '—'}
							>
								{hdr.company || '—'}
							</div>
						</div>
						<div
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
						>
							<div class="text-[9.5px] text-[var(--text-muted)]">Field / Block</div>
							<div
								class="truncate font-semibold text-[var(--text-primary)]"
								title={`${hdr.field || '—'}${hdr.location ? ` • ${hdr.location}` : ''}`}
							>
								{hdr.field || '—'}{hdr.location ? ` • ${hdr.location}` : ''}
							</div>
						</div>
						<div
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
						>
							<div class="text-[9.5px] text-[var(--text-muted)]">Country / Province</div>
							<div
								class="truncate font-semibold text-[var(--text-primary)]"
								title={`${hdr.country || '—'}${hdr.province ? ` (${hdr.province})` : ''}`}
							>
								{hdr.country || '—'}{hdr.province ? ` (${hdr.province})` : ''}
							</div>
						</div>
						<div
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
						>
							<div class="text-[9.5px] text-[var(--text-muted)]">Latitude</div>
							<div
								class="mono-num truncate font-semibold text-[var(--text-primary)]"
								title={hdr.latitude_raw || String(hdr.latitude ?? '—')}
							>
								{hdr.latitude != null ? `${Number(hdr.latitude).toFixed(6)}°` : '—'}
								{#if hdr.latitude_raw && hdr.latitude_raw !== String(hdr.latitude)}
									<span class="text-[9.5px] font-normal text-[var(--text-muted)]"
										>({hdr.latitude_raw})</span
									>
								{/if}
							</div>
						</div>
						<div
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
						>
							<div class="text-[9.5px] text-[var(--text-muted)]">Longitude</div>
							<div
								class="mono-num truncate font-semibold text-[var(--text-primary)]"
								title={hdr.longitude_raw || String(hdr.longitude ?? '—')}
							>
								{hdr.longitude != null ? `${Number(hdr.longitude).toFixed(6)}°` : '—'}
								{#if hdr.longitude_raw && hdr.longitude_raw !== String(hdr.longitude)}
									<span class="text-[9.5px] font-normal text-[var(--text-muted)]"
										>({hdr.longitude_raw})</span
									>
								{/if}
							</div>
						</div>
						{#if hdr.x_coord != null || hdr.y_coord != null}
							<div
								class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
							>
								<div class="text-[9.5px] text-[var(--text-muted)]">X / Easting</div>
								<div class="mono-num truncate font-semibold text-[var(--text-primary)]">
									{hdr.x_coord != null ? fmtNum(hdr.x_coord, 2) : '—'}
								</div>
							</div>
							<div
								class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
							>
								<div class="text-[9.5px] text-[var(--text-muted)]">Y / Northing</div>
								<div class="mono-num truncate font-semibold text-[var(--text-primary)]">
									{hdr.y_coord != null ? fmtNum(hdr.y_coord, 2) : '—'}
								</div>
							</div>
						{/if}
						<div
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
						>
							<div class="text-[9.5px] text-[var(--text-muted)]">KB / DF Elevation</div>
							<div class="mono-num truncate font-semibold text-[var(--text-primary)]">
								KB: {hdr.kb_elev != null ? `${hdr.kb_elev} ${dUnit}` : '—'}
								{#if hdr.df_elev != null}
									• DF: {hdr.df_elev} {dUnit}
								{/if}
							</div>
						</div>
						<div
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
						>
							<div class="text-[9.5px] text-[var(--text-muted)]">GL / Water Depth</div>
							<div class="mono-num truncate font-semibold text-[var(--text-primary)]">
								GL: {hdr.gl_elev != null ? `${hdr.gl_elev} ${dUnit}` : '—'}
								{#if hdr.water_depth != null}
									• WD: {hdr.water_depth} {dUnit}
								{/if}
							</div>
						</div>
						<div
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
						>
							<div class="text-[9.5px] text-[var(--text-muted)]">Service Co. / Date</div>
							<div
								class="truncate font-semibold text-[var(--text-primary)]"
								title={`${hdr.service_company || '—'} • ${hdr.log_date || hdr.spud_date || '—'}`}
							>
								{hdr.service_company || '—'}
								{#if hdr.log_date || hdr.spud_date}
									<span class="font-normal text-[var(--text-secondary)]"
										>• {hdr.log_date || hdr.spud_date}</span
									>
								{/if}
							</div>
						</div>
						<div
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1"
						>
							<div class="text-[9.5px] text-[var(--text-muted)]">
								BHT / Bit Size / Rmf
							</div>
							<div class="mono-num truncate font-semibold text-[var(--text-primary)]">
								{hdr.bht != null ? `${hdr.bht}°` : '—'} • {hdr.bit_size != null
									? `${hdr.bit_size}"`
									: '—'} • {hdr.rmf != null
									? `${hdr.rmf} Ωm${hdr.rmf_temp != null ? ` @ ${hdr.rmf_temp}°C` : ''}`
									: '—'}
							</div>
						</div>
						<div
							class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 col-span-2"
						>
							<div class="text-[9.5px] text-[var(--text-muted)]">
								Mud Filtrate Density / Salinity / Type
							</div>
							<div class="mono-num truncate font-semibold text-[var(--text-primary)]">
								{hdr.mud_filtrate_density != null
									? `${fmtNum(hdr.mud_filtrate_density, 3)} g/cc`
									: '—'} • {hdr.mud_filtrate_salinity != null
									? `${fmtNum(hdr.mud_filtrate_salinity, 0)} ppm`
									: '—'} • {hdr.mud_type ? hdr.mud_type : '—'}
							</div>
						</div>
					</div>

					{#if Array.isArray(hdr.curve_lineage) && hdr.curve_lineage.length > 0}
						<details class="mt-1.5 text-[10px] text-[var(--text-secondary)]">
							<summary
								class="cursor-pointer font-semibold hover:text-[var(--text-primary)]"
							>
								Imported LAS curve lineage ({hdr.curve_lineage.length})
							</summary>
							<div
								class="mt-1 max-h-32 space-y-0.5 overflow-y-auto rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5"
							>
								{#each hdr.curve_lineage as entry}
									<div>
										{entry.filename}: {entry.source_mnemonic}{entry.source_unit
											? ` (${entry.source_unit})`
											: ''}
										→ {entry.stored_curve}{entry.raw_curve
											? ` (raw: ${entry.raw_curve})`
											: ''}
									</div>
								{/each}
							</div>
						</details>
					{/if}
					{#if (hdr.raw_well_header && Object.keys(hdr.raw_well_header).length > 0) || (hdr.raw_parameters && Object.keys(hdr.raw_parameters).length > 0)}
						<details class="mt-1.5 text-[10px] text-[var(--text-secondary)]">
							<summary
								class="cursor-pointer font-semibold hover:text-[var(--text-primary)]"
							>
								Raw LAS/DLIS Header Tags ({Object.keys(hdr.raw_well_header || {})
									.length} ~Well, {Object.keys(hdr.raw_parameters || {}).length} ~Parameter)
							</summary>
							<div
								class="mt-1 grid max-h-32 grid-cols-2 gap-x-2 gap-y-0.5 overflow-y-auto rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 font-mono text-[9.5px]"
							>
								{#each Object.entries( { ...(hdr.raw_well_header || {}), ...(hdr.raw_parameters || {}) } ) as [rk, rv]}
									<div class="truncate" title={`${rk}: ${rv}`}>
										<span class="font-semibold text-[var(--text-secondary)]"
											>{rk}:</span
										>
										<span class="text-[var(--text-primary)]">{rv ?? '—'}</span>
									</div>
								{/each}
							</div>
						</details>
					{/if}
				{:else}
					<div
						class="space-y-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2 text-[10.5px]"
					>
						<div class="grid grid-cols-2 gap-1.5">
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>UWI / API</span
								>
								<input
									type="text"
									bind:value={headerDraft.uwi}
									class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Company / Operator</span
								>
								<input
									type="text"
									bind:value={headerDraft.company}
									class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Field</span
								>
								<input
									type="text"
									bind:value={headerDraft.field}
									class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Country</span
								>
								<input
									type="text"
									bind:value={headerDraft.country}
									class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Location / Block</span
								>
								<input
									type="text"
									bind:value={headerDraft.location}
									class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Province / State</span
								>
								<input
									type="text"
									bind:value={headerDraft.province}
									class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Latitude (°)</span
								>
								<input
									type="number"
									step="0.000001"
									bind:value={headerDraft.latitude}
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Longitude (°)</span
								>
								<input
									type="number"
									step="0.000001"
									bind:value={headerDraft.longitude}
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>KB Elevation ({headerDraft.datum_unit || 'm'})</span
								>
								<input
									type="number"
									step="0.1"
									bind:value={headerDraft.kb_elev}
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>DF Elevation ({headerDraft.datum_unit || 'm'})</span
								>
								<input
									type="number"
									step="0.1"
									bind:value={headerDraft.df_elev}
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>GL / Seabed Elev ({headerDraft.datum_unit || 'm'})</span
								>
								<input
									type="number"
									step="0.1"
									bind:value={headerDraft.gl_elev}
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Service Company</span
								>
								<input
									type="text"
									bind:value={headerDraft.service_company}
									class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Log Date</span
								>
								<input
									type="text"
									bind:value={headerDraft.log_date}
									class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>BHT ({headerDraft.bht_unit || 'degC'})</span
								>
								<input
									type="number"
									step="0.1"
									bind:value={headerDraft.bht}
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Bit Size (in)</span
								>
								<input
									type="number"
									step="0.125"
									bind:value={headerDraft.bit_size}
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Rmf (ohm.m)</span
								>
								<input
									type="number"
									step="0.001"
									min="0.001"
									bind:value={headerDraft.rmf}
									oninput={handleRmfInput}
									placeholder="e.g. 0.5"
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Rmf Meas Temp (°C)</span
								>
								<input
									type="number"
									step="0.5"
									bind:value={headerDraft.rmf_temp}
									oninput={handleRmfInput}
									placeholder="e.g. 20"
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<div class="flex items-center justify-between">
									<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
										>Mud Filtrate Salinity (ppm)</span
									>
									{#if derivedFiltrateSalinity != null}
										<button
											type="button"
											onclick={() => {
												headerDraft.mud_filtrate_salinity =
													derivedFiltrateSalinity;
											}}
											class="cursor-pointer text-[9px] font-semibold text-blue-600 hover:text-blue-800 hover:underline"
											title="Set salinity to Bateman-Konen equivalent from Rmf"
										>
											⚡ Calc ({derivedFiltrateSalinity.toLocaleString()})
										</button>
									{/if}
								</div>
								<input
									type="number"
									min="0"
									step="1000"
									bind:value={headerDraft.mud_filtrate_salinity}
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Mud Filtrate Density (g/cc)</span
								>
								<input
									type="number"
									min="0.1"
									max="2"
									step="0.01"
									bind:value={headerDraft.mud_filtrate_density}
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							<label class="flex flex-col gap-0.5">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>Mud Type</span
								>
								<select
									bind:value={headerDraft.mud_type}
									class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								>
									<option value="">—</option>
									<option value="WBM">WBM (Water-Based)</option>
									<option value="OBM">OBM (Oil-Based)</option>
									<option value="SBM">SBM (Synthetic-Based)</option>
								</select>
							</label>
							<label class="flex flex-col gap-0.5 col-span-2">
								<span class="text-[9.5px] font-medium text-[var(--text-muted)]"
									>OBM Base Oil Density (g/cc)</span
								>
								<input
									type="number"
									min="0.65"
									max="0.95"
									step="0.01"
									bind:value={headerDraft.obm_base_density}
									class="mono-num h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
								/>
							</label>
							{#if filtrateQcWarning}
								<div
									class="col-span-2 flex items-start justify-between gap-1.5 rounded-xs border border-amber-300 bg-amber-50 px-2 py-1 text-[10px] text-amber-900"
								>
									<div class="flex items-start gap-1">
										<span class="font-bold">⚠️ QC:</span>
										<span>{filtrateQcWarning}</span>
									</div>
									{#if derivedFiltrateSalinity != null && Number(headerDraft.mud_filtrate_salinity) !== derivedFiltrateSalinity}
										<button
											type="button"
											onclick={() => {
												headerDraft.mud_filtrate_salinity =
													derivedFiltrateSalinity;
											}}
											class="shrink-0 cursor-pointer font-semibold text-blue-700 underline hover:text-blue-900"
										>
											Use {derivedFiltrateSalinity.toLocaleString()} ppm
										</button>
									{/if}
								</div>
							{/if}
						</div>
						<div class="flex justify-end gap-1.5 pt-1">
							<button
								type="button"
								onclick={() => (editingWellHeader = false)}
								class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[10.5px] font-medium text-[var(--text-secondary)]"
							>
								Cancel
							</button>
							<button
								type="button"
								disabled={savingWellHeader}
								onclick={saveHeaderDraft}
								class="h-6 rounded-xs bg-[var(--accent-primary)] px-2.5 text-[10.5px] font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
							>
								{savingWellHeader ? 'Saving...' : 'Save Header'}
							</button>
						</div>
					</div>
				{/if}
			</div>
		{/if}
	</div>

	<!-- Sub-navigation Tabs (2-row 4-col grid for clear labels across all 8 ancillary tables) -->
	<div
		class="grid shrink-0 grid-cols-4 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] font-semibold"
	>
		{#each [['hole_sections', `Hole (${wellData.holeSections.length})`], ['tops', `Tops (${wellData.tops.length})`], ['contacts', `Contacts (${wellData.contacts.length})`], ['perfs', `Perfs (${wellData.perforations.length})`], ['core', `RCA (${wellData.coreSamples.length})`], ['scal', `SCAL (${wellData.scalPc.length})`], ['pressure', `Press (${wellData.pressureTests.length})`], ['survey', `Survey (${wellData.survey.length})`]] as [key, lbl]}
			<button
				type="button"
				onclick={() => setActiveTab(key as AncillaryTab)}
				class="h-6.5 truncate border-r border-b border-[var(--border-hairline)] px-1 transition-colors [&:nth-child(4n)]:border-r-0 [&:nth-child(n+5)]:border-b-0 {activeTab ===
				key
					? 'bg-[var(--bg-surface)] font-bold text-[var(--accent-primary)] shadow-[inset_0_-2px_0_var(--accent-primary)]'
					: 'text-[var(--text-secondary)] hover:bg-slate-200/60'}"
				title={key === 'hole_sections'
					? 'Hole Sections & Mud Program (Bit Size, Mud Type, Mud Weight, Salinity, Rmf)'
					: key === 'core'
						? 'Routine Core Analysis (CPORE, CPERM, CGD)'
						: key === 'scal'
							? 'SCAL Capillary Pressure (Multiple Pc & Sw pairs per sample/depth)'
							: key === 'perfs'
								? 'Perforation intervals (Name, Top, Bottom, Description)'
								: key === 'pressure'
									? 'Formation pressure tests (Depth, Pressure, Mobility, Permeability, Temperature, Remarks)'
									: lbl}
			>
				{lbl}
			</button>
		{/each}
	</div>

	<!-- Quick Add Bar -->
	<div
		class="flex items-center gap-1 overflow-x-auto border-b border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5"
	>
		{#if activeTab === 'hole_sections'}
			<input
				type="text"
				placeholder='Section (e.g. 8.5" Res)'
				bind:value={newName}
				class="h-6 w-24 min-w-0 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Top"
				title="Section Top MD ({depthDisplay.unit})"
				bind:value={newHoleTop}
				class="mono-num h-6 w-20 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Bottom"
				title="Section Bottom MD ({depthDisplay.unit})"
				bind:value={newHoleBottom}
				class="mono-num h-6 w-20 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-right text-xs"
			/>
			<input
				type="number"
				step="0.125"
				placeholder="Bit (in)"
				title="Bit Size (inches)"
				bind:value={newHoleBitSize}
				class="mono-num h-6 w-16 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<select
				bind:value={newHoleMudType}
				title="Mud System (WBM / OBM / SBM)"
				class="h-6 w-16 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-xs"
			>
				<option value="WBM">WBM</option>
				<option value="OBM">OBM</option>
				<option value="SBM">SBM</option>
			</select>
			<input
				type="number"
				step="0.1"
				placeholder="MW (ppg)"
				title="Mud Weight (ppg)"
				bind:value={newHoleMudWeight}
				class="mono-num h-6 w-16 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			{#if newHoleMudType === 'WBM'}
				<input
					type="number"
					step="1000"
					placeholder="Sal (ppm)"
					title="Mud Filtrate Salinity (ppm)"
					bind:value={newHoleSalinity}
					class="mono-num h-6 w-20 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
				/>
				<input
					type="number"
					step="0.01"
					placeholder="Rmf"
					title="Mud Filtrate Resistivity Rmf (ohm.m)"
					bind:value={newHoleRmf}
					class="mono-num h-6 w-14 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
				/>
				<input
					type="number"
					step="1"
					placeholder="T°C"
					title="Rmf Measurement Temp (°C)"
					bind:value={newHoleRmfTemp}
					class="mono-num h-6 w-12 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
				/>
			{/if}
		{:else if activeTab === 'tops' || activeTab === 'contacts'}
			<input
				type="text"
				placeholder={activeTab === 'tops' ? 'Top Name (e.g. Hugin)' : 'OWC / GOC / FWL'}
				bind:value={newName}
				class="h-6 min-w-0 flex-1 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Depth"
				aria-label="Depth ({depthDisplay.unit})"
				title="Depth ({depthDisplay.unit})"
				bind:value={newDepth}
				class="mono-num h-6 w-24 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-right text-xs"
			/>
		{:else if activeTab === 'perfs'}
			<input
				type="text"
				placeholder="Name"
				bind:value={newName}
				class="h-6 w-20 min-w-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Top"
				title="Perforation Top MD ({depthDisplay.unit})"
				bind:value={newPerfTop}
				class="mono-num h-6 w-24 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Bottom"
				title="Perforation Bottom MD ({depthDisplay.unit})"
				bind:value={newPerfBottom}
				class="mono-num h-6 w-24 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-right text-xs"
			/>
			<input
				type="text"
				placeholder="Description"
				bind:value={newPerfDesc}
				class="h-6 min-w-0 flex-1 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-xs"
			/>
		{:else if activeTab === 'core'}
			<input
				type="text"
				placeholder="Sample"
				bind:value={newName}
				class="h-6 w-16 min-w-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Depth"
				title="Sample Depth ({depthDisplay.unit})"
				bind:value={newDepth}
				class="mono-num h-6 w-24 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="CPORE"
				bind:value={newVal1}
				class="mono-num h-6 w-16 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="CPERM"
				bind:value={newVal2}
				class="mono-num h-6 w-16 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
		{:else if activeTab === 'scal'}
			<input
				type="text"
				placeholder="Sample"
				bind:value={newName}
				title="Sample Name (enter manually; stays sticky for multiple Pc-Sw pairs)"
				class="h-6 w-16 min-w-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Depth"
				bind:value={newDepth}
				title="Sample Depth ({depthDisplay.unit})"
				class="mono-num h-6 w-24 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-right text-xs"
			/>
			<select
				bind:value={newScalExpType}
				title="Experiment Type (Method)"
				class="h-6 w-24 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-xs text-[var(--text-primary)]"
			>
				<option value="porous_plate">Porous Plate</option>
				<option value="micp">MICP</option>
				<option value="centrifuge">Centrifuge</option>
				<option value="oil_brine">Water-Oil</option>
			</select>
			<input
				type="number"
				step="any"
				min="0"
				max="1"
				placeholder="CPORE"
				bind:value={newScalCpore}
				title="Core Porosity CPORE (v/v or %)"
				class="mono-num h-6 w-14 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				min="0"
				placeholder="CPERM"
				bind:value={newScalCperm}
				title="Core Permeability CPERM (mD)"
				class="mono-num h-6 w-14 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Pc"
				bind:value={newScalPc}
				title="Capillary Pressure Pc (psi)"
				class="mono-num h-6 w-14 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				min="0"
				max="1"
				placeholder="Sw"
				bind:value={newScalSw}
				title="Water Saturation Sw (v/v)"
				class="mono-num h-6 w-14 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
		{:else if activeTab === 'pressure'}
			<input
				type="number"
				step="any"
				placeholder="Depth"
				title="Test Depth MD ({depthDisplay.unit})"
				bind:value={newDepth}
				class="mono-num h-6 w-24 min-w-0 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Press"
				title="Formation Pressure (psi)"
				bind:value={newVal2}
				class="mono-num h-6 w-18 min-w-0 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Mob"
				title="Drawdown Mobility (mD/cP) — optional"
				bind:value={newPressMob}
				class="mono-num h-6 w-12 min-w-0 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Perm"
				title="Permeability (mD) — optional"
				bind:value={newPressPerm}
				class="mono-num h-6 w-12 min-w-0 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Temp"
				title="Temperature (degC) — optional"
				bind:value={newPressTemp}
				class="mono-num h-6 w-12 min-w-0 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="text"
				placeholder="Remarks"
				title="Remarks (e.g. Valid, Tight, Supercharged, Lost seal)"
				bind:value={newPressRemarks}
				class="h-6 min-w-0 flex-1 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-xs"
			/>
		{:else if activeTab === 'survey'}
			<input
				type="number"
				step="any"
				placeholder="Depth / MD"
				title="Depth / MD ({depthDisplay.unit})"
				bind:value={newDepth}
				class="mono-num h-6 w-24 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Inc"
				title="Inclination (deg)"
				bind:value={newVal1}
				class="mono-num h-6 w-16 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Azim"
				title="Azimuth (deg)"
				bind:value={newVal2}
				class="mono-num h-6 flex-1 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
		{:else}
			<input
				type="number"
				step="any"
				placeholder="Depth / MD"
				title="Depth / MD ({depthDisplay.unit})"
				bind:value={newDepth}
				class="mono-num h-6 w-24 shrink-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Azim"
				title="Azimuth (deg)"
				bind:value={newVal2}
				class="mono-num h-6 flex-1 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
		{/if}
		<span class="text-[10px] text-[var(--text-muted)]">{depthDisplay.unit}</span>
		<button
			type="button"
			onclick={addQuickRow}
			class="h-6 shrink-0 rounded-sm bg-[var(--accent-primary)] px-2 text-[11px] font-semibold text-white hover:bg-blue-700"
		>
			+ Add
		</button>
	</div>

	{#if activeTab === 'core'}
		<div class="shrink-0 border-b border-[var(--border-hairline)] p-1.5">
			<CoreDepthShiftPanel defaultTarget="combined" />
		</div>
	{:else if activeTab === 'survey'}
		<div
			class="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--border-hairline)] bg-[var(--bg-app,#f8fafc)] px-2.5 py-1.5 text-xs"
		>
			<div class="flex items-center gap-2">
				{#if wellData.datumElevationInfo.hasElevation}
					<span
						class="inline-flex items-center gap-1 rounded bg-emerald-50 px-1.5 py-0.5 text-[11px] font-medium text-emerald-700 ring-1 ring-emerald-600/20 ring-inset"
						data-testid="survey-tvdss-status"
					>
						<span class="inline-block h-1.5 w-1.5 rounded-full bg-emerald-600"></span>
						TVDSS Active &bull; Datum: {wellData.datumElevationInfo.label}
					</span>
				{:else}
					<span
						class="inline-flex items-center gap-1 rounded bg-amber-50 px-1.5 py-0.5 text-[11px] font-medium text-amber-800 ring-1 ring-amber-600/20 ring-inset"
						data-testid="survey-tvdss-status"
						title="No Derrick Floor (DF) or Kelly Bushing (KB) elevation found in well header. TVD is referenced to drill floor (TVDDF)."
					>
						<span class="inline-block h-1.5 w-1.5 rounded-full bg-amber-500"></span>
						TVDDF Mode (No DF/KB Datum)
					</span>
				{/if}
				<span class="text-[11px] text-[var(--text-muted)]">Min. Curvature</span>
			</div>
			<button
				type="button"
				disabled={isRecalculating || wellData.survey.length === 0}
				onclick={recalculateSurvey}
				class="flex items-center gap-1.5 rounded bg-[var(--accent-primary)] px-2.5 py-1 text-[11px] font-semibold text-white shadow-xs hover:bg-blue-700 disabled:opacity-40"
				title="Recalculate True Vertical Depth from surveys using Minimum Curvature and update TVD/TVDSS curves"
			>
				{#if isRecalculating}
					<span
						class="inline-block h-3 w-3 animate-spin rounded-full border-2 border-white border-t-transparent"
					></span>
					Recalculating...
				{:else}
					<svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
						/>
					</svg>
					Recalculate TVD
				{/if}
			</button>
		</div>
	{/if}

	<!-- Editable High-Density Grid (Supports Double-Click Edit, Multi-Row Selection & Excel Paste) -->
	<div class="min-h-0 flex-1">
		{#if activeTab === 'hole_sections'}
			<CompactDataGrid
				columns={HOLE_SECTION_COLS}
				bind:rows={wellData.holeSections}
				editable={true}
				emptyMessage='No hole sections defined. Add intervals above (Bit Size, Mud Type, MW, Salinity, Rmf) or paste from Excel to drive QAQC & Fluid Density.'
				oncellchange={() => saveCurrentTab()}
				onpasteRows={handlePasteRows}
				ondeleteRow={handleDeleteRow}
				ondeleteRows={handleDeleteRows}
			/>
		{:else if activeTab === 'tops'}
			<CompactDataGrid
				columns={TOP_COLS}
				bind:rows={wellData.tops}
				editable={true}
				emptyMessage="No formation tops. Add above, drag markers on log, or paste from Excel."
				oncellchange={() => saveCurrentTab()}
				onpasteRows={handlePasteRows}
				ondeleteRow={handleDeleteRow}
				ondeleteRows={handleDeleteRows}
			/>
		{:else if activeTab === 'contacts'}
			<CompactDataGrid
				columns={CONTACT_COLS}
				bind:rows={wellData.contacts}
				editable={true}
				emptyMessage="No fluid contacts (OWC/GOC/FWL)."
				oncellchange={() => saveCurrentTab()}
				onpasteRows={handlePasteRows}
				ondeleteRow={handleDeleteRow}
				ondeleteRows={handleDeleteRows}
			/>
		{:else if activeTab === 'perfs'}
			<CompactDataGrid
				columns={PERF_COLS}
				bind:rows={wellData.perforations}
				editable={true}
				emptyMessage="No perforation intervals. Add above or paste (Name, Top, Bottom, Description) from Excel."
				oncellchange={() => saveCurrentTab()}
				onpasteRows={handlePasteRows}
				ondeleteRow={handleDeleteRow}
				ondeleteRows={handleDeleteRows}
			/>
		{:else if activeTab === 'core'}
			<CompactDataGrid
				columns={CORE_COLS}
				bind:rows={wellData.coreSamples}
				editable={true}
				emptyMessage="No RCA core plugs (CPORE, CPERM, CGD). Add above or paste from Excel."
				oncellchange={handleCoreCellChange}
				onpasteRows={handlePasteRows}
				ondeleteRow={handleDeleteRow}
				ondeleteRows={handleDeleteRows}
			/>
		{:else if activeTab === 'scal'}
			<CompactDataGrid
				columns={SCAL_PC_COLS}
				bind:rows={wellData.scalPc}
				editable={true}
				emptyMessage="No SCAL capillary pressure (Pc, Sw) pairs. Add multiple pairs per sample/depth above or paste from Excel."
				oncellchange={handleScalCellChange}
				onpasteRows={handlePasteRows}
				ondeleteRow={handleDeleteRow}
				ondeleteRows={handleDeleteRows}
			/>
		{:else if activeTab === 'pressure'}
			<CompactDataGrid
				columns={PRESSURE_COLS}
				bind:rows={wellData.pressureTests}
				editable={true}
				emptyMessage="No formation pressure tests. Add above or paste (Depth, Pressure, Mobility, Perm, Temp, Remarks) from Excel."
				oncellchange={() => saveCurrentTab()}
				onpasteRows={handlePasteRows}
				ondeleteRow={handleDeleteRow}
				ondeleteRows={handleDeleteRows}
			/>
		{:else}
			<CompactDataGrid
				columns={surveyColumns}
				bind:rows={wellData.survey}
				editable={true}
				emptyMessage="No directional survey rows."
				oncellchange={handleSurveyCellChange}
				onpasteRows={handlePasteRows}
				ondeleteRow={handleDeleteRow}
				ondeleteRows={handleDeleteRows}
			/>
		{/if}
	</div>

	<!-- Action Footer -->
	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<span class="truncate text-[11px] text-[var(--text-muted)]">
			{statusMsg ?? 'Double-click cells to edit · Check rows to delete · Ctrl+V from Excel'}
		</span>
		<button
			type="button"
			disabled={isSaving}
			onclick={saveCurrentTab}
			class="h-7 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{isSaving ? 'Saving...' : 'Save All'}
		</button>
	</div>
</div>
