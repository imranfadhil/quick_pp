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
	import { computeMinimumCurvature, interpolateTrajectoryOnGrid } from '$lib/utils/trajectory';

	type AncillaryTab = 'tops' | 'contacts' | 'perfs' | 'core' | 'scal' | 'pressure' | 'survey';
	let { initialTab = 'tops' }: { initialTab?: AncillaryTab } = $props();
	// svelte-ignore state_referenced_locally
	let activeTab = $state<AncillaryTab>(initialTab);
	let isSaving = $state(false);
	let statusMsg = $state<string | null>(null);

	// New item quick-entry drafts (expressed directly in depthDisplay.unit)
	let newName = $state('');
	let newDepth = $state<number | null>(null);
	let newPerfTop = $state<number | null>(null);
	let newPerfBottom = $state<number | null>(null);
	let newPerfDesc = $state('');
	let newVal1 = $state<number | null>(0.2);
	let newVal2 = $state<number | null>(100);
	let newScalPc = $state<number | null>(15);
	let newScalSw = $state<number | null>(0.35);
	let newScalCpore = $state<number | null>(null);
	let newScalCperm = $state<number | null>(null);
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
				newDepth = Number(depthFromMeters(depthToMeters(newDepth, oldUnit), unit).toFixed(2));
			}
			if (newPerfTop !== null && Number.isFinite(newPerfTop)) {
				newPerfTop = Number(depthFromMeters(depthToMeters(newPerfTop, oldUnit), unit).toFixed(2));
			}
			if (newPerfBottom !== null && Number.isFinite(newPerfBottom)) {
				newPerfBottom = Number(depthFromMeters(depthToMeters(newPerfBottom, oldUnit), unit).toFixed(2));
			}
		});
	});

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
					experiment_type: 'micp',
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
			if (activeTab === 'tops') {
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
		if (activeTab === 'tops') {
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
						experiment_type: r.experiment_type || 'micp',
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
			if (activeTab === 'tops') {
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
				// Propagate shared depth, porosity, and permeability for same sample_name if user edited one row
				const depthBySample = new Map<string, number>();
				const cporeBySample = new Map<string, number | null>();
				const cpermBySample = new Map<string, number | null>();
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
						experiment_type: r.experiment_type || 'micp',
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
		if (row && ['porosity', 'permeability', 'depth', 'sample_name'].includes(key)) {
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

				for (const r of wellData.scalPc) {
					if (String(r.sample_name || '').trim() === sname) {
						if (key === 'porosity') r.porosity = normPor;
						if (key === 'permeability') r.permeability = perm;
						if (key === 'depth' && depth !== null) r.depth = depth;
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
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<!-- Sub-navigation Tabs -->
	<div
		class="grid grid-cols-7 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] font-semibold"
	>
		{#each [['tops', `Tops (${wellData.tops.length})`], ['contacts', `Contacts (${wellData.contacts.length})`], ['perfs', `Perfs (${wellData.perforations.length})`], ['core', `RCA (${wellData.coreSamples.length})`], ['scal', `SCAL (${wellData.scalPc.length})`], ['pressure', `Press (${wellData.pressureTests.length})`], ['survey', `Survey (${wellData.survey.length})`]] as [key, lbl]}
			<button
				type="button"
				onclick={() => (activeTab = key as AncillaryTab)}
				class="h-7 truncate border-r border-[var(--border-hairline)] px-1 transition-colors last:border-r-0 {activeTab ===
				key
					? 'border-b-2 border-b-[var(--accent-primary)] bg-[var(--bg-surface)] text-[var(--accent-primary)]'
					: 'text-[var(--text-secondary)] hover:bg-slate-200/60'}"
				title={key === 'core'
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
		{#if activeTab === 'tops' || activeTab === 'contacts'}
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
		{#if activeTab === 'tops'}
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
