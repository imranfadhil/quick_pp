<script lang="ts">
	import { saveAncillaryItems } from '$lib/api/client';
	import CompactDataGrid, {
		type GridColumn
	} from '$lib/components/workbench/CompactDataGrid.svelte';
	import CoreDepthShiftPanel from '$lib/components/workbench/CoreDepthShiftPanel.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	type AncillaryTab = 'tops' | 'contacts' | 'perfs' | 'core' | 'scal' | 'pressure' | 'survey';
	let activeTab = $state<AncillaryTab>('tops');
	let isSaving = $state(false);
	let statusMsg = $state<string | null>(null);

	// New item quick-entry drafts
	let newName = $state('');
	let newDepth = $state<number>(1000);
	let newVal1 = $state<number>(0.2);
	let newVal2 = $state<number>(100);
	let newScalPc = $state<number>(15);
	let newScalSw = $state<number>(0.35);
	let newPerfBottom = $state<number>(1005);
	let newPerfDesc = $state('');
	// Formation pressure test quick-entry drafts (optional fields may be left blank)
	let newPressMob = $state<number | null>(null);
	let newPressPerm = $state<number | null>(null);
	let newPressTemp = $state<number | null>(null);
	let newPressRemarks = $state('');

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
		{ key: 'porosity', label: 'CPORE', unit: 'v/v', numeric: true, decimals: 3, editable: true },
		{ key: 'permeability', label: 'CPERM', unit: 'mD', numeric: true, decimals: 2, editable: true }
	];

	const SCAL_PC_COLS: GridColumn[] = [
		{ key: 'sample_name', label: 'Sample', editable: true },
		{ key: 'depth', label: 'Depth', unit: 'm', numeric: true, decimals: 2, editable: true },
		{ key: 'pc', label: 'Pc', unit: 'psi', numeric: true, decimals: 2, editable: true },
		{ key: 'sw', label: 'Sw', unit: 'v/v', numeric: true, decimals: 3, editable: true }
	];

	const PRESSURE_COLS: GridColumn[] = [
		{ key: 'depth', label: 'Depth', unit: 'm', numeric: true, decimals: 1, editable: true },
		{ key: 'pressure', label: 'Form. Press.', unit: 'psi', numeric: true, decimals: 1, editable: true },
		{ key: 'mobility', label: 'Mobility', unit: 'mD/cP', numeric: true, decimals: 2, editable: true },
		{ key: 'permeability', label: 'Perm', unit: 'mD', numeric: true, decimals: 2, editable: true },
		{ key: 'temperature', label: 'Temp', unit: 'degC', numeric: true, decimals: 1, editable: true },
		{ key: 'remarks', label: 'Remarks', editable: true }
	];

	/** Parse an optional numeric input (blank / invalid -> null). */
	function optNum(v: unknown): number | null {
		if (v === null || v === undefined || (typeof v === 'string' && v.trim() === '')) return null;
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
			remarks: String(r.remarks ?? r.remark ?? '').trim()
		};
	}

	const SURVEY_COLS: GridColumn[] = [
		{ key: 'md', label: 'MD', unit: 'm', numeric: true, decimals: 1, editable: true },
		{ key: 'inc', label: 'Inc', unit: 'deg', numeric: true, decimals: 2, editable: true },
		{ key: 'azim', label: 'Azim', unit: 'deg', numeric: true, decimals: 2, editable: true }
	];

	async function addQuickRow() {
		if (!workbench.projectId || !workbench.wellId) return;
		if (activeTab === 'tops') {
			const name = newName.trim() || `Zone-${wellData.tops.length + 1}`;
			wellData.tops = [
				...wellData.tops,
				{ id: `top-${Date.now()}`, name, depth: Number(newDepth) }
			].sort((a, b) => a.depth - b.depth);
			newName = '';
		} else if (activeTab === 'contacts') {
			const type = (newName.trim() || 'OWC').toUpperCase();
			wellData.contacts = [
				...wellData.contacts,
				{ id: `contact-${Date.now()}`, type, name: type, depth: Number(newDepth) }
			];
			newName = '';
		} else if (activeTab === 'perfs') {
			const t = Number(newDepth);
			const b = Number(newPerfBottom);
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
			newPerfDesc = '';
		} else if (activeTab === 'core') {
			wellData.coreSamples = [
				...wellData.coreSamples,
				{
					sample_name: newName.trim() || `S-${wellData.coreSamples.length + 1}`,
					depth: Number(newDepth),
					porosity: Number(newVal1),
					permeability: Number(newVal2)
				}
			];
			newName = '';
		} else if (activeTab === 'scal') {
			// Keep sample_name & depth sticky so user can enter multiple (Pc, Sw) pairs for the same plug
			const sampleName =
				newName.trim() ||
				(wellData.scalPc.length > 0
					? String(wellData.scalPc[wellData.scalPc.length - 1].sample_name)
					: 'S-1');
			newName = sampleName;
			wellData.scalPc = [
				...wellData.scalPc,
				{
					sample_name: sampleName,
					depth: Number(newDepth),
					pc: Number(newScalPc),
					sw: Number(newScalSw),
					experiment_type: 'micp',
					cycle: 'drainage'
				}
			];
		} else if (activeTab === 'pressure') {
			wellData.pressureTests = [
				...wellData.pressureTests,
				normalizePressureRow({
					depth: newDepth,
					pressure: newVal2,
					mobility: newPressMob,
					permeability: newPressPerm,
					temperature: newPressTemp,
					remarks: newPressRemarks
				})
			].sort((a, b) => a.depth - b.depth);
			newPressRemarks = '';
		} else if (activeTab === 'survey') {
			wellData.survey = [
				...wellData.survey,
				{ md: Number(newDepth), inc: Number(newVal1), azim: Number(newVal2) }
			];
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
					workbench.selectedZones = workbench.selectedZones.filter((z) => remainingNames.has(z));
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
					const t = String(r.type || r.name || 'OWC').trim().toUpperCase();
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
						name: String(r.name || `PERF-${wellData.perforations.length + i + 1}`).trim(),
						top: Math.min(t, b),
						bottom: Math.max(t, b),
						description: String(r.description ?? '').trim()
					};
				})
				.filter((p) => Number.isFinite(p.top) && Number.isFinite(p.bottom));
			wellData.perforations = [...wellData.perforations, ...mapped].sort((a, b) => a.top - b.top);
		} else if (activeTab === 'core') {
			const mapped = pastedRows
				.map((r, i) => ({
					sample_name: String(r.sample_name || `S-${wellData.coreSamples.length + i + 1}`).trim(),
					depth: Number(r.depth ?? 0),
					porosity: r.porosity !== null && r.porosity !== undefined ? Number(r.porosity) : null,
					permeability:
						r.permeability !== null && r.permeability !== undefined
							? Number(r.permeability)
							: null
				}))
				.filter((s) => Number.isFinite(s.depth));
			wellData.coreSamples = [...wellData.coreSamples, ...mapped];
		} else if (activeTab === 'scal') {
			const mapped = pastedRows
				.map((r, i) => ({
					sample_name: String(r.sample_name || `S-${wellData.scalPc.length + i + 1}`).trim(),
					depth: Number(r.depth ?? 0),
					pc: Number(r.pc ?? 0),
					sw: Number(r.sw ?? 0),
					experiment_type: r.experiment_type || 'micp',
					cycle: r.cycle || 'drainage'
				}))
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
					md: Number(r.md ?? 0),
					inc: Number(r.inc ?? 0),
					azim: Number(r.azim ?? 0)
				}))
				.filter((s) => Number.isFinite(s.md));
			wellData.survey = [...wellData.survey, ...mapped];
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
					wellData.contacts.map((c) => ({ name: c.type || c.name, depth: Number(c.depth) }))
				);
			} else if (activeTab === 'perfs') {
				// Re-assign a normalized copy so plots (reference-tracked deps) refresh after inline cell edits
				wellData.perforations = wellData.perforations
					.filter((p) => Number.isFinite(Number(p.top)) && Number.isFinite(Number(p.bottom)))
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
				wellData.syncCoreCurvesFromSamples();
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'core_samples',
					'samples',
					wellData.coreSamples
				);
			} else if (activeTab === 'scal') {
				// Propagate shared depth for same sample_name if user edited depth on one row
				const depthBySample = new Map<string, number>();
				for (const r of wellData.scalPc) {
					if (r.sample_name && Number.isFinite(Number(r.depth))) {
						depthBySample.set(String(r.sample_name), Number(r.depth));
					}
				}
				const normalizedScal = wellData.scalPc.map((r, i) => {
					const sname = String(r.sample_name || `S-${i + 1}`).trim();
					return {
						sample_name: sname,
						depth: Number(r.depth ?? depthBySample.get(sname) ?? 0),
						pc: Number(r.pc ?? 0),
						sw: Number(r.sw ?? 0),
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
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<!-- Sub-navigation Tabs -->
	<div
		class="grid grid-cols-7 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] font-semibold"
	>
		{#each [
			['tops', `Tops (${wellData.tops.length})`],
			['contacts', `Contacts (${wellData.contacts.length})`],
			['perfs', `Perfs (${wellData.perforations.length})`],
			['core', `RCA (${wellData.coreSamples.length})`],
			['scal', `SCAL (${wellData.scalPc.length})`],
			['pressure', `Press (${wellData.pressureTests.length})`],
			['survey', `Survey (${wellData.survey.length})`]
		] as [key, lbl]}
			<button
				type="button"
				onclick={() => (activeTab = key as AncillaryTab)}
				class="h-7 truncate border-r border-[var(--border-hairline)] px-1 last:border-r-0 transition-colors {activeTab ===
				key
					? 'bg-[var(--bg-surface)] text-[var(--accent-primary)] border-b-2 border-b-[var(--accent-primary)]'
					: 'text-[var(--text-secondary)] hover:bg-slate-200/60'}"
				title={key === 'core'
					? 'Routine Core Analysis (CPORE, CPERM)'
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
		class="flex items-center gap-1 border-b border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5"
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
				step="0.5"
				placeholder="Depth (m)"
				bind:value={newDepth}
				class="mono-num h-6 w-20 rounded-sm border border-[var(--border-hairline)] bg-white px-1.5 text-right text-xs"
			/>
		{:else if activeTab === 'perfs'}
			<input
				type="text"
				placeholder="Name"
				bind:value={newName}
				class="h-6 w-16 min-w-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-xs"
			/>
			<input
				type="number"
				step="0.5"
				placeholder="Top"
				title="Perforation Top MD (m)"
				bind:value={newDepth}
				class="mono-num h-6 w-16 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="0.5"
				placeholder="Bottom"
				title="Perforation Bottom MD (m)"
				bind:value={newPerfBottom}
				class="mono-num h-6 w-16 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="text"
				placeholder="Description"
				bind:value={newPerfDesc}
				class="h-6 min-w-0 flex-1 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-xs"
			/>
		{:else if activeTab === 'core'}
			<input
				type="text"
				placeholder="Sample"
				bind:value={newName}
				class="h-6 w-14 min-w-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-xs"
			/>
			<input
				type="number"
				step="0.5"
				placeholder="Depth"
				bind:value={newDepth}
				class="mono-num h-6 w-16 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="0.01"
				placeholder="CPORE"
				bind:value={newVal1}
				class="mono-num h-6 w-14 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="1"
				placeholder="CPERM"
				bind:value={newVal2}
				class="mono-num h-6 w-14 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
		{:else if activeTab === 'scal'}
			<input
				type="text"
				placeholder="Sample"
				bind:value={newName}
				title="Sample Name (enter manually; stays sticky for multiple Pc-Sw pairs)"
				class="h-6 w-14 min-w-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-xs"
			/>
			<input
				type="number"
				step="0.5"
				placeholder="Depth"
				bind:value={newDepth}
				title="Sample Depth (m)"
				class="mono-num h-6 w-16 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="0.5"
				placeholder="Pc (psi)"
				bind:value={newScalPc}
				title="Capillary Pressure Pc (psi)"
				class="mono-num h-6 w-14 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="0.01"
				min="0"
				max="1"
				placeholder="Sw (v/v)"
				bind:value={newScalSw}
				title="Water Saturation Sw (v/v)"
				class="mono-num h-6 w-14 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
		{:else if activeTab === 'pressure'}
			<input
				type="number"
				step="0.5"
				placeholder="Depth"
				title="Test Depth MD (m)"
				bind:value={newDepth}
				class="mono-num h-6 w-14 min-w-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="0.1"
				placeholder="Press"
				title="Formation Pressure (psi)"
				bind:value={newVal2}
				class="mono-num h-6 w-14 min-w-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Mob"
				title="Drawdown Mobility (mD/cP) — optional"
				bind:value={newPressMob}
				class="mono-num h-6 w-12 min-w-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="any"
				placeholder="Perm"
				title="Permeability (mD) — optional"
				bind:value={newPressPerm}
				class="mono-num h-6 w-12 min-w-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="0.1"
				placeholder="Temp"
				title="Temperature (degC) — optional"
				bind:value={newPressTemp}
				class="mono-num h-6 w-12 min-w-0 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="text"
				placeholder="Remarks"
				title="Remarks (e.g. Valid, Tight, Supercharged, Lost seal)"
				bind:value={newPressRemarks}
				class="h-6 min-w-0 flex-1 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-xs"
			/>
		{:else}
			<input
				type="number"
				step="1"
				placeholder="Depth / MD"
				bind:value={newDepth}
				class="mono-num h-6 w-20 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="1"
				placeholder="Azim"
				bind:value={newVal2}
				class="mono-num h-6 flex-1 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
		{/if}
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
				emptyMessage="No RCA core plugs (CPORE, CPERM). Add above or paste from Excel."
				oncellchange={() => saveCurrentTab()}
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
				oncellchange={() => saveCurrentTab()}
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
				columns={SURVEY_COLS}
				bind:rows={wellData.survey}
				editable={true}
				emptyMessage="No directional survey rows."
				oncellchange={() => saveCurrentTab()}
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
