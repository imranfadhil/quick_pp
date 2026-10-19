<script lang="ts">
	import { deleteAncillaryItem, saveAncillaryItems } from '$lib/api/client';
	import CompactDataGrid, { type GridColumn } from '$lib/components/workbench/CompactDataGrid.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	type AncillaryTab = 'tops' | 'contacts' | 'core' | 'pressure' | 'survey';
	let activeTab = $state<AncillaryTab>('tops');
	let isSaving = $state(false);
	let statusMsg = $state<string | null>(null);

	// New item quick-entry drafts
	let newName = $state('');
	let newDepth = $state<number>(1000);
	let newVal1 = $state<number>(0.2);
	let newVal2 = $state<number>(100);

	const TOP_COLS: GridColumn[] = [
		{ key: 'name', label: 'Formation / Zone', editable: true },
		{ key: 'depth', label: 'Depth', unit: 'm', numeric: true, decimals: 1, editable: true }
	];

	const CONTACT_COLS: GridColumn[] = [
		{ key: 'type', label: 'Contact Type', editable: true },
		{ key: 'depth', label: 'Depth', unit: 'm', numeric: true, decimals: 1, editable: true }
	];

	const CORE_COLS: GridColumn[] = [
		{ key: 'sample_name', label: 'Sample', editable: true },
		{ key: 'depth', label: 'Depth', unit: 'm', numeric: true, decimals: 2, editable: true },
		{ key: 'porosity', label: 'CPORE', unit: 'v/v', numeric: true, decimals: 3, editable: true },
		{ key: 'permeability', label: 'CPERM', unit: 'mD', numeric: true, decimals: 2, editable: true }
	];

	const PRESSURE_COLS: GridColumn[] = [
		{ key: 'depth', label: 'Depth', unit: 'm', numeric: true, decimals: 1, editable: true },
		{ key: 'pressure', label: 'Pressure', unit: 'psi', numeric: true, decimals: 1, editable: true }
	];

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
		} else if (activeTab === 'pressure') {
			wellData.pressureTests = [
				...wellData.pressureTests,
				{ depth: Number(newDepth), pressure: Number(newVal2), pressure_uom: 'psi' }
			];
		} else if (activeTab === 'survey') {
			wellData.survey = [
				...wellData.survey,
				{ md: Number(newDepth), inc: Number(newVal1), azim: Number(newVal2) }
			];
		}
		await saveCurrentTab();
	}

	async function handleDeleteRow(idx: number, row: Record<string, any>) {
		if (!workbench.projectId || !workbench.wellId) return;
		try {
			if (activeTab === 'tops') {
				if (row.name) {
					await deleteAncillaryItem(workbench.projectId, workbench.wellId, 'formation_tops', row.name);
				}
				wellData.tops = wellData.tops.filter((_, i) => i !== idx);
			} else if (activeTab === 'contacts') {
				if (row.type || row.name) {
					await deleteAncillaryItem(
						workbench.projectId,
						workbench.wellId,
						'fluid_contacts',
						row.type || row.name
					);
				}
				wellData.contacts = wellData.contacts.filter((_, i) => i !== idx);
			} else if (activeTab === 'core') {
				wellData.coreSamples = wellData.coreSamples.filter((_, i) => i !== idx);
			} else if (activeTab === 'pressure') {
				wellData.pressureTests = wellData.pressureTests.filter((_, i) => i !== idx);
			} else if (activeTab === 'survey') {
				wellData.survey = wellData.survey.filter((_, i) => i !== idx);
			}
		} catch (e: any) {
			statusMsg = e?.message || 'Delete failed';
		}
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
					wellData.contacts.map((c) => ({ name: c.type, depth: Number(c.depth) }))
				);
			} else if (activeTab === 'core') {
				await saveAncillaryItems(
					workbench.projectId,
					workbench.wellId,
					'core_samples',
					'samples',
					wellData.coreSamples
				);
			} else if (activeTab === 'pressure') {
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
			statusMsg = 'Saved ✓';
			wellData.lastSavedMessage = 'Ancillary Saved';
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
		class="grid grid-cols-5 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] font-semibold"
	>
		{#each [
			['tops', `Tops (${wellData.tops.length})`],
			['contacts', `Contacts (${wellData.contacts.length})`],
			['core', `Core (${wellData.coreSamples.length})`],
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
		{:else if activeTab === 'core'}
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
				placeholder="Poro"
				bind:value={newVal1}
				class="mono-num h-6 w-16 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
			/>
			<input
				type="number"
				step="1"
				placeholder="Perm"
				bind:value={newVal2}
				class="mono-num h-6 w-16 rounded-sm border border-[var(--border-hairline)] bg-white px-1 text-right text-xs"
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
				placeholder={activeTab === 'pressure' ? 'Pressure (psi)' : 'Azim'}
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

	<!-- Editable High-Density Grid (Supports Double-Click Edit & Excel Paste) -->
	<div class="min-h-0 flex-1">
		{#if activeTab === 'tops'}
			<CompactDataGrid
				columns={TOP_COLS}
				bind:rows={wellData.tops}
				editable={true}
				emptyMessage="No formation tops. Add above, drag markers on log, or paste from Excel."
				oncellchange={() => saveCurrentTab()}
				ondeleteRow={handleDeleteRow}
			/>
		{:else if activeTab === 'contacts'}
			<CompactDataGrid
				columns={CONTACT_COLS}
				bind:rows={wellData.contacts}
				editable={true}
				emptyMessage="No fluid contacts (OWC/GOC/FWL)."
				oncellchange={() => saveCurrentTab()}
				ondeleteRow={handleDeleteRow}
			/>
		{:else if activeTab === 'core'}
			<CompactDataGrid
				columns={CORE_COLS}
				bind:rows={wellData.coreSamples}
				editable={true}
				emptyMessage="No core plugs loaded."
				oncellchange={() => saveCurrentTab()}
				ondeleteRow={handleDeleteRow}
			/>
		{:else if activeTab === 'pressure'}
			<CompactDataGrid
				columns={PRESSURE_COLS}
				bind:rows={wellData.pressureTests}
				editable={true}
				emptyMessage="No formation pressure tests."
				oncellchange={() => saveCurrentTab()}
				ondeleteRow={handleDeleteRow}
			/>
		{:else}
			<CompactDataGrid
				columns={SURVEY_COLS}
				bind:rows={wellData.survey}
				editable={true}
				emptyMessage="No directional survey rows."
				oncellchange={() => saveCurrentTab()}
				ondeleteRow={handleDeleteRow}
			/>
		{/if}
	</div>

	<!-- Action Footer -->
	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<span class="truncate text-[11px] text-[var(--text-muted)]">
			{statusMsg ?? 'Double-click cells to edit or Ctrl+V from Excel'}
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

