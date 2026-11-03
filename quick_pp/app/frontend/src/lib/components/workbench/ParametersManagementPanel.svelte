<script lang="ts">
	import {
		moduleParams,
		PARAM_MODULE_LABELS,
		currentSelection,
		selectionLabel,
		type ParamModuleKey,
		selectionKey
	} from '$lib/state/moduleParams.svelte';
	import {
		projectParams,
		PROJECT_PARAM_MODULE_LABELS,
		zoneSelectionKey,
		zoneSelectionLabel,
		type ProjectParamModuleKey
	} from '$lib/state/projectParams.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { appStatus } from '$lib/state/status.svelte';

	// Filters
	let filterScope = $state<'active' | 'all'>('active');
	let filterZone = $state<'active' | 'all'>('all');
	let isDeletingContacts = $state(false);
	let confirmDeleteContacts = $state(false);

	let isMultiWell = $derived(workbench.scope === 'multi-well');
	let activeSingleModule = $derived(workbench.singleWellStep as ParamModuleKey);
	let activeMultiModule = $derived(workbench.multiWellStep as ProjectParamModuleKey);

	let activeStepLabel = $derived(
		isMultiWell
			? PROJECT_PARAM_MODULE_LABELS[activeMultiModule] || workbench.multiWellStep
			: PARAM_MODULE_LABELS[activeSingleModule] || workbench.singleWellStep
	);

	let activeZoneLbl = $derived(
		isMultiWell
			? zoneSelectionLabel(workbench.selectedZones)
			: selectionLabel(currentSelection())
	);

	// Count auto-generated contacts across wells
	let autoContactNames = new Set(['GUT', 'GDT', 'OUT', 'ODT', 'WUT']);
	let autoContactsInfo = $derived.by(() => {
		let totalAuto = 0;
		let totalAll = 0;
		const wellsAffected = new Set<string>();

		for (const w of projectData.wells) {
			const contacts = (w.fluid_contacts ?? []) as any[];
			for (const c of contacts) {
				totalAll++;
				const cName = c.name || c.type || '';
				const src = String(c.source || '').toLowerCase();
				const isAuto =
					autoContactNames.has(cName) ||
					src.includes('log evaluation') ||
					src.includes('calculated') ||
					src.includes('optimizer') ||
					src.includes('fpt');
				if (isAuto) {
					totalAuto++;
					wellsAffected.add(w.name);
				}
			}
		}
		// Also check active wellData contacts
		for (const c of (wellData.contacts as any[])) {
			const cName = c.name || c.type || '';
			const src = String(c.source || '').toLowerCase();
			if (
				autoContactNames.has(cName) ||
				src.includes('log evaluation') ||
				src.includes('calculated') ||
				src.includes('optimizer') ||
				src.includes('fpt')
			) {
				wellsAffected.add(wellData.header?.well_name || workbench.wellId || 'Active Well');
			}
		}

		return {
			autoCount: totalAuto,
			totalCount: totalAll,
			wellsCount: wellsAffected.size
		};
	});

	function formatParamSummary(params: Record<string, any>): string {
		if (!params || typeof params !== 'object') return '—';
		const parts: string[] = [];

		if (params.method) parts.push(`Method: ${params.method}`);
		if (params.model) parts.push(`Model: ${params.model}`);
		if (params.modelType) parts.push(`Model: ${params.modelType}`);
		if (params.litho?.model) parts.push(`Litho: ${params.litho.model.toUpperCase()}`);
		if (params.litho?.porosityMethod) parts.push(`Poro: ${params.litho.porosityMethod}`);
		if (params.m !== undefined) parts.push(`m=${params.m}`);
		if (params.n !== undefined) parts.push(`n=${params.n}`);
		if (params.a !== undefined) parts.push(`a=${params.a}`);
		if (params.rwVal !== undefined) parts.push(`Rw=${params.rwVal}`);
		if (params.caliThreshold !== undefined) parts.push(`Cali Thresh=${params.caliThreshold}"`);
		if (params.num_clusters !== undefined) parts.push(`${params.num_clusters} DRTs`);
		if (params.cutoffs && Array.isArray(params.cutoffs)) {
			parts.push(`FZI: [${params.cutoffs.join(', ')}]`);
		}
		if (params.referencePcPsi !== undefined) parts.push(`Ref Pc=${params.referencePcPsi} psi`);
		if (params.fwl !== undefined) parts.push(`FWL=${params.fwl}m`);
		if (params.classifier) parts.push(`Classifier: ${params.classifier}`);
		if (params.vclCutoff !== undefined) parts.push(`Vcl<${params.vclCutoff}`);
		if (params.phieCutoff !== undefined) parts.push(`Φe>${params.phieCutoff}`);
		if (params.swCutoff !== undefined) parts.push(`Sw<${params.swCutoff}`);

		if (parts.length === 0) {
			const keys = Object.keys(params).slice(0, 3);
			return keys.map((k) => `${k}=${params[k]}`).join(', ');
		}
		return parts.slice(0, 4).join(' · ');
	}

	function fmtSavedAt(iso: string): string {
		if (!iso) return '—';
		const d = new Date(iso);
		if (Number.isNaN(d.getTime())) return '—';
		return d.toLocaleString(undefined, {
			month: 'short',
			day: '2-digit',
			hour: '2-digit',
			minute: '2-digit'
		});
	}

	// Table entries
	let tableEntries = $derived.by(() => {
		const curZKey = isMultiWell ? zoneSelectionKey() : selectionKey(currentSelection());

		if (isMultiWell) {
			const all = projectParams.getAllSavedEntries(projectData.projectModuleParams);
			return all.filter((e) => {
				if (filterScope === 'active' && e.module !== activeMultiModule) return false;
				if (filterZone === 'active' && e.key !== curZKey) return false;
				return true;
			});
		} else {
			const all = moduleParams.getAllSavedEntries(wellData.config);
			return all.filter((e) => {
				if (filterScope === 'active' && e.module !== activeSingleModule) return false;
				if (filterZone === 'active' && e.key !== curZKey) return false;
				return true;
			});
		}
	});

	// Reset actions
	function handleResetActiveToDefaults() {
		if (isMultiWell) {
			switch (activeMultiModule) {
				case 'rock-typing':
				case 'rt-classifier':
					projectData.resetRockTyping();
					break;
				case 'swirr-corr':
					projectData.resetScalFits();
					break;
				case 'shf':
					projectData.resetShf();
					break;
				case 'perm-transform':
					projectData.permTransformResult = null;
					if (workbench.projectId) {
						projectParams.deleteAllForModule(workbench.projectId, 'perm-transform', projectData.projectModuleParams);
					}
					appStatus.success('Perm transform reset to regional defaults');
					break;
				default:
					appStatus.info(`Regional defaults applied for ${activeStepLabel}`);
					break;
			}
		} else {
			moduleParams.resetToRegionalDefaults(activeSingleModule);
			appStatus.success(`Reset ${activeStepLabel} parameters to regional defaults`);
		}
	}

	async function handleDeletePreset(entry: any) {
		const pid = workbench.projectId;
		if (!pid) return;

		if (isMultiWell) {
			const ok = await projectParams.deleteSavedEntry(
				pid,
				entry.module,
				entry.key,
				projectData.projectModuleParams
			);
			if (ok) appStatus.success(`Deleted preset for ${entry.moduleLabel} (${entry.label})`);
		} else {
			const wName = wellData.header?.well_name || workbench.wellId || '';
			const ok = await moduleParams.deleteSavedEntry(
				pid,
				wName,
				entry.module,
				entry.key,
				wellData.config
			);
			if (ok) appStatus.success(`Deleted preset for ${entry.moduleLabel} (${entry.label})`);
		}
	}

	async function handleExecuteDeleteAutoContacts() {
		isDeletingContacts = true;
		try {
			await projectData.deleteAutoContacts();
			confirmDeleteContacts = false;
		} finally {
			isDeletingContacts = false;
		}
	}
</script>

<div class="flex h-full flex-col bg-[var(--bg-surface)] text-[11px]">
	<!-- Action & Filter Bar -->
	<div
		class="flex shrink-0 flex-wrap items-center justify-between gap-1.5 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 py-1.5"
	>
		<div class="flex items-center gap-2">
			<!-- Scope filter toggle -->
			<div class="flex h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-0.5 text-[10px]">
				<button
					type="button"
					onclick={() => (filterScope = 'active')}
					class="rounded-2xs px-2 font-medium transition-colors {filterScope === 'active'
						? 'bg-[var(--accent-primary)] text-white shadow-2xs'
						: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
				>
					Current Step ({activeStepLabel})
				</button>
				<button
					type="button"
					onclick={() => (filterScope = 'all')}
					class="rounded-2xs px-2 font-medium transition-colors {filterScope === 'all'
						? 'bg-[var(--accent-primary)] text-white shadow-2xs'
						: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
				>
					All Steps
				</button>
			</div>

			<!-- Zone filter toggle -->
			<div class="flex h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] p-0.5 text-[10px]">
				<button
					type="button"
					onclick={() => (filterZone = 'all')}
					class="rounded-2xs px-2 font-medium transition-colors {filterZone === 'all'
						? 'bg-[var(--bg-subtle)] font-semibold text-[var(--text-primary)]'
						: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
				>
					All Zones
				</button>
				<button
					type="button"
					onclick={() => (filterZone = 'active')}
					class="rounded-2xs px-2 font-medium transition-colors {filterZone === 'active'
						? 'bg-[var(--bg-subtle)] font-semibold text-[var(--text-primary)]'
						: 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'}"
				>
					{activeZoneLbl}
				</button>
			</div>
		</div>

		<!-- Action buttons -->
		<div class="flex items-center gap-1.5">
			<button
				type="button"
				onclick={handleResetActiveToDefaults}
				title="Revert active step parameters to standard regional clastic/sandstone presets"
				class="flex h-6 items-center gap-1 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-[10.5px] font-medium text-[var(--text-primary)] hover:border-amber-500 hover:text-amber-700 shadow-2xs transition-colors"
			>
				<span>⟲</span>
				<span>Reset {activeStepLabel} to Defaults</span>
			</button>

			{#if isMultiWell && (activeMultiModule === 'rock-typing' || activeMultiModule === 'rt-classifier')}
				<button
					type="button"
					onclick={() => projectData.resetRockTyping()}
					title="Reset rock typing DRT cutoffs to 3 clusters and purge trained classifier"
					class="flex h-6 items-center gap-1 rounded-xs bg-amber-500/10 px-2 text-[10.5px] font-medium text-amber-700 border border-amber-500/30 hover:bg-amber-500/20 transition-colors"
				>
					<span>Reset All Rock Types</span>
				</button>
			{/if}

			{#if isMultiWell && (activeMultiModule === 'swirr-corr' || activeMultiModule === 'shf')}
				<button
					type="button"
					onclick={() => projectData.resetScalFits()}
					title="Clear SCAL Swirr correlations, Leverett-J fits, and SWOF deck"
					class="flex h-6 items-center gap-1 rounded-xs bg-amber-500/10 px-2 text-[10.5px] font-medium text-amber-700 border border-amber-500/30 hover:bg-amber-500/20 transition-colors"
				>
					<span>Reset SCAL Fits</span>
				</button>
			{/if}

			{#if autoContactsInfo.autoCount > 0}
				<button
					type="button"
					onclick={() => (confirmDeleteContacts = true)}
					title="Delete auto-generated contacts (GUT, GDT, OUT, ODT, WUT, bracketed OWC/GOC/FWL). Imported contacts remain protected."
					class="flex h-6 items-center gap-1 rounded-xs bg-red-500/10 px-2 text-[10.5px] font-medium text-red-700 border border-red-500/30 hover:bg-red-500/20 transition-colors"
				>
					<span>✕</span>
					<span>Delete Auto Contacts ({autoContactsInfo.autoCount})</span>
				</button>
			{/if}
		</div>
	</div>

	<!-- Delete Contacts Confirmation Modal/Bar -->
	{#if confirmDeleteContacts}
		<div
			class="flex items-center justify-between border-b border-red-500/40 bg-red-500/10 px-3 py-2 text-[11px] text-red-900"
		>
			<div class="flex items-center gap-2">
				<span class="font-bold text-red-700">⚠ Delete Auto-Identified Contacts:</span>
				<span>
					Are you sure you want to delete all <strong>{autoContactsInfo.autoCount}</strong> auto-generated
					contacts across {autoContactsInfo.wellsCount} well(s)? Manually imported contacts will remain
					safe.
				</span>
			</div>
			<div class="flex items-center gap-2">
				<button
					type="button"
					onclick={() => (confirmDeleteContacts = false)}
					class="rounded-xs border border-gray-300 bg-white px-2 py-0.5 text-[10.5px] font-medium text-gray-700 hover:bg-gray-50"
				>
					Cancel
				</button>
				<button
					type="button"
					disabled={isDeletingContacts}
					onclick={handleExecuteDeleteAutoContacts}
					class="rounded-xs bg-red-600 px-2.5 py-0.5 text-[10.5px] font-bold text-white hover:bg-red-700 disabled:opacity-50"
				>
					{isDeletingContacts ? 'Deleting...' : 'Confirm Delete Auto Contacts'}
				</button>
			</div>
		</div>
	{/if}

	<!-- Table Area -->
	<div class="min-h-0 flex-1 overflow-auto">
		{#if tableEntries.length === 0}
			<div class="flex h-full flex-col items-center justify-center gap-2 p-6 text-center text-[var(--text-muted)]">
				<div class="text-2xl opacity-40">⚙</div>
				<p class="text-[12px] font-medium text-[var(--text-secondary)]">No saved parameter presets found</p>
				<p class="max-w-md text-[11px]">
					Parameters are saved automatically on Compute/Save, keyed by zone selection or depth interval.
					You can reset the active step to regional default presets using the toolbar above.
				</p>
				{#if filterScope === 'active'}
					<button
						type="button"
						onclick={() => (filterScope = 'all')}
						class="mt-1 text-[11px] font-medium text-[var(--accent-primary)] hover:underline"
					>
						Show saved parameters for all steps ↗
					</button>
				{/if}
			</div>
		{:else}
			<table class="w-full border-collapse text-left text-[11px]">
				<thead class="sticky top-0 z-10 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] uppercase font-semibold text-[var(--text-secondary)]">
					<tr>
						<th class="px-2.5 py-1.5">Step / Module</th>
						<th class="px-2.5 py-1.5">Zone / Scope</th>
						<th class="px-2.5 py-1.5">Saved Parameters Summary</th>
						<th class="px-2.5 py-1.5">Saved At</th>
						<th class="px-2.5 py-1.5 text-right">Actions</th>
					</tr>
				</thead>
				<tbody class="divide-y divide-[var(--border-hairline)] font-mono text-[10.5px]">
					{#each tableEntries as row (row.module + '::' + row.key)}
						<tr class="hover:bg-[var(--bg-subtle)]/60 transition-colors">
							<td class="whitespace-nowrap px-2.5 py-1.5 font-sans font-medium text-[var(--text-primary)]">
								<div class="flex items-center gap-1.5">
									<span class="inline-block h-2 w-2 rounded-full bg-emerald-500"></span>
									<span>{row.moduleLabel}</span>
								</div>
							</td>
							<td class="whitespace-nowrap px-2.5 py-1.5 text-[var(--text-secondary)] font-sans">
								<span class="rounded-xs bg-[var(--bg-subtle)] px-1.5 py-0.5 border border-[var(--border-hairline)]">
									{row.label}
								</span>
							</td>
							<td class="px-2.5 py-1.5 text-[var(--text-primary)] font-mono text-[10px]">
								{formatParamSummary(row.params)}
							</td>
							<td class="whitespace-nowrap px-2.5 py-1.5 text-[var(--text-muted)] text-[10px]">
								{fmtSavedAt(row.saved_at)}
							</td>
							<td class="whitespace-nowrap px-2.5 py-1.5 text-right">
								<button
									type="button"
									onclick={() => handleDeletePreset(row)}
									title="Delete this saved preset so it reverts to defaults"
									class="rounded-xs px-1.5 py-0.5 text-[10px] text-red-600 hover:bg-red-50 hover:text-red-800 transition-colors"
								>
									Delete
								</button>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
	</div>

	<!-- Footer summary -->
	<div
		class="flex h-6 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-[10px] text-[var(--text-muted)]"
	>
		<div>
			<span>{tableEntries.length} saved preset(s) displayed</span>
			{#if isMultiWell}
				<span class="ml-2">· Scope: Field / Multi-Well</span>
			{:else}
				<span class="ml-2">· Scope: Well ({wellData.header?.well_name || workbench.wellId || 'Active'})</span>
			{/if}
		</div>
		<div>
			<span>Regional Presets: Clastic / Sandstone standard ($m=2.0, n=2.0, \Phi_{`N`}/\rho_B$ SSC endpoints)</span>
		</div>
	</div>
</div>
