<script lang="ts">
	import BulkAncillaryImporter from '$lib/components/BulkAncillaryImporter.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	let activeTab = $state<
		'las' | 'formation_tops' | 'fluid_contacts' | 'core_samples' | 'pressure_tests' | 'well_surveys'
	>('las');
	let lasFiles = $state<FileList | null>(null);
	let uploadingLas = $state(false);
	let uploadStatus = $state<string | null>(null);
	const MAX_DEMO_LAS_BYTES = 5 * 1024 * 1024; // 5 MB in-memory cap per file

	async function handleLasUpload() {
		if (!lasFiles || lasFiles.length === 0) return;
		const fileArr = Array.from(lasFiles);
		const oversized = fileArr.filter((f) => f.size > MAX_DEMO_LAS_BYTES);
		if (oversized.length > 0) {
			uploadStatus = `Error: File exceeds 5 MB Live Demo in-memory cap (${oversized.map((f) => f.name).join(', ')}).`;
			return;
		}
		uploadingLas = true;
		uploadStatus = null;
		try {
			await projectData.uploadProjectFiles(lasFiles);
			uploadStatus = `Loaded ${lasFiles.length} file(s) into ephemeral session (max 5 MB cap; pre-seeded Volve/Costa DB remains read-only).`;
			lasFiles = null;
		} catch (e: any) {
			uploadStatus = `Error: ${e?.message || 'Failed to upload files'}`;
		} finally {
			uploadingLas = false;
		}
	}
</script>

{#if workbench.importerOpen}
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		onclick={() => (workbench.importerOpen = false)}
		class="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-[1px]"
	>
		<div
			onclick={(e) => e.stopPropagation()}
			class="flex max-h-[85vh] w-[820px] flex-col overflow-hidden rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] shadow-xl"
		>
			<!-- Modal Header -->
			<div
				class="flex h-9 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-3"
			>
				<div class="flex items-center gap-2">
					<span class="text-xs font-bold text-[var(--text-primary)]">
						Import Well Logs & Ancillary Data
					</span>
					{#if projectData.summary}
						<span class="rounded-xs bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700">
							Project: {projectData.summary.name}
						</span>
					{/if}
					<span class="rounded-xs border border-amber-200 bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800">
						Ephemeral Demo • Max 5 MB / Read-Only Seed DB
					</span>
				</div>
				<button
					type="button"
					onclick={() => (workbench.importerOpen = false)}
					class="flex h-6 w-6 items-center justify-center rounded-sm text-sm text-[var(--text-muted)] hover:bg-slate-200 hover:text-[var(--text-primary)]"
				>
					×
				</button>
			</div>

			<!-- Modal Sub-Tabs -->
			<div
				class="flex h-8 shrink-0 items-center gap-1 border-b border-[var(--border-hairline)] bg-[var(--bg-app)] px-3"
			>
				{#each [
					['las', 'LAS / Parquet Logs'],
					['formation_tops', 'Formation Tops'],
					['fluid_contacts', 'Fluid Contacts'],
					['core_samples', 'Core Samples'],
					['pressure_tests', 'Pressure Tests'],
					['well_surveys', 'Deviation Survey']
				] as [tabKey, label]}
					<button
						type="button"
						onclick={() => (activeTab = tabKey as any)}
						class="h-6 rounded-sm px-2 text-xs font-medium transition-colors {activeTab === tabKey
							? 'bg-[var(--accent-primary)] text-white'
							: 'text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
					>
						{label}
					</button>
				{/each}
			</div>

			<!-- Modal Body -->
			<div class="flex-1 overflow-y-auto p-4">
				<div class="mb-3 rounded-sm border border-blue-200 bg-blue-50/70 px-3 py-2 text-[11px] text-blue-900">
					<strong>Ephemeral Session Guardrail:</strong> Uploads are capped at <strong>5 MB</strong> in memory so visitors can test custom <code>.las</code>/<code>.csv</code> files without overwriting the pre-seeded Volve &amp; Costa multi-well projects.
				</div>

				{#if activeTab === 'las'}
					<div class="space-y-3">
						<p class="text-xs text-[var(--text-secondary)]">
							Upload one or more <code>.las</code>, <code>.csv</code>, or <code>.parquet</code> well log files (max 5 MB per file) into the ephemeral session.
						</p>
						<input
							type="file"
							multiple
							accept=".las,.LAS,.csv,.parquet"
							onchange={(e) => (lasFiles = (e.currentTarget as HTMLInputElement).files)}
							class="block w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2 text-xs"
						/>
						<div class="flex items-center gap-2">
							<button
								type="button"
								disabled={!lasFiles || uploadingLas || !workbench.projectId}
								onclick={handleLasUpload}
								class="h-7 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white disabled:opacity-50"
							>
								{uploadingLas ? 'Loading in Memory...' : 'Upload Log Files (Max 5 MB)'}
							</button>
							{#if uploadStatus}
								<span class="text-xs font-medium text-[var(--text-secondary)]">
									{uploadStatus}
								</span>
							{/if}
						</div>
					</div>
				{:else if workbench.projectId}
					<BulkAncillaryImporter projectId={workbench.projectId} type={activeTab} />
				{:else}
					<p class="text-xs text-[var(--text-muted)]">
						Please create or select a project first.
					</p>
				{/if}
			</div>
		</div>
	</div>
{/if}
