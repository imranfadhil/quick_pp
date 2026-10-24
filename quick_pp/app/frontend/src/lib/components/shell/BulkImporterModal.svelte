<script lang="ts">
	import { API_BASE } from '$lib/api/client';
	import BulkAncillaryImporter from '$lib/components/BulkAncillaryImporter.svelte';
	import { authState } from '$lib/state/auth.svelte';
	import { projectData } from '$lib/state/projectData.svelte';
	import { appStatus } from '$lib/state/status.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';

	let activeTab = $state<
		| 'las'
		| 'formation_tops'
		| 'fluid_contacts'
		| 'core_samples'
		| 'scal_pc'
		| 'pressure_tests'
		| 'well_surveys'
	>('las');

	let lasFiles = $state<FileList | null>(null);
	let inspectingLas = $state(false);
	let uploadingLas = $state(false);
	let uploadStatus = $state<string | null>(null);
	let inspectedFiles = $state<any[]>([]);
	let expandedFileIdx = $state<number | null>(0);
	let spliceOverlapMode = $state<'keep_last' | 'keep_first' | 'mean'>('keep_last');
	let customTargetStep = $state<string>('');

	const MAX_DEMO_LAS_BYTES = 5 * 1024 * 1024; // 5 MB in-memory cap per file in demo mode

	const STANDARD_MNEMONICS = [
		'DEPTH',
		'GR',
		'RT',
		'RMED',
		'RSHA',
		'RXO',
		'RHOB',
		'NPHI',
		'DTC',
		'DTS',
		'PEF',
		'CALI',
		'BS',
		'SP',
		'DRHO',
		'ROP',
		'TVD',
		'TVDSS',
		'PHIT',
		'PHIE',
		'SW',
		'VSH',
		'VCLAY',
		'PERM'
	];

	async function handleFileSelect(e: Event) {
		const input = e.currentTarget as HTMLInputElement;
		lasFiles = input.files;
		uploadStatus = null;
		if (lasFiles && lasFiles.length > 0) {
			await handleInspectLogs();
		} else {
			inspectedFiles = [];
		}
	}

	async function handleInspectLogs() {
		if (!lasFiles || lasFiles.length === 0) return;
		if (authState.isDemoMode) {
			const fileArr = Array.from(lasFiles);
			const oversized = fileArr.filter((f) => f.size > MAX_DEMO_LAS_BYTES);
			if (oversized.length > 0) {
				uploadStatus = `Error: File exceeds 5 MB Live Demo in-memory cap (${oversized.map((f) => f.name).join(', ')}).`;
				return;
			}
		}

		inspectingLas = true;
		uploadStatus = null;
		try {
			const formData = new FormData();
			for (const f of Array.from(lasFiles)) {
				formData.append('files', f);
			}
			const res = await fetch(`${API_BASE}/quick_pp/import/logs/inspect`, {
				method: 'POST',
				body: formData
			});
			if (!res.ok) {
				const errText = await res.text();
				throw new Error(errText || `HTTP ${res.status}`);
			}
			const data = await res.json();
			inspectedFiles = (data.files || []).map((item: any) => {
				const selectedFrameObj =
					(item.dlis_frames || []).find((f: any) => f.selected) || item.dlis_frames?.[0] || null;
				return {
					...item,
					selectedFrame: selectedFrameObj ? selectedFrameObj.frame_name : null,
					curveOverrides: Object.fromEntries(
						(item.curves || []).map((c: any) => [
							c.raw_mnemonic,
							c.target_mnemonic || c.mapped_mnemonic || c.raw_mnemonic
						])
					),
					unitOverrides: Object.fromEntries(
						(item.curves || []).map((c: any) => [c.raw_mnemonic, c.raw_unit || c.source_unit || ''])
					)
				};
			});
			expandedFileIdx = inspectedFiles.length > 0 ? 0 : null;
		} catch (e: any) {
			uploadStatus = `Inspect Error: ${e?.message || 'Failed to inspect log files'}`;
		} finally {
			inspectingLas = false;
		}
	}

	async function handleCommitLogs() {
		if (!lasFiles || lasFiles.length === 0 || !workbench.projectId) return;
		if (authState.isDemoMode) {
			await handleQuickUpload();
			return;
		}

		uploadingLas = true;
		uploadStatus = null;
		const taskId = appStatus.startTask(`Importing ${lasFiles.length} log file(s)...`);
		try {
			const configs = inspectedFiles.map((f) => ({
				file_id: f.file_id,
				filename: f.filename,
				metadata_overrides: f.metadata,
				mnemonic_map: f.curveOverrides,
				unit_map: f.unitOverrides,
				selected_dlis_frames: f.selectedFrame ? [f.selectedFrame] : null
			}));

			const commitBody = {
				project_id: Number(workbench.projectId),
				target_depth_uom: 'm',
				regularize_grid: Boolean(customTargetStep) || inspectedFiles.length > 1,
				regularize_step: customTargetStep ? Number(customTargetStep) : 0.1524,
				splicing_mode: spliceOverlapMode,
				files: configs
			};

			const res = await fetch(`${API_BASE}/quick_pp/import/logs/commit`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(commitBody)
			});
			if (!res.ok) {
				const errText = await res.text();
				throw new Error(errText || `HTTP ${res.status}`);
			}
			const result = await res.json();
			const committedList = result.wells || result.wells_imported || [];
			const preferredWell = committedList[0]?.well_name || workbench.wellId;
			await projectData.selectProject(workbench.projectId, preferredWell);
			const wellCount = committedList.length || lasFiles.length;
			uploadStatus = `Committed ${wellCount} well(s) to project (${committedList.map((w: any) => w.well_name).join(', ')}).`;
			appStatus.endTask(taskId, {
				level: 'success',
				message: `Imported ${wellCount} well(s)`
			});
			lasFiles = null;
			inspectedFiles = [];
		} catch (e: any) {
			uploadStatus = `Error: ${e?.message || 'Failed to commit log files'}`;
			appStatus.endTask(taskId, {
				level: 'error',
				message: uploadStatus
			});
		} finally {
			uploadingLas = false;
		}
	}

	async function handleQuickUpload() {
		if (!lasFiles || lasFiles.length === 0) return;
		if (authState.isDemoMode) {
			const fileArr = Array.from(lasFiles);
			const oversized = fileArr.filter((f) => f.size > MAX_DEMO_LAS_BYTES);
			if (oversized.length > 0) {
				uploadStatus = `Error: File exceeds 5 MB Live Demo in-memory cap (${oversized.map((f) => f.name).join(', ')}).`;
				return;
			}
		}
		uploadingLas = true;
		uploadStatus = null;
		try {
			await projectData.uploadProjectFiles(lasFiles);
			uploadStatus = authState.isDemoMode
				? `Loaded ${lasFiles.length} file(s) into ephemeral session (max 5 MB cap; pre-seeded Volve/Costa DB remains read-only).`
				: `Uploaded and saved ${lasFiles.length} file(s) to project.`;
			lasFiles = null;
			inspectedFiles = [];
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
			class="flex max-h-[88vh] w-[960px] flex-col overflow-hidden rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] shadow-xl"
		>
			<!-- Modal Header -->
			<div
				class="flex h-9 items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-3"
			>
				<div class="flex items-center gap-2">
					<span class="text-xs font-bold text-[var(--text-primary)]">
						Import Well Logs &amp; Ancillary Data
					</span>
					{#if projectData.summary}
						<span class="rounded-xs bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700">
							Project: {projectData.summary.name}
						</span>
					{/if}
					{#if authState.isDemoMode}
						<span
							class="rounded-xs border border-amber-200 bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800"
						>
							Ephemeral Demo • Max 5 MB / Read-Only Seed DB
						</span>
					{/if}
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
					['las', 'LAS / DLIS / Parquet Logs'],
					['formation_tops', 'Formation Tops'],
					['fluid_contacts', 'Fluid Contacts'],
					['core_samples', 'RCA Core Plugs'],
					['scal_pc', 'SCAL (Pc)'],
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
				{#if authState.isDemoMode}
					<div
						class="mb-3 rounded-sm border border-blue-200 bg-blue-50/70 px-3 py-2 text-[11px] text-blue-900"
					>
						<strong>Ephemeral Session Guardrail:</strong> Uploads are capped at <strong>5 MB</strong>
						in memory so visitors can test custom <code>.las</code>/<code>.dlis</code>/<code>.csv</code>
						files without overwriting the pre-seeded Volve &amp; Costa multi-well projects.
					</div>
				{/if}

				{#if activeTab === 'las'}
					<div class="space-y-3">
						<!-- Step 1: Dropzone / File Input -->
						<div class="flex flex-wrap items-center justify-between gap-2">
							<p class="text-xs text-[var(--text-secondary)]">
								Upload one or more <code>.las</code> (LAS 2.0 / 3.0), <code>.dlis</code>,
								<code>.csv</code>, or <code>.parquet</code> well log files for pre-import QA/QC inspection,
								mnemonic mapping, and multi-run splicing.
							</p>
						</div>

						<div class="flex items-center gap-2">
							<input
								type="file"
								multiple
								accept=".las,.LAS,.dlis,.DLIS,.csv,.parquet"
								onchange={handleFileSelect}
								class="block flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-xs"
							/>
							{#if lasFiles && lasFiles.length > 0}
								<button
									type="button"
									disabled={inspectingLas || uploadingLas}
									onclick={handleInspectLogs}
									class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-xs font-medium text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-50"
								>
									{inspectingLas ? 'Inspecting...' : 'Re-Inspect'}
								</button>
							{/if}
						</div>

						<!-- Step 2: Two-Pass Review & Map Panel -->
						{#if inspectedFiles.length > 0}
							<div
								class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-2.5 space-y-2.5"
							>
								<div class="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-hairline)] pb-2">
									<div class="flex items-center gap-2">
										<span class="text-xs font-bold text-[var(--text-primary)]">
											Step 2: Review Well Metadata, Curve Mappings &amp; QA/QC ({inspectedFiles.length} file{inspectedFiles.length > 1 ? 's' : ''})
										</span>
									</div>
									<div class="flex items-center gap-3 text-[11px]">
										<label class="flex items-center gap-1 text-[var(--text-secondary)]">
											<span>Multi-Run Overlap:</span>
											<select
												bind:value={spliceOverlapMode}
												class="h-6 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
											>
												<option value="keep_last">Keep Last (Deepest Run)</option>
												<option value="keep_first">Keep First (Shallowest Run)</option>
												<option value="mean">Average Overlap</option>
											</select>
										</label>
										<label class="flex items-center gap-1 text-[var(--text-secondary)]">
											<span>Step (m):</span>
											<input
												type="number"
												step="0.05"
												placeholder="Auto (0.1524)"
												bind:value={customTargetStep}
												class="h-6 w-24 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-1.5 text-[11px]"
											/>
										</label>
									</div>
								</div>

								{#each inspectedFiles as fileItem, idx}
									<div class="rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-surface)]">
										<!-- File Summary Row -->
										<div
											class="flex items-center justify-between bg-[var(--bg-subtle)] px-2.5 py-1.5 text-xs"
										>
											<div class="flex flex-wrap items-center gap-2">
												<button
													type="button"
													onclick={() => (expandedFileIdx = expandedFileIdx === idx ? null : idx)}
													class="font-bold text-[var(--text-primary)] hover:underline"
												>
													{expandedFileIdx === idx ? '▼' : '▶'} {fileItem.filename}
												</button>
												<span class="rounded-xs bg-slate-200 px-1.5 py-0.5 text-[10px] font-semibold uppercase">
													{fileItem.format || fileItem.file_format || 'LAS'}
												</span>
												{#if fileItem.metadata}
													<span class="text-[11px] text-[var(--text-secondary)]">
														Well: <strong>{fileItem.metadata.well_name}</strong> (UWI: {fileItem.metadata.uwi})
													</span>
													<span class="text-[11px] text-[var(--text-muted)]">
														Depth: {fileItem.metadata.start_depth ?? '?'} – {fileItem.metadata.stop_depth ?? '?'} {fileItem.metadata.datum_unit || 'm'}
														(step: {fileItem.metadata.step ?? '?'})
													</span>
												{/if}
												{#if (fileItem.qaqc_warnings?.length || 0) > 0}
													<span class="rounded-xs bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800">
														⚠ {fileItem.qaqc_warnings.length} Warning{fileItem.qaqc_warnings.length > 1 ? 's' : ''}
													</span>
												{/if}
											</div>

											{#if fileItem.dlis_frames && fileItem.dlis_frames.length > 0}
												<div class="flex items-center gap-1 text-[11px]">
													<span class="font-medium text-[var(--text-secondary)]">DLIS Frame:</span>
													<select
														bind:value={fileItem.selectedFrame}
														class="h-6 rounded-xs border border-[var(--border-hairline)] bg-white px-1.5 text-[11px]"
													>
														{#each fileItem.dlis_frames as fr}
															<option value={fr.frame_name}>
																{fr.frame_name} ({fr.start_depth?.toFixed(1) ?? '?'}–{fr.stop_depth?.toFixed(1) ?? '?'} {fr.depth_unit}, {fr.scalar_channels?.length ?? 0} ch)
															</option>
														{/each}
													</select>
												</div>
											{/if}
										</div>

										{#if expandedFileIdx === idx}
											<div class="p-2.5 space-y-2.5">
												<!-- Warnings Banner -->
												{#if fileItem.qaqc_warnings && fileItem.qaqc_warnings.length > 0}
													<div class="rounded-xs border border-amber-200 bg-amber-50 px-2.5 py-1.5 text-[11px] text-amber-900">
														{#each fileItem.qaqc_warnings as warn}
															<div>• {warn}</div>
														{/each}
													</div>
												{/if}

												<!-- Editable Well Header Metadata Grid -->
												{#if fileItem.metadata}
													<div class="grid grid-cols-4 gap-2 text-[11px]">
														<label class="flex flex-col gap-0.5">
															<span class="text-[10px] font-semibold text-[var(--text-secondary)]">Well Name</span>
															<input
																type="text"
																bind:value={fileItem.metadata.well_name}
																class="h-6 rounded-xs border border-[var(--border-hairline)] px-1.5"
															/>
														</label>
														<label class="flex flex-col gap-0.5">
															<span class="text-[10px] font-semibold text-[var(--text-secondary)]">UWI</span>
															<input
																type="text"
																bind:value={fileItem.metadata.uwi}
																class="h-6 rounded-xs border border-[var(--border-hairline)] px-1.5"
															/>
														</label>
														<label class="flex flex-col gap-0.5">
															<span class="text-[10px] font-semibold text-[var(--text-secondary)]">Depth Unit</span>
															<select
																bind:value={fileItem.metadata.datum_unit}
																class="h-6 rounded-xs border border-[var(--border-hairline)] bg-white px-1.5"
															>
																<option value="m">m (Meters)</option>
																<option value="ft">ft (Feet → Auto-Convert to m)</option>
															</select>
														</label>
														<label class="flex flex-col gap-0.5">
															<span class="text-[10px] font-semibold text-[var(--text-secondary)]">KB Elevation (m)</span>
															<input
																type="number"
																step="0.1"
																bind:value={fileItem.metadata.kb_elev}
																class="h-6 rounded-xs border border-[var(--border-hairline)] px-1.5"
															/>
														</label>
														<label class="flex flex-col gap-0.5">
															<span class="text-[10px] font-semibold text-[var(--text-secondary)]">GL / Mudline Elev (m)</span>
															<input
																type="number"
																step="0.1"
																bind:value={fileItem.metadata.gl_elev}
																class="h-6 rounded-xs border border-[var(--border-hairline)] px-1.5"
															/>
														</label>
														<label class="flex flex-col gap-0.5">
															<span class="text-[10px] font-semibold text-[var(--text-secondary)]">BHT ({fileItem.metadata.bht_unit || 'degC'})</span>
															<input
																type="number"
																step="0.1"
																bind:value={fileItem.metadata.bht}
																class="h-6 rounded-xs border border-[var(--border-hairline)] px-1.5"
															/>
														</label>
														<label class="flex flex-col gap-0.5">
															<span class="text-[10px] font-semibold text-[var(--text-secondary)]">Rmf (ohm.m)</span>
															<input
																type="number"
																step="0.001"
																bind:value={fileItem.metadata.rmf}
																class="h-6 rounded-xs border border-[var(--border-hairline)] px-1.5"
															/>
														</label>
														<label class="flex flex-col gap-0.5">
															<span class="text-[10px] font-semibold text-[var(--text-secondary)]">Field / Operator</span>
															<input
																type="text"
																bind:value={fileItem.metadata.field}
																placeholder="Field"
																class="h-6 rounded-xs border border-[var(--border-hairline)] px-1.5"
															/>
														</label>
													</div>
												{/if}

												<!-- Curve Mapping Matrix -->
												{#if fileItem.curves && fileItem.curves.length > 0}
													<div class="max-h-52 overflow-y-auto border border-[var(--border-hairline)] rounded-xs">
														<table class="w-full border-collapse text-left text-[11px]">
															<thead class="sticky top-0 bg-[var(--bg-subtle)] text-[10px] font-semibold text-[var(--text-secondary)]">
																<tr>
																	<th class="px-2 py-1">Raw Curve</th>
																	<th class="px-2 py-1">Target Mnemonic</th>
																	<th class="px-2 py-1">Source Unit</th>
																	<th class="px-2 py-1">Target Unit</th>
																	<th class="px-2 py-1">Description</th>
																	<th class="px-2 py-1">Status</th>
																</tr>
															</thead>
															<tbody class="divide-y divide-[var(--border-hairline)]">
																{#each fileItem.curves as c}
																	<tr class="hover:bg-slate-50">
																		<td class="px-2 py-1 font-mono font-semibold">{c.raw_mnemonic}</td>
																		<td class="px-2 py-1">
																			<input
																				type="text"
																				list="standard-mnemonics-list"
																				bind:value={fileItem.curveOverrides[c.raw_mnemonic]}
																				class="h-5 w-28 rounded-xs border border-[var(--border-hairline)] px-1 font-mono text-[11px]"
																			/>
																		</td>
																		<td class="px-2 py-1">
																			<input
																				type="text"
																				bind:value={fileItem.unitOverrides[c.raw_mnemonic]}
																				class="h-5 w-20 rounded-xs border border-[var(--border-hairline)] px-1 font-mono text-[11px]"
																			/>
																		</td>
																		<td class="px-2 py-1 font-mono">
																			{c.target_unit || c.raw_unit || '-'}
																			{#if c.auto_conversion}
																				<span class="ml-1 rounded-xs bg-emerald-100 px-1 py-0.5 text-[9px] font-semibold text-emerald-800">
																					{c.auto_conversion}
																				</span>
																			{/if}
																		</td>
																		<td class="px-2 py-1 text-[10px] text-[var(--text-secondary)] truncate max-w-[180px]">
																			{c.description || '-'}
																		</td>
																		<td class="px-2 py-1">
																			{#if c.is_array_channel}
																				<span class="rounded-xs bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-600">
																					Array ({c.dimensions?.join('×')})
																				</span>
																			{:else if c.warnings && c.warnings.length > 0}
																				<span
																					title={c.warnings.join('; ')}
																					class="rounded-xs bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800"
																				>
																					QA/QC Warn
																				</span>
																			{:else}
																				<span class="rounded-xs bg-emerald-50 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700">
																					Ready
																				</span>
																			{/if}
																		</td>
																	</tr>
																{/each}
															</tbody>
														</table>
													</div>
												{/if}
											</div>
										{/if}
									</div>
								{/each}

								<datalist id="standard-mnemonics-list">
									{#each STANDARD_MNEMONICS as stdMnem}
										<option value={stdMnem}></option>
									{/each}
								</datalist>
							</div>
						{/if}

						<!-- Step 3: Commit & Action Buttons -->
						<div class="flex items-center gap-2">
							<button
								type="button"
								disabled={!lasFiles || uploadingLas || inspectingLas || !workbench.projectId}
								onclick={handleCommitLogs}
								class="h-7 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white disabled:opacity-50"
							>
								{#if uploadingLas}
									{authState.isDemoMode ? 'Loading in Memory...' : 'Committing & Standardizing...'}
								{:else if inspectedFiles.length > 0}
									Commit &amp; Import {inspectedFiles.length} Well File{inspectedFiles.length > 1 ? 's' : ''}
								{:else}
									{authState.isDemoMode ? 'Upload Log Files (Max 5 MB)' : 'Upload Log Files'}
								{/if}
							</button>

							{#if lasFiles && lasFiles.length > 0 && !authState.isDemoMode}
								<button
									type="button"
									disabled={uploadingLas}
									onclick={handleQuickUpload}
									class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2.5 text-xs font-medium text-[var(--text-secondary)] hover:bg-slate-200 disabled:opacity-50"
								>
									Fast 1-Click Upload
								</button>
							{/if}

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
