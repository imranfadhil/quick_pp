<script lang="ts">
	import { api } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { projectData, type DrtMethod } from '$lib/state/projectData.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { getRockTypeColor } from '$lib/utils/petrophysicsColors';

	const DRT_METHOD_META: Record<
		DrtMethod,
		{
			label: string;
			shortName: string;
			badge: string;
			cutoffTitle: string;
			cutoffLabel: string;
			addLabel: string;
			computeLabel: string;
			unit: string;
			min: number;
			max: number;
			step: number;
			formula: string;
			description: string;
			higherIsBetter: boolean;
		}
	> = {
		fzi: {
			label: 'FZI / RQI',
			shortName: 'FZI',
			badge: 'Amaefule (1993)',
			cutoffTitle: 'FZI / RQI Boundary Cutoffs',
			cutoffLabel: 'FZI Cutoff',
			addLabel: '+ Add FZI Boundary',
			computeLabel: 'Compute FZI (⌘+Enter)',
			unit: '',
			min: 0.01,
			max: 50.0,
			step: 0.05,
			formula: 'FZI = 0.0314·√(K/Φ) / [Φ/(1−Φ)]',
			description:
				'Kozeny-Carman Hydraulic Flow Units (HFU). Higher FZI indicates larger hydraulic radius and better reservoir quality (DRT 1 = highest FZI).',
			higherIsBetter: true
		},
		winland: {
			label: 'Winland R35',
			shortName: 'R35',
			badge: 'Winland (1972)',
			cutoffTitle: 'Winland R35 Boundary Cutoffs',
			cutoffLabel: 'R35 Cutoff',
			addLabel: '+ Add R35 Boundary',
			computeLabel: 'Compute Winland R35 (⌘+Enter)',
			unit: 'μm',
			min: 0.02,
			max: 100.0,
			step: 0.1,
			formula: 'log₁₀(R35) = 0.732 + 0.588·log₁₀(K) − 0.864·log₁₀(100·Φ)',
			description:
				'Pore throat radius R35 (μm) at 35% mercury saturation. Classifies pore throats from nanoporous (<0.5 μm) to megaporous (>10 μm) (DRT 1 = largest R35).',
			higherIsBetter: true
		},
		lucia: {
			label: 'Lucia RFN',
			shortName: 'RFN',
			badge: 'Jennings-Lucia (2003)',
			cutoffTitle: 'Lucia RFN Boundary Cutoffs',
			cutoffLabel: 'RFN Cutoff',
			addLabel: '+ Add RFN Boundary',
			computeLabel: 'Compute Lucia RFN (⌘+Enter)',
			unit: 'RFN',
			min: 0.5,
			max: 4.0,
			step: 0.1,
			formula: 'log₁₀(K) = (9.789−12.084·log RFN) + (8.671−8.297·log RFN)·log₁₀(Φ)',
			description:
				'Carbonate Rock Fabric Number (RFN 0.5–4.0): Class 1 Grainstone (0.5–1.5), Class 2 Packstone (1.5–2.5), Class 3 Mud-Dominated (2.5–4.0) (DRT 1 = lowest RFN).',
			higherIsBetter: false
		}
	};

	let activeMeta = $derived(DRT_METHOD_META[projectData.drtMethod]);

	async function selectDrtMethod(m: DrtMethod) {
		if (projectData.drtMethod === m && projectData.rockTypingResult?.method === m) return;
		projectData.drtMethod = m;
		if (workbench.projectId) {
			await executeRockTyping();
		}
	}

	function addCutoff() {
		const cur = projectData.activeDrtCutoffs;
		const last = cur[cur.length - 1] ?? 1.0;
		const nextVal =
			projectData.drtMethod === 'lucia'
				? Math.min(4.0, Number((last + 0.5).toFixed(2)))
				: Number((last * 1.8).toFixed(2));
		projectData.activeDrtCutoffs = [...cur, nextVal];
	}

	function removeCutoff(idx: number) {
		const cur = projectData.activeDrtCutoffs;
		if (cur.length <= 1) return;
		projectData.activeDrtCutoffs = cur.filter((_, i) => i !== idx);
	}

	function formatDrtRange(drtStr: string): string {
		const drtNum = Number(drtStr);
		const sorted = [...projectData.activeDrtCutoffs].sort((a, b) => a - b);
		const n = sorted.length;
		if (!Number.isFinite(drtNum) || n === 0) return '';
		const meta = activeMeta;
		const u = meta.unit ? ` ${meta.unit}` : '';
		if (meta.higherIsBetter) {
			// DRT 1 = >= sorted[n-1], DRT n+1 = < sorted[0]
			const upperIdx = n - (drtNum - 1);
			const lowerIdx = n - drtNum;
			if (drtNum === 1) return `${meta.shortName} ≥ ${sorted[n - 1]}${u}`;
			if (drtNum === n + 1) return `${meta.shortName} < ${sorted[0]}${u}`;
			if (lowerIdx >= 0 && upperIdx < n) {
				return `${sorted[lowerIdx]}–${sorted[upperIdx]}${u}`;
			}
		} else {
			// Lower is better (Lucia RFN): DRT 1 = < sorted[0], DRT n+1 = >= sorted[n-1]
			if (drtNum === 1) return `${meta.shortName} < ${sorted[0]}`;
			if (drtNum === n + 1) return `${meta.shortName} ≥ ${sorted[n - 1]}`;
			if (drtNum - 2 >= 0 && drtNum - 1 < n) {
				return `${sorted[drtNum - 2]}–${sorted[drtNum - 1]}`;
			}
		}
		return '';
	}

	async function executeRockTyping(autoFindCutoffs: boolean = false) {
		if (!workbench.projectId) return;
		projectData.isComputingMultiWell = true;
		projectData.error = null;
		try {
			const method = projectData.drtMethod;
			const meta = DRT_METHOD_META[method];
			const sortedCutoffs = [...projectData.activeDrtCutoffs].sort((a, b) => a - b);
			const { data, error } = await api.POST(
				'/quick_pp/workspace/projects/{project_id}/rock_typing/compute',
				{
					params: { path: { project_id: workbench.projectId } },
					body: {
						well_names:
							projectData.selectedWellNames.length > 0 &&
							projectData.selectedWellNames.length < projectData.wells.length
								? projectData.selectedWellNames
								: null,
						method,
						cutoffs: sortedCutoffs,
						higher_is_better: meta.higherIsBetter,
						auto_find_cutoffs: autoFindCutoffs,
						no_of_cutoffs: sortedCutoffs.length,
						save: false
					} as any
				}
			);
			if (error || !data) {
				throw new Error(String((error as any)?.detail ?? 'Rock typing calculation failed'));
			}
			if (autoFindCutoffs && Array.isArray(data.cutoffs) && data.cutoffs.length > 0) {
				projectData.activeDrtCutoffs = data.cutoffs.map((c: number) => Number(c.toFixed(2)));
			}
			const counts = (data as any).rock_type_counts ?? (data as any).flag_counts ?? {};
			projectData.rockTypingResult = {
				method: (data as any).method ?? method,
				...(data.data as any),
				rock_type_counts: counts
			};
			projectData.markMultiWellComputed('rock-typing');

			// Also update active well's ROCK_FLAG & FZI / R35 / RFN in wellData so LogTrackCanvas updates in real time
			if (workbench.wellId && data.data?.well_names) {
				const wnArr = data.data.well_names;
				const dArr = data.data.depths;
				const rfArr = data.data.rock_flags;
				const fziArr = data.data.fzi;
				const r35Arr = (data.data as any).r35;
				const rfnArr = (data.data as any).rfn;
				const wDepths: number[] = [];
				const wRf: (number | null)[] = [];
				const wFzi: (number | null)[] = [];
				const wR35: (number | null)[] = [];
				const wRfn: (number | null)[] = [];
				for (let i = 0; i < wnArr.length; i++) {
					if (wnArr[i] === workbench.wellId) {
						wDepths.push(dArr[i]);
						wRf.push(rfArr[i]);
						wFzi.push(fziArr?.[i] ?? null);
						wR35.push(r35Arr?.[i] ?? null);
						wRfn.push(rfnArr?.[i] ?? null);
					}
				}
				if (wDepths.length > 0) {
					wellData.mergeComputedCurves({
						DEPTH: wDepths,
						ROCK_FLAG: wRf,
						FZI: wFzi,
						R35: wR35,
						RFN: wRfn
					});
				}
			}
		} catch (e: any) {
			projectData.error = e?.message || 'Rock typing error';
		} finally {
			projectData.isComputingMultiWell = false;
		}
	}

	async function saveRockTyping() {
		await projectData.saveMultiWellComputed('rock-typing');
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection title="Rock Typing Method (DRT)" badge={activeMeta.badge}>
			<div class="grid grid-cols-3 gap-1">
				{#each [['fzi', 'FZI / RQI'], ['winland', 'Winland R35'], ['lucia', 'Lucia RFN']] as [m, lbl]}
					<button
						type="button"
						onclick={() => selectDrtMethod(m as DrtMethod)}
						class="h-6 rounded-sm border text-[10px] font-medium transition-colors {projectData.drtMethod ===
						m
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]'}"
					>
						{lbl}
					</button>
				{/each}
			</div>

			<div
				class="mt-1.5 rounded-xs border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 py-1 text-[10px]"
			>
				<div class="mono-num font-semibold text-[var(--text-primary)]">
					{activeMeta.formula}
				</div>
				<p class="mt-0.5 leading-snug text-[var(--text-secondary)]">
					{activeMeta.description}
				</p>
			</div>
		</ParameterSection>

		<ParameterSection
			title={activeMeta.cutoffTitle}
			badge={`${projectData.activeDrtCutoffs.length + 1} DRTs`}
		>
			<div class="space-y-1">
				{#each projectData.activeDrtCutoffs as _, i}
					<div class="flex items-center gap-1">
						<div class="flex-1">
							{#if projectData.drtMethod === 'winland'}
								<NumericScrubber
									label={`${activeMeta.cutoffLabel} ${i + 1}`}
									bind:value={projectData.winlandCutoffs[i]}
									unit={activeMeta.unit}
									min={activeMeta.min}
									max={activeMeta.max}
									step={activeMeta.step}
									onrun={() => executeRockTyping(false)}
								/>
							{:else if projectData.drtMethod === 'lucia'}
								<NumericScrubber
									label={`${activeMeta.cutoffLabel} ${i + 1}`}
									bind:value={projectData.luciaCutoffs[i]}
									unit={activeMeta.unit}
									min={activeMeta.min}
									max={activeMeta.max}
									step={activeMeta.step}
									onrun={() => executeRockTyping(false)}
								/>
							{:else}
								<NumericScrubber
									label={`${activeMeta.cutoffLabel} ${i + 1}`}
									bind:value={projectData.fziCutoffs[i]}
									unit={activeMeta.unit}
									min={activeMeta.min}
									max={activeMeta.max}
									step={activeMeta.step}
									onrun={() => executeRockTyping(false)}
								/>
							{/if}
						</div>
						<button
							type="button"
							title="Remove cutoff"
							onclick={() => removeCutoff(i)}
							class="flex h-7 w-6 shrink-0 items-center justify-center rounded-sm border border-[var(--border-hairline)] text-xs text-[var(--text-muted)] hover:bg-red-50 hover:text-red-600"
						>
							×
						</button>
					</div>
				{/each}
			</div>

			<div class="mt-1.5 grid grid-cols-2 gap-1">
				<button
					type="button"
					onclick={addCutoff}
					class="h-6 rounded-sm border border-dashed border-[var(--border-hairline)] bg-[var(--bg-app)] text-[10.5px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
				>
					{activeMeta.addLabel}
				</button>
				<button
					type="button"
					disabled={projectData.isComputingMultiWell}
					onclick={() => executeRockTyping(true)}
					title="Automatically find optimal boundary cutoffs using k-Means clustering on {activeMeta.shortName}"
					class="h-6 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[10.5px] font-medium text-[var(--text-secondary)] hover:bg-blue-50 hover:text-[var(--accent-primary)] disabled:opacity-50"
				>
					Auto-Detect (k-Means)
				</button>
			</div>
		</ParameterSection>

		{#if projectData.rockTypingResult?.rock_type_counts}
			<ParameterSection
				title="Discrete Rock Type (DRT) Counts"
				badge={activeMeta.shortName}
			>
				<div class="space-y-1">
					{#each Object.entries(projectData.rockTypingResult.rock_type_counts) as [drt, count]}
						{@const rangeLbl = formatDrtRange(drt)}
						<div
							class="flex h-6 items-center justify-between rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] px-2 text-xs"
						>
							<span class="flex items-center gap-1.5 font-medium">
								<span
									style="background-color: {getRockTypeColor(drt)}"
									class="inline-block h-2.5 w-2.5 rounded-full"
								></span>
								<span>DRT {drt}</span>
								{#if rangeLbl}
									<span class="mono-num text-[10px] font-normal text-[var(--text-muted)]">
										({rangeLbl})
									</span>
								{/if}
							</span>
							<span class="mono-num font-semibold text-[var(--text-primary)]">{count} pts</span>
						</div>
					{/each}
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
		<button
			type="button"
			disabled={projectData.isComputingMultiWell}
			onclick={() => executeRockTyping(false)}
			class="h-7 flex-1 mr-1.5 rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{projectData.isComputingMultiWell ? 'Clustering...' : activeMeta.computeLabel}
		</button>
		<button
			type="button"
			disabled={projectData.isComputingMultiWell || !projectData.hasPendingMultiWell('rock-typing')}
			onclick={saveRockTyping}
			title={projectData.hasPendingMultiWell('rock-typing')
				? `Save computed ROCK_FLAG and ${activeMeta.shortName} curves across all project wells`
				: `Compute ${activeMeta.label} rock typing first to enable saving`}
			class="h-7 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--bg-subtle)] disabled:cursor-not-allowed disabled:opacity-45"
		>
			Save All Wells
		</button>
	</div>
</div>
