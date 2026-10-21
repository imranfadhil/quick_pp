<script lang="ts">
	import { onMount } from 'svelte';
	import { postJson } from '$lib/api/client';
	import NumericScrubber from '$lib/components/workbench/NumericScrubber.svelte';
	import ParameterSection from '$lib/components/workbench/ParameterSection.svelte';
	import { wellData } from '$lib/state/wellData.svelte';
	import { workbench } from '$lib/state/workbench.svelte';
	import { fmtNum } from '$lib/utils/formatters';

	// Core Overburden & Klinkenberg Correction parameters
	let netOverburdenPsi = $state(3500);
	let meanPorePressurePsi = $state(45.0);
	let coreLithology = $state<'sandstone' | 'carbonate'>('sandstone');
	let bMethod = $state<'heins_jones' | 'jones'>('heins_jones');
	let coreCorrSummary = $state<{
		count: number;
		meanPoroAmb: number;
		meanPoroRes: number;
		meanPermGas: number;
		meanPermLiquid: number;
	} | null>(null);

	// Lab-to-Reservoir Capillary Pressure Scaling calculator
	let pcLabInput = $state(25.0);
	let sigmaLab = $state(480.0);
	let thetaLab = $state(140.0);
	let sigmaRes = $state(30.0);
	let thetaRes = $state(30.0);
	let pcResOutput = $state<number | null>(null);

	let copiedSwof = $state(false);

	async function computeScal() {
		wellData.isCalculating = true;
		wellData.lastError = null;
		const sp = workbench.scalParams;
		try {
			const [relpermRes, swofRes] = await Promise.all([
				postJson<{
					model: string;
					data: Array<{ sw: number; swn: number; krw: number; kro: number; fw: number }>;
				}>('/quick_pp/scal/relperm', {
					model: sp.model,
					swirr: sp.swirr,
					sor: sp.sor,
					krw_max: sp.krw_max,
					kro_max: sp.kro_max,
					nw: sp.nw,
					no: sp.no,
					lam: sp.lam,
					lw: sp.lw,
					ew: sp.ew,
					tw: sp.tw,
					lo: sp.lo,
					eo: sp.eo,
					to: sp.to
				}),
				postJson<{
					table: Array<{ SW: number; KRW: number; KROW: number; PCOW: number }>;
					swof_keyword: string;
				}>('/quick_pp/scal/swof', {
					model: sp.model,
					swirr: sp.swirr,
					sor: sp.sor,
					krw_max: sp.krw_max,
					kro_max: sp.kro_max,
					nw: sp.nw,
					no: sp.no,
					lam: sp.lam,
					lw: sp.lw,
					ew: sp.ew,
					tw: sp.tw,
					lo: sp.lo,
					eo: sp.eo,
					to: sp.to,
					pc_entry: sp.pc_entry,
					pc_exponent: sp.pc_exponent,
					pc_max: sp.pc_max,
					n_points: sp.n_points
				})
			]);

			const rows = relpermRes.data || [];
			const swArr = rows.map((r) => r.sw);
			const seArr = rows.map((r) => r.swn);
			const krwArr = rows.map((r) => r.krw);
			const kroArr = rows.map((r) => r.kro);
			const fwArr = rows.map((r, i) => {
				const kw = krwArr[i] / Math.max(1e-6, sp.mu_w);
				const ko = kroArr[i] / Math.max(1e-6, sp.mu_o);
				return kw + ko > 0 ? kw / (kw + ko) : (r.fw ?? 0);
			});

			workbench.scalResult = {
				sw: swArr,
				se: seArr,
				krw: krwArr,
				kro: kroArr,
				fw: fwArr,
				swofTable: swofRes.table,
				swofKeyword: swofRes.swof_keyword
			};
		} catch (e: any) {
			wellData.lastError = e?.message || 'SCAL calculation error';
		} finally {
			wellData.isCalculating = false;
		}
	}

	async function runCoreCorrections() {
		wellData.isCalculating = true;
		wellData.lastError = null;
		try {
			const samples =
				wellData.coreSamples.length > 0
					? wellData.coreSamples
					: [
							{ depth: 2010, porosity: 0.18, perm: 45 },
							{ depth: 2025, porosity: 0.22, perm: 120 },
							{ depth: 2040, porosity: 0.14, perm: 12 }
						];

			const records: Array<{ DEPTH: number; CPORE: number; CPERM: number }> = [];
			const validIndices: number[] = [];

			samples.forEach((s, idx) => {
				const dRaw = Number(s.depth ?? s.DEPTH ?? idx + 1);
				const pRaw = Number(s.porosity ?? s.CPORE ?? s.poro);
				const kRaw = Number(s.perm ?? s.permeability ?? s.CPERM);
				if (Number.isFinite(pRaw) && Number.isFinite(kRaw) && kRaw > 0) {
					records.push({
						DEPTH: dRaw,
						CPORE: pRaw > 1 ? pRaw / 100 : pRaw,
						CPERM: kRaw
					});
					validIndices.push(idx);
				}
			});

			if (records.length === 0) {
				throw new Error('No valid core porosity/permeability samples available.');
			}

			const res = await postJson<{
				data: Array<Record<string, any>>;
			}>('/quick_pp/scal/core_corrections', {
				net_confining_stress: netOverburdenPsi,
				mean_pore_pressure: meanPorePressurePsi,
				compaction_coeff: coreLithology === 'carbonate' ? 2.5e-6 : 4.5e-6,
				replace_ori: false,
				data: records
			});

			const corrRows = res.data || [];
			const mean = (arr: number[]) =>
				arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

			const poroAmb = records.map((r) => r.CPORE);
			const permGas = records.map((r) => r.CPERM);
			const poroRes = corrRows.map((r, i) => Number(r.CPORE ?? poroAmb[i]));
			const permLiq = corrRows.map((r, i) => Number(r.CPERM ?? permGas[i]));

			coreCorrSummary = {
				count: records.length,
				meanPoroAmb: mean(poroAmb),
				meanPoroRes: mean(poroRes),
				meanPermGas: mean(permGas),
				meanPermLiquid: mean(permLiq)
			};

			if (wellData.coreSamples.length > 0) {
				const updated = [...wellData.coreSamples];
				validIndices.forEach((sampleIdx, i) => {
					updated[sampleIdx] = {
						...updated[sampleIdx],
						porosity_res: Number(poroRes[i].toFixed(4)),
						perm_liquid: Number(permLiq[i].toFixed(3))
					};
				});
				wellData.coreSamples = updated;
			}
			wellData.lastSavedMessage = 'Core Overburden & Klinkenberg Applied ✓';
		} catch (e: any) {
			wellData.lastError = e?.message || 'Core corrections error';
		} finally {
			wellData.isCalculating = false;
		}
	}

	async function runPcScaling() {
		try {
			const res = await postJson<{ pc_res: number[] }>('/quick_pp/scal/pc_lab_to_reservoir', {
				pc_lab: [pcLabInput],
				ift_lab: sigmaLab,
				theta_lab: thetaLab,
				ift_res: sigmaRes,
				theta_res: thetaRes
			});
			pcResOutput = res.pc_res?.[0] ?? null;
		} catch (e: any) {
			wellData.lastError = e?.message || 'Pc scaling error';
		}
	}

	async function copySwofKeyword() {
		const kw = workbench.scalResult?.swofKeyword;
		if (!kw) return;
		await navigator.clipboard.writeText(kw);
		copiedSwof = true;
		setTimeout(() => (copiedSwof = false), 2000);
	}

	function downloadSwofInc() {
		const kw = workbench.scalResult?.swofKeyword;
		if (!kw) return;
		const blob = new Blob([kw], { type: 'text/plain;charset=utf-8;' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `${workbench.wellId || 'SCAL'}_SWOF.INC`;
		a.click();
		URL.revokeObjectURL(url);
	}
</script>

<div class="flex h-full flex-col overflow-hidden bg-[var(--bg-surface)]">
	<div class="flex-1 overflow-y-auto">
		<ParameterSection
			title="Relative Permeability Model"
			badge={workbench.scalParams.model.toUpperCase()}
		>
			<div class="grid grid-cols-3 gap-1">
				{#each [['let', 'LET (3-Param)'], ['corey', 'Corey'], ['brooks_corey', 'Brooks-Corey']] as [m, lbl]}
					<button
						type="button"
						onclick={() => {
							workbench.scalParams.model = m as any;
						}}
						class="h-6 rounded-sm border text-[11px] font-medium {workbench.scalParams.model ===
						m
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
					>
						{lbl}
					</button>
				{/each}
			</div>

			<NumericScrubber
				label="Connate Water (Sw,irr)"
				bind:value={workbench.scalParams.swirr}
				unit="v/v"
				min={0.01}
				max={0.5}
				step={0.01}
				onrun={computeScal}
			/>
			<NumericScrubber
				label="Residual Oil (Sor)"
				bind:value={workbench.scalParams.sor}
				unit="v/v"
				min={0.01}
				max={0.5}
				step={0.01}
				onrun={computeScal}
			/>
			<NumericScrubber
				label="Max Water Rel-Perm (krw,max)"
				bind:value={workbench.scalParams.krw_max}
				unit="frac"
				min={0.05}
				max={1.0}
				step={0.02}
				onrun={computeScal}
			/>
			<NumericScrubber
				label="Max Oil Rel-Perm (kro,max)"
				bind:value={workbench.scalParams.kro_max}
				unit="frac"
				min={0.1}
				max={1.0}
				step={0.02}
				onrun={computeScal}
			/>

			{#if workbench.scalParams.model === 'corey'}
				<NumericScrubber
					label="Water Corey Exponent (nw)"
					bind:value={workbench.scalParams.nw}
					min={1.0}
					max={8.0}
					step={0.1}
					onrun={computeScal}
				/>
				<NumericScrubber
					label="Oil Corey Exponent (no)"
					bind:value={workbench.scalParams.no}
					min={1.0}
					max={8.0}
					step={0.1}
					onrun={computeScal}
				/>
			{:else if workbench.scalParams.model === 'brooks_corey'}
				<NumericScrubber
					label="Pore-Size Distribution (λ)"
					bind:value={workbench.scalParams.lam}
					min={0.3}
					max={6.0}
					step={0.1}
					onrun={computeScal}
				/>
			{:else}
				<div class="grid grid-cols-2 gap-x-2">
					<NumericScrubber
						label="Water L (Lw)"
						bind:value={workbench.scalParams.lw}
						min={0.5}
						max={6.0}
						step={0.1}
						onrun={computeScal}
					/>
					<NumericScrubber
						label="Oil L (Lo)"
						bind:value={workbench.scalParams.lo}
						min={0.5}
						max={6.0}
						step={0.1}
						onrun={computeScal}
					/>
					<NumericScrubber
						label="Water E (Ew)"
						bind:value={workbench.scalParams.ew}
						min={0.1}
						max={10.0}
						step={0.2}
						onrun={computeScal}
					/>
					<NumericScrubber
						label="Oil E (Eo)"
						bind:value={workbench.scalParams.eo}
						min={0.1}
						max={10.0}
						step={0.2}
						onrun={computeScal}
					/>
					<NumericScrubber
						label="Water T (Tw)"
						bind:value={workbench.scalParams.tw}
						min={0.5}
						max={6.0}
						step={0.1}
						onrun={computeScal}
					/>
					<NumericScrubber
						label="Oil T (To)"
						bind:value={workbench.scalParams.to}
						min={0.5}
						max={6.0}
						step={0.1}
						onrun={computeScal}
					/>
				</div>
			{/if}

			<div class="grid grid-cols-2 gap-x-2">
				<NumericScrubber
					label="Water Viscosity (μw)"
					bind:value={workbench.scalParams.mu_w}
					unit="cP"
					min={0.1}
					max={5.0}
					step={0.1}
					onrun={computeScal}
				/>
				<NumericScrubber
					label="Oil Viscosity (μo)"
					bind:value={workbench.scalParams.mu_o}
					unit="cP"
					min={0.1}
					max={50.0}
					step={0.2}
					onrun={computeScal}
				/>
			</div>
		</ParameterSection>

		<ParameterSection
			title="Eclipse SWOF Simulation Deck"
			badge={`${workbench.scalResult?.swofTable.length ?? 0} rows`}
		>
			<NumericScrubber
				label="Entry Capillary Pressure (Pd)"
				bind:value={workbench.scalParams.pc_entry}
				unit="psi"
				min={0.1}
				max={25.0}
				step={0.2}
				onrun={computeScal}
			/>
			<NumericScrubber
				label="Capillary Exponent (1/λ)"
				bind:value={workbench.scalParams.pc_exponent}
				min={0.5}
				max={6.0}
				step={0.1}
				onrun={computeScal}
			/>
			<NumericScrubber
				label="Max Capillary Pressure"
				bind:value={workbench.scalParams.pc_max}
				unit="psi"
				min={5.0}
				max={200.0}
				step={5.0}
				onrun={computeScal}
			/>

			<div class="mt-1.5 flex gap-1.5">
				<button
					type="button"
					disabled={!workbench.scalResult?.swofKeyword}
					onclick={copySwofKeyword}
					class="h-7 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-xs font-semibold text-[var(--text-primary)] hover:bg-slate-200 disabled:opacity-40"
				>
					{copiedSwof ? 'Copied SWOF ✓' : 'Copy SWOF Keyword'}
				</button>
				<button
					type="button"
					disabled={!workbench.scalResult?.swofKeyword}
					onclick={downloadSwofInc}
					class="h-7 flex-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2 text-xs font-semibold text-[var(--accent-primary)] hover:bg-blue-50 disabled:opacity-40"
				>
					Download SWOF.INC
				</button>
			</div>
		</ParameterSection>

		<ParameterSection
			title="Core Overburden & Klinkenberg"
			badge={`${wellData.coreSamples.length} plugs`}
			defaultOpen={false}
		>
			<div class="grid grid-cols-2 gap-1">
				{#each [['sandstone', 'Sandstone'], ['carbonate', 'Carbonate']] as [l, lbl]}
					<button
						type="button"
						onclick={() => (coreLithology = l as any)}
						class="h-6 rounded-sm border text-[11px] font-medium {coreLithology === l
							? 'border-[var(--accent-primary)] bg-blue-50 font-semibold text-[var(--accent-primary)]'
							: 'border-[var(--border-hairline)] bg-[var(--bg-surface)] text-[var(--text-secondary)]'}"
					>
						{lbl}
					</button>
				{/each}
			</div>

			<NumericScrubber
				label="Net Overburden Stress"
				bind:value={netOverburdenPsi}
				unit="psi"
				min={500}
				max={12000}
				step={250}
			/>
			<NumericScrubber
				label="Lab Mean Gas Pressure"
				bind:value={meanPorePressurePsi}
				unit="psi"
				min={15}
				max={200}
				step={5}
			/>

			<button
				type="button"
				onclick={runCoreCorrections}
				class="mt-1 h-7 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-2 text-xs font-semibold text-[var(--text-primary)] hover:bg-slate-200"
			>
				Apply Stress & Klinkenberg Corrections
			</button>

			{#if coreCorrSummary}
				<div class="mt-1.5 grid grid-cols-2 gap-1 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-app)] p-1.5 text-[10px]">
					<div>
						<span class="text-[var(--text-muted)]">Mean Φ (Amb → Res):</span>
						<div class="mono-num font-semibold">
							{fmtNum(coreCorrSummary.meanPoroAmb, 3)} → {fmtNum(coreCorrSummary.meanPoroRes, 3)}
						</div>
					</div>
					<div>
						<span class="text-[var(--text-muted)]">Mean K (Gas → Liq):</span>
						<div class="mono-num font-semibold">
							{fmtNum(coreCorrSummary.meanPermGas, 1)} → {fmtNum(coreCorrSummary.meanPermLiquid, 1)} mD
						</div>
					</div>
				</div>
			{/if}
		</ParameterSection>

		<ParameterSection title="Lab-to-Reservoir Pc Scaling" defaultOpen={false}>
			<NumericScrubber
				label="Lab Capillary Pressure"
				bind:value={pcLabInput}
				unit="psi"
				min={0.1}
				max={2000}
				step={5.0}
				onrun={runPcScaling}
			/>
			<div class="grid grid-cols-2 gap-x-2">
				<NumericScrubber
					label="σ Lab (Hg/Air=480)"
					bind:value={sigmaLab}
					unit="dyn/cm"
					min={10}
					max={500}
					step={5}
					onrun={runPcScaling}
				/>
				<NumericScrubber
					label="θ Lab (Hg/Air=140)"
					bind:value={thetaLab}
					unit="deg"
					min={0}
					max={180}
					step={5}
					onrun={runPcScaling}
				/>
				<NumericScrubber
					label="σ Reservoir"
					bind:value={sigmaRes}
					unit="dyn/cm"
					min={1}
					max={80}
					step={1}
					onrun={runPcScaling}
				/>
				<NumericScrubber
					label="θ Reservoir"
					bind:value={thetaRes}
					unit="deg"
					min={0}
					max={90}
					step={1}
					onrun={runPcScaling}
				/>
			</div>
			<button
				type="button"
				onclick={runPcScaling}
				class="mt-1 h-6 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[11px] font-semibold text-[var(--text-primary)] hover:bg-slate-200"
			>
				Convert Lab Pc → Reservoir Pc
			</button>
			{#if pcResOutput !== null}
				<div class="mono-num mt-1 rounded-sm border border-blue-200 bg-blue-50 px-2 py-1 text-center text-xs font-semibold text-blue-900">
					Reservoir Pc = {fmtNum(pcResOutput, 3)} psi
				</div>
			{/if}
		</ParameterSection>

		{#if wellData.lastError}
			<div class="m-2 rounded-sm border border-red-200 bg-red-50 p-2 text-xs text-[var(--status-error)]">
				{wellData.lastError}
			</div>
		{/if}
	</div>

	<div
		class="flex h-9 shrink-0 items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-app)] px-2.5"
	>
		<button
			type="button"
			disabled={wellData.isCalculating}
			onclick={computeScal}
			class="h-7 w-full rounded-sm bg-[var(--accent-primary)] px-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
		>
			{wellData.isCalculating ? 'Computing SCAL...' : 'Compute Rel-Perm & SWOF (⌘+Enter)'}
		</button>
	</div>
</div>
