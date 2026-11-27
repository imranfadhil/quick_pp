/**
 * Borehole Environmental Correction QC multi-track plot builder and summary helpers.
 *
 * Combines causal hole-quality logs (Caliper vs. Bit Size with washout/mudcake fill + DRHO
 * overlay) with raw-vs-corrected sensor tracks (GR, Resistivity + Rw/Rmf/Rm & invasion,
 * NPHI, RHOB, and DTC cycle-skip removal), supporting both Overlay (±1σ band) and Δ Shift modes.
 */

export type EnvQcViewMode = 'overlay' | 'delta' | 'invasion';
export type EnvQcTrackFilter = 'all' | 'GR' | 'RT' | 'NPHI_RHOB' | 'DT';

export interface EnvQcParamsInput {
	bitsizeDefault: number;
	mudWeight: number;
	surfaceTemp: number;
	tempGrad: number;
	salinityPpm: number;
	toolType: 'WL' | 'LWD';
	mudType: 'WBM' | 'OBM' | 'SBM';
	offshore: boolean;
	waterDepth: number;
	seabedTempC: number;
	rmf: number;
	rmfTempC: number;
	formationSalinityPpm: number;
	washoutThold: number;
}

export const ENV_QC_TRACK_FILTERS: ReadonlyArray<{ id: EnvQcTrackFilter; label: string }> = [
	{ id: 'all', label: 'All Tracks' },
	{ id: 'GR', label: 'GR' },
	{ id: 'RT', label: 'Resistivity' },
	{ id: 'NPHI_RHOB', label: 'NPHI · RHOB' },
	{ id: 'DT', label: 'Sonic (DTC)' }
];

export const ENV_QC_TOLERANCES: Record<'GR' | 'RT' | 'NPHI' | 'RHOB' | 'DT', number> = {
	GR: 15.0,
	RT: 0.25,
	NPHI: 0.04,
	RHOB: 0.06,
	DT: 10.0
};

export const ENV_QC_UNITS: Record<'GR' | 'RT' | 'NPHI' | 'RHOB' | 'DT', string> = {
	GR: 'gAPI',
	RT: 'dec',
	NPHI: 'v/v',
	RHOB: 'g/cc',
	DT: 'us/ft'
};

export const ENV_QC_REASON_BITS: ReadonlyArray<{ bit: number; label: string }> = [
	{ bit: 1, label: 'Washout' },
	{ bit: 2, label: 'High DRHO' },
	{ bit: 4, label: 'Badhole' },
	{ bit: 8, label: 'Invasion' },
	{ bit: 16, label: 'No Caliper' },
	{ bit: 32, label: 'Spike' },
	{ bit: 64, label: 'Flatline' },
	{ bit: 128, label: 'Cycle Skip' },
	{ bit: 256, label: 'Assumed BS' },
	{ bit: 512, label: 'Res/Mud Mismatch' },
	{ bit: 1024, label: 'Off-Depth' }
];

const M_TO_FT = 3.280839895013123;
const DRHO_LIMIT_DEFAULT = 0.15;

export function decodeEnvQcReason(mask: number | null | undefined): string {
	if (typeof mask !== 'number' || !Number.isFinite(mask) || mask <= 0) return 'Clean';
	const flags: string[] = [];
	const intMask = Math.trunc(mask);
	for (const item of ENV_QC_REASON_BITS) {
		if ((intMask & item.bit) !== 0) flags.push(item.label);
	}
	return flags.length > 0 ? flags.join(', ') : 'Clean';
}

export function decodeEnvQcStatus(status: number | null | undefined): string {
	if (status === 0) return 'Unchanged';
	if (status === 1) return 'Corrected';
	if (status === 2) return 'Unreliable';
	return 'No Data';
}

export function decodeInvasionProfile(code: number | null | undefined): string {
	switch (code) {
		case 1:
			return 'Water-like (Rxo/Rt ≈ Rmf/Rw)';
		case 2:
			return 'HC-like (Rxo/Rt < Rmf/Rw)';
		case 3:
			return 'Anomalous contrast';
		case 4:
			return 'Tight / No invasion';
		default:
			return 'Uninformative / None';
	}
}

function parseFiniteHeaderNum(val: unknown): number | null {
	if (val === null || val === undefined || val === '') return null;
	const n = Number(val);
	return Number.isFinite(n) ? n : null;
}

/**
 * Resolve the rig depth datum above MSL (KB / DFE play the identical petrophysical role:
 * ``TVDSS = TVD - rigDatum``, and offshore ``airGap = rigDatum``).
 */
export function resolveHeaderRigDatum(
	hdr: Record<string, any> | null | undefined
): { value: number | null; source: 'DFE' | 'KB' | 'KB/DFE' | null } {
	if (!hdr || typeof hdr !== 'object') return { value: null, source: null };
	const df = parseFiniteHeaderNum(hdr.df_elev ?? hdr.dfe_elevation);
	const kb = parseFiniteHeaderNum(hdr.kb_elev ?? hdr.kb_elevation);
	if (df !== null && kb !== null) {
		if (Math.abs(df - kb) <= 0.1) return { value: kb, source: 'KB/DFE' };
		return { value: df, source: 'DFE' };
	}
	if (kb !== null) return { value: kb, source: 'KB' };
	if (df !== null) return { value: df, source: 'DFE' };
	return { value: null, source: null };
}

/**
 * Resolve whether a well header represents Onshore (Ground Level above MSL) vs.
 * Offshore (Water Depth from MSL to Seabed), handling legacy LAS conventions where
 * seabed depth was stored in `gl_elev` (either as `gl_elev < 0` below MSL or
 * `gl_elev > rigDatum` such as UKCS `KB=109 ft, GL=117 ft`).
 */
export function resolveHeaderSurfaceAndWaterDepth(
	hdr: Record<string, any> | null | undefined
): {
	offshore: boolean;
	waterDepth: number | null;
	glElev: number | null;
	inferredFromGl: boolean;
} {
	if (!hdr || typeof hdr !== 'object') {
		return { offshore: false, waterDepth: null, glElev: null, inferredFromGl: false };
	}
	const rawWd = parseFiniteHeaderNum(hdr.water_depth);
	const rawGl = parseFiniteHeaderNum(hdr.gl_elev ?? hdr.ground_elevation);
	const rig = resolveHeaderRigDatum(hdr).value;

	if (rawWd !== null && rawWd > 0) {
		return { offshore: true, waterDepth: rawWd, glElev: null, inferredFromGl: false };
	}
	if (rawWd === 0) {
		return { offshore: false, waterDepth: null, glElev: rawGl, inferredFromGl: false };
	}
	if (rawGl !== null) {
		if (rawGl < 0) {
			return {
				offshore: true,
				waterDepth: Math.abs(rawGl),
				glElev: null,
				inferredFromGl: true
			};
		}
		if (rawGl > 0 && rig !== null && rig > 0 && rawGl > rig) {
			return {
				offshore: true,
				waterDepth: rawGl,
				glElev: null,
				inferredFromGl: true
			};
		}
	}
	return { offshore: false, waterDepth: null, glElev: rawGl, inferredFromGl: false };
}

/**
 * Parse water depth from a well header into canonical meters, honoring
 * `water_depth_unit` / `datum_unit` / `depth_unit` as well as subsea/offshore `gl_elev`
 * conventions. Returns null when onshore or absent.
 */
export function parseHeaderWaterDepthMeters(
	hdr: Record<string, any> | null | undefined
): number | null {
	if (!hdr || typeof hdr !== 'object') return null;
	const surf = resolveHeaderSurfaceAndWaterDepth(hdr);
	if (!surf.offshore || surf.waterDepth === null || surf.waterDepth <= 0) {
		return null;
	}
	const u = String(hdr.water_depth_unit ?? hdr.datum_unit ?? hdr.depth_unit ?? 'm')
		.trim()
		.toLowerCase();
	const isFt = u === 'ft' || u === 'f' || u === 'feet' || u === 'foot';
	return isFt ? surf.waterDepth / M_TO_FT : surf.waterDepth;
}

/**
 * Synchronize well header water depth (converted to meters) into borehole environmental
 * correction parameters.
 */
export function syncHeaderToEnvParams(
	hdr: Record<string, any> | null | undefined,
	ep: { offshore: boolean; waterDepth: number }
): void {
	if (!hdr || typeof hdr !== 'object') return;
	const wdM = parseHeaderWaterDepthMeters(hdr);
	if (wdM !== null && wdM > 0) {
		ep.offshore = true;
		ep.waterDepth = Number(wdM.toFixed(2));
	} else {
		ep.offshore = false;
		ep.waterDepth = 0;
	}
}

function firstCurve(
	curves: Record<string, Array<number | null> | undefined>,
	names: readonly string[]
): { name: string; values: Array<number | null> } | null {
	for (const name of names) {
		const arr = curves[name];
		if (arr && arr.some((v) => typeof v === 'number' && Number.isFinite(v))) {
			return { name, values: arr };
		}
	}
	return null;
}

function isFiniteNum(v: unknown): v is number {
	return typeof v === 'number' && Number.isFinite(v);
}

function rwFromSalinity(tempC: number, salinityPpm: number): number {
	const sal = Math.max(salinityPpm, 100.0);
	const tempF = Math.max((tempC * 9.0) / 5.0 + 32.0, 40.0);
	return Math.pow(400000.0 / (sal * tempF), 0.88);
}

function arpsResistivity(r1: number, t1C: number, t2C: number): number {
	return (r1 * (t1C + 21.5)) / Math.max(t2C + 21.5, 1.0);
}

/**
 * Resolve formation-temperature Rw, Rmf, and Rm arrays:
 * prefers backend-exported ENV_RW / ENV_RMF / ENV_RM curves and falls back to
 * Arps / Bateman-Konen from ENV_TEMP_C or thermal gradient parameters.
 */
export function resolveFluidResistivities(
	curves: Record<string, Array<number | null> | undefined>,
	depthsMeters: Array<number | null>,
	envParams: EnvQcParamsInput
): {
	rw: Array<number | null>;
	rmf: Array<number | null>;
	rm: Array<number | null>;
} {
	const n = depthsMeters.length;
	const envRw = curves['ENV_RW'];
	const envRmf = curves['ENV_RMF'];
	const envRm = curves['ENV_RM'];
	const envTempC = curves['ENV_TEMP_C'];

	const hasRw = Boolean(envRw && envRw.some(isFiniteNum));
	const hasRmf = Boolean(envRmf && envRmf.some(isFiniteNum));
	const hasRm = Boolean(envRm && envRm.some(isFiniteNum));

	const gradCPerKm =
		(isFiniteNum(envParams.tempGrad) ? envParams.tempGrad : 1.5) * (5.0 / 9.0) * (1000.0 / 30.48);
	const surfTempC = ((isFiniteNum(envParams.surfaceTemp) ? envParams.surfaceTemp : 75.0) - 32.0) * (5.0 / 9.0);
	const waterDepthM = envParams.offshore ? Math.max(0, envParams.waterDepth || 0) : 0;
	const topTempC = envParams.offshore
		? isFiniteNum(envParams.seabedTempC)
			? envParams.seabedTempC
			: 4.0
		: surfTempC;
	const isOil = envParams.mudType !== 'WBM';

	const rw: Array<number | null> = new Array(n).fill(null);
	const rmf: Array<number | null> = new Array(n).fill(null);
	const rm: Array<number | null> = new Array(n).fill(null);

	for (let i = 0; i < n; i++) {
		const dM = depthsMeters[i];
		if (!isFiniteNum(dM)) continue;

		const tCurve = envTempC?.[i];
		const tempC = isFiniteNum(tCurve)
			? tCurve
			: topTempC + (gradCPerKm * Math.max(0, dM - waterDepthM)) / 1000.0;

		const rwVal = hasRw ? envRw![i] : rwFromSalinity(tempC, envParams.formationSalinityPpm || 35000);
		rw[i] = isFiniteNum(rwVal) && rwVal > 0 ? rwVal : null;

		if (isOil) {
			rmf[i] = hasRmf && isFiniteNum(envRmf![i]) ? envRmf![i] : null;
			rm[i] = hasRm && isFiniteNum(envRm![i]) ? envRm![i] : null;
		} else {
			let rmfVal: number | null = null;
			if (hasRmf) {
				rmfVal = isFiniteNum(envRmf![i]) ? envRmf![i] : null;
			} else if (isFiniteNum(envParams.rmf) && envParams.rmf > 0) {
				rmfVal = arpsResistivity(
					envParams.rmf,
					isFiniteNum(envParams.rmfTempC) ? envParams.rmfTempC : 25.0,
					tempC
				);
			} else {
				const mudSal = isFiniteNum(envParams.salinityPpm) && envParams.salinityPpm > 0
					? envParams.salinityPpm
					: 20000;
				rmfVal = rwFromSalinity(tempC, mudSal);
			}
			rmf[i] = isFiniteNum(rmfVal) && rmfVal > 0 ? rmfVal : null;

			const rmVal = hasRm
				? envRm![i]
				: rmf[i] !== null
					? rmf[i]! / 0.75
					: null;
			rm[i] = isFiniteNum(rmVal) && rmVal > 0 ? rmVal : null;
		}
	}

	return { rw, rmf, rm };
}

export interface EnvQcCurveRow {
	CURVE: 'GR' | 'RT' | 'NPHI' | 'RHOB' | 'DT';
	RAW_CURVE: string;
	CORR_CURVE: string;
	SAMPLES: number;
	GOOD_PCT: number;
	CORRECTED_PCT: number;
	UNRELIABLE_PCT: number;
	MEAN_CONF: number | null;
	MEAN_DELTA: number | null;
	MEAN_ABS_DELTA: number | null;
	P95_ABS_DELTA: number | null;
	MEAN_SIGMA: number | null;
	UNIT: string;
	DOMINANT_REASON: string;
}

export interface EnvQcInvasionSummary {
	invadedSamples: number;
	totalSamples: number;
	waterLikePct: number;
	hcLikePct: number;
	anomalousPct: number;
	tightPct: number;
	medianDiIn: number | null;
	p90DiIn: number | null;
}

export interface EnvQcSummary {
	hasEnvCorr: boolean;
	totalSamples: number;
	washoutPct: number;
	drhoHighPct: number;
	badholePct: number;
	invadedPct: number;
	cycleSkipCount: number;
	invasion: EnvQcInvasionSummary;
	rows: EnvQcCurveRow[];
}

export function computeEnvCorrQcSummary(
	curves: Record<string, Array<number | null> | undefined>,
	indices: number[],
	envParams: EnvQcParamsInput
): EnvQcSummary {
	const cali = firstCurve(curves, ['CALI', 'CAL', 'HCAL'])?.values;
	const bsArr = firstCurve(curves, ['BS', 'ENV_BS', 'BIT', 'BITSIZE'])?.values;
	const drho = firstCurve(curves, ['DRHO', 'HDRA', 'DCOR'])?.values;
	const badhole = firstCurve(curves, ['BADHOLE', 'BAD_HOLE', 'BAD_HOLE_FLAG'])?.values;
	const invFlag = curves['INVASION_FLAG'];
	const invProfile = curves['INVASION_PROFILE'];
	const diEst = curves['DI_EST'];
	const reasonArr = curves['ENV_QC_REASON'];
	const dtRaw = firstCurve(curves, ['DTC', 'DT'])?.values;
	const dtCorr = curves['DT_ENV_CORR'];

	const bsDefault = isFiniteNum(envParams.bitsizeDefault) && envParams.bitsizeDefault > 0
		? envParams.bitsizeDefault
		: 8.5;
	const washThold = isFiniteNum(envParams.washoutThold) && envParams.washoutThold > 0
		? envParams.washoutThold
		: 2.0;

	let washoutCount = 0;
	let drhoHighCount = 0;
	let badholeCount = 0;
	let invadedCount = 0;
	let cycleSkipCount = 0;
	let waterLikeCnt = 0;
	let hcLikeCnt = 0;
	let anomalousCnt = 0;
	let tightCnt = 0;
	const diValues: number[] = [];
	const n = indices.length;

	for (const idx of indices) {
		const c = cali?.[idx];
		const b = isFiniteNum(bsArr?.[idx]) ? (bsArr![idx] as number) : bsDefault;
		if (isFiniteNum(c) && c - b >= washThold) washoutCount++;
		const dr = drho?.[idx];
		if (isFiniteNum(dr) && Math.abs(dr) > DRHO_LIMIT_DEFAULT) drhoHighCount++;
		const bh = badhole?.[idx];
		if (isFiniteNum(bh) && bh >= 0.5) badholeCount++;
		const inv = invFlag?.[idx];
		if (isFiniteNum(inv) && inv > 0) invadedCount++;
		const ip = invProfile?.[idx];
		if (ip === 1) waterLikeCnt++;
		else if (ip === 2) hcLikeCnt++;
		else if (ip === 3) anomalousCnt++;
		else if (ip === 4) tightCnt++;
		const di = diEst?.[idx];
		if (isFiniteNum(di) && di > 0) diValues.push(di);

		const rMask = reasonArr?.[idx];
		const dtVal = dtRaw?.[idx];
		const dtCorrVal = dtCorr?.[idx];
		if (
			(isFiniteNum(rMask) && (Math.trunc(rMask) & 128) !== 0) ||
			(dtCorr !== undefined && isFiniteNum(dtVal) && !isFiniteNum(dtCorrVal))
		) {
			cycleSkipCount++;
		}
	}

	const specs: ReadonlyArray<{
		key: 'GR' | 'RT' | 'NPHI' | 'RHOB' | 'DT';
		rawCandidates: readonly string[];
	}> = [
		{ key: 'GR', rawCandidates: ['GR'] },
		{ key: 'RT', rawCandidates: ['RT', 'RD', 'RDEP', 'LLD'] },
		{ key: 'NPHI', rawCandidates: ['NPHI'] },
		{ key: 'RHOB', rawCandidates: ['RHOB'] },
		{ key: 'DT', rawCandidates: ['DTC', 'DT'] }
	];

	let hasEnvCorr = false;
	const rows: EnvQcCurveRow[] = [];

	for (const spec of specs) {
		const rawHit = firstCurve(curves, spec.rawCandidates);
		const corrName = `${spec.key}_ENV_CORR`;
		const corrArr = curves[corrName];
		if (corrArr && corrArr.some(isFiniteNum)) {
			hasEnvCorr = true;
		}
		if (!rawHit && !(corrArr && corrArr.some(isFiniteNum))) continue;

		const statusArr = curves[`${spec.key}_QC_STATUS`];
		const confArr = curves[`${spec.key}_QC_CONF`];
		const deltaArr = curves[`${spec.key}_CORR_DELTA`];
		const sigmaArr = curves[`${spec.key}_SIGMA`];

		let valid = 0;
		let good = 0;
		let corrected = 0;
		let unreliable = 0;
		let confSum = 0;
		let confN = 0;
		let deltaSum = 0;
		let absDeltas: number[] = [];
		let sigmaSum = 0;
		let sigmaN = 0;
		const reasonCounts = new Map<string, number>();

		for (const idx of indices) {
			const rawVal = rawHit?.values[idx];
			const corrVal = corrArr?.[idx];
			const st = statusArr?.[idx];

			const samplePresent = isFiniteNum(rawVal) || isFiniteNum(corrVal);
			if (!samplePresent) continue;
			valid++;

			if (isFiniteNum(st) && st >= 0) {
				if (st === 0) good++;
				else if (st === 1) corrected++;
				else if (st === 2) unreliable++;
			} else if (isFiniteNum(corrVal) && isFiniteNum(rawVal)) {
				const rawDelta =
					spec.key === 'RT'
						? Math.log10(Math.max(corrVal, 1e-3)) - Math.log10(Math.max(rawVal, 1e-3))
						: corrVal - rawVal;
				if (Math.abs(rawDelta) > 0.05 * ENV_QC_TOLERANCES[spec.key]) corrected++;
				else good++;
			}

			const cf = confArr?.[idx];
			if (isFiniteNum(cf)) {
				confSum += cf;
				confN++;
			}

			let dl = deltaArr?.[idx];
			if (!isFiniteNum(dl) && isFiniteNum(corrVal) && isFiniteNum(rawVal)) {
				dl =
					spec.key === 'RT'
						? Math.log10(Math.max(corrVal, 1e-3)) - Math.log10(Math.max(rawVal, 1e-3))
						: corrVal - rawVal;
			}
			if (isFiniteNum(dl)) {
				deltaSum += dl;
				absDeltas.push(Math.abs(dl));
			}

			const sg = sigmaArr?.[idx];
			if (isFiniteNum(sg)) {
				sigmaSum += sg;
				sigmaN++;
			}

			const rm = reasonArr?.[idx];
			if (isFiniteNum(rm) && rm > 0) {
				for (const bitDef of ENV_QC_REASON_BITS) {
					if ((Math.trunc(rm) & bitDef.bit) !== 0) {
						reasonCounts.set(bitDef.label, (reasonCounts.get(bitDef.label) ?? 0) + 1);
					}
				}
			}
		}

		if (valid === 0) continue;
		absDeltas = absDeltas.sort((a, b) => a - b);
		const p95Idx = absDeltas.length > 0 ? Math.min(absDeltas.length - 1, Math.floor(0.95 * (absDeltas.length - 1))) : -1;
		let topReason = 'Clean';
		let topReasonCnt = 0;
		for (const [lbl, cnt] of reasonCounts) {
			if (cnt > topReasonCnt) {
				topReason = lbl;
				topReasonCnt = cnt;
			}
		}

		rows.push({
			CURVE: spec.key,
			RAW_CURVE: rawHit?.name ?? spec.key,
			CORR_CURVE: corrName,
			SAMPLES: valid,
			GOOD_PCT: Number(((good * 100) / valid).toFixed(1)),
			CORRECTED_PCT: Number(((corrected * 100) / valid).toFixed(1)),
			UNRELIABLE_PCT: Number(((unreliable * 100) / valid).toFixed(1)),
			MEAN_CONF: confN > 0 ? Number((confSum / confN).toFixed(3)) : null,
			MEAN_DELTA: absDeltas.length > 0 ? Number((deltaSum / absDeltas.length).toFixed(4)) : null,
			MEAN_ABS_DELTA:
				absDeltas.length > 0
					? Number((absDeltas.reduce((a, b) => a + b, 0) / absDeltas.length).toFixed(4))
					: null,
			P95_ABS_DELTA: p95Idx >= 0 ? Number(absDeltas[p95Idx].toFixed(4)) : null,
			MEAN_SIGMA: sigmaN > 0 ? Number((sigmaSum / sigmaN).toFixed(4)) : null,
			UNIT: ENV_QC_UNITS[spec.key],
			DOMINANT_REASON: topReason
		});
	}

	diValues.sort((a, b) => a - b);
	const medianDiIn =
		diValues.length > 0 ? Number(diValues[Math.floor(0.5 * (diValues.length - 1))].toFixed(1)) : null;
	const p90DiIn =
		diValues.length > 0 ? Number(diValues[Math.floor(0.9 * (diValues.length - 1))].toFixed(1)) : null;

	return {
		hasEnvCorr,
		totalSamples: n,
		washoutPct: n > 0 ? Number(((washoutCount * 100) / n).toFixed(1)) : 0,
		drhoHighPct: n > 0 ? Number(((drhoHighCount * 100) / n).toFixed(1)) : 0,
		badholePct: n > 0 ? Number(((badholeCount * 100) / n).toFixed(1)) : 0,
		invadedPct: n > 0 ? Number(((invadedCount * 100) / n).toFixed(1)) : 0,
		cycleSkipCount,
		invasion: {
			invadedSamples: invadedCount,
			totalSamples: n,
			waterLikePct: n > 0 ? Number(((waterLikeCnt * 100) / n).toFixed(1)) : 0,
			hcLikePct: n > 0 ? Number(((hcLikeCnt * 100) / n).toFixed(1)) : 0,
			anomalousPct: n > 0 ? Number(((anomalousCnt * 100) / n).toFixed(1)) : 0,
			tightPct: n > 0 ? Number(((tightCnt * 100) / n).toFixed(1)) : 0,
			medianDiIn,
			p90DiIn
		},
		rows
	};
}

export interface BuildEnvCorrQcPlotOptions {
	curves: Record<string, Array<number | null> | undefined>;
	depthsMeters: Array<number | null>;
	indices: number[];
	depthUnit: 'm' | 'ft';
	viewMode: EnvQcViewMode;
	trackFilter: EnvQcTrackFilter;
	showSigma: boolean;
	showFluidRes: boolean;
	envParams: EnvQcParamsInput;
}

export function buildEnvCorrQcPlot(opts: BuildEnvCorrQcPlotOptions): {
	traces: Record<string, any>[];
	layout: Record<string, any>;
	trackCount: number;
	hasEnvCorr: boolean;
} {
	const {
		curves,
		depthsMeters,
		indices,
		depthUnit,
		viewMode,
		trackFilter,
		showSigma,
		showFluidRes,
		envParams
	} = opts;

	const toDisplayDepth = (dM: number) => (depthUnit === 'ft' ? dM * M_TO_FT : dM);
	const traces: Record<string, any>[] = [];

	// Build contiguous-segment projected depth & index arrays (inserts null on index jumps)
	const yDisplay: Array<number | null> = [];
	const customIdx: Array<number | null> = [];
	const slotIndices: Array<number | null> = [];
	let prevIdx = -2;
	for (const idx of indices) {
		const dM = depthsMeters[idx];
		if (!isFiniteNum(dM)) continue;
		if (prevIdx !== -2 && idx !== prevIdx + 1) {
			yDisplay.push(null);
			customIdx.push(null);
			slotIndices.push(null);
		}
		yDisplay.push(toDisplayDepth(dM));
		customIdx.push(idx);
		slotIndices.push(idx);
		prevIdx = idx;
	}

	const mapSlots = (fn: (idx: number) => number | null): Array<number | null> =>
		slotIndices.map((idx) => (idx === null ? null : fn(idx)));

	const minDepth = yDisplay.reduce<number>(
		(acc, v) => (v !== null && v < acc ? v : acc),
		Number.POSITIVE_INFINITY
	);
	const maxDepth = yDisplay.reduce<number>(
		(acc, v) => (v !== null && v > acc ? v : acc),
		Number.NEGATIVE_INFINITY
	);
	const hasDepthSpan = Number.isFinite(minDepth) && Number.isFinite(maxDepth);

	// Discover available curves
	const caliHit = firstCurve(curves, ['CALI', 'CAL', 'HCAL']);
	const bsHit = firstCurve(curves, ['BS', 'ENV_BS', 'BIT', 'BITSIZE']);
	const drhoHit = firstCurve(curves, ['DRHO', 'HDRA', 'DCOR']);
	const grRawHit = firstCurve(curves, ['GR']);
	const rtRawHit = firstCurve(curves, ['RT', 'RD', 'RDEP', 'LLD']);
	const rmedHit = firstCurve(curves, ['RM', 'ILM', 'RMED', 'RLA3']);
	const rxoHit = firstCurve(curves, ['RXO', 'MSFL', 'RSHAL', 'RS', 'RLA1']);
	const nphiRawHit = firstCurve(curves, ['NPHI']);
	const rhobRawHit = firstCurve(curves, ['RHOB']);
	const dtRawHit = firstCurve(curves, ['DTC', 'DT']);

	const hasFinite = (name: string) => Boolean(curves[name]?.some(isFiniteNum));
	const hasEnvCorr = ['GR_ENV_CORR', 'RT_ENV_CORR', 'NPHI_ENV_CORR', 'RHOB_ENV_CORR', 'DT_ENV_CORR'].some(
		hasFinite
	);

	type TrackKey = 'hole' | 'GR' | 'RT' | 'NPHI' | 'RHOB' | 'DT';
	const activeTracks: Array<{ key: TrackKey; weight: number }> = [];

	const hasAnyData =
		Boolean(caliHit || bsHit || drhoHit || grRawHit || rtRawHit || nphiRawHit || rhobRawHit || dtRawHit) ||
		hasEnvCorr;

	if (hasAnyData && hasDepthSpan && viewMode !== 'invasion') {
		activeTracks.push({ key: 'hole', weight: trackFilter === 'all' ? 0.95 : 0.72 });

		const wantTrack = (k: Exclude<TrackKey, 'hole'>) => {
			if (trackFilter === 'all') return true;
			if (trackFilter === 'NPHI_RHOB') return k === 'NPHI' || k === 'RHOB';
			return trackFilter === k;
		};

		if (wantTrack('GR') && (grRawHit || hasFinite('GR_ENV_CORR') || trackFilter === 'GR')) {
			activeTracks.push({ key: 'GR', weight: 1.0 });
		}
		if (wantTrack('RT') && (rtRawHit || hasFinite('RT_ENV_CORR') || trackFilter === 'RT')) {
			activeTracks.push({ key: 'RT', weight: trackFilter === 'all' ? 1.25 : 1.4 });
		}
		if (wantTrack('NPHI') && (nphiRawHit || hasFinite('NPHI_ENV_CORR') || trackFilter === 'NPHI_RHOB')) {
			activeTracks.push({ key: 'NPHI', weight: 1.0 });
		}
		if (wantTrack('RHOB') && (rhobRawHit || hasFinite('RHOB_ENV_CORR') || trackFilter === 'NPHI_RHOB')) {
			activeTracks.push({ key: 'RHOB', weight: 1.0 });
		}
		if (wantTrack('DT') && (dtRawHit || hasFinite('DT_ENV_CORR') || trackFilter === 'DT')) {
			activeTracks.push({ key: 'DT', weight: 1.0 });
		}
	}

	const layout: Record<string, any> = {
		autosize: true,
		paper_bgcolor: '#FFFFFF',
		plot_bgcolor: '#FFFFFF',
		dragmode: 'pan',
		margin: { l: 54, r: 12, t: 46, b: drhoHit ? 48 : 36 },
		legend: {
			orientation: 'h',
			x: 0,
			y: drhoHit ? -0.11 : -0.06,
			xanchor: 'left',
			yanchor: 'top',
			font: { family: 'Inter', size: 9.5, color: '#334155' }
		},
		yaxis: {
			title: { text: `Depth (${depthUnit})`, font: { family: 'Inter', size: 11, color: '#334155' } },
			autorange: 'reversed',
			showgrid: true,
			gridcolor: '#E2E8F0',
			zeroline: false,
			tickfont: { family: 'JetBrains Mono', size: 9.5, color: '#64748B' }
		},
		shapes: [] as Record<string, any>[]
	};

	if (activeTracks.length === 0 && !(viewMode === 'invasion' && hasAnyData && hasDepthSpan)) {
		return { traces, layout, trackCount: 0, hasEnvCorr };
	}

	const trackCount = viewMode === 'invasion' ? 3 : activeTracks.length;
	const gap = trackCount <= 2 ? 0.04 : trackCount <= 3 ? 0.032 : 0.02;
	const usableWidth = Math.max(0.2, 1 - gap * (trackCount - 1));
	const invasionWeights = [1.05, 1.25, 1.0];
	const totalWeight =
		viewMode === 'invasion'
			? invasionWeights.reduce((a, b) => a + b, 0)
			: activeTracks.reduce((acc, t) => acc + t.weight, 0);

	const xRefFor = (ti: number) => (ti === 0 ? 'x' : `x${ti + 1}`);
	const xLayoutFor = (ti: number) => (ti === 0 ? 'xaxis' : `xaxis${ti + 1}`);

	let cursor = 0;
	const domains: Array<[number, number]> = (
		viewMode === 'invasion' ? invasionWeights : activeTracks.map((t) => t.weight)
	).map((weight, i) => {
		const w = (usableWidth * weight) / totalWeight;
		const start = Number(cursor.toFixed(4));
		const end = i === trackCount - 1 ? 1.0 : Number(Math.min(1, cursor + w).toFixed(4));
		cursor += w + gap;
		return [start, end];
	});

	const reasonArr = curves['ENV_QC_REASON'];
	const severityArr = curves['HOLE_SEVERITY'];
	const bsDefault =
		isFiniteNum(envParams.bitsizeDefault) && envParams.bitsizeDefault > 0
			? envParams.bitsizeDefault
			: 8.5;
	const resolvedBsSample = bsHit?.values.find((v): v is number => isFiniteNum(v) && v > 0);
	const bsLegendName = bsHit
		? bsHit.name === 'ENV_BS' && resolvedBsSample !== undefined
			? `BS (${Number(resolvedBsSample.toFixed(2))}")`
			: bsHit.name
		: `BS (${bsDefault}")`;
	const washThold =
		isFiniteNum(envParams.washoutThold) && envParams.washoutThold > 0
			? envParams.washoutThold
			: 2.0;

	// Helper to add a vertical constant reference line on a track
	const addRefLine = (
		xref: string,
		xVal: number,
		color: string,
		dash: string = 'dot',
		width: number = 1
	) => {
		if (!hasDepthSpan) return;
		traces.push({
			type: 'scatter',
			mode: 'lines',
			x: [xVal, xVal],
			y: [minDepth, maxDepth],
			xaxis: xref,
			yaxis: 'y',
			line: { color, width, dash },
			hoverinfo: 'skip',
			showlegend: false
		});
	};

	// Build self-contained closed polygon subpaths (separated by null) for `fill: 'toself'`.
	// Unlike Plotly's `tonextx` / `tozerox` (which emit `Z M ... Z` chords across any null gap and
	// produce giant diagonal wedges across the track), `toself` closes each contiguous valid
	// interval independently onto itself with zero cross-gap artifacts.
	type RibbonSample = { low: number; high: number; y: number };

	const emitPolygonSubpaths = (
		xref: string,
		segments: RibbonSample[][],
		fillcolor: string,
		name?: string
	) => {
		const polyX: Array<number | null> = [];
		const polyY: Array<number | null> = [];

		for (const seg of segments) {
			if (seg.length < 2) continue;
			if (!seg.some((pt) => Math.abs(pt.high - pt.low) > 1e-9)) continue;

			// Trim leading/trailing zero-width runs while keeping the immediate pinch point
			let i = 0;
			while (i < seg.length) {
				if (Math.abs(seg[i].high - seg[i].low) <= 1e-12) {
					i++;
					continue;
				}
				const start = Math.max(0, i - 1);
				let j = i;
				while (j + 1 < seg.length && Math.abs(seg[j + 1].high - seg[j + 1].low) > 1e-12) {
					j++;
				}
				const end = Math.min(seg.length - 1, j + 1);
				if (end > start) {
					if (polyX.length > 0) {
						polyX.push(null);
						polyY.push(null);
					}
					for (let k = start; k <= end; k++) {
						polyX.push(seg[k].low);
						polyY.push(seg[k].y);
					}
					for (let k = end; k >= start; k--) {
						polyX.push(seg[k].high);
						polyY.push(seg[k].y);
					}
					polyX.push(seg[start].low);
					polyY.push(seg[start].y);
				}
				i = j + 1;
			}
		}

		if (polyX.length === 0) return;

		traces.push({
			type: 'scatter',
			mode: 'lines',
			name: name ?? 'Band',
			x: polyX,
			y: polyY,
			xaxis: xref,
			yaxis: 'y',
			fill: 'toself',
			fillcolor,
			line: { width: 0, color: 'rgba(0,0,0,0)', simplify: false },
			connectgaps: false,
			hoverinfo: 'skip',
			showlegend: Boolean(name)
		});
	};

	const addRibbon = (
		xref: string,
		xLow: Array<number | null>,
		xHigh: Array<number | null>,
		fillcolor: string,
		name?: string
	) => {
		const segments: RibbonSample[][] = [];
		let current: RibbonSample[] = [];
		for (let i = 0; i < yDisplay.length; i++) {
			const lo = xLow[i];
			const hi = xHigh[i];
			const yv = yDisplay[i];
			if (lo !== null && hi !== null && yv !== null && Number.isFinite(lo) && Number.isFinite(hi)) {
				current.push({ low: lo, high: hi, y: yv });
			} else if (current.length > 0) {
				segments.push(current);
				current = [];
			}
		}
		if (current.length > 0) segments.push(current);
		emitPolygonSubpaths(xref, segments, fillcolor, name);
	};

	// One-sided threshold fill ('above': signal > threshold, 'below': signal < threshold)
	// with exact linear interpolation at threshold crossings so washout/mudcake lobes meet cleanly.
	const addThresholdRibbon = (
		xref: string,
		signal: Array<number | null>,
		threshold: Array<number | null>,
		direction: 'above' | 'below',
		fillcolor: string,
		name?: string
	) => {
		const segments: RibbonSample[][] = [];
		let current: RibbonSample[] = [];

		const activeDiff = (s: number, t: number) => (direction === 'above' ? s - t : t - s);
		const makeSample = (s: number, t: number, yv: number): RibbonSample => {
			if (direction === 'above') {
				return { low: t, high: Math.max(s, t), y: yv };
			}
			return { low: Math.min(s, t), high: t, y: yv };
		};

		for (let i = 0; i < yDisplay.length; i++) {
			const s = signal[i];
			const t = threshold[i];
			const yv = yDisplay[i];
			if (s !== null && t !== null && yv !== null && Number.isFinite(s) && Number.isFinite(t)) {
				if (current.length > 0) {
					const prevS = signal[i - 1]!;
					const prevT = threshold[i - 1]!;
					const prevY = yDisplay[i - 1]!;
					const d0 = activeDiff(prevS, prevT);
					const d1 = activeDiff(s, t);
					if ((d0 > 0 && d1 < 0) || (d0 < 0 && d1 > 0)) {
						const frac = d0 / (d0 - d1);
						const crossT = prevT + frac * (t - prevT);
						const crossY = prevY + frac * (yv - prevY);
						current.push({ low: crossT, high: crossT, y: crossY });
					}
				}
				current.push(makeSample(s, t, yv));
			} else if (current.length > 0) {
				segments.push(current);
				current = [];
			}
		}
		if (current.length > 0) segments.push(current);
		emitPolygonSubpaths(xref, segments, fillcolor, name);
	};

	// Helper for standard linear/log sensor tracks (GR, NPHI, RHOB, DT)
	const buildLinearSensorTrack = (
		ti: number,
		logKey: 'GR' | 'NPHI' | 'RHOB' | 'DT',
		titleLabel: string,
		rawHit: { name: string; values: Array<number | null> } | null,
		color: string,
		bandFill: string,
		diffFill: string,
		defaultRange: [number, number],
		reversed: boolean
	) => {
		const xref = xRefFor(ti);
		const xLayoutKey = xLayoutFor(ti);
		const domain = domains[ti];
		const corrName = `${logKey}_ENV_CORR`;
		const corrArr = curves[corrName];
		const sigmaArr = curves[`${logKey}_SIGMA`];
		const confArr = curves[`${logKey}_QC_CONF`];
		const statusArr = curves[`${logKey}_QC_STATUS`];
		const deltaArr = curves[`${logKey}_CORR_DELTA`];
		const unit = ENV_QC_UNITS[logKey];
		const tol = ENV_QC_TOLERANCES[logKey];

		if (viewMode === 'delta') {
			const deltaSeries = mapSlots((idx) => {
				const dl = deltaArr?.[idx];
				if (isFiniteNum(dl)) return dl;
				const r = rawHit?.values[idx];
				const c = corrArr?.[idx];
				return isFiniteNum(r) && isFiniteNum(c) ? c - r : null;
			});
			const sigmaLow = mapSlots((idx) => {
				const sg = sigmaArr?.[idx];
				return isFiniteNum(sg) ? -sg : null;
			});
			const sigmaHigh = mapSlots((idx) => {
				const sg = sigmaArr?.[idx];
				return isFiniteNum(sg) ? sg : null;
			});

			if (showSigma) {
				addRibbon(xref, sigmaLow, sigmaHigh, bandFill, ti === 1 ? '±1σ Uncertainty' : undefined);
			}
			addRefLine(xref, 0, '#94A3B8', 'solid', 1);
			addRefLine(xref, -tol, '#F59E0B', 'dot', 1);
			addRefLine(xref, tol, '#F59E0B', 'dot', 1);

			const hoverMeta = slotIndices.map((idx) => {
				if (idx === null) return '';
				const cf = confArr?.[idx];
				const st = statusArr?.[idx];
				return `Conf: ${isFiniteNum(cf) ? cf.toFixed(2) : '—'} (${decodeEnvQcStatus(st)})`;
			});

			const zeroSeries = deltaSeries.map((v) => (v !== null ? 0 : null));
			addRibbon(xref, zeroSeries, deltaSeries, diffFill);

			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: `Δ${logKey}`,
				x: deltaSeries,
				y: yDisplay,
				customdata: customIdx,
				text: hoverMeta,
				xaxis: xref,
				yaxis: 'y',
				connectgaps: false,
				line: { color, width: 1.6 },
				showlegend: false,
				hovertemplate: `Δ${logKey}: %{x:+.4g} ${unit}<br>Depth: %{y:.2f} ${depthUnit}<br>%{text}<extra></extra>`
			});

			// Mark unreliable samples (status === 2) or cycle skips on DT
			const unrelX: number[] = [];
			const unrelY: number[] = [];
			const unrelIdx: number[] = [];
			for (let i = 0; i < slotIndices.length; i++) {
				const idx = slotIndices[i];
				if (idx === null) continue;
				const st = statusArr?.[idx];
				const dv = deltaSeries[i];
				if (st === 2 && dv !== null) {
					unrelX.push(dv);
					unrelY.push(yDisplay[i]!);
					unrelIdx.push(idx);
				} else if (
					logKey === 'DT' &&
					isFiniteNum(rawHit?.values[idx]) &&
					corrArr !== undefined &&
					!isFiniteNum(corrArr[idx])
				) {
					unrelX.push(tol * 1.15);
					unrelY.push(yDisplay[i]!);
					unrelIdx.push(idx);
				}
			}
			if (unrelX.length > 0) {
				traces.push({
					type: 'scatter',
					mode: 'markers',
					name: logKey === 'DT' ? 'Cycle Skip / Unreliable' : `${logKey} Unreliable`,
					x: unrelX,
					y: unrelY,
					customdata: unrelIdx,
					xaxis: xref,
					yaxis: 'y',
					marker: { color: '#DC2626', symbol: logKey === 'DT' ? 'x' : 'circle-open', size: 5.5 },
					showlegend: false,
					hovertemplate: `${logKey} flagged<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
				});
			}

			let maxAbs = tol * 1.35;
			for (const v of deltaSeries) {
				if (v !== null && Math.abs(v) * 1.15 > maxAbs) maxAbs = Math.abs(v) * 1.15;
			}
			if (showSigma) {
				for (const v of sigmaHigh) {
					if (v !== null && Math.abs(v) * 1.15 > maxAbs) maxAbs = Math.abs(v) * 1.15;
				}
			}

			layout[xLayoutKey] = {
				domain,
				anchor: 'y',
				side: 'top',
				range: reversed ? [maxAbs, -maxAbs] : [-maxAbs, maxAbs],
				showgrid: true,
				gridcolor: '#E2E8F0',
				zeroline: false,
				fixedrange: true,
				title: {
					text: `Δ${logKey} (${unit})`,
					font: { family: 'Inter', size: 10, color }
				},
				tickfont: { family: 'JetBrains Mono', size: 8.5, color },
				linecolor: color,
				tickcolor: color
			};
			return;
		}

		// OVERLAY MODE
		const rawSeries = mapSlots((idx) => {
			const v = rawHit?.values[idx];
			return isFiniteNum(v) ? v : null;
		});
		const corrSeries = mapSlots((idx) => {
			const v = corrArr?.[idx];
			return isFiniteNum(v) ? v : null;
		});
		const hasCorr = corrSeries.some((v) => v !== null);

		if (showSigma && hasCorr && sigmaArr) {
			const sigLow = mapSlots((idx) => {
				const c = corrArr?.[idx];
				const s = sigmaArr[idx];
				return isFiniteNum(c) && isFiniteNum(s) ? c - s : null;
			});
			const sigHigh = mapSlots((idx) => {
				const c = corrArr?.[idx];
				const s = sigmaArr[idx];
				return isFiniteNum(c) && isFiniteNum(s) ? c + s : null;
			});
			addRibbon(xref, sigLow, sigHigh, bandFill);
		}

		if (hasCorr && rawHit) {
			addRibbon(xref, rawSeries, corrSeries, diffFill);
		}

		if (rawHit) {
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: `${rawHit.name} (raw)`,
				x: rawSeries,
				y: yDisplay,
				...(hasCorr ? {} : { customdata: customIdx }),
				xaxis: xref,
				yaxis: 'y',
				connectgaps: false,
				line: { color: '#64748B', width: 1.1, dash: 'dot' },
				opacity: 0.85,
				showlegend: ti === 1,
				hovertemplate: `${rawHit.name} raw: %{x:.4g} ${unit}<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
			});
		}

		if (hasCorr) {
			const hoverDetails = slotIndices.map((idx) => {
				if (idx === null) return '';
				const r = rawHit?.values[idx];
				const c = corrArr?.[idx];
				const dl = isFiniteNum(deltaArr?.[idx])
					? deltaArr![idx]!
					: isFiniteNum(r) && isFiniteNum(c)
						? c - r
						: null;
				const sg = sigmaArr?.[idx];
				const cf = confArr?.[idx];
				const st = statusArr?.[idx];
				const parts: string[] = [];
				if (dl !== null) parts.push(`Δ: ${dl >= 0 ? '+' : ''}${dl.toFixed(4)} ${unit}`);
				if (isFiniteNum(sg)) parts.push(`±1σ: ${sg.toFixed(4)}`);
				if (isFiniteNum(cf)) parts.push(`Conf: ${cf.toFixed(2)} (${decodeEnvQcStatus(st)})`);
				return parts.join(' · ');
			});

			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: corrName,
				x: corrSeries,
				y: yDisplay,
				customdata: customIdx,
				text: hoverDetails,
				xaxis: xref,
				yaxis: 'y',
				connectgaps: false,
				line: { color, width: 1.75 },
				showlegend: true,
				hovertemplate: `${corrName}: %{x:.4g} ${unit}<br>Depth: %{y:.2f} ${depthUnit}<br>%{text}<extra></extra>`
			});
		}

		// Highlight removed sonic cycle skips on Track 6 (DTC finite, DT_ENV_CORR NaN or reason bit 128)
		if (logKey === 'DT' && rawHit && (hasCorr || reasonArr)) {
			const csX: number[] = [];
			const csY: number[] = [];
			const csIdx: number[] = [];
			for (let i = 0; i < slotIndices.length; i++) {
				const idx = slotIndices[i];
				if (idx === null) continue;
				const rv = rawHit.values[idx];
				const cv = corrArr?.[idx];
				const rMask = reasonArr?.[idx];
				const isCs =
					isFiniteNum(rv) &&
					((corrArr !== undefined && !isFiniteNum(cv)) ||
						(isFiniteNum(rMask) && (Math.trunc(rMask) & 128) !== 0));
				if (isCs) {
					csX.push(rv);
					csY.push(yDisplay[i]!);
					csIdx.push(idx);
				}
			}
			if (csX.length > 0) {
				traces.push({
					type: 'scatter',
					mode: 'markers',
					name: 'Cycle Skip Removed',
					x: csX,
					y: csY,
					customdata: csIdx,
					xaxis: xref,
					yaxis: 'y',
					marker: { color: '#DC2626', symbol: 'x', size: 6.5, line: { width: 1.2, color: '#991B1B' } },
					showlegend: true,
					hovertemplate: `Cycle Skip Removed<br>${rawHit.name}: %{x:.1f} us/ft<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
				});
			}
		}

		let lo = Math.min(defaultRange[0], defaultRange[1]);
		let hi = Math.max(defaultRange[0], defaultRange[1]);
		for (const v of [...rawSeries, ...corrSeries]) {
			if (v !== null) {
				if (v < lo) lo = v;
				if (v > hi) hi = v;
			}
		}

		layout[xLayoutKey] = {
			domain,
			anchor: 'y',
			side: 'top',
			range: reversed ? [hi, lo] : [lo, hi],
			showgrid: true,
			gridcolor: '#E2E8F0',
			zeroline: false,
			fixedrange: true,
			title: {
				text: `${titleLabel} (${unit})`,
				font: { family: 'Inter', size: 10, color }
			},
			tickfont: { family: 'JetBrains Mono', size: 8.5, color },
			linecolor: color,
			tickcolor: color
		};
	};

	// Dedicated 3-Track Invasion Profile Mode:
	// Track 1: Borehole & Invasion Diameter (BS, CALI, DI_EST + 10"/30"/80" tool investigation zones)
	// Track 2: Radial Resistivity Array (Rw, Rmf, Rxo/RXO_EST, Rmed, Rt/RT_ENV_CORR)
	// Track 3: Log10(Rxo/Rt) vs Expected Log10(Rmf/Rw) Contrast (Movable HC & Anomaly Quick-Look)
	if (viewMode === 'invasion') {
		layout.margin = { l: 54, r: 12, t: 46, b: 40 };
		layout.legend.y = -0.07;

		const diEstArr = curves['DI_EST'];
		const rxoEstArr = curves['RXO_EST'];
		const rtCorrArr = curves['RT_ENV_CORR'];
		const invProfArr = curves['INVASION_PROFILE'];

		const bsSeries = mapSlots((idx) => {
			const b = bsHit?.values[idx];
			return isFiniteNum(b) && b > 0 ? b : bsDefault;
		});
		const caliSeries = mapSlots((idx) => {
			const c = caliHit?.values[idx];
			return isFiniteNum(c) && c > 0 ? c : bsSeries[0] ?? bsDefault;
		});
		const wallSeries = slotIndices.map((idx, i) => {
			if (idx === null) return null;
			const c = caliHit?.values[idx];
			return isFiniteNum(c) && c > 0 ? c : bsSeries[i];
		});
		const diSeries = mapSlots((idx) => {
			const v = diEstArr?.[idx];
			return isFiniteNum(v) && v > 0 ? v : null;
		});

		// --- TRACK 1: Borehole & Invasion Diameter (in) ---
		const zeroWall = wallSeries.map((w) => (w !== null ? 0 : null));
		addRibbon('x', zeroWall, wallSeries, 'rgba(148, 163, 184, 0.20)', 'Borehole');

		// Shade invaded annulus (wall -> DI_EST) by INVASION_PROFILE regime
		const regimeSpecs: ReadonlyArray<{
			codes: readonly number[];
			fill: string;
			label: string;
		}> = [
			{ codes: [2], fill: 'rgba(22, 163, 74, 0.34)', label: 'HC-like Invasion' },
			{ codes: [1], fill: 'rgba(37, 99, 235, 0.28)', label: 'Water-like Invasion' },
			{ codes: [3], fill: 'rgba(217, 119, 6, 0.36)', label: 'Anomalous Contrast' },
			{ codes: [0, 4], fill: 'rgba(100, 116, 139, 0.22)', label: 'Tight / Unclassified' }
		];

		for (const reg of regimeSpecs) {
			const regHigh = slotIndices.map((idx, i) => {
				if (idx === null) return null;
				const w = wallSeries[i];
				const di = diSeries[i];
				if (w === null || di === null) return null;
				const code = isFiniteNum(invProfArr?.[idx]) ? Math.trunc(invProfArr![idx] as number) : 0;
				return reg.codes.includes(code) ? Math.max(w, di) : w;
			});
			addRibbon('x', wallSeries, regHigh, reg.fill, reg.label);
		}

		// Tool radial characteristic diameters (Shallow 10", Medium 30", Deep 80")
		for (const [dTool, dColor] of [
			[10, '#0D9488'],
			[30, '#2563EB'],
			[80, '#DC2626']
		] as const) {
			addRefLine('x', dTool, dColor, 'dot', 1.0);
		}

		traces.push({
			type: 'scatter',
			mode: 'lines',
			name: bsLegendName,
			x: bsSeries,
			y: yDisplay,
			xaxis: 'x',
			yaxis: 'y',
			connectgaps: false,
			line: { color: '#475569', width: 1.1, dash: 'dash' },
			showlegend: false,
			hovertemplate: `Bit Size: %{x:.2f} in<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
		});

		if (caliHit) {
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: caliHit.name,
				x: caliSeries,
				y: yDisplay,
				customdata: customIdx,
				xaxis: 'x',
				yaxis: 'y',
				connectgaps: false,
				line: { color: '#0F172A', width: 1.4 },
				showlegend: false,
				hovertemplate: `${caliHit.name}: %{x:.2f} in<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
			});
		}

		if (diSeries.some((v) => v !== null)) {
			const diHover = slotIndices.map((idx, i) => {
				if (idx === null) return '';
				const di = diSeries[i];
				const w = wallSeries[i];
				const dr = di !== null && w !== null ? Math.max(0, (di - w) / 2.0) : null;
				const ip = invProfArr?.[idx];
				const parts: string[] = [];
				if (dr !== null) parts.push(`Radial depth: ${dr.toFixed(1)} in`);
				parts.push(decodeInvasionProfile(ip));
				return parts.join(' · ');
			});
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'DI_EST (di)',
				x: diSeries,
				y: yDisplay,
				customdata: customIdx,
				text: diHover,
				xaxis: 'x',
				yaxis: 'y',
				connectgaps: false,
				line: { color: '#7C3AED', width: 1.75 },
				showlegend: true,
				hovertemplate: `Invasion di: %{x:.1f} in<br>Depth: %{y:.2f} ${depthUnit}<br>%{text}<extra></extra>`
			});
		}

		let maxDiPlot = 90;
		for (const v of [...diSeries, ...wallSeries]) {
			if (v !== null && v + 5 > maxDiPlot) maxDiPlot = Math.min(130, v + 5);
		}
		layout.xaxis = {
			domain: domains[0],
			anchor: 'y',
			side: 'top',
			range: [0, maxDiPlot],
			showgrid: true,
			gridcolor: '#E2E8F0',
			zeroline: false,
			fixedrange: true,
			title: {
				text: 'Hole & Invasion di (in)',
				font: { family: 'Inter', size: 10, color: '#7C3AED' }
			},
			tickfont: { family: 'JetBrains Mono', size: 8.5, color: '#475569' },
			linecolor: '#7C3AED',
			tickcolor: '#7C3AED'
		};

		// --- TRACK 2: Radial Resistivities + Fluid Rw/Rmf (log10 Ω·m) ---
		const fluids = resolveFluidResistivities(curves, depthsMeters, envParams);
		const rwSeries = mapSlots((idx) => fluids.rw[idx]);
		const rmfSeries = mapSlots((idx) => fluids.rmf[idx]);
		const rtRawSeries = mapSlots((idx) => {
			const v = rtRawHit?.values[idx];
			return isFiniteNum(v) && v > 0 ? v : null;
		});
		const rtBestSeries = mapSlots((idx) => {
			const c = rtCorrArr?.[idx];
			if (isFiniteNum(c) && c > 0) return c;
			const r = rtRawHit?.values[idx];
			return isFiniteNum(r) && r > 0 ? r : null;
		});
		const rxoBestSeries = mapSlots((idx) => {
			const e = rxoEstArr?.[idx];
			if (isFiniteNum(e) && e > 0) return e;
			const r = rxoHit?.values[idx];
			return isFiniteNum(r) && r > 0 ? r : null;
		});

		addRibbon('x2', rxoBestSeries, rtBestSeries, 'rgba(13, 148, 136, 0.18)', 'Rxo–Rt Separation');
		if (rtRawHit && rtCorrArr?.some(isFiniteNum)) {
			addRibbon('x2', rtRawSeries, rtBestSeries, 'rgba(220, 38, 38, 0.18)');
		}

		let minR = 0.05;
		let maxR = 200.0;
		const trackRTraces: Array<{
			arr: Array<number | null>;
			name: string;
			color: string;
			dash: string;
			width: number;
			withCustom?: boolean;
		}> = [
			{ arr: rwSeries, name: 'Rw (T)', color: '#1D4ED8', dash: 'longdash', width: 1.2 },
			{ arr: rmfSeries, name: 'Rmf (T)', color: '#7C3AED', dash: 'dash', width: 1.2 }
		];
		if (rmedHit) {
			trackRTraces.push({
				arr: mapSlots((idx) => {
					const v = rmedHit.values[idx];
					return isFiniteNum(v) && v > 0 ? v : null;
				}),
				name: `${rmedHit.name} (med)`,
				color: '#2563EB',
				dash: 'dashdot',
				width: 1.05
			});
		}
		if (rxoBestSeries.some((v) => v !== null)) {
			trackRTraces.push({
				arr: rxoBestSeries,
				name: rxoEstArr?.some(isFiniteNum) ? 'RXO_EST' : (rxoHit?.name ?? 'RXO'),
				color: '#0F766E',
				dash: 'dot',
				width: 1.35
			});
		}
		if (rtRawHit && rtCorrArr?.some(isFiniteNum)) {
			trackRTraces.push({
				arr: rtRawSeries,
				name: `${rtRawHit.name} (raw)`,
				color: '#334155',
				dash: 'dot',
				width: 1.25
			});
		}
		if (rtBestSeries.some((v) => v !== null)) {
			trackRTraces.push({
				arr: rtBestSeries,
				name: rtCorrArr?.some(isFiniteNum) ? 'RT_ENV_CORR' : (rtRawHit?.name ?? 'RT'),
				color: '#DC2626',
				dash: 'solid',
				width: 1.75,
				withCustom: true
			});
		}

		for (const tr of trackRTraces) {
			if (!tr.arr.some((v) => v !== null)) continue;
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: tr.name,
				x: tr.arr,
				y: yDisplay,
				...(tr.withCustom ? { customdata: customIdx } : {}),
				xaxis: 'x2',
				yaxis: 'y',
				connectgaps: false,
				line: { color: tr.color, width: tr.width, dash: tr.dash },
				showlegend: true,
				hovertemplate: `${tr.name}: %{x:.4g} Ω·m<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
			});
			for (const v of tr.arr) {
				if (v !== null && v > 0) {
					if (v < minR) minR = v;
					if (v > maxR) maxR = v;
				}
			}
		}

		layout.xaxis2 = {
			domain: domains[1],
			anchor: 'y',
			side: 'top',
			type: 'log',
			range: [
				Math.floor(Math.log10(Math.max(1e-3, minR * 0.85))),
				Math.ceil(Math.log10(Math.max(10, maxR * 1.15)))
			],
			showgrid: true,
			gridcolor: '#E2E8F0',
			zeroline: false,
			fixedrange: true,
			title: {
				text: 'Rw · Rmf · Rxo · Rt (Ω·m)',
				font: { family: 'Inter', size: 10, color: '#DC2626' }
			},
			tickfont: { family: 'JetBrains Mono', size: 8.5, color: '#DC2626' },
			linecolor: '#DC2626',
			tickcolor: '#DC2626'
		};

		// --- TRACK 3: Log10(Rxo/Rt) vs Expected Log10(Rmf/Rw) Ratio Track ---
		const obsRatio = slotIndices.map((idx, i) => {
			if (idx === null) return null;
			const rxoV = rxoBestSeries[i];
			const rtV = rtBestSeries[i];
			return rxoV !== null && rtV !== null && rxoV > 0 && rtV > 0
				? Math.log10(rxoV / rtV)
				: null;
		});
		const expRatio = slotIndices.map((idx, i) => {
			if (idx === null) return null;
			const rmfV = rmfSeries[i];
			const rwV = rwSeries[i];
			return rmfV !== null && rwV !== null && rmfV > 0 && rwV > 0
				? Math.log10(rmfV / rwV)
				: null;
		});

		const expLowBand = expRatio.map((e) => (e !== null ? e - 0.15 : null));
		const expHighBand = expRatio.map((e) => (e !== null ? e + 0.18 : null));
		addRibbon('x3', expLowBand, expHighBand, 'rgba(37, 99, 235, 0.12)');
		addThresholdRibbon(
			'x3',
			obsRatio,
			expLowBand,
			'below',
			'rgba(22, 163, 74, 0.34)',
			'Movable HC (Rxo/Rt < Rmf/Rw)'
		);
		addThresholdRibbon(
			'x3',
			obsRatio,
			expHighBand,
			'above',
			'rgba(217, 119, 6, 0.32)'
		);

		addRefLine('x3', 0, '#94A3B8', 'dot', 1.0);

		if (expRatio.some((v) => v !== null)) {
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'log₁₀(Rmf/Rw)',
				x: expRatio,
				y: yDisplay,
				xaxis: 'x3',
				yaxis: 'y',
				connectgaps: false,
				line: { color: '#1D4ED8', width: 1.3, dash: 'dash' },
				showlegend: true,
				hovertemplate: `Expected log₁₀(Rmf/Rw): %{x:+.2f} dec<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
			});
		}

		if (obsRatio.some((v) => v !== null)) {
			const ratioHover = slotIndices.map((idx) =>
				idx === null ? '' : decodeInvasionProfile(invProfArr?.[idx])
			);
			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: 'log₁₀(Rxo/Rt)',
				x: obsRatio,
				y: yDisplay,
				customdata: customIdx,
				text: ratioHover,
				xaxis: 'x3',
				yaxis: 'y',
				connectgaps: false,
				line: { color: '#0F172A', width: 1.65 },
				showlegend: true,
				hovertemplate: `Observed log₁₀(Rxo/Rt): %{x:+.2f} dec<br>Depth: %{y:.2f} ${depthUnit}<br>%{text}<extra></extra>`
			});
		}

		let minRat = -1.5;
		let maxRat = 1.5;
		for (const v of [...obsRatio, ...expLowBand, ...expHighBand]) {
			if (v !== null) {
				if (v - 0.15 < minRat) minRat = v - 0.15;
				if (v + 0.15 > maxRat) maxRat = v + 0.15;
			}
		}

		layout.xaxis3 = {
			domain: domains[2],
			anchor: 'y',
			side: 'top',
			range: [minRat, maxRat],
			showgrid: true,
			gridcolor: '#E2E8F0',
			zeroline: false,
			fixedrange: true,
			title: {
				text: 'log₁₀(Rxo/Rt) (dec)',
				font: { family: 'Inter', size: 10, color: '#0F172A' }
			},
			tickfont: { family: 'JetBrains Mono', size: 8.5, color: '#334155' },
			linecolor: '#475569',
			tickcolor: '#475569'
		};

		return { traces, layout, trackCount: 3, hasEnvCorr };
	}

	activeTracks.forEach((track, ti) => {
		const xref = xRefFor(ti);
		const xLayoutKey = xLayoutFor(ti);
		const domain = domains[ti];

		if (track.key === 'hole') {
			const bsSeries = mapSlots((idx) => {
				const b = bsHit?.values[idx];
				return isFiniteNum(b) && b > 0 ? b : bsDefault;
			});
			const caliSeries = mapSlots((idx) => {
				const c = caliHit?.values[idx];
				return isFiniteNum(c) ? c : null;
			});

			if (caliHit) {
				// 1. Washout fill (CALI > BS)
				addThresholdRibbon(
					xref,
					caliSeries,
					bsSeries,
					'above',
					'rgba(245, 158, 11, 0.34)',
					'Washout'
				);

				// 2. Severe washout fill (CALI > BS + washoutThold)
				const severeBase = bsSeries.map((b) => (b !== null ? b + washThold : null));
				addThresholdRibbon(xref, caliSeries, severeBase, 'above', 'rgba(220, 38, 38, 0.35)');

				// 3. Mudcake fill (CALI < BS)
				addThresholdRibbon(
					xref,
					caliSeries,
					bsSeries,
					'below',
					'rgba(120, 53, 15, 0.30)',
					'Mudcake'
				);
			}

			traces.push({
				type: 'scatter',
				mode: 'lines',
				name: bsLegendName,
				x: bsSeries,
				y: yDisplay,
				xaxis: xref,
				yaxis: 'y',
				connectgaps: false,
				line: { color: '#475569', width: 1.2, dash: 'dash' },
				showlegend: true,
				hovertemplate: `Bit Size: %{x:.2f} in<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
			});

			if (caliHit) {
				const holeHover = slotIndices.map((idx) => {
					if (idx === null) return '';
					const c = caliHit.values[idx];
					const b = isFiniteNum(bsHit?.values[idx]) ? (bsHit!.values[idx] as number) : bsDefault;
					const dc = isFiniteNum(c) ? c - b : null;
					const dr = drhoHit?.values[idx];
					const sev = severityArr?.[idx];
					const rsn = decodeEnvQcReason(reasonArr?.[idx]);
					const parts: string[] = [];
					if (dc !== null) parts.push(`ΔCALI: ${dc >= 0 ? '+' : ''}${dc.toFixed(2)} in`);
					if (isFiniteNum(dr)) parts.push(`DRHO: ${dr >= 0 ? '+' : ''}${dr.toFixed(3)} g/cc`);
					if (isFiniteNum(sev)) parts.push(`Sev: ${sev}/3`);
					parts.push(`QC: ${rsn}`);
					return parts.join(' · ');
				});

				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: caliHit.name,
					x: caliSeries,
					y: yDisplay,
					customdata: customIdx,
					text: holeHover,
					xaxis: xref,
					yaxis: 'y',
					connectgaps: false,
					line: { color: '#0F172A', width: 1.6 },
					showlegend: true,
					hovertemplate: `${caliHit.name}: %{x:.2f} in<br>Depth: %{y:.2f} ${depthUnit}<br>%{text}<extra></extra>`
				});
			}

			let minHole = Math.min(6.0, bsDefault - 2.0);
			let maxHole = Math.max(16.0, bsDefault + washThold + 4.0);
			for (const v of [...bsSeries, ...caliSeries]) {
				if (v !== null) {
					if (v - 0.5 < minHole) minHole = Math.max(2.0, v - 0.5);
					if (v + 0.5 > maxHole) maxHole = v + 0.5;
				}
			}

			layout[xLayoutKey] = {
				domain,
				anchor: 'y',
				side: 'top',
				range: [minHole, maxHole],
				showgrid: true,
				gridcolor: '#E2E8F0',
				zeroline: false,
				fixedrange: true,
				title: {
					text: 'CALI · BS (in)',
					font: { family: 'Inter', size: 10, color: '#0F172A' }
				},
				tickfont: { family: 'JetBrains Mono', size: 8.5, color: '#334155' },
				linecolor: '#475569',
				tickcolor: '#475569'
			};

			// Secondary overlay axis on Track 1 for DRHO
			if (drhoHit) {
				const drhoAxisIdx = trackCount + 1;
				const drhoXRef = `x${drhoAxisIdx}`;
				const drhoLayoutKey = `xaxis${drhoAxisIdx}`;
				const drhoSeries = mapSlots((idx) => {
					const v = drhoHit.values[idx];
					return isFiniteNum(v) ? v : null;
				});

				// Highlight |DRHO| > 0.15 g/cc
				const posBase = yDisplay.map((y) => (y !== null ? DRHO_LIMIT_DEFAULT : null));
				addThresholdRibbon(drhoXRef, drhoSeries, posBase, 'above', 'rgba(220, 38, 38, 0.24)');

				const negBase = yDisplay.map((y) => (y !== null ? -DRHO_LIMIT_DEFAULT : null));
				addThresholdRibbon(drhoXRef, drhoSeries, negBase, 'below', 'rgba(220, 38, 38, 0.24)');

				addRefLine(drhoXRef, -DRHO_LIMIT_DEFAULT, '#D97706', 'dot', 0.9);
				addRefLine(drhoXRef, DRHO_LIMIT_DEFAULT, '#D97706', 'dot', 0.9);

				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: drhoHit.name,
					x: drhoSeries,
					y: yDisplay,
					customdata: customIdx,
					xaxis: drhoXRef,
					yaxis: 'y',
					connectgaps: false,
					line: { color: '#D97706', width: 1.25 },
					showlegend: true,
					hovertemplate: `${drhoHit.name}: %{x:+.3f} g/cc<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
				});

				let maxAbsDrho = 0.35;
				for (const v of drhoSeries) {
					if (v !== null && Math.abs(v) * 1.15 > maxAbsDrho) maxAbsDrho = Math.abs(v) * 1.15;
				}

				layout[drhoLayoutKey] = {
					domain,
					anchor: 'y',
					overlaying: xref,
					side: 'bottom',
					range: [-maxAbsDrho, maxAbsDrho],
					showgrid: false,
					zeroline: false,
					fixedrange: true,
					title: {
						text: `${drhoHit.name} (g/cc)`,
						font: { family: 'Inter', size: 9.5, color: '#B45309' },
						standoff: 2
					},
					tickfont: { family: 'JetBrains Mono', size: 8, color: '#B45309' },
					linecolor: '#D97706',
					tickcolor: '#D97706'
				};
			}
		} else if (track.key === 'GR') {
			buildLinearSensorTrack(
				ti,
				'GR',
				'GR',
				grRawHit,
				'#15803D',
				'rgba(22, 163, 74, 0.14)',
				'rgba(245, 158, 11, 0.28)',
				[0, 150],
				false
			);
		} else if (track.key === 'RT') {
			const corrArr = curves['RT_ENV_CORR'];
			const sigmaArr = curves['RT_SIGMA'];
			const confArr = curves['RT_QC_CONF'];
			const statusArr = curves['RT_QC_STATUS'];
			const deltaArr = curves['RT_CORR_DELTA'];
			const rxoEstArr = curves['RXO_EST'];
			const diEstArr = curves['DI_EST'];
			const invProfArr = curves['INVASION_PROFILE'];

			if (viewMode === 'delta') {
				const tol = ENV_QC_TOLERANCES.RT;
				const deltaSeries = mapSlots((idx) => {
					const dl = deltaArr?.[idx];
					if (isFiniteNum(dl)) return dl;
					const r = rtRawHit?.values[idx];
					const c = corrArr?.[idx];
					return isFiniteNum(r) && isFiniteNum(c) && r > 0 && c > 0
						? Math.log10(c) - Math.log10(r)
						: null;
				});
				const sigLow = mapSlots((idx) => {
					const s = sigmaArr?.[idx];
					return isFiniteNum(s) ? -s : null;
				});
				const sigHigh = mapSlots((idx) => {
					const s = sigmaArr?.[idx];
					return isFiniteNum(s) ? s : null;
				});

				if (showSigma) {
					addRibbon(xref, sigLow, sigHigh, 'rgba(220, 38, 38, 0.14)');
				}
				addRefLine(xref, 0, '#94A3B8', 'solid', 1);
				addRefLine(xref, -tol, '#F59E0B', 'dot', 1);
				addRefLine(xref, tol, '#F59E0B', 'dot', 1);

				const zeroRt = deltaSeries.map((v) => (v !== null ? 0 : null));
				addRibbon(xref, zeroRt, deltaSeries, 'rgba(220, 38, 38, 0.24)');

				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: 'Δlog₁₀(Rt)',
					x: deltaSeries,
					y: yDisplay,
					customdata: customIdx,
					xaxis: xref,
					yaxis: 'y',
					connectgaps: false,
					line: { color: '#DC2626', width: 1.6 },
					showlegend: false,
					hovertemplate: `Δlog₁₀(Rt): %{x:+.3f} dec<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
				});

				let maxAbs = tol * 1.4;
				for (const v of [...deltaSeries, ...(showSigma ? sigHigh : [])]) {
					if (v !== null && Math.abs(v) * 1.15 > maxAbs) maxAbs = Math.abs(v) * 1.15;
				}

				layout[xLayoutKey] = {
					domain,
					anchor: 'y',
					side: 'top',
					range: [-maxAbs, maxAbs],
					showgrid: true,
					gridcolor: '#E2E8F0',
					zeroline: false,
					fixedrange: true,
					title: {
						text: 'Δlog₁₀Rt (dec)',
						font: { family: 'Inter', size: 10, color: '#DC2626' }
					},
					tickfont: { family: 'JetBrains Mono', size: 8.5, color: '#DC2626' },
					linecolor: '#DC2626',
					tickcolor: '#DC2626'
				};
				return;
			}

			// OVERLAY MODE for Resistivity (log10 axis)
			const rtRawSeries = mapSlots((idx) => {
				const v = rtRawHit?.values[idx];
				return isFiniteNum(v) && v > 0 ? v : null;
			});
			const rtCorrSeries = mapSlots((idx) => {
				const v = corrArr?.[idx];
				return isFiniteNum(v) && v > 0 ? v : null;
			});
			const hasRtCorr = rtCorrSeries.some((v) => v !== null);

			if (showSigma && hasRtCorr && sigmaArr) {
				// RT_SIGMA is in log10 decades
				const sigLow = mapSlots((idx) => {
					const c = corrArr?.[idx];
					const s = sigmaArr[idx];
					return isFiniteNum(c) && c > 0 && isFiniteNum(s)
						? Math.pow(10, Math.log10(c) - s)
						: null;
				});
				const sigHigh = mapSlots((idx) => {
					const c = corrArr?.[idx];
					const s = sigmaArr[idx];
					return isFiniteNum(c) && c > 0 && isFiniteNum(s)
						? Math.pow(10, Math.log10(c) + s)
						: null;
				});
				addRibbon(xref, sigLow, sigHigh, 'rgba(220, 38, 38, 0.14)');
			}

			if (hasRtCorr && rtRawHit) {
				addRibbon(xref, rtRawSeries, rtCorrSeries, 'rgba(220, 38, 38, 0.22)');
			}

			// Fluid resistivities (Rw, Rmf, Rm_mud at formation temperature) when toggled on
			let minPosVal = 0.2;
			let maxPosVal = 200.0;
			if (showFluidRes) {
				const fluids = resolveFluidResistivities(curves, depthsMeters, envParams);
				const rwSeries = mapSlots((idx) => fluids.rw[idx]);
				const rmfSeries = mapSlots((idx) => fluids.rmf[idx]);
				const rmMudSeries = mapSlots((idx) => fluids.rm[idx]);

				for (const [fSeries, fName, fColor, fDash] of [
					[rwSeries, 'Rw (T)', '#1D4ED8', 'longdash'],
					[rmfSeries, 'Rmf (T)', '#7C3AED', 'dash'],
					[rmMudSeries, 'Rm mud (T)', '#B45309', 'dot']
				] as const) {
					if (fSeries.some((v) => v !== null)) {
						traces.push({
							type: 'scatter',
							mode: 'lines',
							name: fName,
							x: fSeries,
							y: yDisplay,
							xaxis: xref,
							yaxis: 'y',
							connectgaps: false,
							line: { color: fColor, width: 1.25, dash: fDash },
							showlegend: true,
							hovertemplate: `${fName}: %{x:.4g} Ω·m<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
						});
						for (const v of fSeries) {
							if (v !== null && v > 0 && v < minPosVal) minPosVal = v;
						}
					}
				}
			}

			if (rmedHit) {
				const rmedSeries = mapSlots((idx) => {
					const v = rmedHit.values[idx];
					return isFiniteNum(v) && v > 0 ? v : null;
				});
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: `${rmedHit.name} (med)`,
					x: rmedSeries,
					y: yDisplay,
					customdata: customIdx,
					xaxis: xref,
					yaxis: 'y',
					connectgaps: false,
					line: { color: '#2563EB', width: 1.05, dash: 'dashdot' },
					opacity: 0.72,
					showlegend: true,
					hovertemplate: `${rmedHit.name}: %{x:.3g} Ω·m<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
				});
				for (const v of rmedSeries) {
					if (v !== null) {
						if (v < minPosVal) minPosVal = v;
						if (v > maxPosVal) maxPosVal = v;
					}
				}
			}

			if (rxoHit) {
				const rxoSeries = mapSlots((idx) => {
					const v = rxoHit.values[idx];
					return isFiniteNum(v) && v > 0 ? v : null;
				});
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: `${rxoHit.name} (shal)`,
					x: rxoSeries,
					y: yDisplay,
					customdata: customIdx,
					xaxis: xref,
					yaxis: 'y',
					connectgaps: false,
					line: { color: '#0D9488', width: 1.05, dash: 'dot' },
					opacity: 0.8,
					showlegend: true,
					hovertemplate: `${rxoHit.name}: %{x:.3g} Ω·m<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
				});
				for (const v of rxoSeries) {
					if (v !== null) {
						if (v < minPosVal) minPosVal = v;
						if (v > maxPosVal) maxPosVal = v;
					}
				}
			}

			if (rxoEstArr && rxoEstArr.some((v) => isFiniteNum(v) && v > 0)) {
				const rxoEstSeries = mapSlots((idx) => {
					const v = rxoEstArr[idx];
					return isFiniteNum(v) && v > 0 ? v : null;
				});
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: 'RXO_EST',
					x: rxoEstSeries,
					y: yDisplay,
					customdata: customIdx,
					xaxis: xref,
					yaxis: 'y',
					connectgaps: false,
					line: { color: '#0F766E', width: 1.35, dash: 'dash' },
					showlegend: true,
					hovertemplate: `RXO_EST: %{x:.3g} Ω·m<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
				});
				for (const v of rxoEstSeries) {
					if (v !== null) {
						if (v < minPosVal) minPosVal = v;
						if (v > maxPosVal) maxPosVal = v;
					}
				}
			}

			if (rtRawHit) {
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: `${rtRawHit.name} (raw)`,
					x: rtRawSeries,
					y: yDisplay,
					...(hasRtCorr ? {} : { customdata: customIdx }),
					xaxis: xref,
					yaxis: 'y',
					connectgaps: false,
					line: { color: '#64748B', width: 1.1, dash: 'dot' },
					showlegend: ti === 1,
					hovertemplate: `${rtRawHit.name} raw: %{x:.4g} Ω·m<br>Depth: %{y:.2f} ${depthUnit}<extra></extra>`
				});
			}

			if (hasRtCorr) {
				const rtHover = slotIndices.map((idx) => {
					if (idx === null) return '';
					const di = diEstArr?.[idx];
					const ip = invProfArr?.[idx];
					const cf = confArr?.[idx];
					const st = statusArr?.[idx];
					const parts: string[] = [];
					if (isFiniteNum(di)) parts.push(`di: ${di.toFixed(0)} in`);
					if (isFiniteNum(ip) && ip > 0) parts.push(decodeInvasionProfile(ip));
					if (isFiniteNum(cf)) parts.push(`Conf: ${cf.toFixed(2)} (${decodeEnvQcStatus(st)})`);
					return parts.join(' · ');
				});
				traces.push({
					type: 'scatter',
					mode: 'lines',
					name: 'RT_ENV_CORR',
					x: rtCorrSeries,
					y: yDisplay,
					customdata: customIdx,
					text: rtHover,
					xaxis: xref,
					yaxis: 'y',
					connectgaps: false,
					line: { color: '#DC2626', width: 1.8 },
					showlegend: true,
					hovertemplate: `RT_ENV_CORR: %{x:.4g} Ω·m<br>Depth: %{y:.2f} ${depthUnit}<br>%{text}<extra></extra>`
				});
			}

			for (const v of [...rtRawSeries, ...rtCorrSeries]) {
				if (v !== null) {
					if (v < minPosVal) minPosVal = v;
					if (v > maxPosVal) maxPosVal = v;
				}
			}

			const logMin = Math.floor(Math.log10(Math.max(1e-3, minPosVal * 0.85)));
			const logMax = Math.ceil(Math.log10(Math.max(10, maxPosVal * 1.15)));

			layout[xLayoutKey] = {
				domain,
				anchor: 'y',
				side: 'top',
				type: 'log',
				range: [logMin, logMax],
				showgrid: true,
				gridcolor: '#E2E8F0',
				zeroline: false,
				fixedrange: true,
				title: {
					text: showFluidRes ? 'Rt · Rxo · Rw/Rmf (Ω·m)' : 'Rt · Rxo (Ω·m)',
					font: { family: 'Inter', size: 10, color: '#DC2626' }
				},
				tickfont: { family: 'JetBrains Mono', size: 8.5, color: '#DC2626' },
				linecolor: '#DC2626',
				tickcolor: '#DC2626'
			};
		} else if (track.key === 'NPHI') {
			buildLinearSensorTrack(
				ti,
				'NPHI',
				'NPHI',
				nphiRawHit,
				'#2563EB',
				'rgba(37, 99, 235, 0.14)',
				'rgba(37, 99, 235, 0.28)',
				[0.45, -0.05],
				true
			);
		} else if (track.key === 'RHOB') {
			buildLinearSensorTrack(
				ti,
				'RHOB',
				'RHOB',
				rhobRawHit,
				'#DC2626',
				'rgba(220, 38, 38, 0.17)',
				'rgba(220, 38, 38, 0.28)',
				[1.95, 2.95],
				false
			);
		} else if (track.key === 'DT') {
			buildLinearSensorTrack(
				ti,
				'DT',
				dtRawHit?.name ?? 'DTC',
				dtRawHit,
				'#059669',
				'rgba(5, 150, 105, 0.14)',
				'rgba(5, 150, 105, 0.26)',
				[140, 40],
				true
			);
		}
	});

	return { traces, layout, trackCount, hasEnvCorr };
}
