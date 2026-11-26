/**
 * Combined 2-Track Temperature & Pressure Modelling Profile builder for Single-Well QC.
 *
 * Indexed strictly in True Vertical Depth Sub-Sea (TVDSS, positive downward below MSL),
 * matching reservoir engineering & FPT conventions:
 * - Track 1: Geothermal / Offshore Seabed Temperature Model vs. Measured TEMP, TEMP_FL,
 *            Header BHT, and MDT/RFT gauge temperatures vs. TVDSS.
 * - Track 2: Hydrostatic Pore Pressure vs. Borehole Mud Column Pressure (with closed-polygon
 *            overbalance / underbalance shading) + Single-Well RFT/MDT pressure_tests vs. TVDSS.
 */

import type { EnvQcParamsInput } from './envCorrQc.ts';
import { computeMinimumCurvature } from './trajectory.ts';

export type TempDisplayUnit = 'degC' | 'degF';
export type PressureDisplayUnit = 'psi' | 'MPa';
export type TempPresDepthBasis =
	| 'TVDSS'
	| 'TVD_MINUS_KB'
	| 'TVD_NO_DATUM'
	| 'MD_MINUS_KB'
	| 'MD_FALLBACK';

const M_TO_FT = 3.280839895013123;
const PSI_TO_MPA = 0.006894757293168361;
const SEAWATER_PSI_PER_FT = 0.4446;
const BRINE_PSI_PER_FT = 0.465;
const PPG_PER_GCC = 8.3454;

function isFiniteNum(v: unknown): v is number {
	return typeof v === 'number' && Number.isFinite(v);
}

/**
 * Safely parse an optional numeric field without coercing `null`, `undefined`, or `""` to `0`
 * (`Number(null) === 0` in JS, which otherwise pins null `tvdss` fields to surface depth 0).
 */
function parseOptionalFinite(v: unknown): number | null {
	if (v === null || v === undefined || typeof v === 'boolean') return null;
	if (typeof v === 'string' && v.trim() === '') return null;
	const n = typeof v === 'number' ? v : Number(v);
	return Number.isFinite(n) ? n : null;
}

export function cToDisplayTemp(tempC: number, unit: TempDisplayUnit): number {
	return unit === 'degF' ? (tempC * 9.0) / 5.0 + 32.0 : tempC;
}

export function psiToDisplayPressure(psi: number, unit: PressureDisplayUnit): number {
	return unit === 'MPa' ? psi * PSI_TO_MPA : psi;
}

export function normalizePressureToPsi(value: number, uom?: string | null): number {
	const u = String(uom ?? 'psi')
		.trim()
		.toLowerCase();
	if (u === 'mpa') return value / PSI_TO_MPA;
	if (u === 'kpa') return value / (PSI_TO_MPA * 1000.0);
	if (u === 'bar' || u === 'bara' || u === 'barg') return value * 14.5037738;
	return value;
}

export function normalizeTempToC(value: number, uom?: string | null): number {
	const u = String(uom ?? 'degC')
		.trim()
		.toLowerCase();
	if (u === 'degf' || u === 'f' || u === '°f') return ((value - 32.0) * 5.0) / 9.0;
	return value;
}

export interface TempPressureModelSummary {
	hasTrueTvdss: boolean;
	depthBasis: TempPresDepthBasis;
	depthBasisLabel: string;
	yAxisLabel: string;
	datumElevM: number;
	offshore: boolean;
	seabedTvdssM: number;
	topTempDisplay: number;
	gradientCPerKm: number;
	gradientFPer100Ft: number;
	mudWeightPpg: number;
	bhtDisplay: number | null;
	bhtTvdssDisplay: number | null;
	meanOverbalanceDisplay: number | null;
	minOverbalanceDisplay: number | null;
	underbalancedCount: number;
	mdtCount: number;
	tempUnitLabel: string;
	presUnitLabel: string;
	tableTitle: string;
	tableKind: 'mdt' | 'zonal';
	tableRows: Array<Record<string, any>>;
}

export interface BuildTempPressurePlotOptions {
	curves: Record<string, Array<number | null> | undefined>;
	depthsMeters: Array<number | null>;
	indices: number[];
	depthUnit: 'm' | 'ft';
	tempUnit: TempDisplayUnit;
	presUnit: PressureDisplayUnit;
	envParams: EnvQcParamsInput;
	header?: Record<string, any>;
	config?: Record<string, any>;
	survey?: Array<Record<string, any>>;
	pressureTests?: Array<Record<string, any>>;
	tops?: Array<{ name: string; depth: number }>;
}

function resolveDatumElevationMeters(
	header: Record<string, any>,
	config: Record<string, any> = {}
): number {
	const raw =
		header?.df_elev ??
		header?.kb_elev ??
		config?.df_elev ??
		config?.kb_elev ??
		header?.dfe_elevation ??
		header?.kb_elevation ??
		header?.raw_well_header?.EDF ??
		header?.raw_well_header?.DF ??
		header?.raw_well_header?.DFE ??
		header?.raw_well_header?.EKB ??
		header?.raw_well_header?.KB ??
		header?.raw_well_header?.RKB ??
		header?.raw_well_header?.APD;
	const num = parseOptionalFinite(raw);
	if (num === null || num <= 0) return 0;
	const hdrDatumFt = String(header?.datum_unit ?? config?.datum_unit ?? '')
		.trim()
		.toLowerCase()
		.startsWith('f');
	return hdrDatumFt ? num / M_TO_FT : num;
}

function resolveEnvironmentGeometry(
	envParams: EnvQcParamsInput,
	header: Record<string, any>,
	depthUnit: 'm' | 'ft',
	kbM: number
) {
	const hdrDatumFt = String(header?.datum_unit ?? '')
		.trim()
		.toLowerCase()
		.startsWith('f');
	const hdrWd = Number(header?.water_depth);
	const paramWd = isFiniteNum(envParams.waterDepth) ? envParams.waterDepth : 0;
	const offshore =
		Boolean(envParams.offshore && paramWd > 0) ||
		(!envParams.offshore && Number.isFinite(hdrWd) && hdrWd > 0 && paramWd > 0);
	const wdM =
		envParams.offshore && paramWd > 0
			? depthUnit === 'ft'
				? paramWd / M_TO_FT
				: paramWd
			: offshore && Number.isFinite(hdrWd) && hdrWd > 0
				? hdrDatumFt
					? hdrWd / M_TO_FT
					: hdrWd
				: 0;
	// In TVD (from KB/DF), seabed is at water_depth + air_gap (KB).
	// In TVDSS (below MSL), seabed is directly at water_depth!
	const seabedTvdM = offshore ? wdM + kbM : 0;
	const seabedTvdssM = offshore ? wdM : 0;

	const surfTempC =
		(((isFiniteNum(envParams.surfaceTemp) ? envParams.surfaceTemp : 75.0) - 32.0) * 5.0) / 9.0;
	const seabedTempC = isFiniteNum(envParams.seabedTempC) ? envParams.seabedTempC : 4.0;
	const topTempC = offshore ? seabedTempC : surfTempC;

	const gradFPer100Ft = isFiniteNum(envParams.tempGrad) ? envParams.tempGrad : 1.5;
	const gradCPerKm = gradFPer100Ft * (5.0 / 9.0) * (1000.0 / 30.48);

	const rawMw =
		isFiniteNum(envParams.mudWeight) && envParams.mudWeight > 0
			? envParams.mudWeight
			: Number(header?.mud_density ?? 9.5);
	const mwVal = Number.isFinite(rawMw) && rawMw > 0 ? rawMw : 9.5;
	const mudWeightPpg = mwVal <= 3.0 ? mwVal * PPG_PER_GCC : mwVal;

	const tempCAtTvdM = (tvdM: number) =>
		topTempC + (gradCPerKm * Math.max(0, tvdM - seabedTvdM)) / 1000.0;

	const hydroPsiAtTvdssM = (tvdssM: number, tvdM: number) => {
		if (offshore && wdM > 0) {
			const seaColFt = Math.min(Math.max(0, tvdssM), seabedTvdssM) * M_TO_FT;
			const belowFt = Math.max(0, tvdssM - seabedTvdssM) * M_TO_FT;
			return SEAWATER_PSI_PER_FT * seaColFt + BRINE_PSI_PER_FT * belowFt;
		}
		// Onshore: hydrostatic column referenced to surface/TVDSS (falls back to TVD when KB=0)
		const effVertM = kbM > 0 ? Math.max(0, tvdssM + kbM) : Math.max(0, tvdM);
		return BRINE_PSI_PER_FT * effVertM * M_TO_FT;
	};

	// Static mud column always acts from Kelly Bushing / Rig Floor (TVD from KB/DF)
	const mudPsiAtTvdM = (tvdM: number) => 0.052 * mudWeightPpg * Math.max(0, tvdM) * M_TO_FT;

	return {
		offshore,
		wdM,
		seabedTvdM,
		seabedTvdssM,
		topTempC,
		gradCPerKm,
		gradFPer100Ft,
		mudWeightPpg,
		tempCAtTvdM,
		hydroPsiAtTvdssM,
		mudPsiAtTvdM
	};
}

export function buildTempPressurePlot(opts: BuildTempPressurePlotOptions): {
	traces: Record<string, any>[];
	layout: Record<string, any>;
	summary: TempPressureModelSummary;
} {
	const {
		curves,
		depthsMeters,
		indices,
		depthUnit,
		tempUnit,
		presUnit,
		envParams,
		header = {},
		config = {},
		survey = [],
		pressureTests = [],
		tops = []
	} = opts;

	const tempLabel = tempUnit === 'degF' ? '°F' : '°C';
	const presLabel = presUnit === 'MPa' ? 'MPa' : 'psi';
	const toDispDepth = (dM: number) => (depthUnit === 'ft' ? dM * M_TO_FT : dM);

	const tvdssCurve = curves['TVDSS'];
	const tvdCurve = curves['TVD'] ?? curves['DEPT_TVD'] ?? curves['DEPTH_TVD'];
	const hasTvdssCurve = Boolean(tvdssCurve && tvdssCurve.some(isFiniteNum));
	const hasTvdCurve = Boolean(tvdCurve && tvdCurve.some(isFiniteNum));

	let kbM = resolveDatumElevationMeters(header, config);
	// If KB/DF datum is missing in header/config, infer it when TVDSS at surface (MD ≈ 0) is negative
	if (kbM <= 0 && hasTvdssCurve) {
		for (let i = 0; i < depthsMeters.length; i++) {
			const md = depthsMeters[i];
			const tss = tvdssCurve?.[i];
			if (isFiniteNum(md) && isFiniteNum(tss)) {
				const tvd = tvdCurve?.[i];
				if (isFiniteNum(tvd) && tvd - tss > 0.5) {
					kbM = tvd - tss;
				} else if (md <= 5.0 && md - tss > 0.5) {
					kbM = md - tss;
				}
				break;
			}
		}
	}

	const hdrDatumFt = String(header?.datum_unit ?? config?.datum_unit ?? '')
		.trim()
		.toLowerCase()
		.startsWith('f');
	const geom = resolveEnvironmentGeometry(envParams, header, depthUnit, kbM);

	// Compute directional survey trajectory stations when `survey` table rows are available
	const computedSurvey =
		survey && survey.length > 0 ? computeMinimumCurvature(survey, kbM) : [];
	const surveyStations: Array<{ md: number; tvdss: number; tvd: number }> = [];
	for (const st of computedSurvey) {
		const md = parseOptionalFinite(st?.md);
		if (md === null) continue;
		const stTvd = parseOptionalFinite(st?.tvd);
		const stTvdss = parseOptionalFinite(st?.tvdss);
		if (stTvdss !== null || stTvd !== null) {
			const tvd = stTvd !== null ? Math.max(0, stTvd) : Math.max(0, stTvdss! + kbM);
			const tvdss = stTvdss !== null ? stTvdss : kbM > 0 ? tvd - kbM : tvd;
			surveyStations.push({ md, tvdss, tvd });
		}
	}
	surveyStations.sort((a, b) => a.md - b.md);
	const hasSurveyTrajectory = surveyStations.length > 0;

	let depthBasis: TempPresDepthBasis;
	let depthBasisLabel: string;
	let yAxisLabel: string;
	const hasTrueTvdss =
		hasTvdssCurve ||
		(hasTvdCurve && kbM > 0) ||
		(hasSurveyTrajectory && (kbM > 0 || surveyStations.some((s) => s.tvdss < s.md - 0.5)));

	if (hasTvdssCurve) {
		depthBasis = 'TVDSS';
		depthBasisLabel = 'TVDSS (subsea curve)';
		yAxisLabel = `TVDSS (${depthUnit})`;
	} else if (hasSurveyTrajectory && kbM > 0) {
		depthBasis = 'TVDSS';
		depthBasisLabel = `TVDSS (survey − ${toDispDepth(kbM).toFixed(1)}${depthUnit} KB/DF)`;
		yAxisLabel = `TVDSS (${depthUnit})`;
	} else if (hasTvdCurve && kbM > 0) {
		depthBasis = 'TVD_MINUS_KB';
		depthBasisLabel = `TVDSS (TVD − ${toDispDepth(kbM).toFixed(1)}${depthUnit} KB/DF)`;
		yAxisLabel = `TVDSS (${depthUnit})`;
	} else if (hasSurveyTrajectory) {
		depthBasis = 'TVD_NO_DATUM';
		depthBasisLabel = 'TVDSS ≈ TVD (survey, KB/DF = 0)';
		yAxisLabel = `TVDSS / TVD (${depthUnit})`;
	} else if (hasTvdCurve) {
		depthBasis = 'TVD_NO_DATUM';
		depthBasisLabel = 'TVDSS ≈ TVD (KB/DF = 0)';
		yAxisLabel = `TVDSS / TVD (${depthUnit})`;
	} else if (kbM > 0) {
		depthBasis = 'MD_MINUS_KB';
		depthBasisLabel = `TVDSS est. (MD − ${toDispDepth(kbM).toFixed(1)}${depthUnit} KB, vertical assumed)`;
		yAxisLabel = `TVDSS est. (${depthUnit}, vertical assumed)`;
	} else {
		depthBasis = 'MD_FALLBACK';
		depthBasisLabel = 'MD used as TVDSS (no survey / KB datum)';
		yAxisLabel = `TVDSS* (${depthUnit}, MD fallback)`;
	}

	// Build monotonic (MD -> TVDSS, TVD) lookup strictly from valid curve/survey trajectory stations
	// so trailing NaN curve rows never pollute interpolation or cause end-of-well datum jumps.
	const mdLookup: Array<{ md: number; tvdss: number; tvd: number }> = [];
	if (hasTvdssCurve || hasTvdCurve) {
		for (let i = 0; i < depthsMeters.length; i++) {
			const md = depthsMeters[i];
			if (!isFiniteNum(md)) continue;
			const tss = tvdssCurve?.[i];
			const tvd = tvdCurve?.[i];
			if (isFiniteNum(tss)) {
				const resolvedTvd = isFiniteNum(tvd) && tvd >= 0 ? tvd : Math.max(0, tss + kbM);
				mdLookup.push({ md, tvdss: tss, tvd: resolvedTvd });
			} else if (isFiniteNum(tvd)) {
				mdLookup.push({
					md,
					tvdss: kbM > 0 ? tvd - kbM : tvd,
					tvd: Math.max(0, tvd)
				});
			}
		}
	}
	if (mdLookup.length === 0 && hasSurveyTrajectory) {
		for (const s of surveyStations) {
			mdLookup.push(s);
		}
	}
	mdLookup.sort((a, b) => a.md - b.md);

	const interpolateAtMdM = (mdM: number): { tvdssM: number; tvdM: number } => {
		if (mdLookup.length === 0) {
			return { tvdssM: kbM > 0 ? mdM - kbM : mdM, tvdM: Math.max(0, mdM) };
		}
		if (mdLookup.length === 1) {
			const dMd = mdM - mdLookup[0].md;
			return {
				tvdssM: mdLookup[0].tvdss + dMd,
				tvdM: Math.max(0, mdLookup[0].tvd + dMd)
			};
		}
		if (mdM <= mdLookup[0].md) {
			const p0 = mdLookup[0];
			const p1 = mdLookup[1];
			const span = p1.md - p0.md;
			const slopeTvdss = Math.abs(span) > 1e-4 ? (p1.tvdss - p0.tvdss) / span : 1.0;
			const slopeTvd = Math.abs(span) > 1e-4 ? (p1.tvd - p0.tvd) / span : 1.0;
			return {
				tvdssM: p0.tvdss + (mdM - p0.md) * slopeTvdss,
				tvdM: Math.max(0, p0.tvd + (mdM - p0.md) * slopeTvd)
			};
		}
		const last = mdLookup.length - 1;
		if (mdM >= mdLookup[last].md) {
			const p0 = mdLookup[last - 1];
			const p1 = mdLookup[last];
			const span = p1.md - p0.md;
			const slopeTvdss = Math.abs(span) > 1e-4 ? (p1.tvdss - p0.tvdss) / span : 1.0;
			const slopeTvd = Math.abs(span) > 1e-4 ? (p1.tvd - p0.tvd) / span : 1.0;
			return {
				tvdssM: p1.tvdss + (mdM - p1.md) * slopeTvdss,
				tvdM: Math.max(0, p1.tvd + (mdM - p1.md) * slopeTvd)
			};
		}
		let lo = 0;
		let hi = last;
		while (lo <= hi) {
			const mid = (lo + hi) >> 1;
			if (mdLookup[mid].md < mdM) lo = mid + 1;
			else hi = mid - 1;
		}
		const i0 = Math.max(0, lo - 1);
		const i1 = Math.min(last, i0 + 1);
		const p0 = mdLookup[i0];
		const p1 = mdLookup[i1];
		const span = p1.md - p0.md;
		const frac = Math.abs(span) > 1e-6 ? (mdM - p0.md) / span : 0;
		return {
			tvdssM: p0.tvdss + frac * (p1.tvdss - p0.tvdss),
			tvdM: Math.max(0, p0.tvd + frac * (p1.tvd - p0.tvd))
		};
	};

	const resolveSampleDepthsM = (idx: number, mdM: number): { tvdssM: number; tvdM: number } => {
		const tss = tvdssCurve?.[idx];
		const tvd = tvdCurve?.[idx];
		if (isFiniteNum(tss)) {
			const resolvedTvd = isFiniteNum(tvd) && tvd >= 0 ? tvd : Math.max(0, tss + kbM);
			return { tvdssM: tss, tvdM: resolvedTvd };
		}
		if (isFiniteNum(tvd)) {
			return { tvdssM: kbM > 0 ? tvd - kbM : tvd, tvdM: Math.max(0, tvd) };
		}
		if (mdLookup.length > 0) {
			return interpolateAtMdM(mdM);
		}
		return { tvdssM: kbM > 0 ? mdM - kbM : mdM, tvdM: Math.max(0, mdM) };
	};

	const traces: Record<string, any>[] = [];

	// Build contiguous-segment projected TVDSS depth & index arrays
	const yTvdssDisplay: Array<number | null> = [];
	const mdDisplayArr: Array<number | null> = [];
	const slotIndices: Array<number | null> = [];
	const customIdx: Array<number | null> = [];
	let prevIdx = -2;
	for (const idx of indices) {
		const mdM = depthsMeters[idx];
		if (!isFiniteNum(mdM)) continue;
		if (prevIdx !== -2 && idx !== prevIdx + 1) {
			yTvdssDisplay.push(null);
			mdDisplayArr.push(null);
			slotIndices.push(null);
			customIdx.push(null);
		}
		const { tvdssM } = resolveSampleDepthsM(idx, mdM);
		yTvdssDisplay.push(toDispDepth(tvdssM));
		mdDisplayArr.push(toDispDepth(mdM));
		slotIndices.push(idx);
		customIdx.push(idx);
		prevIdx = idx;
	}

	// Closed-polygon `toself` ribbon helper (gap-safe)
	type RibbonSample = { low: number; high: number; y: number };
	const emitClosedRibbon = (
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

	const addDirectionalRibbon = (
		xref: string,
		signal: Array<number | null>,
		baseline: Array<number | null>,
		direction: 'above' | 'below',
		fillcolor: string,
		name?: string
	) => {
		const segments: RibbonSample[][] = [];
		let current: RibbonSample[] = [];
		const diffFn = (s: number, b: number) => (direction === 'above' ? s - b : b - s);
		for (let i = 0; i < yTvdssDisplay.length; i++) {
			const s = signal[i];
			const b = baseline[i];
			const yv = yTvdssDisplay[i];
			if (s !== null && b !== null && yv !== null && Number.isFinite(s) && Number.isFinite(b)) {
				if (current.length > 0) {
					const ps = signal[i - 1]!;
					const pb = baseline[i - 1]!;
					const py = yTvdssDisplay[i - 1]!;
					const d0 = diffFn(ps, pb);
					const d1 = diffFn(s, b);
					if ((d0 > 0 && d1 < 0) || (d0 < 0 && d1 > 0)) {
						const frac = d0 / (d0 - d1);
						const crossX = pb + frac * (b - pb);
						const crossY = py + frac * (yv - py);
						current.push({ low: crossX, high: crossX, y: crossY });
					}
				}
				current.push(
					direction === 'above'
						? { low: b, high: Math.max(s, b), y: yv }
						: { low: Math.min(s, b), high: b, y: yv }
				);
			} else if (current.length > 0) {
				segments.push(current);
				current = [];
			}
		}
		if (current.length > 0) segments.push(current);
		emitClosedRibbon(xref, segments, fillcolor, name);
	};

	const mdHoverText = mdDisplayArr.map((md) =>
		md !== null ? `MD: ${md.toFixed(1)} ${depthUnit}` : ''
	);

	// --- TRACK 1: Formation & Borehole Temperature vs. TVDSS ---
	const envTempCArr = curves['ENV_TEMP_C'];
	const measTempArr = curves['TEMP'];
	const tempFlArr = curves['TEMP_FL'];
	const hasEnvTemp = Boolean(envTempCArr && envTempCArr.some(isFiniteNum));

	// Check whether stored ENV_TEMP_C agrees with live Geothermal Model (e.g. before vs after unit change)
	let maxEnvVsLiveDiffC = 0;
	if (hasEnvTemp) {
		for (const idx of slotIndices) {
			if (idx === null) continue;
			const ec = envTempCArr?.[idx];
			const mdM = depthsMeters[idx];
			if (isFiniteNum(ec) && isFiniteNum(mdM)) {
				const { tvdM } = resolveSampleDepthsM(idx, mdM);
				const diff = Math.abs(ec - geom.tempCAtTvdM(tvdM));
				if (diff > maxEnvVsLiveDiffC) maxEnvVsLiveDiffC = diff;
			}
		}
	}
	const envTempMatchesLive = maxEnvVsLiveDiffC <= 5.0;

	const liveGeoTempSeries = slotIndices.map((idx) => {
		if (idx === null) return null;
		const mdM = depthsMeters[idx];
		if (!isFiniteNum(mdM)) return null;
		const { tvdM } = resolveSampleDepthsM(idx, mdM);
		return cToDisplayTemp(geom.tempCAtTvdM(tvdM), tempUnit);
	});

	const modTempSeries = slotIndices.map((idx, i) => {
		if (idx === null) return null;
		if (!hasEnvTemp) return liveGeoTempSeries[i];
		const ec = envTempCArr?.[idx];
		if (isFiniteNum(ec)) return cToDisplayTemp(ec, tempUnit);
		// Only fill NaN gaps in ENV_TEMP_C from liveGeoTempSeries when they share the same model scale;
		// otherwise keep null so stale stored ENV_TEMP_C never jumps horizontally at NaN boundaries.
		return envTempMatchesLive ? liveGeoTempSeries[i] : null;
	});

	const measTempSeries = slotIndices.map((idx) => {
		if (idx === null) return null;
		const v = measTempArr?.[idx];
		return isFiniteNum(v) ? cToDisplayTemp(v, tempUnit) : null;
	});

	const tempFlSeries = slotIndices.map((idx) => {
		if (idx === null) return null;
		const v = tempFlArr?.[idx];
		return isFiniteNum(v) ? cToDisplayTemp(v, tempUnit) : null;
	});

	if (measTempSeries.some((v) => v !== null)) {
		addDirectionalRibbon('x', measTempSeries, modTempSeries, 'above', 'rgba(220, 38, 38, 0.16)');
		addDirectionalRibbon('x', measTempSeries, modTempSeries, 'below', 'rgba(37, 99, 235, 0.16)');
		traces.push({
			type: 'scatter',
			mode: 'lines',
			name: 'TEMP (log)',
			x: measTempSeries,
			y: yTvdssDisplay,
			text: mdHoverText,
			xaxis: 'x',
			yaxis: 'y',
			connectgaps: false,
			line: { color: '#64748B', width: 1.25, dash: 'dot' },
			showlegend: true,
			hovertemplate: `TEMP log: %{x:.1f} ${tempLabel}<br>TVDSS: %{y:.2f} ${depthUnit} (%{text})<extra></extra>`
		});
	}

	if (tempFlSeries.some((v) => v !== null)) {
		traces.push({
			type: 'scatter',
			mode: 'lines',
			name: 'TEMP_FL',
			x: tempFlSeries,
			y: yTvdssDisplay,
			text: mdHoverText,
			xaxis: 'x',
			yaxis: 'y',
			connectgaps: false,
			line: { color: '#EA580C', width: 1.25, dash: 'dash' },
			showlegend: true,
			hovertemplate: `TEMP_FL: %{x:.1f} ${tempLabel}<br>TVDSS: %{y:.2f} ${depthUnit} (%{text})<extra></extra>`
		});
	}

	if (hasEnvTemp && !envTempMatchesLive && liveGeoTempSeries.some((v) => v !== null)) {
		traces.push({
			type: 'scatter',
			mode: 'lines',
			name: 'Geothermal Model',
			x: liveGeoTempSeries,
			y: yTvdssDisplay,
			text: mdHoverText,
			xaxis: 'x',
			yaxis: 'y',
			connectgaps: false,
			line: { color: '#991B1B', width: 1.4, dash: 'dashdot' },
			showlegend: true,
			hovertemplate: `Geothermal Model: %{x:.1f} ${tempLabel}<br>TVDSS: %{y:.2f} ${depthUnit} (%{text})<extra></extra>`
		});
	}

	if (modTempSeries.some((v) => v !== null)) {
		traces.push({
			type: 'scatter',
			mode: 'lines',
			name: hasEnvTemp ? 'ENV_TEMP' : 'Geothermal Model',
			x: modTempSeries,
			y: yTvdssDisplay,
			customdata: customIdx,
			text: mdHoverText,
			xaxis: 'x',
			yaxis: 'y',
			connectgaps: false,
			line: { color: '#DC2626', width: 1.9 },
			showlegend: true,
			hovertemplate: `Modeled T: %{x:.1f} ${tempLabel}<br>TVDSS: %{y:.2f} ${depthUnit} (%{text})<extra></extra>`
		});
	}

	// Header BHT anchor point (projected onto TVDSS)
	let bhtDisplay: number | null = null;
	let bhtTvdssDisplay: number | null = null;
	const rawBht = parseOptionalFinite(header?.bht);
	if (rawBht !== null && rawBht > 0) {
		const bhtC = normalizeTempToC(rawBht, header?.bht_unit || 'degC');
		bhtDisplay = Number(cToDisplayTemp(bhtC, tempUnit).toFixed(1));
		const rawBhtDepth = parseOptionalFinite(header?.bht_depth);
		let bhtMdM: number | null = null;
		if (rawBhtDepth !== null && rawBhtDepth > 0) {
			bhtMdM = hdrDatumFt ? rawBhtDepth / M_TO_FT : rawBhtDepth;
		} else {
			const finiteDepths = depthsMeters.filter(isFiniteNum);
			if (finiteDepths.length > 0) bhtMdM = Math.max(...finiteDepths);
		}
		if (bhtMdM !== null) {
			const { tvdssM: bhtTvdssM } = interpolateAtMdM(bhtMdM);
			bhtTvdssDisplay = Number(toDispDepth(bhtTvdssM).toFixed(1));
			const bhtMdDisp = Number(toDispDepth(bhtMdM).toFixed(1));
			traces.push({
				type: 'scatter',
				mode: 'markers',
				name: 'Header BHT',
				x: [bhtDisplay],
				y: [bhtTvdssDisplay],
				text: [`MD: ${bhtMdDisp} ${depthUnit}`],
				xaxis: 'x',
				yaxis: 'y',
				marker: {
					color: '#B91C1C',
					symbol: 'diamond',
					size: 9,
					line: { color: '#FFFFFF', width: 1.2 }
				},
				showlegend: true,
				hovertemplate: `Header BHT: %{x:.1f} ${tempLabel}<br>TVDSS: %{y:.1f} ${depthUnit} (%{text})<extra></extra>`
			});
		}
	}

	// Gauge temperatures from pressureTests (projected onto TVDSS)
	const testTempX: number[] = [];
	const testTempY: number[] = [];
	const testTempMdText: string[] = [];
	for (const pt of pressureTests) {
		const mdM = parseOptionalFinite(pt?.depth ?? pt?.md);
		const tRaw = parseOptionalFinite(pt?.temperature);
		const explicitTvdss = parseOptionalFinite(pt?.tvdss);
		if (tRaw !== null && (mdM !== null || explicitTvdss !== null)) {
			const tC = normalizeTempToC(tRaw, pt?.temperature_uom || 'degC');
			const tvdssM =
				explicitTvdss !== null
					? explicitTvdss
					: interpolateAtMdM(mdM!).tvdssM;
			testTempX.push(cToDisplayTemp(tC, tempUnit));
			testTempY.push(toDispDepth(tvdssM));
			testTempMdText.push(
				mdM !== null ? `MD: ${toDispDepth(mdM).toFixed(1)} ${depthUnit}` : 'TVDSS override'
			);
		}
	}
	if (testTempX.length > 0) {
		traces.push({
			type: 'scatter',
			mode: 'markers',
			name: 'MDT Gauge T',
			x: testTempX,
			y: testTempY,
			text: testTempMdText,
			xaxis: 'x',
			yaxis: 'y',
			marker: {
				color: '#F59E0B',
				symbol: 'circle',
				size: 6.5,
				line: { color: '#0F172A', width: 0.9 }
			},
			showlegend: true,
			hovertemplate: `MDT Gauge T: %{x:.1f} ${tempLabel}<br>TVDSS: %{y:.1f} ${depthUnit} (%{text})<extra></extra>`
		});
	}

	// --- TRACK 2: Pore Pressure, Borehole Mud Column & MDT Points vs. TVDSS ---
	const measPresArr = curves['PRESSURE'];
	const presFlArr = curves['PRES_FL']; // MPa

	const hydroSeries: Array<number | null> = [];
	const mudSeries: Array<number | null> = [];
	let overbalSumPsi = 0;
	let overbalMinPsi = Number.POSITIVE_INFINITY;
	let overbalN = 0;
	let underbalCount = 0;

	for (const idx of slotIndices) {
		if (idx === null) {
			hydroSeries.push(null);
			mudSeries.push(null);
			continue;
		}
		const mdM = depthsMeters[idx];
		if (!isFiniteNum(mdM)) {
			hydroSeries.push(null);
			mudSeries.push(null);
			continue;
		}
		const { tvdssM, tvdM } = resolveSampleDepthsM(idx, mdM);
		const hPsi = geom.hydroPsiAtTvdssM(tvdssM, tvdM);
		const mPsiCurve = measPresArr?.[idx];
		const mPsi = isFiniteNum(mPsiCurve) ? mPsiCurve : geom.mudPsiAtTvdM(tvdM);
		const obPsi = mPsi - hPsi;
		overbalSumPsi += obPsi;
		if (obPsi < overbalMinPsi) overbalMinPsi = obPsi;
		if (obPsi < -1e-3) underbalCount++;
		overbalN++;

		hydroSeries.push(psiToDisplayPressure(hPsi, presUnit));
		mudSeries.push(psiToDisplayPressure(mPsi, presUnit));
	}

	// Overbalance (P_mud > P_hydro) vs Underbalance (P_mud < P_hydro) closed-polygon shading
	addDirectionalRibbon(
		'x2',
		mudSeries,
		hydroSeries,
		'above',
		'rgba(14, 165, 233, 0.18)',
		'Mud Overbalance'
	);
	addDirectionalRibbon(
		'x2',
		mudSeries,
		hydroSeries,
		'below',
		'rgba(220, 38, 38, 0.30)',
		'Underbalanced'
	);

	if (hydroSeries.some((v) => v !== null)) {
		traces.push({
			type: 'scatter',
			mode: 'lines',
			name: 'Hydrostatic P_fm',
			x: hydroSeries,
			y: yTvdssDisplay,
			text: mdHoverText,
			xaxis: 'x2',
			yaxis: 'y',
			connectgaps: false,
			line: { color: '#2563EB', width: 1.5, dash: 'dash' },
			showlegend: true,
			hovertemplate: `Hydrostatic P: %{x:.1f} ${presLabel}<br>TVDSS: %{y:.2f} ${depthUnit} (%{text})<extra></extra>`
		});
	}

	if (presFlArr && presFlArr.some(isFiniteNum)) {
		const presFlDisp = slotIndices.map((idx) => {
			if (idx === null) return null;
			const vMpa = presFlArr[idx];
			if (!isFiniteNum(vMpa)) return null;
			return presUnit === 'MPa' ? vMpa : vMpa / PSI_TO_MPA;
		});
		traces.push({
			type: 'scatter',
			mode: 'lines',
			name: 'PRES_FL',
			x: presFlDisp,
			y: yTvdssDisplay,
			text: mdHoverText,
			xaxis: 'x2',
			yaxis: 'y',
			connectgaps: false,
			line: { color: '#7C3AED', width: 1.25, dash: 'dot' },
			showlegend: true,
			hovertemplate: `PRES_FL: %{x:.1f} ${presLabel}<br>TVDSS: %{y:.2f} ${depthUnit} (%{text})<extra></extra>`
		});
	}

	if (mudSeries.some((v) => v !== null)) {
		const mudHover = slotIndices.map((idx, i) => {
			if (idx === null) return '';
			const m = mudSeries[i];
			const h = hydroSeries[i];
			const mdDisp = mdDisplayArr[i];
			if (m === null || h === null) return '';
			const ob = m - h;
			return `MD: ${mdDisp !== null ? mdDisp.toFixed(1) : '—'} ${depthUnit}<br>Overbalance: ${ob >= 0 ? '+' : ''}${ob.toFixed(presUnit === 'MPa' ? 2 : 0)} ${presLabel} (${geom.mudWeightPpg.toFixed(1)} ppg)`;
		});
		traces.push({
			type: 'scatter',
			mode: 'lines',
			name: `Mud Column (${geom.mudWeightPpg.toFixed(1)} ppg)`,
			x: mudSeries,
			y: yTvdssDisplay,
			customdata: customIdx,
			text: mudHover,
			xaxis: 'x2',
			yaxis: 'y',
			connectgaps: false,
			line: { color: '#0F766E', width: 1.85 },
			showlegend: true,
			hovertemplate: `Mud Column P: %{x:.1f} ${presLabel}<br>TVDSS: %{y:.2f} ${depthUnit} · %{text}<extra></extra>`
		});
	}

	// Single-Well RFT/MDT Pressure Test Points + Table Rows (plotted in TVDSS, showing both TVDSS & MD)
	const mdtX: number[] = [];
	const mdtY: number[] = [];
	const mdtColors: string[] = [];
	const mdtText: string[] = [];
	const mdtRows: Array<Record<string, any>> = [];

	const colorForFluid = (fl: string) => {
		const u = fl.toUpperCase();
		if (u.includes('GAS')) return '#DC2626';
		if (u.includes('OIL')) return '#16A34A';
		if (u.includes('WAT') || u.includes('BRINE')) return '#2563EB';
		return '#0F172A';
	};

	const sortedTests = [...pressureTests]
		.filter((pt) => {
			const md = parseOptionalFinite(pt?.depth ?? pt?.md);
			const tss = parseOptionalFinite(pt?.tvdss);
			const pres = parseOptionalFinite(pt?.pressure);
			return pres !== null && (md !== null || tss !== null);
		})
		.sort((a, b) => {
			const da = parseOptionalFinite(a.depth ?? a.md) ?? parseOptionalFinite(a.tvdss) ?? 0;
			const db = parseOptionalFinite(b.depth ?? b.md) ?? parseOptionalFinite(b.tvdss) ?? 0;
			return da - db;
		});

	for (const pt of sortedTests) {
		const mdOpt = parseOptionalFinite(pt.depth ?? pt.md);
		const explicitTvdss = parseOptionalFinite(pt.tvdss);
		const mdM = mdOpt ?? explicitTvdss!;
		const pPsi = normalizePressureToPsi(parseOptionalFinite(pt.pressure)!, pt.pressure_uom || 'psi');
		const interp = mdOpt !== null ? interpolateAtMdM(mdOpt) : null;
		const tvdssM = explicitTvdss !== null ? explicitTvdss : interp!.tvdssM;
		const tvdM =
			explicitTvdss !== null
				? Math.max(0, explicitTvdss + kbM)
				: interp!.tvdM;
		const tvdFt = Math.max(1e-3, tvdM * M_TO_FT);
		const hydroPsi = geom.hydroPsiAtTvdssM(tvdssM, tvdM);
		const mudPsi = geom.mudPsiAtTvdM(tvdM);
		const excessPsi = pPsi - hydroPsi;
		const overbalPsi = mudPsi - pPsi;
		const emwPpg = pPsi / (0.052 * tvdFt);
		const fluid = String(pt.fluid_type ?? '').trim() || '—';
		const qc = String(pt.test_quality ?? '').trim() || '—';
		const mob = parseOptionalFinite(pt.mobility);
		const tempRaw = parseOptionalFinite(pt.temperature);
		const tempDisp = isFiniteNum(tempRaw)
			? Number(cToDisplayTemp(normalizeTempToC(tempRaw, pt.temperature_uom), tempUnit).toFixed(1))
			: null;

		const pDisp = psiToDisplayPressure(pPsi, presUnit);
		const tvdssDisp = toDispDepth(tvdssM);
		const mdDisp = toDispDepth(mdM);
		mdtX.push(pDisp);
		mdtY.push(tvdssDisp);
		mdtColors.push(colorForFluid(fluid));
		mdtText.push(
			`MD: ${mdDisp.toFixed(1)} ${depthUnit} · Excess vs Hydro: ${excessPsi >= 0 ? '+' : ''}${psiToDisplayPressure(excessPsi, presUnit).toFixed(presUnit === 'MPa' ? 2 : 0)} ${presLabel} · EMW: ${emwPpg.toFixed(2)} ppg${mob !== null ? ` · Mob: ${mob.toFixed(1)} mD/cP` : ''}${fluid !== '—' ? ` · ${fluid}` : ''}`
		);

		mdtRows.push({
			TVDSS: Number(tvdssDisp.toFixed(1)),
			MD: Number(mdDisp.toFixed(1)),
			P_TEST: Number(pDisp.toFixed(presUnit === 'MPa' ? 2 : 1)),
			P_HYDRO: Number(psiToDisplayPressure(hydroPsi, presUnit).toFixed(presUnit === 'MPa' ? 2 : 1)),
			EXCESS_DP: Number(psiToDisplayPressure(excessPsi, presUnit).toFixed(presUnit === 'MPa' ? 2 : 1)),
			MUD_OB: Number(psiToDisplayPressure(overbalPsi, presUnit).toFixed(presUnit === 'MPa' ? 2 : 1)),
			EMW_PPG: Number(emwPpg.toFixed(2)),
			MOBILITY: mob,
			TEMP: tempDisp,
			FLUID: fluid,
			QC: qc,
			REMARKS: String(pt.remarks ?? '').trim()
		});
	}

	if (mdtX.length > 0) {
		traces.push({
			type: 'scatter',
			mode: 'markers',
			name: `MDT / RFT (${mdtX.length})`,
			x: mdtX,
			y: mdtY,
			text: mdtText,
			xaxis: 'x2',
			yaxis: 'y',
			marker: {
				color: mdtColors,
				symbol: 'circle',
				size: 7.5,
				line: { color: '#FFFFFF', width: 1.1 }
			},
			showlegend: true,
			hovertemplate: `MDT P: %{x:.1f} ${presLabel}<br>TVDSS: %{y:.1f} ${depthUnit} · %{text}<extra></extra>`
		});
	}

	// Build fallback Zonal / Checkpoint Summary Rows (with both TVDSS & MD) when no MDT points exist
	const zonalRows: Array<Record<string, any>> = [];
	if (mdtRows.length === 0) {
		const sortedTops = [...tops].filter((t) => isFiniteNum(t.depth)).sort((a, b) => a.depth - b.depth);
		if (sortedTops.length > 0) {
			for (const t of sortedTops) {
				const { tvdssM, tvdM } = interpolateAtMdM(t.depth);
				const tDisp = cToDisplayTemp(geom.tempCAtTvdM(tvdM), tempUnit);
				const hDisp = psiToDisplayPressure(geom.hydroPsiAtTvdssM(tvdssM, tvdM), presUnit);
				const mDisp = psiToDisplayPressure(geom.mudPsiAtTvdM(tvdM), presUnit);
				zonalRows.push({
					HORIZON: t.name,
					TVDSS: Number(toDispDepth(tvdssM).toFixed(1)),
					MD: Number(toDispDepth(t.depth).toFixed(1)),
					TEMP: Number(tDisp.toFixed(1)),
					P_HYDRO: Number(hDisp.toFixed(presUnit === 'MPa' ? 2 : 0)),
					P_MUD: Number(mDisp.toFixed(presUnit === 'MPa' ? 2 : 0)),
					OVERBALANCE: Number((mDisp - hDisp).toFixed(presUnit === 'MPa' ? 2 : 0))
				});
			}
		} else {
			const finiteD = indices.map((i) => depthsMeters[i]).filter(isFiniteNum);
			if (finiteD.length > 0) {
				const dMin = Math.min(...finiteD);
				const dMax = Math.max(...finiteD);
				const dMid = 0.5 * (dMin + dMax);
				for (const [lbl, mdM] of [
					['Interval Top', dMin],
					['Interval Mid', dMid],
					['Interval Base', dMax]
				] as const) {
					const { tvdssM, tvdM } = interpolateAtMdM(mdM);
					const tDisp = cToDisplayTemp(geom.tempCAtTvdM(tvdM), tempUnit);
					const hDisp = psiToDisplayPressure(geom.hydroPsiAtTvdssM(tvdssM, tvdM), presUnit);
					const mDisp = psiToDisplayPressure(geom.mudPsiAtTvdM(tvdM), presUnit);
					zonalRows.push({
						HORIZON: lbl,
						TVDSS: Number(toDispDepth(tvdssM).toFixed(1)),
						MD: Number(toDispDepth(mdM).toFixed(1)),
						TEMP: Number(tDisp.toFixed(1)),
						P_HYDRO: Number(hDisp.toFixed(presUnit === 'MPa' ? 2 : 0)),
						P_MUD: Number(mDisp.toFixed(presUnit === 'MPa' ? 2 : 0)),
						OVERBALANCE: Number((mDisp - hDisp).toFixed(presUnit === 'MPa' ? 2 : 0))
					});
				}
			}
		}
	}

	const shapes: Record<string, any>[] = [];
	const annotations: Record<string, any>[] = [];
	if (geom.offshore && geom.seabedTvdssM > 0) {
		const sbDisp = toDispDepth(geom.seabedTvdssM);
		shapes.push({
			type: 'line',
			xref: 'paper',
			yref: 'y',
			x0: 0,
			x1: 1,
			y0: sbDisp,
			y1: sbDisp,
			line: { color: '#0284C7', width: 1.2, dash: 'dot' }
		});
		annotations.push({
			xref: 'paper',
			yref: 'y',
			x: 0.01,
			y: sbDisp,
			xanchor: 'left',
			yanchor: 'bottom',
			text: `Mudline (${sbDisp.toFixed(0)} ${depthUnit} TVDSS)`,
			showarrow: false,
			font: { family: 'Inter', size: 9, color: '#0369A1' }
		});
	}

	const layout: Record<string, any> = {
		autosize: true,
		paper_bgcolor: '#FFFFFF',
		plot_bgcolor: '#FFFFFF',
		dragmode: 'pan',
		margin: { l: 58, r: 14, t: 46, b: 40 },
		legend: {
			orientation: 'h',
			x: 0,
			y: -0.07,
			xanchor: 'left',
			yanchor: 'top',
			font: { family: 'Inter', size: 9.5, color: '#334155' }
		},
		xaxis: {
			domain: [0, 0.45],
			anchor: 'y',
			side: 'top',
			showgrid: true,
			gridcolor: '#E2E8F0',
			zeroline: false,
			fixedrange: true,
			title: {
				text: `Temperature (${tempLabel})`,
				font: { family: 'Inter', size: 10.5, color: '#DC2626' }
			},
			tickfont: { family: 'JetBrains Mono', size: 9, color: '#DC2626' },
			linecolor: '#DC2626',
			tickcolor: '#DC2626'
		},
		xaxis2: {
			domain: [0.51, 1.0],
			anchor: 'y',
			side: 'top',
			showgrid: true,
			gridcolor: '#E2E8F0',
			zeroline: false,
			fixedrange: true,
			title: {
				text: `Pressure (${presLabel})`,
				font: { family: 'Inter', size: 10.5, color: '#0F766E' }
			},
			tickfont: { family: 'JetBrains Mono', size: 9, color: '#0F766E' },
			linecolor: '#0F766E',
			tickcolor: '#0F766E'
		},
		yaxis: {
			title: { text: yAxisLabel, font: { family: 'Inter', size: 11, color: '#0F172A' } },
			autorange: 'reversed',
			showgrid: true,
			gridcolor: '#E2E8F0',
			zeroline: false,
			tickfont: { family: 'JetBrains Mono', size: 9.5, color: '#475569' }
		},
		shapes,
		annotations
	};

	const meanObDisp =
		overbalN > 0
			? Number(
					psiToDisplayPressure(overbalSumPsi / overbalN, presUnit).toFixed(
						presUnit === 'MPa' ? 2 : 0
					)
				)
			: null;
	const minObDisp =
		overbalN > 0 && Number.isFinite(overbalMinPsi)
			? Number(psiToDisplayPressure(overbalMinPsi, presUnit).toFixed(presUnit === 'MPa' ? 2 : 0))
			: null;

	return {
		traces,
		layout,
		summary: {
			hasTrueTvdss,
			depthBasis,
			depthBasisLabel,
			yAxisLabel,
			datumElevM: kbM,
			offshore: geom.offshore,
			seabedTvdssM: geom.seabedTvdssM,
			topTempDisplay: Number(cToDisplayTemp(geom.topTempC, tempUnit).toFixed(1)),
			gradientCPerKm: Number(geom.gradCPerKm.toFixed(1)),
			gradientFPer100Ft: Number(geom.gradFPer100Ft.toFixed(2)),
			mudWeightPpg: Number(geom.mudWeightPpg.toFixed(1)),
			bhtDisplay,
			bhtTvdssDisplay,
			meanOverbalanceDisplay: meanObDisp,
			minOverbalanceDisplay: minObDisp,
			underbalancedCount: underbalCount,
			mdtCount: mdtRows.length,
			tempUnitLabel: tempLabel,
			presUnitLabel: presLabel,
			tableKind: mdtRows.length > 0 ? 'mdt' : 'zonal',
			tableTitle:
				mdtRows.length > 0
					? `Single-Well Formation Pressure Tests vs. TVDSS (${mdtRows.length} pts)`
					: 'Temperature & Pressure Checkpoints by Horizon (TVDSS)',
			tableRows: mdtRows.length > 0 ? mdtRows : zonalRows
		}
	};
}
