import type { SaturationModelParams } from '$lib/state/workbench.svelte';
import type { DepthUnit } from '$lib/utils/depthUnits';

export interface WaterLegCalibrationResult {
	xs: number[]; // porosity points
	ys: number[]; // inverted m points
	vshs: number[];
	depths: number[];
	customIdx: number[];
	meanM: number;
	medianM: number;
	slope: number;
	intercept: number;
	r2: number;
	validCount: number;
}

/**
 * Robustly normalizes depth bounds to meters matching well curve array.
 */
export function resolveDepthToMeters(
	val: number | null | undefined,
	unit: DepthUnit,
	deptAll: number[]
): number | null {
	if (val === null || val === undefined || !Number.isFinite(val)) return null;
	if (deptAll.length === 0) return val;

	const firstD = deptAll[0];
	const lastD = deptAll[deptAll.length - 1];
	const dMin = Math.min(firstD, lastD);
	const dMax = Math.max(firstD, lastD);

	// If deptAll itself is in feet (> 5000 ft)
	const isCurveInFeet = dMin > 5000;
	if (isCurveInFeet) {
		if (unit === 'm' && val < dMin * 0.6) {
			return val / 0.3048;
		}
		return val;
	}

	// Typical: deptAll is in meters (~1000 - 5000 m).
	// If val is entered in feet (e.g. 12558 ft while dMax is ~4000 m), convert to meters.
	if (val > dMax * 1.5) {
		return val * 0.3048;
	}
	if (unit === 'ft') {
		return val * 0.3048;
	}
	return val;
}

/**
 * Computes formation water resistivity Rw at depth from geothermal gradient & salinity,
 * or returns pickett line override.
 */
export function getRwAtDepth(
	depthM: number,
	sp: SaturationModelParams
): number {
	if (sp.rwMode === 'pickett' && Number.isFinite(sp.rwPickett) && sp.rwPickett > 0) {
		return sp.rwPickett;
	}
	const sTemp = Number.isFinite(sp.surfaceTemp) ? sp.surfaceTemp : 25.0;
	const tGrad = Number.isFinite(sp.tempGradient) ? sp.tempGradient : 30.0;
	let tempC: number;
	if (sp.measSystem === 'imperial') {
		const tvdKft = (depthM * 3.28084) / 1000.0;
		const tempF = sTemp + tGrad * tvdKft;
		tempC = Math.max(5, Math.min(250, ((tempF - 32.0) * 5.0) / 9.0));
	} else {
		tempC = Math.max(5, Math.min(250, sTemp + tGrad * (depthM / 1000.0)));
	}
	const tempF = (tempC * 9.0) / 5.0 + 32.0;
	const sal = Math.max(
		100,
		Number.isFinite(sp.waterSalinity) ? sp.waterSalinity : 35000
	);
	return Math.pow(400000.0 / (sal * tempF), 0.88);
}

/**
 * Reactively inverts cementation factor m in the 100% water-bearing leg
 * and fits a linear trend m = slope * phi + intercept.
 */
export function calculateWaterLegCalibration(
	deptAll: number[],
	phitRaw: number[],
	rtRaw: number[],
	vshRaw: number[],
	rwCurve: number[],
	indices: number[],
	depthUnit: DepthUnit,
	sp: SaturationModelParams,
	resolveZoneAtDepth?: (d: number) => string
): WaterLegCalibrationResult {
	const emptyResult: WaterLegCalibrationResult = {
		xs: [],
		ys: [],
		vshs: [],
		depths: [],
		customIdx: [],
		meanM: 2.0,
		medianM: 2.0,
		slope: 0,
		intercept: 2.0,
		r2: 0,
		validCount: 0
	};

	if (deptAll.length === 0 || phitRaw.length === 0 || rtRaw.length === 0) {
		return emptyResult;
	}

	const poolIndices = indices && indices.length > 0 ? indices : deptAll.map((_, i) => i);
	const waterLegIndices: number[] = [];

	let topM: number | null = null;
	let botM: number | null = null;
	if (sp.waterLegIntervalMode === 'depth') {
		const rawTop = resolveDepthToMeters(sp.waterLegTopDepth, depthUnit, deptAll);
		const rawBot = resolveDepthToMeters(sp.waterLegBottomDepth, depthUnit, deptAll);
		if (rawTop !== null && rawBot !== null) {
			topM = Math.min(rawTop, rawBot);
			botM = Math.max(rawTop, rawBot);
		} else if (rawTop !== null) {
			topM = rawTop;
		} else if (rawBot !== null) {
			botM = rawBot;
		}
	}

	for (const idx of poolIndices) {
		const d = deptAll[idx];
		if (!Number.isFinite(d)) continue;

		if (sp.waterLegIntervalMode === 'zone' && sp.waterLegSelectedZone && resolveZoneAtDepth) {
			const z = resolveZoneAtDepth(d);
			if (z !== sp.waterLegSelectedZone) continue;
		} else if (sp.waterLegIntervalMode === 'depth') {
			if (topM !== null && d < topM) continue;
			if (botM !== null && d > botM) continue;
		} else if (Number.isFinite(sp.fwl) && sp.fwl > 0) {
			if (d < sp.fwl) continue;
		}
		waterLegIndices.push(idx);
	}

	const xs: number[] = [];
	const ys: number[] = [];
	const vshs: number[] = [];
	const depths: number[] = [];
	const customIdx: number[] = [];

	const aFactor = Number.isFinite(sp.a) && sp.a > 0 ? sp.a : 1.0;
	const rsh = Math.max(0.1, sp.rsh || 2.0);
	const vshCleanCutoff = sp.waterLegVshCutoff ?? 0.3;

	for (const idx of waterLegIndices) {
		const d = deptAll[idx];
		const phi = phitRaw[idx];
		const rt = rtRaw[idx];
		const vsh = Number.isFinite(vshRaw[idx]) ? Math.max(0, Math.min(1, vshRaw[idx])) : 0;

		if (!Number.isFinite(phi) || phi < 0.02 || phi > 0.45) continue;
		if (!Number.isFinite(rt) || rt <= 0) continue;

		const rw =
			Number.isFinite(rwCurve[idx]) && rwCurve[idx] > 0
				? rwCurve[idx]
				: getRwAtDepth(d, sp);
		if (!Number.isFinite(rw) || rw <= 0) continue;

		let mVal: number | null = null;
		if (sp.waterLegMethod === 'indonesian') {
			const shTerm = Math.pow(vsh, 1 - 0.5 * vsh) / Math.sqrt(rsh);
			const diff = 1.0 / Math.sqrt(rt) - shTerm;
			if (diff > 0) {
				mVal = (2.0 / Math.log(phi)) * Math.log(Math.sqrt(aFactor * rw) * diff);
			}
		} else {
			// Archie clean water sand
			if (vsh <= vshCleanCutoff) {
				mVal = Math.log((aFactor * rw) / rt) / Math.log(phi);
			}
		}

		if (mVal !== null && Number.isFinite(mVal) && mVal >= 1.0 && mVal <= 3.5) {
			xs.push(phi);
			ys.push(mVal);
			vshs.push(vsh);
			depths.push(d);
			customIdx.push(idx);
		}
	}

	if (xs.length === 0) {
		return emptyResult;
	}

	let meanM = xs.reduce((acc, _, i) => acc + ys[i], 0) / xs.length;
	const sortedYs = [...ys].sort((a, b) => a - b);
	let medianM = sortedYs[Math.floor(sortedYs.length / 2)];
	let slope = 0;
	let intercept = meanM;
	let r2 = 0;

	if (xs.length >= 3) {
		const nPts = xs.length;
		const xMean = xs.reduce((a, b) => a + b, 0) / nPts;
		const yMean = meanM;
		let num = 0;
		let den = 0;
		for (let i = 0; i < nPts; i++) {
			num += (xs[i] - xMean) * (ys[i] - yMean);
			den += (xs[i] - xMean) ** 2;
		}
		if (den > 1e-9) {
			slope = num / den;
			intercept = yMean - slope * xMean;
			let ssRes = 0;
			let ssTot = 0;
			for (let i = 0; i < nPts; i++) {
				const fit = slope * xs[i] + intercept;
				ssRes += (ys[i] - fit) ** 2;
				ssTot += (ys[i] - yMean) ** 2;
			}
			r2 = ssTot > 0 ? Math.max(0, 1 - ssRes / ssTot) : 0;
		}
	}

	return {
		xs,
		ys,
		vshs,
		depths,
		customIdx,
		meanM: Number(meanM.toFixed(3)),
		medianM: Number(medianM.toFixed(3)),
		slope: Number(slope.toFixed(3)),
		intercept: Number(intercept.toFixed(3)),
		r2: Number(r2.toFixed(3)),
		validCount: xs.length
	};
}
