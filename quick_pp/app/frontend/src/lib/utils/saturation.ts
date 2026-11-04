export interface BqvCrossplotPoint {
	index: number;
	x: number;
	y: number;
	depth: number;
}

export interface BqvCrossplotFit {
	points: BqvCrossplotPoint[];
	slope: number | null;
	intercept: number | null;
	r2: number | null;
}

export interface BqvRwSettings {
	rwMode: 'salinity_temp' | 'pickett';
	rwPickett: number;
	measSystem: 'metric' | 'imperial';
	surfaceTemp: number;
	tempGradient: number;
	waterSalinity: number;
}

export function getBqvFitIndices(
	visibleIndices: number[],
	brushedIndices: ReadonlySet<number>
): number[] {
	return brushedIndices.size > 0
		? visibleIndices.filter((index) => brushedIndices.has(index))
		: visibleIndices;
}

export function fillMissingCurve(primary: number[] | undefined, fallback: number[]): number[] {
	if (!primary) return fallback;
	return fallback.map((value, index) =>
		Number.isFinite(primary[index]) ? primary[index] : value
	);
}

export function estimateBqvReferenceRw(
	settings: BqvRwSettings,
	measuredDepths: number[],
	tvdDepths: number[],
	fallbackDepths: number[],
	visibleIndices: number[],
	referenceDepth: number
): { rw: number; tempC: number } {
	let closestIndex = -1;
	let closestDistance = Infinity;
	for (const index of visibleIndices) {
		const measuredDepth = measuredDepths[index];
		if (
			Number.isFinite(measuredDepth) &&
			Math.abs(measuredDepth - referenceDepth) < closestDistance
		) {
			closestIndex = index;
			closestDistance = Math.abs(measuredDepth - referenceDepth);
		}
	}

	const temperatureDepth =
		closestIndex >= 0
			? Number.isFinite(tvdDepths[closestIndex])
				? tvdDepths[closestIndex]
				: (fallbackDepths[closestIndex] ?? referenceDepth)
			: referenceDepth;
	let tempC: number;
	if (settings.measSystem === 'imperial') {
		const tvdKft = (temperatureDepth * 3.28084) / 1000;
		const tempF = settings.surfaceTemp + settings.tempGradient * tvdKft;
		tempC = Math.max(5, Math.min(250, ((tempF - 32) * 5) / 9));
	} else {
		tempC = Math.max(
			5,
			Math.min(250, settings.surfaceTemp + settings.tempGradient * (temperatureDepth / 1000))
		);
	}

	const rw =
		settings.rwMode === 'pickett'
			? settings.rwPickett
			: Math.pow(400000 / (Math.max(100, settings.waterSalinity) * (tempC * 9 / 5 + 32)), 0.88);
	return { rw, tempC };
}

export function fitBqvCrossplotLine(
	points: BqvCrossplotPoint[],
	slope: number,
	intercept: number
): BqvCrossplotFit {
	if (points.length < 2 || !Number.isFinite(slope) || !Number.isFinite(intercept)) {
		return { points, slope: null, intercept: null, r2: null };
	}

	const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
	let totalVarianceY = 0;
	let residualSumSquares = 0;
	for (const point of points) {
		totalVarianceY += Math.pow(point.y - meanY, 2);
		residualSumSquares += Math.pow(point.y - (slope * point.x + intercept), 2);
	}

	return {
		points,
		slope,
		intercept,
		r2: totalVarianceY > 0 ? 1 - residualSumSquares / totalVarianceY : null
	};
}

export function fitBqvCrossplotWithFixedIntercept(
	points: BqvCrossplotPoint[],
	intercept: number
): BqvCrossplotFit {
	if (!Number.isFinite(intercept)) {
		return { points, slope: null, intercept: null, r2: null };
	}
	if (points.length < 2) {
		return { points, slope: null, intercept, r2: null };
	}

	let numerator = 0;
	let denominator = 0;
	for (const point of points) {
		numerator += point.x * (point.y - intercept);
		denominator += point.x * point.x;
	}
	if (denominator <= 0) {
		return { points, slope: null, intercept, r2: null };
	}

	const slope = numerator / denominator;
	const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
	let totalVarianceY = 0;
	let residualSumSquares = 0;
	for (const point of points) {
		totalVarianceY += Math.pow(point.y - meanY, 2);
		residualSumSquares += Math.pow(point.y - (slope * point.x + intercept), 2);
	}

	return {
		points,
		slope,
		intercept,
		r2: totalVarianceY > 0 ? 1 - residualSumSquares / totalVarianceY : null
	};
}

export function fitBqvCrossplot(
	rt: number[],
	phit: number[],
	vclay: number[],
	depths: number[],
	indices: number[],
	m: number
): BqvCrossplotFit {
	const points: BqvCrossplotPoint[] = [];
	for (const index of indices) {
		const resistivity = rt[index];
		const porosity = phit[index];
		const clayVolume = vclay[index];
		const depth = depths[index];
		if (
			!Number.isFinite(resistivity) ||
			resistivity <= 0 ||
			!Number.isFinite(porosity) ||
			porosity <= 0 ||
			!Number.isFinite(clayVolume) ||
			clayVolume < 0 ||
			clayVolume > 1 ||
			!Number.isFinite(depth)
		) {
			continue;
		}

		const x = clayVolume / porosity;
		const y = 1 / (resistivity * Math.pow(porosity, m));
		if (Number.isFinite(x) && Number.isFinite(y)) {
			points.push({ index, x, y, depth });
		}
	}

	if (points.length < 2) {
		return { points, slope: null, intercept: null, r2: null };
	}

	const meanX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
	const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
	let covariance = 0;
	let varianceX = 0;
	let totalVarianceY = 0;
	let residualSumSquares = 0;
	for (const point of points) {
		covariance += (point.x - meanX) * (point.y - meanY);
		varianceX += Math.pow(point.x - meanX, 2);
		totalVarianceY += Math.pow(point.y - meanY, 2);
	}
	if (varianceX <= 0) {
		return { points, slope: null, intercept: null, r2: null };
	}

	const slope = covariance / varianceX;
	const intercept = meanY - slope * meanX;
	for (const point of points) {
		residualSumSquares += Math.pow(point.y - (slope * point.x + intercept), 2);
	}

	return {
		points,
		slope,
		intercept,
		r2: totalVarianceY > 0 ? 1 - residualSumSquares / totalVarianceY : null
	};
}
