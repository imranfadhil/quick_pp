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

export function fillMissingCurve(primary: number[] | undefined, fallback: number[]): number[] {
	if (!primary) return fallback;
	return fallback.map((value, index) =>
		Number.isFinite(primary[index]) ? primary[index] : value
	);
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
