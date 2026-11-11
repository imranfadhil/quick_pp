export function normalizedScalSaturation(sw: number, swirr: number): number {
	return Math.min(0.99, Math.max(0.005, (sw - swirr) / Math.max(0.05, 1 - swirr)));
}

export interface PoreThroatPoint {
	logRadius: number;
	distribution: number;
}

/** Washburn radius and the same nonuniform finite differences as numpy.gradient. */
export function poreThroatDistribution(
	pc: number[],
	swn: number[],
	ift: number,
	theta: number
): PoreThroatPoint[] {
	const factor = 2 * 0.145 * ift * Math.abs(Math.cos((theta * Math.PI) / 180));
	if (!Number.isFinite(factor) || ift <= 0 || factor < 1e-12) return [];

	const points = pc
		.slice(0, Math.min(pc.length, swn.length))
		.map((pressure, index) => ({ pressure, saturation: swn[index] }))
		.filter((point) => Number.isFinite(point.pressure) && point.pressure > 0 && Number.isFinite(point.saturation))
		.sort((a, b) => a.pressure - b.pressure)
		.filter((point, index, sorted) => index === 0 || point.pressure !== sorted[index - 1].pressure)
		.map((point) => ({
			logRadius: Math.log10(factor / point.pressure),
			saturation: point.saturation
		}))
		.filter((point) => Number.isFinite(point.logRadius));
	if (points.length < 2) return [];

	return points.map((point, i) => {
		let derivative: number;
		if (i === 0 || i === points.length - 1) {
			const a = i === 0 ? points[0] : points[i - 1];
			const b = i === 0 ? points[1] : points[i];
			derivative = (b.saturation - a.saturation) / (b.logRadius - a.logRadius);
		} else {
			const previous = points[i - 1];
			const next = points[i + 1];
			const h1 = point.logRadius - previous.logRadius;
			const h2 = next.logRadius - point.logRadius;
			derivative =
				(-h2 / (h1 * (h1 + h2))) * previous.saturation +
				((h2 - h1) / (h1 * h2)) * point.saturation +
				(h1 / (h2 * (h1 + h2))) * next.saturation;
		}
		return { logRadius: point.logRadius, distribution: derivative };
	}).filter((point) => Number.isFinite(point.distribution));
}
