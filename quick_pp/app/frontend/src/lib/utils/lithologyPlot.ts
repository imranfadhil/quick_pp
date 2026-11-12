export function scaleLithologyForPlot(
	volumes: (number | null)[][],
	phit: (number | null)[]
): (number | null)[][] {
	const totals = phit.map((_, i) =>
		volumes.reduce((sum, curve) => {
			const value = curve[i];
			return sum + (value !== null && Number.isFinite(value) ? Math.max(value, 0) : 0);
		}, 0)
	);
	return volumes.map((curve) =>
		curve.map((value, i) => {
			if (value === null || !Number.isFinite(value)) return null;
			const phi = phit[i];
			const matrix = phi !== null && Number.isFinite(phi) ? 1 - Math.min(1, Math.max(0, phi)) : 1;
			return totals[i] > 0 ? Math.max(value, 0) * matrix / totals[i] : 0;
		})
	);
}
