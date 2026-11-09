export const MOBILITY_RANGE = [0.01, 10000] as const;
export const PERMEABILITY_TRACK_RANGE = [0.01, 10000] as const;

export interface PressureTestMobility {
	depth?: unknown;
	mobility?: unknown;
	fluid_type?: unknown;
	test_quality?: unknown;
	remarks?: unknown;
}

export interface MobilityPoint {
	depth: number;
	mobility: number;
	details: string;
}

export function buildMobilityTrace(
	points: readonly MobilityPoint[],
	formatDepth: (depth: number) => string = (depth) => `${depth.toFixed(2)}m`
) {
	const [mobMin, mobMax] = MOBILITY_RANGE.map(Math.log10);
	const [permMin, permMax] = PERMEABILITY_TRACK_RANGE.map(Math.log10);
	return {
		type: 'scatter',
		mode: 'markers',
		name: 'FPT mobility',
		// Map between display scales only; this does not convert mobility to permeability.
		x: points.map((point) => {
			const fraction = (Math.log10(point.mobility) - mobMin) / (mobMax - mobMin);
			return 10 ** (permMin + fraction * (permMax - permMin));
		}),
		y: points.map((point) => point.depth),
		customdata: points.map((point) => [point.mobility, point.details, formatDepth(point.depth)]),
		xaxis: 'x5',
		yaxis: 'y',
		marker: {
			color: '#DB2777',
			size: 8,
			symbol: 'triangle-up',
			line: { color: '#FFFFFF', width: 0.9 }
		},
		hovertemplate:
			'FPT mobility: %{customdata[0]:.4g} mD/cP<br>Depth: %{customdata[2]}%{customdata[1]}<extra></extra>'
	};
}

function finiteNumber(value: unknown): number | null {
	if (typeof value !== 'number' && typeof value !== 'string') return null;
	if (typeof value === 'string' && !value.trim()) return null;
	const n = Number(value);
	return Number.isFinite(n) ? n : null;
}

function escapeHtml(value: string): string {
	return value.replace(/[&<>"']/g, (char) => {
		switch (char) {
			case '&':
				return '&amp;';
			case '<':
				return '&lt;';
			case '>':
				return '&gt;';
			case '"':
				return '&quot;';
			default:
				return '&#39;';
		}
	});
}

export function buildMobilityPoints(
	tests: readonly PressureTestMobility[],
	isDepthIncluded: (depth: number) => boolean
): { points: MobilityPoint[]; outsideRange: number } {
	const points: MobilityPoint[] = [];
	let outsideRange = 0;
	for (const test of tests) {
		const depth = finiteNumber(test.depth);
		const mobility = finiteNumber(test.mobility);
		if (depth === null || mobility === null || mobility <= 0 || !isDepthIncluded(depth))
			continue;
		if (mobility < MOBILITY_RANGE[0] || mobility > MOBILITY_RANGE[1]) {
			outsideRange++;
			continue;
		}
		const details = [
			['Fluid', test.fluid_type],
			['QC', test.test_quality],
			['Remarks', test.remarks]
		]
			.filter(([, value]) => typeof value === 'string' && value.trim())
			.map(([label, value]) => `<br>${label}: ${escapeHtml(String(value).trim())}`)
			.join('');
		points.push({ depth, mobility, details });
	}
	return { points, outsideRange };
}
