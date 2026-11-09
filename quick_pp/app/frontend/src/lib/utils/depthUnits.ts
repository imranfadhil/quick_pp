export type DepthUnit = 'm' | 'ft';

export const METERS_PER_FOOT = 0.3048;

export function depthToMeters(value: number, unit: DepthUnit): number {
	return unit === 'ft' ? value * METERS_PER_FOOT : value;
}

export function depthFromMeters(value: number, unit: DepthUnit): number {
	return unit === 'ft' ? value / METERS_PER_FOOT : value;
}

export function depthAxisTicks(range: readonly [number, number], unit: DepthUnit) {
	const low = depthFromMeters(Math.min(...range), unit);
	const high = depthFromMeters(Math.max(...range), unit);
	const span = high - low;
	if (!Number.isFinite(span) || span <= 0) return { tickvals: [], ticktext: [] };
	const rawStep = span / 7;
	const magnitude = 10 ** Math.floor(Math.log10(rawStep));
	const normalized = rawStep / magnitude;
	const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;
	const decimals = Math.max(0, Math.min(10, -Math.floor(Math.log10(step))));
	const tickvals: number[] = [];
	const ticktext: string[] = [];
	const firstIndex = Math.ceil(low / step);
	const lastIndex = Math.floor(high / step);
	for (let i = firstIndex; i <= lastIndex && tickvals.length < 20; i++) {
		const display = i * step;
		tickvals.push(depthToMeters(display, unit));
		ticktext.push(display.toFixed(decimals));
	}
	return { tickvals, ticktext };
}
