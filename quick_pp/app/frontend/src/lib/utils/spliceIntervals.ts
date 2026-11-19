export type SpliceInterval = {
	top: number | null;
	bottom: number | null;
};

export function selectSpliceInterval(
	depths: number[],
	values: number[],
	interval?: SpliceInterval
): number[] {
	return depths.map((depth, index) => {
		if (
			!Number.isFinite(depth) ||
			(interval?.top != null && depth < interval.top) ||
			(interval?.bottom != null && depth > interval.bottom)
		)
			return Number.NaN;
		return values[index];
	});
}

export function validateSpliceInterval(
	name: string,
	depths: number[],
	values: number[],
	interval?: SpliceInterval
): void {
	if (
		(interval?.top != null && !Number.isFinite(interval.top)) ||
		(interval?.bottom != null && !Number.isFinite(interval.bottom))
	) {
		throw new Error(`Enter finite depth limits for ${name}.`);
	}
	if (interval?.top != null && interval.bottom != null && interval.top > interval.bottom) {
		throw new Error(`Top depth must not exceed bottom depth for ${name}.`);
	}
	if (
		(interval?.top != null || interval?.bottom != null) &&
		!selectSpliceInterval(depths, values, interval).some(Number.isFinite)
	) {
		throw new Error(`No valid data for ${name} in the selected depth interval.`);
	}
}
