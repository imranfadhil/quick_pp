export type CurveDepthComparison = {
	target: string;
	sources: string[];
	kind: 'math' | 'splice';
};

export function curveDepthSeries(
	curves: Record<string, number[]>,
	depths: number[],
	indices: number[],
	selected: string,
	mode: 'math' | 'splice' | 'manage',
	runs: string[],
	comparison: CurveDepthComparison | null
) {
	const result = comparison?.target === selected ? comparison : null;
	const sources = mode === 'splice' ? runs : (result?.sources ?? []);
	const series: Array<{ name: string; values: number[]; role: 'source' | 'result' }> = [];
	const seen = new Set<string>();

	for (const source of sources) {
		if (!source || seen.has(source)) continue;
		seen.add(source);
		const isOverwritten = result && source === result.target;
		const backup = isOverwritten ? `${source}_ORI` : source;
		if (curves[backup] && (!isOverwritten || backup !== source)) {
			series.push({
				name: isOverwritten ? `${backup} (original)` : `${source} (source)`,
				values: curves[backup],
				role: 'source'
			});
		}
	}

	const original = `${selected}_ORI`;
	if (curves[original] && !series.some((item) => item.values === curves[original])) {
		series.push({ name: `${original} (original)`, values: curves[original], role: 'source' });
	}
	if (curves[selected] && (!seen.has(selected) || result || mode !== 'splice')) {
		series.push({
			name: result ? `${selected} (result)` : selected,
			values: curves[selected],
			role: 'result'
		});
	}

	return series.map((item) => {
		const x: Array<number | null> = [];
		const y: Array<number | null> = [];
		const customdata: Array<number | null> = [];
		let previous = -2;
		for (const index of indices) {
			if (previous !== -2 && index !== previous + 1) {
				x.push(null);
				y.push(null);
				customdata.push(null);
			}
			const depth = depths[index];
			const value = item.values[index];
			x.push(Number.isFinite(value) && Number.isFinite(depth) ? value : null);
			y.push(Number.isFinite(depth) ? depth : null);
			customdata.push(index);
			previous = index;
		}
		return { name: item.name, role: item.role, x, y, customdata };
	});
}
