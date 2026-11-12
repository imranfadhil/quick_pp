export function comparisonXAxis(kind: 'porosity' | 'permeability') {
	if (kind === 'porosity') {
		return { title: { text: 'Total porosity (v/v)' }, range: [0, 0.5], fixedrange: true };
	}
	return {
		title: { text: 'Permeability (mD)' }, type: 'log' as const,
		range: [Math.log10(0.1), Math.log10(1e5)], fixedrange: true
	};
}

export const comparisonPlotConfig = { responsive: true, displaylogo: false, scrollZoom: true };
