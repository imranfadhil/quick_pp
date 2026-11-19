export type ComparisonKind = 'porosity' | 'permeability';

/** Fixed value-axis ranges; depth remains scroll-zoomable. */
export const POROSITY_RANGE: [number, number] = [0, 0.5];
export const PERM_RANGE: [number, number] = [Math.log10(0.1), Math.log10(1e5)];
export const MATRIX_DENSITY_RANGE: [number, number] = [1.8, 3.0];
export const VSHALE_RANGE: [number, number] = [0, 1];

export function comparisonXAxis(kind: ComparisonKind) {
	if (kind === 'porosity') {
		return { title: { text: 'Total porosity (v/v)' }, range: POROSITY_RANGE, fixedrange: true };
	}
	return {
		title: { text: 'Permeability (mD)' },
		type: 'log' as const,
		range: PERM_RANGE,
		fixedrange: true
	};
}

/** Leading track: matrix/fluid density for porosity, VSHALE for permeability. */
export function comparisonLeadingAxis(kind: ComparisonKind, rhoFluid?: number | null) {
	if (kind === 'porosity') {
		const range: [number, number] = [MATRIX_DENSITY_RANGE[0], MATRIX_DENSITY_RANGE[1]];
		if (typeof rhoFluid === 'number' && Number.isFinite(rhoFluid) && rhoFluid < range[0]) {
			range[0] = Math.floor(rhoFluid * 10) / 10;
		}
		return { title: { text: 'Density (g/cc)' }, range, fixedrange: true };
	}
	return { title: { text: 'VSHALE (v/v)' }, range: VSHALE_RANGE, fixedrange: true };
}

export const comparisonPlotConfig = { responsive: true, displaylogo: false, scrollZoom: true };
