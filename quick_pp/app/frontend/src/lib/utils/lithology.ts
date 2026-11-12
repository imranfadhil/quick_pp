import type { LithoEndpointParams } from '../state/workbench.svelte';

type GammaParams = Pick<LithoEndpointParams,
	'model' | 'gammaSource' | 'gammaCurve' | 'thoriumCurve' | 'potassiumCurve' |
	'cgrThCoefficient' | 'cgrKCoefficient' | 'grClean' | 'grClay' |
	'gammaClean' | 'gammaShale' | 'gammaEndpoints'>;

export function gammaKey(lp: GammaParams): string {
	if (lp.model !== 'vsh_gr' || lp.gammaSource === 'gr') return 'gr';
	return lp.gammaSource === 'cgr' ? `curve:${lp.gammaCurve.toUpperCase()}` : 'derived_cgr';
}

export function gammaEndpoints(lp: GammaParams): { clean: number; shale: number } {
	return gammaKey(lp) === 'gr'
		? { clean: lp.grClean, shale: lp.grClay }
		: { clean: lp.gammaClean, shale: lp.gammaShale };
}

export function setGammaEndpoints(lp: GammaParams, clean: number, shale: number): void {
	const key = gammaKey(lp);
	if (key === 'gr') {
		lp.grClean = clean;
		lp.grClay = shale;
	} else {
		lp.gammaClean = clean;
		lp.gammaShale = shale;
		lp.gammaEndpoints[key] = { clean, shale };
	}
}

export function selectGamma(
	lp: GammaParams,
	selection: Partial<Pick<GammaParams, 'gammaSource' | 'gammaCurve'>>
): void {
	const oldKey = gammaKey(lp);
	if (oldKey !== 'gr') lp.gammaEndpoints[oldKey] = gammaEndpoints(lp);
	Object.assign(lp, selection);
	const key = gammaKey(lp);
	if (key !== 'gr' && key !== oldKey) {
		const endpoints = lp.gammaEndpoints[key] ?? (Object.keys(lp.gammaEndpoints).length === 0
			? { clean: lp.gammaClean, shale: lp.gammaShale }
			: { clean: 20, shale: 120 });
		lp.gammaClean = endpoints.clean;
		lp.gammaShale = endpoints.shale;
	}
}

export function selectedGamma(lp: GammaParams, curves: Record<string, number[]>): number[] {
	const key = gammaKey(lp);
	if (key === 'gr') return curves['GR'] ?? [];
	if (lp.gammaSource === 'cgr') return curves[lp.gammaCurve.toUpperCase()] ?? [];
	const th = curves[lp.thoriumCurve.toUpperCase()] ?? [];
	const k = curves[lp.potassiumCurve.toUpperCase()] ?? [];
	if (!Number.isFinite(lp.cgrThCoefficient) || lp.cgrThCoefficient <= 0 ||
		!Number.isFinite(lp.cgrKCoefficient) || lp.cgrKCoefficient <= 0) return [];
	return th.map((value, i) =>
		Number.isFinite(value) && value >= 0 && Number.isFinite(k[i]) && k[i] >= 0
			? value * lp.cgrThCoefficient + k[i] * lp.cgrKCoefficient
			: Number.NaN
	);
}

export function gammaLabel(lp: GammaParams): string {
	if (gammaKey(lp) === 'gr') return 'GR';
	return lp.gammaSource === 'cgr' ? lp.gammaCurve.toUpperCase() : 'Derived CGR';
}

export function calibrateGamma(
	groups: { points: { GR: number; VSH_DN: number }[]; weight: number }[],
	current: { clean: number; shale: number },
	holdClean: boolean
): { clean: number; shale: number } {
	const points = groups.flatMap((group) => group.points);
	const minGamma = points.reduce((min, point) => Math.min(min, point.GR), Infinity);
	const maxGamma = points.reduce((max, point) => Math.max(max, point.GR), 0);
	const minSpan = Math.max(0.1, (maxGamma - minGamma) * 0.01);
	const maxClean = Math.max(maxGamma, 0.1);
	const maxShale = Math.max(maxGamma * 1.5 + 1, current.clean + minSpan * 2);
	const score = (clean: number, shale: number) => {
		if (shale - clean < minSpan) return Infinity;
		return groups.reduce((sum, group) => {
			const mse = group.points.reduce((acc, point) => {
				const vsh = Math.max(0, Math.min(1, (point.GR - clean) / (shale - clean)));
				return acc + (vsh - point.VSH_DN) ** 2;
			}, 0) / group.points.length;
			return sum + group.weight * mse;
		}, 0);
	};
	let best = { ...current };
	let bestScore = score(best.clean, best.shale);
	const cleanStep = maxClean / 40;
	const shaleStep = maxShale / 80;
	for (let i = 0; i <= (holdClean ? 0 : 40); i++) {
		const clean = holdClean ? current.clean : i * cleanStep;
		for (let j = 1; j <= 80; j++) {
			const shale = j * shaleStep;
			const mse = score(clean, shale);
			if (mse < bestScore) {
				best = { clean, shale };
				bestScore = mse;
			}
		}
	}
	for (let level = 1; level <= 3; level++) {
		const centre = { ...best };
		const cStep = cleanStep / 4 ** level;
		const sStep = shaleStep / 4 ** level;
		for (let i = holdClean ? 0 : -4; i <= (holdClean ? 0 : 4); i++) {
			for (let j = -4; j <= 4; j++) {
				const clean = Math.max(0, centre.clean + i * cStep);
				const shale = centre.shale + j * sStep;
				const mse = score(clean, shale);
				if (mse < bestScore) {
					best = { clean, shale };
					bestScore = mse;
				}
			}
		}
	}
	const clean = Number(best.clean.toFixed(1));
	return { clean, shale: Math.max(Number((clean + 0.1).toFixed(1)), Number(best.shale.toFixed(1))) };
}

export function thomasStieberAvailable(lithologyModel: string, hasThomasStieber: boolean): boolean {
	return lithologyModel !== 'vsh_bulk' && hasThomasStieber;
}

export function gammaParameters(lp: LithoEndpointParams) {
	return {
		gamma_source: lp.gammaSource,
		gamma_curve: lp.gammaCurve,
		thorium_curve: lp.thoriumCurve,
		potassium_curve: lp.potassiumCurve,
		gamma_clean: lp.gammaSource === 'gr' ? lp.grClean : lp.gammaClean,
		gamma_shale: lp.gammaSource === 'gr' ? lp.grClay : lp.gammaShale,
		cgr_th_coefficient: lp.cgrThCoefficient,
		cgr_k_coefficient: lp.cgrKCoefficient
	};
}
