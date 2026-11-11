import type { LithoEndpointParams } from '../state/workbench.svelte';

export function bulkShaleParameters(lp: LithoEndpointParams) {
	return {
		gamma_source: lp.gammaSource,
		gamma_curve: lp.gammaCurve,
		thorium_curve: lp.thoriumCurve,
		potassium_curve: lp.potassiumCurve,
		gamma_clean: lp.gammaSource === 'gr' ? lp.bulkGrClean : lp.gammaClean,
		gamma_shale: lp.gammaSource === 'gr' ? lp.bulkGrShale : lp.gammaShale,
		cgr_th_coefficient: lp.cgrThCoefficient,
		cgr_k_coefficient: lp.cgrKCoefficient,
		shale_solid_density: lp.shaleSolidDensity,
		shale_porosity: lp.shalePorosity,
		shale_ineffective_fraction: lp.shaleIneffectiveFraction
	};
}
