import type { LithoEndpointParams } from '../state/workbench.svelte';

export function bulkShaleConflicts(
	lp: Pick<LithoEndpointParams, 'porosityMethod' | 'logCorrectionMode' | 'badholeTreatment'>,
	thomasStieberEnabled: boolean
): string[] {
	const conflicts: string[] = [];
	if (lp.porosityMethod !== 'den') conflicts.push('Porosity method must be Density.');
	if (lp.logCorrectionMode !== 'none') conflicts.push('Log correction must be Off.');
	if (!['none', 'ignore'].includes(lp.badholeTreatment)) {
		conflicts.push('Badhole Action must be Use Normally or Ignore / Nullify.');
	}
	if (thomasStieberEnabled) conflicts.push('Thomas-Stieber must be disabled.');
	return conflicts;
}

export function thomasStieberAvailable(lithologyModel: string, hasThomasStieber: boolean): boolean {
	return lithologyModel !== 'vsh_bulk' && hasThomasStieber;
}

export function applyBulkShaleCompatibility(
	lp: Pick<LithoEndpointParams, 'porosityMethod' | 'logCorrectionMode' | 'enableHcCorrection' | 'useHcCorrected' | 'badholeTreatment'>
): void {
	lp.porosityMethod = 'den';
	lp.logCorrectionMode = 'none';
	lp.enableHcCorrection = false;
	lp.useHcCorrected = false;
	if (!['none', 'ignore'].includes(lp.badholeTreatment)) lp.badholeTreatment = 'ignore';
}

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
