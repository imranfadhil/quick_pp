import { test } from 'node:test';
import assert from 'node:assert/strict';
import { calibrateGamma, gammaEndpoints, selectedGamma, selectGamma, setGammaEndpoints } from '../src/lib/utils/lithology.ts';

function params() {
	return {
		model: 'vsh_gr', gammaSource: 'gr', gammaCurve: 'CGR', thoriumCurve: 'THOR',
		potassiumCurve: 'POTA', cgrThCoefficient: 4, cgrKCoefficient: 16,
		grClean: 20, grClay: 120, gammaClean: 20, gammaShale: 120, gammaEndpoints: {}
	};
}

test('switching gamma sources retains distinct calibrated endpoints', () => {
	const lp = params();
	selectGamma(lp, { gammaSource: 'cgr' });
	setGammaEndpoints(lp, 10, 70);
	selectGamma(lp, { gammaCurve: 'SGR' });
	setGammaEndpoints(lp, 30, 150);
	selectGamma(lp, { gammaSource: 'derived_cgr' });
	setGammaEndpoints(lp, 12, 60);
	selectGamma(lp, { gammaSource: 'gr' });
	assert.deepEqual(gammaEndpoints(lp), { clean: 20, shale: 120 });
	selectGamma(lp, { gammaSource: 'cgr', gammaCurve: 'CGR' });
	assert.deepEqual(gammaEndpoints(lp), { clean: 10, shale: 70 });
	selectGamma(lp, { gammaCurve: 'SGR' });
	assert.deepEqual(gammaEndpoints(lp), { clean: 30, shale: 150 });
	selectGamma(lp, { gammaSource: 'derived_cgr' });
	assert.deepEqual(gammaEndpoints(lp), { clean: 12, shale: 60 });
});

test('selected and derived gamma use their inputs rather than GR', () => {
	const lp = params();
	const curves = { GR: [20, 70, 120], CGR: [10, 40, 70], THOR: [1, 4, -1], POTA: [0.5, 1.5, 2] };
	selectGamma(lp, { gammaSource: 'cgr' });
	assert.deepEqual(selectedGamma(lp, curves), [10, 40, 70]);
	selectGamma(lp, { gammaSource: 'derived_cgr' });
	assert.deepEqual(selectedGamma(lp, curves), [12, 40, NaN]);
	lp.cgrThCoefficient = 0;
	assert.deepEqual(selectedGamma(lp, curves), []);
	lp.model = 'ssc';
	assert.deepEqual(selectedGamma(lp, curves), curves.GR);
});

test('auto-calibration fits selected gamma even for a narrow response range', () => {
	const points = [
		{ GR: 4, VSH_DN: 0 }, { GR: 6, VSH_DN: 0.25 },
		{ GR: 8, VSH_DN: 0.5 }, { GR: 10, VSH_DN: 0.75 },
		{ GR: 12, VSH_DN: 1 }
	];
	const fit = calibrateGamma([{ points, weight: 1 }], { clean: 0, shale: 100 }, false);
	assert.ok(Math.abs(fit.clean - 4) <= 0.3);
	assert.ok(Math.abs(fit.shale - 12) <= 0.3);
	const held = calibrateGamma([{ points, weight: 1 }], { clean: 4, shale: 100 }, true);
	assert.equal(held.clean, 4);
	assert.ok(Math.abs(held.shale - 12) <= 0.3);
});
