import assert from 'node:assert/strict';
import test from 'node:test';
import {
	normalizedScalSaturation,
	poreThroatDistribution,
	resolvePlugLabAndResPc,
	resolveScalFluidSystem
} from '../src/lib/utils/poreThroat.ts';

test('Washburn radius and nonuniform derivatives match numpy gradient', () => {
	const points = poreThroatDistribution([1000, 1, 10], [9, 0, 1], 1 / 0.29, 0);
	assert.deepEqual(points.map((point) => point.logRadius), [0, -1, -3]);
	assert.deepEqual(points.map((point) => point.distribution), [-1, -2, -4]);
});

test('two points use the same one-sided derivative at both ends', () => {
	const points = poreThroatDistribution([1, 10], [0.8, 0.3], 1 / 0.29, 0);
	assert.deepEqual(points.map((point) => point.distribution), [0.5, 0.5]);
});

test('invalid pressure, duplicate pressure, and undersized curves are handled', () => {
	const points = poreThroatDistribution([0, 1, 1, 10, NaN], [1, 0.8, 0.6, 0.3, 0.2], 1 / 0.29, 0);
	assert.equal(points.length, 2);
	assert.deepEqual(points.map((point) => point.distribution), [0.5, 0.5]);
	assert.deepEqual(poreThroatDistribution([1, 1], [1, 0.5], 30, 30), []);
	assert.deepEqual(poreThroatDistribution([1, 10], [1, 0.5], 30, 90), []);
	assert.deepEqual(poreThroatDistribution([1, 10], [1, 0.5], -1, 30), []);
});

test('Swn uses the existing SCAL normalization and clamps', () => {
	assert.equal(normalizedScalSaturation(0.6, 0.2), 0.49999999999999994);
	assert.equal(normalizedScalSaturation(1, 0.2), 0.99);
	assert.equal(normalizedScalSaturation(0.1, 0.2), 0.005);
});

test('Multi-method SCAL fluid resolution keeps Leverett-J invariant across Lab and Reservoir domains', () => {
	const micp = resolveScalFluidSystem('micp', 2000, 30, 30);
	const pp = resolveScalFluidSystem('porous_plate', 150, 30, 30);
	const centOil = resolveScalFluidSystem('centrifuge', 80, 30, 30, 'lab_oil_brine');

	assert.equal(micp.system, 'micp');
	assert.equal(pp.system, 'air_brine');
	assert.equal(centOil.system, 'lab_oil_brine');

	// Same pore throat measured by MICP vs Air-Brine porous plate
	const targetResPc = 20.0;
	const pcLabMicp = targetResPc * (micp.labIftCos / micp.resIftCos);
	const pcLabPp = targetResPc * (pp.labIftCos / pp.resIftCos);

	// Resolve from lab domain AND from reservoir domain
	const plugMicpLab = resolvePlugLabAndResPc(
		{ experiment_type: 'micp', pc_domain: 'lab', pc_curve: [pcLabMicp] },
		30,
		30
	);
	const plugMicpRes = resolvePlugLabAndResPc(
		{
			experiment_type: 'micp',
			pc_domain: 'reservoir',
			pc_curve: [targetResPc],
			pc_lab_curve: [pcLabMicp]
		},
		45, // Even if user scrubs Res IFT to 45, J must stay invariant!
		15
	);
	const plugPpLab = resolvePlugLabAndResPc(
		{ experiment_type: 'porous_plate', pc_domain: 'lab', pc_curve: [pcLabPp] },
		30,
		30
	);

	const jMicpLab = (0.2166 * plugMicpLab.pcLabCurve[0] * Math.sqrt(100 / 0.2)) / plugMicpLab.fluid.labIftCos;
	const jMicpRes = (0.2166 * plugMicpRes.pcLabCurve[0] * Math.sqrt(100 / 0.2)) / plugMicpRes.fluid.labIftCos;
	const jPpLab = (0.2166 * plugPpLab.pcLabCurve[0] * Math.sqrt(100 / 0.2)) / plugPpLab.fluid.labIftCos;

	assert.ok(Math.abs(jMicpLab - jPpLab) < 1e-9);
	assert.ok(Math.abs(jMicpLab - jMicpRes) < 1e-9);
});
