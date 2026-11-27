import assert from 'node:assert/strict';
import test from 'node:test';
import {
	buildDepthAlignmentQcPlot,
	computeDepthAlignmentQcSummary
} from '../src/lib/utils/depthAlignmentQc.ts';

function buildSyntheticCurves(shiftDtcM = 0.6) {
	const depths = [];
	for (let z = 2000; z < 2060; z += 0.15) {
		depths.push(Number(z.toFixed(4)));
	}
	const beds = depths.map(
		(z) =>
			(Math.sin((2 * Math.PI * (z - 2000)) / 8.5) > 0 ? 1 : 0) +
			0.5 * (Math.sin((2 * Math.PI * (z - 2000)) / 3.4) > 0.2 ? 1 : 0)
	);
	const gr = beds.map((b) => 30 + 60 * b);
	const rhob = beds.map((b) => 2.25 + 0.2 * b);
	const dtcTrue = beds.map((b) => 95 - 18 * b);

	// Linearly interpolate dtcTrue at z + shiftDtcM (simulating curve recorded shallower by shiftDtcM)
	const dtc = depths.map((z) => {
		const targetZ = z + shiftDtcM;
		const pos = (targetZ - 2000) / 0.15;
		const i0 = Math.max(0, Math.min(depths.length - 1, Math.floor(pos)));
		const i1 = Math.max(0, Math.min(depths.length - 1, Math.ceil(pos)));
		if (i0 === i1) return dtcTrue[i0];
		const frac = pos - i0;
		return dtcTrue[i0] + frac * (dtcTrue[i1] - dtcTrue[i0]);
	});

	return {
		depths,
		indices: depths.map((_, i) => i),
		curves: {
			DEPT: depths,
			GR: gr,
			RHOB: rhob,
			DTC: dtc
		}
	};
}

test('computeDepthAlignmentQcSummary detects off-depth curve and converts display units', () => {
	const { depths, indices, curves } = buildSyntheticCurves(0.6);

	const sumM = computeDepthAlignmentQcSummary({
		curves,
		depthsMeters: depths,
		indices,
		depthUnit: 'm',
		referenceCurve: 'GR'
	});
	const rhob = sumM.rows.find((r) => r.CURVE === 'RHOB');
	const dtc = sumM.rows.find((r) => r.CURVE === 'DTC');
	assert.equal(rhob?.STATUS, 'ALIGNED');
	assert.equal(dtc?.STATUS, 'OFF_DEPTH');
	assert.ok(dtc && Math.abs(dtc.SHIFT_M - 0.6) < 0.1);

	const sumFt = computeDepthAlignmentQcSummary({
		curves,
		depthsMeters: depths,
		indices,
		depthUnit: 'ft',
		referenceCurve: 'GR'
	});
	const dtcFt = sumFt.rows.find((r) => r.CURVE === 'DTC');
	assert.ok(dtcFt && Math.abs(dtcFt.SHIFT_DISPLAY - 0.6 * 3.28084) < 0.35);
});

test('buildDepthAlignmentQcPlot builds both 3-track drift composite and correlogram views', () => {
	const { depths, indices, curves } = buildSyntheticCurves(0.45);

	const driftPlot = buildDepthAlignmentQcPlot({
		curves,
		depthsMeters: depths,
		indices,
		depthUnit: 'm',
		viewMode: 'drift',
		referenceCurve: 'GR'
	});
	assert.equal(driftPlot.activeTarget, 'DTC');
	assert.ok(driftPlot.layout.xaxis && driftPlot.layout.xaxis2 && driftPlot.layout.xaxis3);
	assert.equal(driftPlot.layout.dragmode, 'pan');
	assert.equal(driftPlot.layout.xaxis.fixedrange, true);
	assert.equal(driftPlot.layout.xaxis2.fixedrange, true);
	assert.equal(driftPlot.layout.xaxis3.fixedrange, true);
	assert.ok(Array.isArray(driftPlot.layout.xaxis2.range) && driftPlot.layout.xaxis2.range.length === 2);
	assert.ok(driftPlot.traces.length >= 6);

	const corrPlot = buildDepthAlignmentQcPlot({
		curves,
		depthsMeters: depths,
		indices,
		depthUnit: 'ft',
		viewMode: 'correlogram',
		referenceCurve: 'GR'
	});
	assert.ok(corrPlot.traces.length >= 4);
	assert.ok(corrPlot.layout.xaxis.title.text.includes('ft'));
});
