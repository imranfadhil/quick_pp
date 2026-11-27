import assert from 'node:assert/strict';
import test from 'node:test';
import {
	buildEnvCorrQcPlot,
	computeEnvCorrQcSummary,
	decodeEnvQcReason,
	decodeEnvQcStatus,
	decodeInvasionProfile,
	resolveFluidResistivities
} from '../src/lib/utils/envCorrQc.ts';

const baseParams = {
	bitsizeDefault: 8.5,
	mudWeight: 9.5,
	surfaceTemp: 75.0,
	tempGrad: 1.5,
	salinityPpm: 25000.0,
	toolType: 'WL',
	mudType: 'WBM',
	offshore: false,
	waterDepth: 0,
	seabedTempC: 4.0,
	rmf: 0.35,
	rmfTempC: 25.0,
	formationSalinityPpm: 60000,
	washoutThold: 2.0
};

function sampleCurves() {
	return {
		CALI: [8.5, 11.2, 8.1, 8.5],
		BS: [8.5, 8.5, 8.5, 8.5],
		DRHO: [0.02, 0.22, -0.01, 0.03],
		GR: [55, 62, 70, 58],
		GR_ENV_CORR: [55, 74, 69, 58],
		GR_SIGMA: [3.2, 9.5, 3.1, 3.2],
		GR_QC_CONF: [0.95, 0.65, 0.96, 0.95],
		GR_QC_STATUS: [0, 1, 0, 0],
		GR_CORR_DELTA: [0, 12, -1, 0],
		RT: [25, 18, 30, 28],
		RM: [12, 10, 14, 13],
		RXO: [4, 3.5, 4.5, 4.2],
		RT_ENV_CORR: [31, 23, 37, 34],
		RXO_EST: [3.8, 3.3, 4.2, 4.0],
		DI_EST: [30, 35, 28, 30],
		INVASION_FLAG: [1, 1, 1, 1],
		INVASION_PROFILE: [2, 2, 2, 2],
		RT_SIGMA: [0.04, 0.09, 0.04, 0.04],
		RT_QC_CONF: [0.97, 0.88, 0.97, 0.97],
		RT_QC_STATUS: [1, 1, 1, 1],
		NPHI: [0.22, 0.28, 0.2, 0.21],
		NPHI_ENV_CORR: [0.22, 0.245, 0.205, 0.21],
		NPHI_SIGMA: [0.012, 0.038, 0.012, 0.012],
		NPHI_QC_CONF: [0.91, 0.52, 0.92, 0.91],
		NPHI_QC_STATUS: [0, 1, 0, 0],
		RHOB: [2.35, 2.18, 2.38, 2.36],
		RHOB_ENV_CORR: [2.35, 2.18, 2.38, 2.36],
		RHOB_SIGMA: [0.015, 0.095, 0.015, 0.015],
		RHOB_QC_CONF: [0.94, 0.28, 0.94, 0.94],
		RHOB_QC_STATUS: [0, 2, 0, 0],
		DTC: [78, 128, 76, 79],
		DT_ENV_CORR: [78, null, 76, 79],
		DT_SIGMA: [2.0, 22.0, 2.0, 2.0],
		DT_QC_CONF: [0.96, 0.0, 0.96, 0.96],
		DT_QC_STATUS: [0, 2, 0, 0],
		ENV_QC_REASON: [8, 1 + 2 + 8 + 128, 8, 8],
		HOLE_SEVERITY: [0, 3, 0, 0],
		ENV_TEMP_C: [75, 75.5, 76, 76.5],
		ENV_RM: [0.22, 0.219, 0.218, 0.217],
		ENV_RMF: [0.165, 0.164, 0.163, 0.162],
		ENV_RW: [0.055, 0.054, 0.054, 0.053]
	};
}

test('decodeEnvQcReason and decodeInvasionProfile format bitmasks and profile codes', () => {
	assert.equal(decodeEnvQcReason(0), 'Clean');
	assert.equal(decodeEnvQcReason(1 + 2 + 128), 'Washout, High DRHO, Cycle Skip');
	assert.equal(decodeEnvQcStatus(0), 'Unchanged');
	assert.equal(decodeEnvQcStatus(1), 'Corrected');
	assert.equal(decodeEnvQcStatus(2), 'Unreliable');
	assert.equal(decodeInvasionProfile(2), 'HC-like (Rxo/Rt < Rmf/Rw)');
	assert.equal(decodeInvasionProfile(0), 'Uninformative / None');
});

test('resolveFluidResistivities prefers backend ENV_RM/RMF/RW and falls back cleanly for WBM vs OBM', () => {
	const curves = sampleCurves();
	const depths = [2000, 2000.5, 2001, 2001.5];
	const direct = resolveFluidResistivities(curves, depths, baseParams);
	assert.equal(direct.rm[0], 0.22);
	assert.equal(direct.rmf[0], 0.165);
	assert.equal(direct.rw[0], 0.055);

	const fallbackWbm = resolveFluidResistivities({}, depths, baseParams);
	assert.ok(fallbackWbm.rw[0] > 0 && fallbackWbm.rmf[0] > fallbackWbm.rw[0]);
	assert.ok(fallbackWbm.rm[0] > fallbackWbm.rmf[0]);

	const fallbackObm = resolveFluidResistivities({}, depths, { ...baseParams, mudType: 'OBM' });
	assert.ok(fallbackObm.rw[0] > 0);
	assert.equal(fallbackObm.rmf[0], null);
	assert.equal(fallbackObm.rm[0], null);
});

test('buildEnvCorrQcPlot creates 6-track composite with dual Hole+DRHO, fluid R toggle, ±1σ bands, and cycle-skip markers', () => {
	const curves = sampleCurves();
	const depths = [2000, 2000.5, 2001, 2001.5];
	const indices = [0, 1, 2, 3];

	const res = buildEnvCorrQcPlot({
		curves,
		depthsMeters: depths,
		indices,
		depthUnit: 'm',
		viewMode: 'overlay',
		trackFilter: 'all',
		showSigma: true,
		showFluidRes: true,
		envParams: baseParams
	});

	assert.equal(res.trackCount, 6);
	assert.equal(res.hasEnvCorr, true);
	// 6 primary x-axes + 1 secondary DRHO overlay axis (xaxis7)
	for (const axis of ['xaxis', 'xaxis2', 'xaxis3', 'xaxis4', 'xaxis5', 'xaxis6', 'xaxis7']) {
		assert.ok(res.layout[axis], `Expected ${axis} in layout`);
	}
	assert.equal(res.layout.xaxis7.overlaying, 'x');
	assert.equal(res.layout.xaxis3.type, 'log');

	const traceNames = res.traces.map((t) => t.name).filter(Boolean);
	for (const expected of [
		'CALI',
		'DRHO',
		'GR_ENV_CORR',
		'RT_ENV_CORR',
		'RXO_EST',
		'Rw (T)',
		'Rmf (T)',
		'Rm mud (T)',
		'NPHI_ENV_CORR',
		'RHOB_ENV_CORR',
		'DT_ENV_CORR',
		'Cycle Skip Removed'
	]) {
		assert.ok(traceNames.includes(expected), `Missing trace ${expected}`);
	}

	const csTrace = res.traces.find((t) => t.name === 'Cycle Skip Removed');
	assert.deepEqual(csTrace.x, [128]);
	assert.deepEqual(csTrace.y, [2000.5]);
});

test('buildEnvCorrQcPlot supports delta view mode and track focus filtering', () => {
	const curves = sampleCurves();
	const depths = [2000, 2000.5, 2001, 2001.5];
	const deltaPlot = buildEnvCorrQcPlot({
		curves,
		depthsMeters: depths,
		indices: [0, 1, 2, 3],
		depthUnit: 'ft',
		viewMode: 'delta',
		trackFilter: 'NPHI_RHOB',
		showSigma: true,
		showFluidRes: false,
		envParams: baseParams
	});

	// Hole + NPHI + RHOB = 3 tracks
	assert.equal(deltaPlot.trackCount, 3);
	assert.equal(deltaPlot.layout.yaxis.title.text, 'Depth (ft)');
	const names = deltaPlot.traces.map((t) => t.name).filter(Boolean);
	assert.ok(names.includes('ΔNPHI'));
	assert.ok(names.includes('ΔRHOB'));
});

test('computeEnvCorrQcSummary aggregates hole flags and per-curve QC metrics', () => {
	const summary = computeEnvCorrQcSummary(sampleCurves(), [0, 1, 2, 3], baseParams);
	assert.equal(summary.hasEnvCorr, true);
	assert.equal(summary.washoutPct, 25);
	assert.equal(summary.drhoHighPct, 25);
	assert.equal(summary.cycleSkipCount, 1);
	assert.equal(summary.rows.length, 5);
	const rhobRow = summary.rows.find((r) => r.CURVE === 'RHOB');
	assert.equal(rhobRow?.UNRELIABLE_PCT, 25);
	assert.equal(summary.invasion.hcLikePct, 100);
	assert.equal(summary.invasion.medianDiIn, 30);
});

test('buildEnvCorrQcPlot supports 3-track invasion profile mode with di, resistivity array, and Rxo/Rt ratio', () => {
	const curves = sampleCurves();
	const depths = [2000, 2000.5, 2001, 2001.5];
	const invPlot = buildEnvCorrQcPlot({
		curves,
		depthsMeters: depths,
		indices: [0, 1, 2, 3],
		depthUnit: 'm',
		viewMode: 'invasion',
		trackFilter: 'all',
		showSigma: true,
		showFluidRes: false,
		envParams: baseParams
	});

	assert.equal(invPlot.trackCount, 3);
	assert.ok(invPlot.layout.xaxis && invPlot.layout.xaxis2 && invPlot.layout.xaxis3);
	assert.equal(invPlot.layout.xaxis2.type, 'log');
	const names = invPlot.traces.map((t) => t.name).filter(Boolean);
	for (const expected of [
		'DI_EST (di)',
		'Rxo–Rt Separation',
		'RT_ENV_CORR',
		'RXO_EST',
		'log₁₀(Rmf/Rw)',
		'log₁₀(Rxo/Rt)'
	]) {
		assert.ok(names.includes(expected), `Missing invasion trace ${expected}`);
	}
	const sepFill = invPlot.traces.find((t) => t.name === 'Rxo–Rt Separation');
	assert.equal(sepFill.fillcolor, 'rgba(13, 148, 136, 0.18)');
	for (const ft of invPlot.traces.filter((t) => t.fill && t.fill !== 'none')) {
		assert.equal(ft.fill, 'toself');
	}
});

test('ENV_BS resolves auto-detected bit size when raw BS curve is absent', () => {
	const curves = sampleCurves();
	delete curves.BS;
	curves.ENV_BS = [8.5, 8.5, 8.5, 8.5];
	curves.CALI = [8.65, 8.7, 8.68, 8.72];

	// Even if UI scrubber has bitsizeDefault = 6.0, ENV_BS (8.5") prevents false washout
	const paramsWith6InDefault = { ...baseParams, bitsizeDefault: 6.0, washoutThold: 2.0 };
	const summary = computeEnvCorrQcSummary(curves, [0, 1, 2, 3], paramsWith6InDefault);
	assert.equal(summary.washoutPct, 0);

	const plot = buildEnvCorrQcPlot({
		curves,
		depthsMeters: [2000, 2000.5, 2001, 2001.5],
		indices: [0, 1, 2, 3],
		depthUnit: 'm',
		viewMode: 'overlay',
		trackFilter: 'all',
		showSigma: false,
		showFluidRes: false,
		envParams: paramsWith6InDefault
	});
	const bsTrace = plot.traces.find((t) => t.name === 'BS (8.5")');
	assert.ok(bsTrace, 'Expected BS (8.5") legend label from ENV_BS');
	assert.deepEqual(bsTrace.x, [8.5, 8.5, 8.5, 8.5]);
});

test('all shaded fills use closed toself polygons split cleanly across null gaps without tonextx/tozerox wedges', () => {
	const curves = {
		CALI: [10.5, 11.0, null, 7.8, 8.0, 9.5],
		BS: [8.5, 8.5, 8.5, 8.5, 8.5, 8.5],
		DRHO: [0.02, 0.24, null, -0.22, -0.05, 0.01],
		GR: [50, 60, null, 70, 80, 65],
		GR_ENV_CORR: [58, 72, null, 76, 85, 70],
		GR_SIGMA: [4, 8, null, 5, 5, 4],
		RHOB: [2.35, 2.30, null, 2.40, 2.42, 2.38],
		RHOB_ENV_CORR: [2.35, 2.30, null, 2.40, 2.42, 2.38],
		RHOB_SIGMA: [0.02, 0.09, null, 0.02, 0.02, 0.02]
	};
	const depths = [3450, 3451, 3452, 3453, 3454, 3455];
	const indices = [0, 1, 2, 3, 4, 5];

	for (const viewMode of ['overlay', 'delta']) {
		const res = buildEnvCorrQcPlot({
			curves,
			depthsMeters: depths,
			indices,
			depthUnit: 'm',
			viewMode,
			trackFilter: 'all',
			showSigma: true,
			showFluidRes: false,
			envParams: baseParams
		});

		const filledTraces = res.traces.filter((t) => t.fill && t.fill !== 'none');
		assert.ok(filledTraces.length > 0, `Expected filled traces in ${viewMode} mode`);
		for (const ft of filledTraces) {
			assert.equal(ft.fill, 'toself', `Expected toself instead of ${ft.fill} on ${ft.name}`);
			// Verify every null-separated polygon subpath explicitly closes back onto its start vertex
			let segStart = 0;
			for (let k = 0; k <= ft.x.length; k++) {
				if (k === ft.x.length || ft.x[k] === null) {
					const len = k - segStart;
					assert.ok(len >= 5, 'Closed polygon must have at least 5 vertices');
					assert.equal(ft.x[segStart], ft.x[k - 1]);
					assert.equal(ft.y[segStart], ft.y[k - 1]);
					// Ensure the gap depth (3452) never appears inside any polygon subpath
					for (let m = segStart; m < k; m++) {
						assert.notEqual(ft.y[m], 3452);
					}
					segStart = k + 1;
				}
			}
		}
	}
});
