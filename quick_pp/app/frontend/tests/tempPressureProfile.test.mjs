import assert from 'node:assert/strict';
import test from 'node:test';
import { buildTempPressurePlot } from '../src/lib/utils/tempPressureProfile.ts';

const baseParams = {
	bitsizeDefault: 8.5,
	mudWeight: 10.5,
	surfaceTemp: 77.0,
	tempGrad: 1.6,
	salinityPpm: 25000.0,
	toolType: 'WL',
	mudType: 'WBM',
	offshore: false,
	waterDepth: 0,
	seabedTempC: 4.0,
	rmf: 0.3,
	rmfTempC: 25.0,
	formationSalinityPpm: 50000,
	washoutThold: 2.0
};

test('buildTempPressurePlot plots against TVDSS when TVDSS curve or TVD-KB is present, and flags MD fallback clearly', () => {
	// Deviated well: MD = 2000..2300 m, TVDSS = 1800..2040 m, KB = 25 m
	const md = [2000, 2100, 2200, 2300];
	const tvdss = [1800, 1880, 1960, 2040];
	const curves = {
		TVDSS: tvdss,
		ENV_TEMP_C: [80, 83, null, 89],
		TEMP: [78, 82, null, 88]
	};
	const header = { bht: 92, bht_unit: 'degC', bht_depth: 2300, kb_elev: 25, datum_unit: 'm' };
	const pressureTests = [
		{ depth: 2050, pressure: 3250, pressure_uom: 'psi', mobility: 45, temperature: 81.5, fluid_type: 'GAS', test_quality: 'Good' },
		{ depth: 2250, pressure: 3580, pressure_uom: 'psi', mobility: 18, temperature: 87.2, fluid_type: 'WATER', test_quality: 'Good' }
	];

	const metricPlot = buildTempPressurePlot({
		curves,
		depthsMeters: md,
		indices: [0, 1, 2, 3],
		depthUnit: 'm',
		tempUnit: 'degC',
		presUnit: 'psi',
		envParams: baseParams,
		header,
		pressureTests
	});

	assert.equal(metricPlot.summary.hasTrueTvdss, true);
	assert.equal(metricPlot.summary.depthBasis, 'TVDSS');
	assert.equal(metricPlot.layout.yaxis.title.text, 'TVDSS (m)');
	assert.equal(metricPlot.summary.bhtDisplay, 92);
	// BHT at MD=2300 should project onto TVDSS=2040, NOT MD=2300!
	assert.equal(metricPlot.summary.bhtTvdssDisplay, 2040);

	//ENV_TEMP trace y coordinates must match TVDSS [1800, 1880, 1960, 2040]
	const envTempTrace = metricPlot.traces.find((t) => t.name === 'ENV_TEMP');
	assert.deepEqual(envTempTrace.y, [1800, 1880, 1960, 2040]);

	// MDT at MD=2050 (midpoint of 2000..2100) must interpolate to TVDSS=1840
	assert.equal(metricPlot.summary.tableKind, 'mdt');
	assert.equal(metricPlot.summary.tableRows[0].MD, 2050);
	assert.equal(metricPlot.summary.tableRows[0].TVDSS, 1840);

	// Ensure all filled ribbons use gap-safe closed toself polygons
	for (const ft of metricPlot.traces.filter((t) => t.fill && t.fill !== 'none')) {
		assert.equal(ft.fill, 'toself');
	}

	// Verify fallback labeling when no TVDSS or TVD curve exists (MD - KB vertical assumed)
	const offshoreFallback = buildTempPressurePlot({
		curves: {},
		depthsMeters: [500, 1000, 2000],
		indices: [0, 1, 2],
		depthUnit: 'ft',
		tempUnit: 'degF',
		presUnit: 'MPa',
		envParams: { ...baseParams, offshore: true, waterDepth: 1500, seabedTempC: 4.0 },
		header: { kb_elev: 25, datum_unit: 'm' },
		pressureTests: [],
		tops: [{ name: 'Sand A', depth: 1200 }, { name: 'Sand B', depth: 1800 }]
	});

	assert.equal(offshoreFallback.summary.hasTrueTvdss, false);
	assert.equal(offshoreFallback.summary.depthBasis, 'MD_MINUS_KB');
	assert.ok(offshoreFallback.layout.yaxis.title.text.includes('TVDSS est.'));
	assert.equal(offshoreFallback.summary.tableKind, 'zonal');
	assert.ok('TVDSS' in offshoreFallback.summary.tableRows[0] && 'MD' in offshoreFallback.summary.tableRows[0]);
});

test('pressureTests with tvdss: null resolve TVDSS via TVDSS curve or directional survey instead of coercing null to 0', () => {
	// Reproduce 44-22-1: DB pressure_tests have `tvdss: null` and directional survey provides trajectory
	const md = [0, 2000, 3495, 3550, 4264, 4264.8];
	const survey = [
		{ md: 0, inc: 0, azim: 0 },
		{ md: 2000, inc: 2, azim: 90 },
		{ md: 3500, inc: 5, azim: 90 },
		{ md: 4264, inc: 5, azim: 90 }
	];
	const pressureTests = [
		{
			depth: 3495.14,
			tvdss: null,
			pressure: 6136.2,
			pressure_uom: 'psi',
			mobility: null,
			temperature: null,
			fluid_type: 'GAS',
			test_quality: 'VALID'
		},
		{
			depth: 3517.7,
			tvdss: null,
			pressure: 6173.1,
			pressure_uom: 'psi',
			mobility: 12.4,
			temperature: 131.0,
			fluid_type: 'GAS',
			test_quality: 'VALID'
		}
	];

	// 1. With survey table only (no TVDSS curve yet) + DF elevation 33.2 m
	const fromSurvey = buildTempPressurePlot({
		curves: {},
		depthsMeters: md,
		indices: [0, 1, 2, 3, 4, 5],
		depthUnit: 'm',
		tempUnit: 'degC',
		presUnit: 'psi',
		envParams: baseParams,
		header: {},
		config: { df_elev: 33.22, datum_unit: 'm' },
		survey,
		pressureTests
	});

	assert.equal(fromSurvey.summary.hasTrueTvdss, true);
	assert.equal(fromSurvey.summary.depthBasis, 'TVDSS');
	const mdtTrace = fromSurvey.traces.find((t) => t.name?.startsWith('MDT / RFT'));
	assert.ok(mdtTrace, 'Expected MDT / RFT scatter trace');
	// Must be ~3458 m TVDSS (NOT 0!)
	assert.ok(mdtTrace.y[0] > 3400 && mdtTrace.y[0] < 3500, `Expected ~3460m TVDSS, got ${mdtTrace.y[0]}`);
	assert.ok(mdtTrace.y[1] > mdtTrace.y[0]);
	assert.ok(fromSurvey.summary.tableRows[0].TVDSS > 3400);

	// Gauge temperature with tvdss: null must also project to ~3480 m TVDSS (not 0)
	const gaugeTempTrace = fromSurvey.traces.find((t) => t.name === 'MDT Gauge T');
	assert.ok(gaugeTempTrace && gaugeTempTrace.y[0] > 3400);

	// 2. With TVDSS curve having trailing NaN + stale ENV_TEMP_C having trailing NaN (no horizontal jump)
	const withTrailingNan = buildTempPressurePlot({
		curves: {
			TVDSS: [-33.22, 1966.5, 3461.8, 3516.6, 4229.6, null],
			ENV_TEMP_C: [23.9, 41.0, 52.8, 53.3, 59.4, null]
		},
		depthsMeters: md,
		indices: [0, 1, 2, 3, 4, 5],
		depthUnit: 'm',
		tempUnit: 'degC',
		presUnit: 'psi',
		envParams: baseParams,
		header: {},
		pressureTests
	});
	const envTrace = withTrailingNan.traces.find((t) => t.name === 'ENV_TEMP');
	// Last slot is null rather than jumping from 59.4°C to ~149°C!
	assert.equal(envTrace.x[5], null);
	// MDT points still land at ~3461.9 m TVDSS
	const mdtTrace2 = withTrailingNan.traces.find((t) => t.name?.startsWith('MDT / RFT'));
	assert.ok(Math.abs(mdtTrace2.y[0] - 3461.94) < 1.0, `Got ${mdtTrace2.y[0]}`);
});
