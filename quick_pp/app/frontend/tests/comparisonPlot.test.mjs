import assert from 'node:assert/strict';
import test from 'node:test';
import { comparisonPlotConfig, comparisonXAxis } from '../src/lib/utils/comparisonPlot.ts';

test('comparison tracks retain their fixed scales while allowing scroll zoom', () => {
	const porosity = comparisonXAxis('porosity');
	const permeability = comparisonXAxis('permeability');
	assert.deepEqual(porosity.range, [0, 0.5]);
	assert.equal(porosity.fixedrange, true);
	assert.equal(permeability.type, 'log');
	assert.deepEqual(permeability.range, [Math.log10(0.1), Math.log10(1e5)]);
	assert.equal(permeability.fixedrange, true);
	assert.equal(comparisonPlotConfig.scrollZoom, true);
});
