import assert from 'node:assert/strict';
import test from 'node:test';
import {
	comparisonPlotConfig,
	comparisonLeadingAxis,
	comparisonXAxis,
	MATRIX_DENSITY_RANGE
} from '../src/lib/utils/comparisonPlot.ts';

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

test('leading tracks carry density for porosity and VSHALE for permeability', () => {
	const density = comparisonLeadingAxis('porosity');
	const vshale = comparisonLeadingAxis('permeability');
	assert.deepEqual(density.range, MATRIX_DENSITY_RANGE);
	assert.equal(density.fixedrange, true);
	assert.deepEqual(vshale.range, [0, 1]);
	assert.equal(vshale.fixedrange, true);
});

test('density track expands to include a light fluid without becoming auto-ranged', () => {
	const gas = comparisonLeadingAxis('porosity', 0.25);
	assert.deepEqual(gas.range, [0.2, 3.0]);
	assert.equal(gas.fixedrange, true);
	assert.deepEqual(comparisonLeadingAxis('porosity', 1.0).range, [1, 3]);
	assert.deepEqual(comparisonLeadingAxis('porosity', 2.0).range, MATRIX_DENSITY_RANGE);
});
