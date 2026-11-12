import assert from 'node:assert/strict';
import test from 'node:test';
import {
	CROSSPLOTS,
	crossplotGroups,
	crossplotLabel,
	isCrossplotAvailable,
	resolveCrossplot
} from '../src/lib/utils/crossplots.ts';

const singleSteps = [
	'qaqc',
	'ancillary',
	'litho-poro',
	'perm',
	'saturation',
	'ressum',
	'scal',
	'report'
];
const multiSteps = [
	'inventory',
	'qaqc',
	'rock-typing',
	'perm-transform',
	'swirr-corr',
	'shf',
	'rt-classifier',
	'ml-training',
	'fpt',
	'volumetrics',
	'report'
];

test('every context lists each supported plot exactly once and only supported plots', () => {
	for (const [scope, steps] of [
		['single-well', singleSteps],
		['multi-well', multiSteps]
	]) {
		for (const step of steps) {
			const context = { scope, step };
			const ids = crossplotGroups(context).flatMap((group) =>
				group.plots.map((plot) => plot.id)
			);
			assert.equal(new Set(ids).size, ids.length, `${scope} ${step}`);
			assert.deepEqual(
				ids.toSorted(),
				CROSSPLOTS.filter((plot) => isCrossplotAvailable(plot.id, scope))
					.map((plot) => plot.id)
					.toSorted(),
				`${scope} ${step}`
			);
			const auto = resolveCrossplot(context, 'auto');
			if (auto !== null) assert.ok(ids.includes(auto));
		}
	}
});

test('auto is specific to the current step, including steps without a recommended plot', () => {
	for (const [scope, steps] of [
		['single-well', singleSteps],
		['multi-well', multiSteps]
	]) {
		for (const step of steps) {
			const key = scope === 'single-well' ? 'single' : 'multi';
			assert.ok(
				CROSSPLOTS.filter((plot) => plot[key]?.autoFor?.includes(step)).length <= 1,
				`Multiple Auto recommendations for ${scope} ${step}`
			);
		}
	}
	assert.equal(resolveCrossplot({ scope: 'single-well', step: 'qaqc' }, 'auto'), 'histogram');
	assert.equal(
		resolveCrossplot({ scope: 'single-well', step: 'litho-poro' }, 'auto'),
		'nphi-rhob'
	);
	assert.equal(resolveCrossplot({ scope: 'single-well', step: 'scal' }, 'auto'), 'core-qc');
	assert.equal(resolveCrossplot({ scope: 'single-well', step: 'ancillary' }, 'auto'), null);
	assert.equal(resolveCrossplot({ scope: 'multi-well', step: 'inventory' }, 'auto'), null);
	assert.equal(
		resolveCrossplot({ scope: 'multi-well', step: 'qaqc' }, 'auto'),
		'mw-distribution'
	);
	assert.equal(resolveCrossplot({ scope: 'multi-well', step: 'rt-classifier' }, 'auto'), 'phi-k');
	assert.equal(
		resolveCrossplot({ scope: 'multi-well', step: 'swirr-corr' }, 'auto'),
		'swirr-fit'
	);
	assert.equal(resolveCrossplot({ scope: 'multi-well', step: 'shf' }, 'auto'), 'shf');
});

test('manual plots survive step changes but incompatible scopes resolve to auto', () => {
	assert.equal(resolveCrossplot({ scope: 'single-well', step: 'perm' }, 'porosity-comparison'), 'porosity-comparison');
	assert.equal(resolveCrossplot({ scope: 'single-well', step: 'litho-poro' }, 'permeability-comparison'), 'permeability-comparison');
	assert.equal(isCrossplotAvailable('porosity-comparison', 'multi-well'), false);
	assert.equal(isCrossplotAvailable('permeability-comparison', 'multi-well'), false);
	assert.ok(crossplotGroups({ scope: 'single-well', step: 'litho-poro' })[0].plots.some((plot) => plot.id === 'porosity-comparison'));
	assert.ok(crossplotGroups({ scope: 'single-well', step: 'perm' })[0].plots.some((plot) => plot.id === 'permeability-comparison'));
	assert.equal(resolveCrossplot({ scope: 'single-well', step: 'qaqc' }, 'core-qc'), 'core-qc');
	assert.equal(resolveCrossplot({ scope: 'single-well', step: 'perm' }, 'core-qc'), 'core-qc');
	assert.equal(
		resolveCrossplot({ scope: 'multi-well', step: 'qaqc' }, 'core-qc'),
		'mw-distribution'
	);
	assert.equal(
		resolveCrossplot({ scope: 'single-well', step: 'ancillary' }, 'mw-distribution'),
		null
	);
	assert.equal(resolveCrossplot({ scope: 'single-well', step: 'qaqc' }, 'shf'), 'shf');
	assert.equal(resolveCrossplot({ scope: 'multi-well', step: 'qaqc' }, 'shf'), 'shf');
});

test('current-step plots are promoted and other plots retain workflow and standalone groups', () => {
	const groups = crossplotGroups({ scope: 'single-well', step: 'saturation' });
	assert.equal(groups[0].label, 'Current step');
	assert.deepEqual(
		groups[0].plots.map((plot) => plot.id),
		['sw-comparison', 'pickett', 'bqv', 'shf']
	);
	assert.ok(
		groups
			.find((group) => group.label === 'Standalone · Core & Pc Prep')
			?.plots.some((plot) => plot.id === 'core-qc')
	);
	const scal = crossplotGroups({ scope: 'multi-well', step: 'swirr-corr' });
	assert.deepEqual(
		scal[0].plots.map((plot) => plot.id),
		['swirr-fit', 'relperm']
	);
	assert.equal(crossplotLabel('phi-k', 'winland'), 'Porosity vs. Perm (Winland R35)');
});
