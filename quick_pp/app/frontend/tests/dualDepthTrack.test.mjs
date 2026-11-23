import assert from 'node:assert/strict';
import test from 'node:test';
import { createServer } from 'vite';

test('LogTrackCanvas renders dual depth track header and ticks when TVDSS is present', async () => {
	const previousCancelAnimationFrame = globalThis.cancelAnimationFrame;
	globalThis.cancelAnimationFrame = () => {};
	const server = await createServer({
		server: { middlewareMode: true },
		appType: 'custom'
	});

	try {
		const { wellData } = await server.ssrLoadModule('/src/lib/state/wellData.svelte.ts');
		const { workbench } = await server.ssrLoadModule('/src/lib/state/workbench.svelte.ts');
		const { default: LogTrackCanvas } = await server.ssrLoadModule(
			'/src/lib/components/workbench/LogTrackCanvas.svelte'
		);
		const { render } = await server.ssrLoadModule('svelte/server');

		wellData.curves = {
			DEPT: [1000, 1050, 1100],
			TVDSS: [800, 840, 880],
			GR: [45, 60, 55]
		};
		workbench.cursorDepth = 1050;

		const html = render(LogTrackCanvas).body;

		// 1. Dual Depth Track scale headers
		assert.match(html, /Reference/);
		assert.match(html, /TVDSS/);

		// 2. Primary Index Selector and Dual Depth toggle
		assert.match(html, /Index/);
		assert.match(html, /Dual Depth/);

		// 3. Ticks rendered in gutter
		assert.match(html, /1000/);
		assert.match(html, /800/);
	} finally {
		await server.close();
		globalThis.cancelAnimationFrame = previousCancelAnimationFrame;
	}
});

test('LogTrackCanvas handles TVDSS with leading/trailing NaNs and partial intervals gracefully', async () => {
	const previousCancelAnimationFrame = globalThis.cancelAnimationFrame;
	globalThis.cancelAnimationFrame = () => {};
	const server = await createServer({
		server: { middlewareMode: true },
		appType: 'custom'
	});

	try {
		const { wellData } = await server.ssrLoadModule('/src/lib/state/wellData.svelte.ts');
		const { workbench } = await server.ssrLoadModule('/src/lib/state/workbench.svelte.ts');
		const { default: LogTrackCanvas } = await server.ssrLoadModule(
			'/src/lib/components/workbench/LogTrackCanvas.svelte'
		);
		const { render } = await server.ssrLoadModule('svelte/server');

		// Simulating well 44-22-1: TVDSS only defined for sub-interval, with NaNs at boundaries
		wellData.curves = {
			DEPT: [11300, 11350, 11400, 11450, 11500, 11550, 11600],
			TVDSS: [Number.NaN, Number.NaN, 11400, 11450, 11500, Number.NaN, Number.NaN],
			GR: [45, 60, 55, 70, 80, 65, 50]
		};
		workbench.cursorDepth = 11450;

		const html = render(LogTrackCanvas).body;
		assert.match(html, /Index/);
		assert.match(html, /Dual Depth/);
		assert.match(html, /Reference/);
		assert.match(html, /TVDSS/);
	} finally {
		await server.close();
		globalThis.cancelAnimationFrame = previousCancelAnimationFrame;
	}
});

test('trajectory Minimum Curvature and depth grid interpolation works correctly', async () => {
	const server = await createServer({
		server: { middlewareMode: true },
		appType: 'custom'
	});

	try {
		const { computeMinimumCurvature, interpolateTrajectoryOnGrid } = await server.ssrLoadModule(
			'/src/lib/utils/trajectory.ts'
		);

		const rawSurveys = [
			{ md: 1000, inc: 0, azim: 0 },
			{ md: 1100, inc: 10, azim: 45 },
			{ md: 1200, inc: 20, azim: 45 }
		];

		// 1. With elevation (DF = 25m): TVDSS = TVD - 25
		const withElev = computeMinimumCurvature(rawSurveys, 25);
		assert.equal(withElev.length, 3);
		assert.equal(withElev[0].tvd, 1000);
		assert.equal(withElev[0].tvdss, 975);
		assert.ok(withElev[1].tvd > 1000 && withElev[1].tvd < 1100);
		assert.equal(Number((withElev[1].tvd - withElev[1].tvdss).toFixed(2)), 25);

		// 2. Without elevation (0): TVDSS = TVD (TVDDF mode)
		const noElev = computeMinimumCurvature(rawSurveys, 0);
		assert.equal(noElev[0].tvd, 1000);
		assert.equal(noElev[0].tvdss, 1000);
		assert.equal(noElev[1].tvdss, noElev[1].tvd);

		// 3. Grid interpolation
		const grid = [1000, 1050, 1100, 1150, 1200];
		const interp = interpolateTrajectoryOnGrid(grid, rawSurveys, 25);
		assert.ok(interp);
		assert.equal(interp.tvd.length, 5);
		assert.equal(interp.tvdss.length, 5);
		assert.equal(interp.tvd[0], 1000);
		assert.equal(interp.tvdss[0], 975);
	} finally {
		await server.close();
	}
});

test('InspectorAncillary displays TVDSS when datum elevation is present and TVDDF when missing', async () => {
	const server = await createServer({
		server: { middlewareMode: true },
		appType: 'custom'
	});

	try {
		const { wellData } = await server.ssrLoadModule('/src/lib/state/wellData.svelte.ts');
		const { workbench } = await server.ssrLoadModule('/src/lib/state/workbench.svelte.ts');
		const { default: InspectorAncillary } = await server.ssrLoadModule(
			'/src/lib/components/modules/InspectorAncillary.svelte'
		);
		const { render } = await server.ssrLoadModule('svelte/server');

		workbench.projectId = 'proj-1';
		workbench.wellId = 'well-1';
		wellData.survey = [
			{ md: 1000, inc: 0, azim: 0 },
			{ md: 1100, inc: 5, azim: 30 }
		];

		// Case A: Missing DF/KB elevation datum -> TVDDF
		wellData.header = {};
		wellData.config = {};
		let html = render(InspectorAncillary, { props: { initialTab: 'survey' } }).body;
		assert.match(html, /TVDDF/);
		assert.match(html, /Recalculate TVD/);

		// Case B: Present DF/KB elevation datum (e.g. 82 ft DF) -> TVDSS
		wellData.header = { df_elev: 82, datum_unit: 'ft' };
		html = render(InspectorAncillary, { props: { initialTab: 'survey' } }).body;
		assert.match(html, /TVDSS/);
		assert.match(html, /TVDSS Active/);
	} finally {
		await server.close();
	}
});
