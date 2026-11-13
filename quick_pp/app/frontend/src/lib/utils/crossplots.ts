import type { MultiWellStep, SingleWellStep } from '../state/workbench.svelte';

export type CrossplotId =
	| 'nphi-rhob'
	| 'mn-plot'
	| 'thomas-stieber'
	| 'phi-k'
	| 'pickett'
	| 'sw-comparison'
	| 'porosity-comparison'
	| 'permeability-comparison'
	| 'bqv'
	| 'histogram'
	| 'curve-depth'
	| 'enhanced-vs-raw'
	| 'mw-distribution'
	| 'swirr-fit'
	| 'shf'
	| 'relperm'
	| 'core-qc'
	| 'vsh-qc'
	| 'th-k'
	| 'pef-k'
	| 'pef-th-k';

export type CrossplotContext =
	| { scope: 'single-well'; step: SingleWellStep }
	| { scope: 'multi-well'; step: MultiWellStep };

type PlotPlacement<Step extends string> = {
	group: string;
	steps: readonly Step[];
	autoFor?: readonly Step[];
};

type CrossplotDefinition = {
	id: CrossplotId;
	label: string;
	single?: PlotPlacement<SingleWellStep>;
	multi?: PlotPlacement<MultiWellStep>;
};

export const CROSSPLOTS: readonly CrossplotDefinition[] = [
	{
		id: 'histogram',
		label: 'Single-Well Histogram',
		single: { group: 'Step 1 · Data & QAQC', steps: ['qaqc'], autoFor: ['qaqc'] }
	},
	{
		id: 'curve-depth',
		label: 'Curve Editor · Log vs Depth',
		single: { group: 'Step 1 · Data & QAQC', steps: ['qaqc'] }
	},
	{
		id: 'enhanced-vs-raw',
		label: 'Resolution Enhanced vs. Raw',
		single: { group: 'Step 1 · Data & QAQC', steps: ['qaqc'] }
	},
	{
		id: 'nphi-rhob',
		label: 'NPHI vs. RHOB (SSC)',
		single: {
			group: 'Step 3 · Lithology & Porosity',
			steps: ['litho-poro'],
			autoFor: ['litho-poro']
		}
	},
	{
		id: 'mn-plot',
		label: 'M-N Plot (Mineral ID)',
		single: { group: 'Step 3 · Lithology & Porosity', steps: ['litho-poro'] }
	},
	{
		id: 'vsh-qc',
		label: 'Vsh Calibration (VSH_GR × VSH_DN)',
		single: { group: 'Step 3 · Lithology & Porosity', steps: ['litho-poro'] }
	},
	{
		id: 'porosity-comparison',
		label: 'Porosity Methods vs Depth + Core',
		single: { group: 'Step 3 · Lithology & Porosity', steps: ['litho-poro'] }
	},
	{
		id: 'th-k',
		label: 'Thorium vs. Potassium',
		single: { group: 'Step 3 · Lithology & Porosity', steps: ['litho-poro'] }
	},
	{
		id: 'pef-k',
		label: 'PEF vs. Potassium',
		single: { group: 'Step 3 · Lithology & Porosity', steps: ['litho-poro'] }
	},
	{
		id: 'pef-th-k',
		label: 'PEF vs. Th/K',
		single: { group: 'Step 3 · Lithology & Porosity', steps: ['litho-poro'] }
	},
	{
		id: 'thomas-stieber',
		label: 'Thomas-Stieber (Vsh vs. Φ)',
		single: { group: 'Step 3 · Lithology & Porosity', steps: ['litho-poro'] }
	},
	{
		id: 'phi-k',
		label: 'Porosity vs. Perm',
		single: { group: 'Step 4 · Permeability', steps: ['perm'], autoFor: ['perm'] },
		multi: {
			group: 'Steps 3–4 · Rock Typing & Perm Transform',
			steps: ['rock-typing', 'perm-transform', 'rt-classifier'],
			autoFor: ['rock-typing', 'perm-transform', 'rt-classifier']
		}
	},
	{
		id: 'sw-comparison',
		label: 'Saturation vs Depth + VSHALE',
		single: { group: 'Step 5 · Saturation', steps: ['saturation'], autoFor: ['saturation'] }
	},
	{
		id: 'permeability-comparison',
		label: 'Permeability Methods vs Depth + Core',
		single: { group: 'Step 4 · Permeability', steps: ['perm'] }
	},
	{
		id: 'pickett',
		label: 'Pickett Plot (Rt vs Φ)',
		single: { group: 'Step 5 · Saturation', steps: ['saturation'] }
	},
	{
		id: 'bqv',
		label: 'Waxman–Smits BQv',
		single: { group: 'Step 5 · Saturation', steps: ['saturation'] }
	},
	{
		id: 'core-qc',
		label: 'Core Parity & Error (Φ & K)',
		single: {
			group: 'Standalone · Core & Pc Prep',
			steps: ['scal', 'litho-poro', 'perm'],
			autoFor: ['scal']
		}
	},
	{
		id: 'mw-distribution',
		label: 'Multi-Well Distribution (QAQC)',
		multi: { group: 'Step 2 · Data & QAQC', steps: ['qaqc'], autoFor: ['qaqc'] }
	},
	{
		id: 'shf',
		label: 'Sw vs. Height (SHF)',
		single: { group: 'Step 5 · Saturation', steps: ['saturation'] },
		multi: { group: 'Step 5 · Saturation Height', steps: ['shf'], autoFor: ['shf'] }
	},
	{
		id: 'swirr-fit',
		label: 'Swirr & Pc Fit (SCAL)',
		multi: {
			group: 'Standalone · SCAL & Rel-Perm',
			steps: ['swirr-corr'],
			autoFor: ['swirr-corr']
		}
	},
	{
		id: 'relperm',
		label: 'Rel-Perm & Fractional Flow (SCAL)',
		multi: { group: 'Standalone · SCAL & Rel-Perm', steps: ['swirr-corr'] }
	}
];

function placement(plot: CrossplotDefinition, context: CrossplotContext) {
	return context.scope === 'single-well' ? plot.single : plot.multi;
}

export function isCrossplotAvailable(id: CrossplotId, scope: CrossplotContext['scope']): boolean {
	const plot = CROSSPLOTS.find((item) => item.id === id);
	return Boolean(plot && (scope === 'single-well' ? plot.single : plot.multi));
}

export function resolveCrossplot(
	context: CrossplotContext,
	override: CrossplotId | 'auto'
): CrossplotId | null {
	if (override !== 'auto' && isCrossplotAvailable(override, context.scope)) return override;
	const plot = CROSSPLOTS.find((item) => {
		if (context.scope === 'single-well') return item.single?.autoFor?.includes(context.step);
		return item.multi?.autoFor?.includes(context.step);
	});
	return plot?.id ?? null;
}

export function crossplotLabel(id: CrossplotId, drtMethod: string): string {
	if (id === 'phi-k') {
		const method =
			drtMethod === 'winland' ? 'Winland R35' : drtMethod === 'lucia' ? 'Lucia RFN' : 'FZI';
		return `Porosity vs. Perm (${method})`;
	}
	return CROSSPLOTS.find((plot) => plot.id === id)!.label;
}

export function crossplotGroups(context: CrossplotContext): Array<{
	label: string;
	plots: readonly CrossplotDefinition[];
}> {
	const current: CrossplotDefinition[] = [];
	const other = new Map<string, CrossplotDefinition[]>();
	for (const plot of CROSSPLOTS) {
		const entry = placement(plot, context);
		if (!entry) continue;
		const isCurrent =
			context.scope === 'single-well'
				? plot.single?.steps.includes(context.step)
				: plot.multi?.steps.includes(context.step);
		if (isCurrent) {
			current.push(plot);
		} else {
			const group = other.get(entry.group) ?? [];
			group.push(plot);
			other.set(entry.group, group);
		}
	}
	return [
		...(current.length ? [{ label: 'Current step', plots: current }] : []),
		...Array.from(other, ([label, plots]) => ({ label, plots }))
	];
}
