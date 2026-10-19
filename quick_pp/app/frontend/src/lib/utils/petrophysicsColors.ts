/**
 * Subsurface petrophysical color palette, curve metadata, and fill helpers
 * aligned with the "Precision Daylight" design system.
 */

export const PETRO_COLORS = {
	// Lithology fills & strokes
	sandFill: '#FDE047',
	sandStroke: '#CA8A04',
	shaleFill: '#94A3B8',
	shaleStroke: '#475569',
	siltFill: '#FDBA74',
	siltStroke: '#EA580C',
	calcFill: '#38BDF8',
	calcStroke: '#0284C7',
	doloFill: '#C084FC',
	doloStroke: '#7E22CE',

	// Fluids & saturation
	oil: '#16A34A',
	gas: '#DC2626',
	water: '#2563EB',

	// Standard log curves
	gr: '#15803D',
	rt: '#DC2626',
	rmed: '#2563EB',
	rshal: '#0D9488',
	rhob: '#DC2626',
	nphi: '#2563EB',
	dt: '#9333EA',
	cali: '#475569',
	sp: '#7C3AED',
	core: '#0F172A',
	perm: '#0284C7',
	phie: '#2563EB',
	phit: '#0F172A'
} as const;

export const ROCK_TYPE_PALETTE = [
	'#2563EB', // RT 1 - Cobalt
	'#16A34A', // RT 2 - Emerald
	'#D97706', // RT 3 - Amber
	'#DC2626', // RT 4 - Crimson
	'#9333EA', // RT 5 - Purple
	'#0891B2', // RT 6 - Cyan
	'#EA580C', // RT 7 - Orange
	'#64748B' // RT 8 / Non-reservoir - Slate
];

export const ZONE_BAND_COLORS = [
	'rgba(37, 99, 235, 0.08)',
	'rgba(22, 163, 74, 0.08)',
	'rgba(217, 119, 6, 0.08)',
	'rgba(147, 51, 234, 0.08)',
	'rgba(8, 145, 178, 0.08)',
	'rgba(234, 88, 12, 0.08)'
];

export interface CurveMeta {
	mnemonic: string;
	label: string;
	unit: string;
	min: number;
	max: number;
	color: string;
	logScale?: boolean;
	reversed?: boolean;
}

export const STANDARD_CURVE_META: Record<string, CurveMeta> = {
	GR: { mnemonic: 'GR', label: 'Gamma Ray', unit: 'gAPI', min: 0, max: 150, color: PETRO_COLORS.gr },
	CALI: { mnemonic: 'CALI', label: 'Caliper', unit: 'in', min: 6, max: 16, color: PETRO_COLORS.cali },
	SP: { mnemonic: 'SP', label: 'Spontaneous Potential', unit: 'mV', min: -100, max: 20, color: PETRO_COLORS.sp },
	RT: {
		mnemonic: 'RT',
		label: 'Deep Resistivity',
		unit: 'ohm.m',
		min: 0.2,
		max: 2000,
		color: PETRO_COLORS.rt,
		logScale: true
	},
	NPHI: {
		mnemonic: 'NPHI',
		label: 'Neutron Porosity',
		unit: 'v/v',
		min: -0.15,
		max: 0.45,
		color: PETRO_COLORS.nphi,
		reversed: true
	},
	RHOB: {
		mnemonic: 'RHOB',
		label: 'Bulk Density',
		unit: 'g/cc',
		min: 1.95,
		max: 2.95,
		color: PETRO_COLORS.rhob
	},
	DTC: { mnemonic: 'DTC', label: 'Compressional Sonic', unit: 'us/ft', min: 40, max: 140, color: PETRO_COLORS.dt, reversed: true },
	PEF: { mnemonic: 'PEF', label: 'Photoelectric Factor', unit: 'b/e', min: 0, max: 10, color: '#0284C7' },
	VSAND: { mnemonic: 'VSAND', label: 'Sand Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.sandStroke },
	VSILT: { mnemonic: 'VSILT', label: 'Silt Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.siltStroke },
	VCLAY: { mnemonic: 'VCLAY', label: 'Clay Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.shaleStroke },
	PHIT: { mnemonic: 'PHIT', label: 'Total Porosity', unit: 'v/v', min: 0, max: 0.4, color: PETRO_COLORS.phit, reversed: true },
	PHIE: { mnemonic: 'PHIE', label: 'Effective Porosity', unit: 'v/v', min: 0, max: 0.4, color: PETRO_COLORS.phie, reversed: true },
	PERM: {
		mnemonic: 'PERM',
		label: 'Permeability',
		unit: 'mD',
		min: 0.01,
		max: 10000,
		color: PETRO_COLORS.perm,
		logScale: true
	},
	SWT: { mnemonic: 'SWT', label: 'Total Water Saturation', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.water, reversed: true },
	SWE: { mnemonic: 'SWE', label: 'Effective Water Saturation', unit: 'v/v', min: 0, max: 1, color: '#0284C7', reversed: true }
};

export function getCurveUnit(mnemonic: string): string {
	const upper = mnemonic.toUpperCase();
	if (STANDARD_CURVE_META[upper]) return STANDARD_CURVE_META[upper].unit;
	if (upper.includes('NPHI') || upper.includes('PHI') || upper.startsWith('V') || upper.startsWith('SW') || upper === 'BVW') {
		return 'v/v';
	}
	if (upper.includes('RHOB') || upper.includes('DEN')) return 'g/cc';
	if (upper.includes('PERM')) return 'mD';
	if (upper.startsWith('R') && upper.length <= 5) return 'ohm.m';
	if (upper.includes('DEPTH') || upper.includes('DEPT') || upper === 'TVD') return 'm';
	return '';
}

export function getRockTypeColor(rockFlag: number | string | null | undefined): string {
	const idx = Math.max(0, (Number(rockFlag) || 1) - 1);
	return ROCK_TYPE_PALETTE[idx % ROCK_TYPE_PALETTE.length];
}

