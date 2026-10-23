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
	anhyFill: '#F472B6',
	anhyStroke: '#BE185D',
	haliFill: '#4ADE80',
	haliStroke: '#15803D',
	gypsFill: '#A78BFA',
	gypsStroke: '#6D28D9',
	kaolFill: '#CBD5E1',
	kaolStroke: '#64748B',
	feldFill: '#FBCFE8',
	feldStroke: '#DB2777',
	pyriFill: '#FACC15',
	pyriStroke: '#854D0E',
	coalFill: '#1E293B',
	coalStroke: '#0F172A',
	vshLamFill: '#475569',
	vshLamStroke: '#1E293B',
	vshDisFill: '#CBD5E1',
	vshDisStroke: '#64748B',

	// Fluids & saturation
	oil: '#16A34A',
	gas: '#DC2626',
	water: '#2563EB',
	waterFill: 'rgba(56, 189, 248, 0.32)',

	// Standard log curves
	gr: '#78350F',
	rt: '#DC2626',
	rm: '#2563EB',
	rs: '#0D9488',
	rmed: '#2563EB',
	rshal: '#0D9488',
	rtMod: '#9333EA',
	rhob: '#DC2626',
	nphi: '#2563EB',
	pef: '#A855F7',
	drho: '#D97706',
	dt: '#9333EA',
	cali: '#475569',
	sp: '#7C3AED',
	core: '#0F172A',
	perm: '#0284C7',
	phie: '#2563EB',
	phit: '#0F172A',
	phitSand: '#0D9488'
} as const;

export interface GrEarthShadeBand {
	maxGr: number;
	fillRgba: string;
}

/**
 * Earth-tone horizontal gradient stops for Gamma Ray (0–200 gAPI, 10 gAPI bins):
 * - Low GR (0–50 gAPI, clean sand/carbonate): light warm cream -> golden sand
 * - Medium GR (60–110 gAPI, silty/argillaceous): warm ochre -> earthy amber -> sienna
 * - High GR (120–150 gAPI, shale): chestnut -> anchored dark umber shale (#78350F at 150)
 * - Very High GR (160–200 gAPI, hot/organic shale): deep espresso -> charcoal earth
 */
export const GR_EARTH_SHADE_BANDS: GrEarthShadeBand[] = [
	{ maxGr: 10, fillRgba: 'rgba(254, 249, 195, 0.80)' }, // #FEF9C3 Pale warm cream
	{ maxGr: 20, fillRgba: 'rgba(254, 243, 199, 0.78)' }, // #FEF3C7 Very light warm sand
	{ maxGr: 30, fillRgba: 'rgba(253, 230, 138, 0.74)' }, // #FDE68A Light sand
	{ maxGr: 40, fillRgba: 'rgba(253, 224, 71, 0.68)' }, // #FDE047 Golden sand
	{ maxGr: 50, fillRgba: 'rgba(250, 204, 21, 0.64)' }, // #FACC15 Rich golden sand
	{ maxGr: 60, fillRgba: 'rgba(251, 191, 36, 0.62)' }, // #FBBF24 Warm yellow-ochre
	{ maxGr: 70, fillRgba: 'rgba(245, 158, 11, 0.60)' }, // #F59E0B Warm ochre
	{ maxGr: 80, fillRgba: 'rgba(234, 138, 8, 0.62)' }, // #EA8A08 Deep ochre
	{ maxGr: 90, fillRgba: 'rgba(217, 119, 6, 0.64)' }, // #D97706 Earthy amber
	{ maxGr: 100, fillRgba: 'rgba(194, 97, 8, 0.66)' }, // #C26108 Raw sienna
	{ maxGr: 110, fillRgba: 'rgba(180, 83, 9, 0.68)' }, // #B45309 Sienna brown
	{ maxGr: 120, fillRgba: 'rgba(161, 71, 11, 0.70)' }, // #A1470B Burnt sienna
	{ maxGr: 130, fillRgba: 'rgba(146, 64, 14, 0.72)' }, // #92400E Chestnut brown
	{ maxGr: 140, fillRgba: 'rgba(133, 58, 15, 0.74)' }, // #853A0F Warm umber
	{ maxGr: 150, fillRgba: 'rgba(120, 53, 15, 0.76)' }, // #78350F Dark umber shale (anchor at 150)
	{ maxGr: 160, fillRgba: 'rgba(102, 44, 13, 0.80)' }, // #662C0D Deep umber
	{ maxGr: 170, fillRgba: 'rgba(84, 36, 11, 0.83)' }, // #54240B Dark bistre
	{ maxGr: 180, fillRgba: 'rgba(69, 26, 3, 0.86)' }, // #451A03 Espresso shale
	{ maxGr: 190, fillRgba: 'rgba(48, 20, 4, 0.89)' }, // #301404 Dark roast earth
	{ maxGr: 200, fillRgba: 'rgba(28, 13, 4, 0.92)' } // #1C0D04 Very dark organic shale
];

export interface LithologyTrackSpec {
	mnemonic: string;
	altMnemonic?: string;
	mineralName: string;
	shortLabel: string;
	fillRgba: string;
	fillHex: string;
	strokeHex: string;
	nphiEndpoint: number;
	rhobEndpoint: number;
}

export const LITHOLOGY_TRACK_SPECS: LithologyTrackSpec[] = [
	{
		mnemonic: 'VCLAY',
		altMnemonic: 'VSHALE',
		mineralName: 'SHALE',
		shortLabel: 'Shale',
		fillRgba: 'rgba(148, 163, 184, 0.58)',
		fillHex: PETRO_COLORS.shaleFill,
		strokeHex: PETRO_COLORS.shaleStroke,
		nphiEndpoint: 0.3,
		rhobEndpoint: 2.55
	},
	{
		mnemonic: 'VKAOL',
		mineralName: 'KAOLINITE',
		shortLabel: 'Kaolinite',
		fillRgba: 'rgba(203, 213, 225, 0.58)',
		fillHex: PETRO_COLORS.kaolFill,
		strokeHex: PETRO_COLORS.kaolStroke,
		nphiEndpoint: 0.37,
		rhobEndpoint: 2.41
	},
	{
		mnemonic: 'VSILT',
		mineralName: 'SILT',
		shortLabel: 'Silt',
		fillRgba: 'rgba(253, 186, 116, 0.58)',
		fillHex: PETRO_COLORS.siltFill,
		strokeHex: PETRO_COLORS.siltStroke,
		nphiEndpoint: 0.12,
		rhobEndpoint: 2.68
	},
	{
		mnemonic: 'VFELD',
		mineralName: 'FELDSPAR',
		shortLabel: 'Feldspar',
		fillRgba: 'rgba(251, 207, 232, 0.58)',
		fillHex: PETRO_COLORS.feldFill,
		strokeHex: PETRO_COLORS.feldStroke,
		nphiEndpoint: 0.02,
		rhobEndpoint: 2.56
	},
	{
		mnemonic: 'VCALC',
		mineralName: 'CALCITE',
		shortLabel: 'Calcite',
		fillRgba: 'rgba(56, 189, 248, 0.52)',
		fillHex: PETRO_COLORS.calcFill,
		strokeHex: PETRO_COLORS.calcStroke,
		nphiEndpoint: 0.0,
		rhobEndpoint: 2.71
	},
	{
		mnemonic: 'VDOLO',
		mineralName: 'DOLOMITE',
		shortLabel: 'Dolomite',
		fillRgba: 'rgba(192, 132, 252, 0.52)',
		fillHex: PETRO_COLORS.doloFill,
		strokeHex: PETRO_COLORS.doloStroke,
		nphiEndpoint: 0.01,
		rhobEndpoint: 2.85
	},
	{
		mnemonic: 'VANHY',
		mineralName: 'ANHYDRITE',
		shortLabel: 'Anhydrite',
		fillRgba: 'rgba(244, 114, 182, 0.52)',
		fillHex: PETRO_COLORS.anhyFill,
		strokeHex: PETRO_COLORS.anhyStroke,
		nphiEndpoint: -0.02,
		rhobEndpoint: 2.98
	},
	{
		mnemonic: 'VGYPS',
		mineralName: 'GYPSUM',
		shortLabel: 'Gypsum',
		fillRgba: 'rgba(167, 139, 250, 0.52)',
		fillHex: PETRO_COLORS.gypsFill,
		strokeHex: PETRO_COLORS.gypsStroke,
		nphiEndpoint: 0.6,
		rhobEndpoint: 2.35
	},
	{
		mnemonic: 'VHALI',
		mineralName: 'HALITE',
		shortLabel: 'Halite',
		fillRgba: 'rgba(74, 222, 128, 0.48)',
		fillHex: PETRO_COLORS.haliFill,
		strokeHex: PETRO_COLORS.haliStroke,
		nphiEndpoint: -0.03,
		rhobEndpoint: 2.04
	},
	{
		mnemonic: 'VPYRI',
		mineralName: 'PYRITE',
		shortLabel: 'Pyrite',
		fillRgba: 'rgba(250, 204, 21, 0.62)',
		fillHex: PETRO_COLORS.pyriFill,
		strokeHex: PETRO_COLORS.pyriStroke,
		nphiEndpoint: -0.02,
		rhobEndpoint: 5.0
	},
	{
		mnemonic: 'VCOAL',
		mineralName: 'COAL',
		shortLabel: 'Coal',
		fillRgba: 'rgba(30, 41, 59, 0.72)',
		fillHex: PETRO_COLORS.coalFill,
		strokeHex: PETRO_COLORS.coalStroke,
		nphiEndpoint: 0.5,
		rhobEndpoint: 1.3
	},
	{
		mnemonic: 'VSAND',
		mineralName: 'QUARTZ',
		shortLabel: 'Quartz/Sand',
		fillRgba: 'rgba(253, 224, 71, 0.58)',
		fillHex: PETRO_COLORS.sandFill,
		strokeHex: PETRO_COLORS.sandStroke,
		nphiEndpoint: -0.02,
		rhobEndpoint: 2.64
	}
];

export const MINERAL_TO_CURVE_MNEMONIC: Record<string, string> = {
	QUARTZ: 'VSAND',
	CALCITE: 'VCALC',
	DOLOMITE: 'VDOLO',
	SHALE: 'VCLAY',
	KAOLINITE: 'VKAOL',
	FELDSPAR: 'VFELD',
	ANHYDRITE: 'VANHY',
	GYPSUM: 'VGYPS',
	HALITE: 'VHALI',
	PYRITE: 'VPYRI',
	COAL: 'VCOAL',
	SILT: 'VSILT'
};

export const ALL_LITHOLOGY_CURVES = [
	'VSAND',
	'VSILT',
	'VCLAY',
	'VSHALE',
	'VCALC',
	'VDOLO',
	'VKAOL',
	'VFELD',
	'VANHY',
	'VGYPS',
	'VHALI',
	'VPYRI',
	'VCOAL'
] as const;

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
	RM: {
		mnemonic: 'RM',
		label: 'Medium Resistivity',
		unit: 'ohm.m',
		min: 0.2,
		max: 2000,
		color: PETRO_COLORS.rm,
		logScale: true
	},
	RS: {
		mnemonic: 'RS',
		label: 'Shallow Resistivity',
		unit: 'ohm.m',
		min: 0.2,
		max: 2000,
		color: PETRO_COLORS.rs,
		logScale: true
	},
	RMED: {
		mnemonic: 'RMED',
		label: 'Medium Resistivity',
		unit: 'ohm.m',
		min: 0.2,
		max: 2000,
		color: PETRO_COLORS.rmed,
		logScale: true
	},
	RSHAL: {
		mnemonic: 'RSHAL',
		label: 'Shallow Resistivity',
		unit: 'ohm.m',
		min: 0.2,
		max: 2000,
		color: PETRO_COLORS.rshal,
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
	PEF: { mnemonic: 'PEF', label: 'Photoelectric Factor', unit: 'b/e', min: 0, max: 10, color: PETRO_COLORS.pef },
	VSAND: { mnemonic: 'VSAND', label: 'Quartz/Sand Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.sandStroke },
	VSILT: { mnemonic: 'VSILT', label: 'Silt Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.siltStroke },
	VCLAY: { mnemonic: 'VCLAY', label: 'Clay/Shale Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.shaleStroke },
	VCALC: { mnemonic: 'VCALC', label: 'Calcite Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.calcStroke },
	VDOLO: { mnemonic: 'VDOLO', label: 'Dolomite Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.doloStroke },
	VANHY: { mnemonic: 'VANHY', label: 'Anhydrite Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.anhyStroke },
	VHALI: { mnemonic: 'VHALI', label: 'Halite Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.haliStroke },
	VGYPS: { mnemonic: 'VGYPS', label: 'Gypsum Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.gypsStroke },
	VKAOL: { mnemonic: 'VKAOL', label: 'Kaolinite Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.kaolStroke },
	VFELD: { mnemonic: 'VFELD', label: 'Feldspar Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.feldStroke },
	VPYRI: { mnemonic: 'VPYRI', label: 'Pyrite Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.pyriStroke },
	VCOAL: { mnemonic: 'VCOAL', label: 'Coal Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.coalStroke },
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
	SWE: { mnemonic: 'SWE', label: 'Effective Water Saturation', unit: 'v/v', min: 0, max: 1, color: '#0284C7', reversed: true },
	BS: { mnemonic: 'BS', label: 'Bit Size', unit: 'in', min: 6, max: 16, color: '#64748B' },
	DCALI: { mnemonic: 'DCALI', label: 'Differential Caliper', unit: 'in', min: -2, max: 8, color: '#DC2626' },
	DRHO: { mnemonic: 'DRHO', label: 'Density Correction', unit: 'g/cc', min: -0.25, max: 0.25, color: '#D97706' },
	BADHOLE: { mnemonic: 'BADHOLE', label: 'Badhole Flag', unit: 'flag', min: 0, max: 1, color: '#DC2626' },
	VSH_LAM: { mnemonic: 'VSH_LAM', label: 'Laminated Shale Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.vshLamStroke },
	VSH_DIS: { mnemonic: 'VSH_DIS', label: 'Dispersed Shale Volume', unit: 'v/v', min: 0, max: 1, color: PETRO_COLORS.vshDisStroke },
	PHIT_SAND: { mnemonic: 'PHIT_SAND', label: 'Sand Lamina Porosity', unit: 'v/v', min: 0, max: 0.45, color: PETRO_COLORS.phitSand },
	RT_MOD: { mnemonic: 'RT_MOD', label: 'Hagiwara Modeled Resistivity', unit: 'ohm.m', min: 0.2, max: 2000, color: PETRO_COLORS.rtMod, logScale: true }
};

export function getCurveUnit(mnemonic: string): string {
	const upper = mnemonic.toUpperCase();
	if (STANDARD_CURVE_META[upper]) return STANDARD_CURVE_META[upper].unit;
	if (upper.includes('CALI') || upper === 'BS' || upper === 'BITSIZE') return 'in';
	if (upper === 'BADHOLE' || upper === 'BAD_HOLE' || upper.endsWith('_FLAG')) return 'flag';
	if (upper.includes('NPHI') || upper.includes('PHI') || upper.startsWith('V') || upper.startsWith('SW') || upper === 'BVW') {
		return 'v/v';
	}
	if (upper.includes('RHOB') || upper.includes('DEN') || upper === 'DRHO') return 'g/cc';
	if (upper.includes('PERM')) return 'mD';
	if (upper.startsWith('R') && upper.length <= 5) return 'ohm.m';
	if (upper.includes('DEPTH') || upper.includes('DEPT') || upper === 'TVD') return 'm';
	return '';
}

export function getRockTypeColor(rockFlag: number | string | null | undefined): string {
	const idx = Math.max(0, (Number(rockFlag) || 1) - 1);
	return ROCK_TYPE_PALETTE[idx % ROCK_TYPE_PALETTE.length];
}

