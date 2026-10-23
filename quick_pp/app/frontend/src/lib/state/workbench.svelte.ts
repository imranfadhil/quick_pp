export type SingleWellStep =
	| 'qaqc'
	| 'ancillary'
	| 'litho-poro'
	| 'perm'
	| 'saturation'
	| 'ressum'
	| 'scal';
export type MultiWellStep =
	| 'inventory'
	| 'qaqc'
	| 'rock-typing'
	| 'perm-transform'
	| 'shf'
	| 'ml-training';

export type BadholeTreatmentMode = 'none' | 'vsh_gr' | 'interpolate' | 'ignore';

export interface BadholeFlaggingParams {
	useCali: boolean;
	caliThreshold: number;
	autoBitsize: boolean;
	bitsizeDefault: number;
	useDrho: boolean;
	drhoThreshold: number;
	useRhobNphi: boolean;
	rhobMin: number;
	rhobMax: number;
	nphiMax: number;
	excludeCoal: boolean;
	useRugosity: boolean;
	rugosityWindow: number;
	rhobTrendFactor: number;
	caliSpikeThreshold: number;
}

export interface LithoEndpointParams {
	model: 'ssc' | 'multi_mineral' | 'vsh_gr';
	porosityMethod: 'neu_den' | 'den';
	useHcCorrected: boolean;
	enableHcCorrection: boolean;
	rhoWater: number;
	rhoHc: number;
	hiHc: number;
	hcBuffer: number;
	drySandNphi: number;
	drySandRhob: number;
	drySiltNphi: number;
	drySiltRhob: number;
	dryClayNphi: number;
	dryClayRhob: number;
	fluidNphi: number;
	fluidRhob: number;
	siltLineAngle: number;
	grClean: number;
	grClay: number;
	minerals: string[];
	enableCoalFlagging: boolean;
	coalRhobMax: number;
	coalNphiMin: number;
	coalWindowSize: number;
	coalTrendFactor: number;
	badholeTreatment: BadholeTreatmentMode;
}

export interface ThomasStieberParams {
	enabled: boolean;
	useInPerm: boolean;
	useInSaturation: boolean;
	drySandPoro: number;
	dryShalePoro: number;
	rsand: number;
	rvShale: number;
	rhShale: number;
	theta: number;
}

export interface SaturationModelParams {
	method: 'waxman_smits' | 'archie';
	rwMode: 'salinity_temp' | 'pickett';
	measSystem: 'metric' | 'imperial';
	waterSalinity: number;
	rwPickett: number;
	a: number;
	m: number;
	n: number;
	tempGradient: number;
	surfaceTemp: number;
	bMethod: string;
	qvMethod: string;
	cecClay: number;
	rhoDryClay: number;
}

export interface ScalModelParams {
	model: 'let' | 'corey' | 'brooks_corey';
	swirr: number;
	sor: number;
	krw_max: number;
	kro_max: number;
	nw: number;
	no: number;
	lam: number;
	lw: number;
	ew: number;
	tw: number;
	lo: number;
	eo: number;
	to: number;
	mu_w: number;
	mu_o: number;
	pc_entry: number;
	pc_exponent: number;
	pc_max: number;
	n_points: number;
}

export interface ScalComputedResult {
	sw: number[];
	se: number[];
	krw: number[];
	kro: number[];
	fw: number[];
	swofTable: Array<{ SW: number; KRW: number; KROW: number; PCOW: number }>;
	swofKeyword: string;
}

class WorkbenchState {
	projectId = $state<string | null>(null);
	wellId = $state<string | null>(null);
	scope = $state<'single-well' | 'multi-well'>('single-well');
	singleWellStep = $state<SingleWellStep>('qaqc');
	multiWellStep = $state<MultiWellStep>('inventory');

	// Global depth & zone filter
	depthFilter = $state<{ min: number; max: number } | null>(null);
	selectedZones = $state<string[]>([]);

	// Synchronized cursor & crossplot lasso brushing
	cursorDepth = $state<number | null>(null);
	brushedDepthIndices = $state<Set<number>>(new Set());
	targetScrollDepth = $state<number | null>(null);

	// Right pane view mode
	rightPaneTab = $state<'crossplot' | 'grid' | 'split'>('split');
	crossplotTemplateOverride = $state<
		| 'auto'
		| 'nphi-rhob'
		| 'thomas-stieber'
		| 'phi-k'
		| 'pickett'
		| 'histogram'
		| 'mw-distribution'
		| 'shf'
		| 'relperm'
	>('auto');

	// Active curve selected in QAQC for histogram / multi-well distribution inspection
	selectedQaqcCurve = $state<string>('GR');

	// Multi-well QAQC distribution & visualization controls
	mwQaqcGroupBy = $state<'well' | 'zone' | 'well_zone'>('well');
	mwQaqcPlotType = $state<'box' | 'violin' | 'histogram' | 'cdf' | 'nphi-rhob'>('box');
	mwQaqcCompareMode = $state<'none' | 'norm' | 'corr'>('norm');
	mwQaqcExcludeBadhole = $state<boolean>(false);
	mwQaqcLogScale = $state<boolean>(false);

	// Global modal / drawer flags
	importerOpen = $state<boolean>(false);
	aiDrawerOpen = $state<boolean>(false);

	// Single-well left pane section collapse / expand / focus controls
	sectionFocusMode = $state<boolean>(false);
	focusedSectionTitle = $state<string | null>(null);
	sectionBulkAction = $state<{
		type: 'expand-all' | 'collapse-all' | 'focus';
		seq: number;
	}>({ type: 'collapse-all', seq: 0 });

	// Shared interactive parameters between Left Pane Inspectors & Right Pane Crossplot
	lithoParams = $state<LithoEndpointParams>({
		model: 'ssc',
		porosityMethod: 'neu_den',
		useHcCorrected: true,
		enableHcCorrection: true,
		rhoWater: 1.0,
		rhoHc: 0.8,
		hiHc: 0.9,
		hcBuffer: 0.0,
		drySandNphi: -0.02,
		drySandRhob: 2.65,
		drySiltNphi: 0.12,
		drySiltRhob: 2.68,
		dryClayNphi: 0.33,
		dryClayRhob: 2.7,
		fluidNphi: 1.0,
		fluidRhob: 1.0,
		siltLineAngle: 119.0,
		grClean: 15.0,
		grClay: 115.0,
		minerals: ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE'],
		enableCoalFlagging: true,
		coalRhobMax: 2.0,
		coalNphiMin: 0.30,
		coalWindowSize: 41,
		coalTrendFactor: 0.05,
		badholeTreatment: 'vsh_gr'
	});

	tsParams = $state<ThomasStieberParams>({
		enabled: false,
		useInPerm: true,
		useInSaturation: true,
		drySandPoro: 0.28,
		dryShalePoro: 0.10,
		rsand: 20.0,
		rvShale: 4.0,
		rhShale: 1.5,
		theta: 0.0
	});

	badholeParams = $state<BadholeFlaggingParams>({
		useCali: true,
		caliThreshold: 2.0,
		autoBitsize: true,
		bitsizeDefault: 8.5,
		useDrho: true,
		drhoThreshold: 0.15,
		useRhobNphi: false,
		rhobMin: 1.95,
		rhobMax: 2.95,
		nphiMax: 0.45,
		excludeCoal: true,
		useRugosity: false,
		rugosityWindow: 21,
		rhobTrendFactor: 0.05,
		caliSpikeThreshold: 0.5
	});

	saturationParams = $state<SaturationModelParams>({
		method: 'waxman_smits',
		rwMode: 'salinity_temp',
		measSystem: 'metric',
		waterSalinity: 35000,
		rwPickett: 0.08,
		a: 1.0,
		m: 2.0,
		n: 2.0,
		tempGradient: 30.0,
		surfaceTemp: 25.0,
		bMethod: 'waxman_smits',
		qvMethod: 'cec',
		cecClay: 0.1,
		rhoDryClay: 2.71
	});

	scalParams = $state<ScalModelParams>({
		model: 'let',
		swirr: 0.15,
		sor: 0.2,
		krw_max: 0.45,
		kro_max: 0.9,
		nw: 2.5,
		no: 2.5,
		lam: 2.0,
		lw: 2.0,
		ew: 1.5,
		tw: 2.0,
		lo: 2.0,
		eo: 1.5,
		to: 2.0,
		mu_w: 0.5,
		mu_o: 1.5,
		pc_entry: 1.5,
		pc_exponent: 2.0,
		pc_max: 50.0,
		n_points: 25
	});

	scalResult = $state<ScalComputedResult | null>(null);

	clearFilters() {
		this.depthFilter = null;
		this.selectedZones = [];
		this.brushedDepthIndices = new Set();
	}

	setBrushedIndices(indices: number[]) {
		this.brushedDepthIndices = new Set(indices);
	}

	scrollToDepth(depth: number) {
		this.targetScrollDepth = depth;
		this.cursorDepth = depth;
	}

	expandAllSections() {
		this.sectionFocusMode = false;
		this.sectionBulkAction = {
			type: 'expand-all',
			seq: this.sectionBulkAction.seq + 1
		};
	}

	collapseAllSections() {
		this.focusedSectionTitle = null;
		this.sectionBulkAction = {
			type: 'collapse-all',
			seq: this.sectionBulkAction.seq + 1
		};
	}

	toggleFocusMode() {
		this.sectionFocusMode = !this.sectionFocusMode;
		if (this.sectionFocusMode) {
			this.sectionBulkAction = {
				type: 'focus',
				seq: this.sectionBulkAction.seq + 1
			};
		}
	}

	triggerSectionFocus(title: string) {
		this.focusedSectionTitle = title;
		this.sectionBulkAction = {
			type: 'focus',
			seq: this.sectionBulkAction.seq + 1
		};
	}
}

export const workbench = new WorkbenchState();
