export type SingleWellStep = 'qaqc' | 'ancillary' | 'litho-poro' | 'perm' | 'saturation' | 'ressum';
export type MultiWellStep = 'inventory' | 'rock-typing' | 'perm-transform' | 'shf';

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
}

export interface SaturationModelParams {
	method: 'waxman_smits' | 'archie';
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

class WorkbenchState {
	projectId = $state<string | null>(null);
	wellId = $state<string | null>(null);
	scope = $state<'single-well' | 'multi-well'>('single-well');
	singleWellStep = $state<SingleWellStep>('litho-poro');
	multiWellStep = $state<MultiWellStep>('rock-typing');

	// Global depth & zone filter
	depthFilter = $state<{ min: number; max: number } | null>(null);
	selectedZones = $state<string[]>([]);

	// Synchronized cursor & crossplot lasso brushing
	cursorDepth = $state<number | null>(null);
	brushedDepthIndices = $state<Set<number>>(new Set());
	targetScrollDepth = $state<number | null>(null);

	// Right pane view mode
	rightPaneTab = $state<'crossplot' | 'grid' | 'split'>('split');

	// Active curve selected in QAQC for histogram inspection
	selectedQaqcCurve = $state<string>('GR');

	// Global modal / drawer flags
	importerOpen = $state<boolean>(false);
	aiDrawerOpen = $state<boolean>(false);

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
		minerals: ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE']
	});

	saturationParams = $state<SaturationModelParams>({
		method: 'waxman_smits',
		measSystem: 'metric',
		waterSalinity: 35000,
		rwPickett: 0.08,
		a: 1.0,
		m: 2.0,
		n: 2.0,
		tempGradient: 30.0,
		surfaceTemp: 25.0,
		bMethod: 'waxman_smits',
		qvMethod: 'hill',
		cecClay: 0.1,
		rhoDryClay: 2.71
	});

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
}

export const workbench = new WorkbenchState();

