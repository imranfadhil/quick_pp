export type SingleWellStep =
	| 'qaqc'
	| 'ancillary'
	| 'litho-poro'
	| 'perm'
	| 'saturation'
	| 'ressum'
	| 'scal'
	| 'report';
export type MultiWellStep =
	| 'inventory'
	| 'qaqc'
	| 'rock-typing'
	| 'perm-transform'
	| 'swirr-corr'
	| 'shf'
	| 'rt-classifier'
	| 'ml-training'
	| 'fpt'
	| 'volumetrics'
	| 'report';

export type ReportType = 'FDP_FULL' | 'SINGLE_WELL_CPI' | 'QAQC_SUMMARY' | 'RESSUM_ONLY';

export interface ReportSectionConfig {
	section_id: string;
	title: string;
	enabled: boolean;
	include_plots: boolean;
	include_tables: boolean;
	custom_notes: string;
}

export interface ReportConfigParams {
	reportType: ReportType;
	reportTitle: string;
	author: string;
	company: string;
	fieldName: string;
	includeAiSummary: boolean;
	cutoffs: { vsh: number; phie: number; sw: number };
	sections: ReportSectionConfig[];
}

export interface ActiveReportState {
	report_id: string;
	project_id: string;
	project_name?: string;
	title: string;
	author: string;
	report_type: string;
	well_names: string[];
	sections: Array<{
		section_id: string;
		title: string;
		enabled: boolean;
		include_ai_commentary?: boolean;
		custom_notes?: string | null;
	}>;
	section_contents: Record<string, string>;
	markdown_content: string;
	html_content: string;
	asset_urls: Record<string, any>;
	pdf_download_url?: string | null;
	print_html_url?: string | null;
	context_summary: Record<string, any>;
	created_at: string;
	updated_at?: string | null;
}

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
	logCorrectionMode: 'none' | 'hc' | 'den';
	useHcCorrected: boolean;
	enableHcCorrection: boolean;
	rhoWater: number;
	rhoHc: number;
	hiHc: number;
	hcBuffer: number;
	hcFlagMode: 'strict' | 'vsh_gr';
	hcVshTol: number;
	denCorrScope: 'full' | 'badhole_only' | 'vsh_mismatch';
	denCorrSyncEndpoints: boolean;
	denCorrRhoMa: number;
	denCorrPhinSh: number;
	denCorrPhidSh: number;
	denCorrMismatchThreshold: number;
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
	fwl: number;
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
	// Core Overburden & Klinkenberg Correction Parameters
	core_lithology: 'sandstone' | 'friable_sandstone' | 'carbonate';
	net_confining_stress: number;
	mean_pore_pressure: number;
	compaction_coeff: number;
	perm_compaction_coeff: number;
	klinkenberg_method: 'jones' | 'heithaus';
	core_corrected: boolean;
	core_poro_amb_mean: number | null;
	core_poro_res_mean: number | null;
	core_perm_air_mean: number | null;
	core_perm_liq_mean: number | null;
	// Lab-to-Reservoir Pc Scaling Parameters
	pc_lab_input: number;
	ift_lab: number;
	theta_lab: number;
	ift_res: number;
	theta_res: number;
	pc_res_output: number | null;
	pc_scale_factor: number | null;
	pc_scaling_applied: boolean;
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

export interface EnvCorrectionParams {
	bitsizeDefault: number;
	mudWeight: number;
	surfaceTemp: number;
	tempGrad: number;
	salinityPpm: number;
}

export interface PermModelParams {
	method: 'choo' | 'timur' | 'tixier' | 'coates' | 'kozeny_carman';
	m: number;
	A: number;
	B: number;
	C: number;
	s: number;
	swirr: number;
}

export interface ResSumParams {
	vshMax: number;
	phitMin: number;
	swtMax: number;
	includeUpscaling: boolean;
	byZone: boolean;
	cellSize: number;
	permMethod: 'geometric' | 'arithmetic' | 'harmonic' | 'power_law';
	omega: number;
	includeUncertainty: boolean;
	nSamples: number;
	uncMetric: 'stoiip' | 'net' | 'phit' | 'swt' | 'bvhc';
	areaMean: number;
	areaStd: number;
	thickMean: number;
	thickStd: number;
	ntgMean: number;
	poroMean: number;
	swMean: number;
	boP50: number;
}

export interface VolumetricPercentiles {
	p90: number;
	p50: number;
	p10: number;
	mean: number;
	std: number;
}

export interface TornadoItem {
	param: string;
	label: string;
	unit: string;
	low_param?: number;
	high_param?: number;
	vol_at_low: number;
	vol_at_high: number;
	base_vol: number;
	swing: number;
}

export interface ExceedancePoint {
	volume: number;
	prob: number;
}

export interface HistBin {
	bin_start: number;
	bin_end: number;
	count: number;
}

export interface ShfProfilePoint {
	depth_tvd: number;
	height_above_fwl: number;
	sw_shf: number;
	pc_psi: number;
	area_fraction: number;
}

export interface ShfComparison {
	log_mean_sw: number;
	shf_p50_sw: number;
	delta_sw: number;
	delta_volume_pct: number;
	model: string;
	parameters: {
		a: number;
		b: number;
		swirr: number;
		fwl_p50: number;
		crest: number;
	};
}

export interface InputDistSummary {
	param: string;
	label: string;
	unit: string;
	dist_type: string;
	n_samples?: number;
	mean: number;
	std: number;
	min: number;
	max: number;
	p90: number;
	p50: number;
	p10: number;
	histogram: HistBin[];
}

export interface AssumptionsRegister {
	evaluation_standard: string;
	reservoir_type_applied: string;
	contact_source_applied: string;
	fluid_contacts: {
		goc_tvdss: number | null;
		owc_tvdss: number | null;
		gwc_tvdss: number | null;
		source: string;
	};
	cut_offs: {
		vsh_max: number;
		phit_min: number;
		swt_max: number;
	};
	monte_carlo_engine: {
		method: string;
		realizations: number;
		random_seed: number;
	};
	parameter_assumptions: Array<{
		parameter: string;
		distribution: string;
		justification: string;
	}>;
}

export interface ZonalVolumetricSummary {
	zone: string;
	well_count: number;
	mean_gross: number;
	mean_net_res: number;
	ntg_res: number;
	mean_net_pay: number;
	ntg_pay: number;
	mean_phie: number;
	mean_sw: number;
	mean_perm_geom: number;
	mean_hcpt: number;
	mode: 'area' | 'grv';
	area_mean?: number | null;
	grv_mean?: number | null;
	stoiip: VolumetricPercentiles;
	rec_oil: VolumetricPercentiles;
	giip: VolumetricPercentiles;
	rec_gas: VolumetricPercentiles;
	assoc_gas: VolumetricPercentiles;
	exceedance_curve: ExceedancePoint[];
	histogram: HistBin[];
	tornado: TornadoItem[];
	input_distributions?: Record<string, InputDistSummary>;
	assumptions_register?: AssumptionsRegister;
	contacts?: {
		goc: number | null;
		owc: number | null;
		gwc: number | null;
		reservoir_type: string;
		source: string;
	};
	shf_profile?: ShfProfilePoint[];
	shf_comparison?: ShfComparison | null;
	saturation_method?: 'log_average' | 'shf';
}

export interface FieldVolumetricTotals {
	total_wells: number;
	total_zones: number;
	stoiip: VolumetricPercentiles;
	rec_oil: VolumetricPercentiles;
	giip: VolumetricPercentiles;
	rec_gas: VolumetricPercentiles;
	assoc_gas: VolumetricPercentiles;
	total_gas: VolumetricPercentiles;
	exceedance_curve: ExceedancePoint[];
	histogram: HistBin[];
	tornado: TornadoItem[];
	input_distributions?: Record<string, InputDistSummary>;
	assumptions_register?: AssumptionsRegister;
	shf_profile?: ShfProfilePoint[];
	shf_comparison?: ShfComparison | null;
	saturation_method?: 'log_average' | 'shf';
}

export interface MultiWellVolumetricResult {
	project_id: number;
	mode: 'area' | 'grv';
	cut_offs: Record<string, number>;
	n_samples: number;
	well_summary: Array<{
		well_name: string;
		zone: string;
		top: number;
		bottom: number;
		gross: number;
		net_res: number;
		ntg_res: number;
		net_pay: number;
		ntg_pay: number;
		phie: number;
		sw: number;
		perm_geom: number;
		hcpt: number;
	}>;
	zonal_summary: ZonalVolumetricSummary[];
	field_totals: FieldVolumetricTotals;
	contacts?: {
		goc: number | null;
		owc: number | null;
		gwc: number | null;
		source: string;
	};
	assumptions_register?: AssumptionsRegister;
}

export interface VolumetricZoneParamConfig {
	areaMean: number;
	areaStd: number;
	grvMean: number;
	grvStd: number;
	thickMean: number;
	thickStd: number;
	ntgMean: number;
	ntgStd: number;
	poroMean: number;
	swMean: number;
	boP50: number;
	bgP50: number;
	rs: number;
	rfOil: number;
	rfGas: number;
}

export const DEFAULT_VOLUMETRIC_ZONE_CONFIG: VolumetricZoneParamConfig = {
	areaMean: 1200.0,
	areaStd: 300.0,
	grvMean: 50.0,
	grvStd: 15.0,
	thickMean: 25.0,
	thickStd: 5.0,
	ntgMean: 0.65,
	ntgStd: 0.10,
	poroMean: 0.18,
	swMean: 0.35,
	boP50: 1.25,
	bgP50: 0.005,
	rs: 500.0,
	rfOil: 0.30,
	rfGas: 0.70
};

export interface MultiWellVolumetricsState {
	mode: 'area' | 'grv';
	computeOil: boolean;
	computeGas: boolean;
	contactSource: 'inherit' | 'custom';
	reservoirType: 'auto' | 'oil_only' | 'gas_only' | 'oil_and_gas';
	owc: number;
	goc: number;
	gwc: number;
	contactsByZone: Record<string, { owc?: number | null; goc?: number | null; gwc?: number | null }>;
	saturationMethod: 'log_average' | 'shf';
	fwlUncertainty: number;
	crestDepth: number | null;
	hypsometryModel: 'structural_dome' | 'well_tracks';
	shfA: number;
	shfB: number;
	shfSwirr: number;
	vshMax: number;
	phitMin: number;
	swtMax: number;
	nSamples: number;
	seed: number;
	activeTab: 'exceedance' | 'inputs' | 'shf_profile' | 'assumptions' | 'tornado' | 'summary' | 'well_matrix';
	rightTab: 'crossplot' | 'tornado' | 'summary' | 'well_matrix';
	selectedZone: string;
	fluidMetric: 'oil' | 'gas';
	zoneConfigs: Record<string, VolumetricZoneParamConfig>;
}

/**
 * Calculates the intersection point between:
 * 1. The silt line emanating from the water/fluid point (fluidNphi, fluidRhob) at siltLineAngle (in degrees).
 * 2. The rock line connecting the dry sand point (sandNphi, sandRhob) and dry clay point (clayNphi, clayRhob).
 */
export function computeSiltIntersection(
	sand: [number, number],
	clay: [number, number],
	fluid: [number, number],
	siltLineAngleDeg: number
): [number, number] {
	const rad = (siltLineAngleDeg * Math.PI) / 180;
	const pSilt: [number, number] = [fluid[0] + Math.cos(rad), fluid[1] + Math.sin(rad)];

	const xdiff0 = sand[0] - clay[0];
	const ydiff0 = sand[1] - clay[1];
	const d0 = sand[0] * clay[1] - sand[1] * clay[0];

	const xdiff1 = fluid[0] - pSilt[0];
	const ydiff1 = fluid[1] - pSilt[1];
	const d1 = fluid[0] * pSilt[1] - fluid[1] * pSilt[0];

	const div = xdiff0 * ydiff1 - xdiff1 * ydiff0;
	if (Math.abs(div) < 1e-12) {
		return [(sand[0] + clay[0]) / 2, (sand[1] + clay[1]) / 2];
	}

	const varX = (d0 * xdiff1 - d1 * xdiff0) / div;
	const varY = (d0 * ydiff1 - d1 * ydiff0) / div;
	return [varX, varY];
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
		| 'swirr-fit'
		| 'shf'
		| 'shf'
		| 'relperm'
		| 'core-qc'
		| 'vsh-qc'
	>('auto');

	// Active curve selected in QAQC for histogram / multi-well distribution inspection
	selectedQaqcCurve = $state<string>('GR');
	scalSubMode = $state<'static' | 'relperm'>('static');
	swirrCrossplotTab = $state<'swirr-corr' | 'pc-qc' | 'scal-fit' | 'relperm'>('swirr-corr');
	coreQcView = $state<'dual' | 'poro' | 'perm' | 'depth-error'>('dual');
	vshQcView = $state<'dual' | 'gr-fit' | 'depth-error'>('dual');
	vshCalibScope = $state<'shale_sand' | 'shale_only' | 'all' | 'brushed'>('shale_sand');
	vshShaleCutoff = $state<number>(0.6);
	vshSandCutoff = $state<number>(0.2);

	// Multi-well QAQC distribution & visualization controls
	mwQaqcGroupBy = $state<'well' | 'zone' | 'well_zone'>('well');
	mwQaqcPlotType = $state<'box' | 'violin' | 'histogram' | 'cdf' | 'nphi-rhob'>('box');
	mwQaqcCompareMode = $state<'none' | 'norm' | 'corr'>('norm');
	mwQaqcExcludeBadhole = $state<boolean>(false);
	mwQaqcLogScale = $state<boolean>(false);

	// Multi-well Center Pane view ('logs' | 'stick-plot') & Stick Plot controls
	centerPaneMultiWellView = $state<'logs' | 'stick-plot'>('logs');
	stickPlotSortMode = $state<'project' | 'crest-tvd' | 'chronological' | 'custom'>('crest-tvd');
	stickPlotCustomOrder = $state<string[]>([]);
	stickPlotZoneFilter = $state<string>('ALL');
	stickPlotShowPhit = $state<boolean>(false);
	stickPlotShowContacts = $state<boolean>(true);
	stickPlotShowTops = $state<boolean>(true);
	stickPlotShowPerfs = $state<boolean>(true);
	logShowContacts = $state<boolean>(true);

	// Global modal / drawer flags
	importerOpen = $state<boolean>(false);
	importerTab = $state<
		| 'las'
		| 'formation_tops'
		| 'fluid_contacts'
		| 'perforations'
		| 'core_samples'
		| 'scal_pc'
		| 'pressure_tests'
		| 'well_surveys'
		| 'handover'
	>('las');
	aiDrawerOpen = $state<boolean>(false);
	workflowGuideOpen = $state<boolean>(false);
	workflowGuideScope = $state<'single-well' | 'multi-well'>('single-well');
	workflowGuideTargetAnchor = $state<string | null>(null);

	openWorkflowGuide(scope?: 'single-well' | 'multi-well', anchor?: string) {
		this.workflowGuideScope = scope ?? this.scope;
		this.workflowGuideTargetAnchor = anchor ?? null;
		this.workflowGuideOpen = true;
	}

	closeWorkflowGuide() {
		this.workflowGuideOpen = false;
		this.workflowGuideTargetAnchor = null;
	}

	openImporter(
		tab?:
			| 'las'
			| 'formation_tops'
			| 'fluid_contacts'
			| 'perforations'
			| 'core_samples'
			| 'scal_pc'
			| 'pressure_tests'
			| 'well_surveys'
			| 'handover'
	) {
		if (tab) {
			this.importerTab = tab;
		}
		this.importerOpen = true;
	}

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
		porosityMethod: 'den',
		logCorrectionMode: 'none',
		useHcCorrected: false,
		enableHcCorrection: false,
		rhoWater: 1.0,
		rhoHc: 0.8,
		hiHc: 0.9,
		hcBuffer: 0.0,
		hcFlagMode: 'strict',
		hcVshTol: 0.05,
		denCorrScope: 'vsh_mismatch',
		denCorrSyncEndpoints: true,
		denCorrRhoMa: 2.65,
		denCorrPhinSh: 0.33,
		denCorrPhidSh: 0.05,
		denCorrMismatchThreshold: 0.15,
		drySandNphi: -0.02,
		drySandRhob: 2.65,
		drySiltNphi: 0.078,
		drySiltRhob: 2.664,
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
		coalNphiMin: 0.3,
		coalWindowSize: 41,
		coalTrendFactor: 0.05,
		badholeTreatment: 'vsh_gr'
	});

	tsParams = $state<ThomasStieberParams>({
		enabled: false,
		useInPerm: true,
		useInSaturation: true,
		drySandPoro: 0.32,
		dryShalePoro: 0.1,
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
		rhoDryClay: 2.71,
		fwl: 0
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
		n_points: 25,
		core_lithology: 'sandstone',
		net_confining_stress: 3500,
		mean_pore_pressure: 45,
		compaction_coeff: 6e-6,
		perm_compaction_coeff: 3e-5,
		klinkenberg_method: 'jones',
		core_corrected: false,
		core_poro_amb_mean: null,
		core_poro_res_mean: null,
		core_perm_air_mean: null,
		core_perm_liq_mean: null,
		pc_lab_input: 25.0,
		ift_lab: 480.0,
		theta_lab: 140.0,
		ift_res: 30.0,
		theta_res: 30.0,
		pc_res_output: null,
		pc_scale_factor: null,
		pc_scaling_applied: false
	});

	scalResult = $state<ScalComputedResult | null>(null);

	envCorrParams = $state<EnvCorrectionParams>({
		bitsizeDefault: 8.5,
		mudWeight: 9.5,
		surfaceTemp: 75.0,
		tempGrad: 1.5,
		salinityPpm: 30000.0
	});

	permParams = $state<PermModelParams>({
		method: 'choo',
		m: 2.0,
		A: 1000000,
		B: 3.0,
		C: 2.0,
		s: 0.01,
		swirr: 0.2
	});

	resSumParams = $state<ResSumParams>({
		vshMax: 0.4,
		phitMin: 0.05,
		swtMax: 0.75,
		includeUpscaling: true,
		byZone: false,
		cellSize: 2.0,
		permMethod: 'geometric',
		omega: 0.33,
		includeUncertainty: false,
		nSamples: 300,
		uncMetric: 'stoiip',
		areaMean: 640.0,
		areaStd: 80.0,
		thickMean: 25.0,
		thickStd: 4.0,
		ntgMean: 0.65,
		poroMean: 0.18,
		swMean: 0.28,
		boP50: 1.22
	});

	multiWellVolumetrics = $state<MultiWellVolumetricsState>({
		mode: 'area',
		computeOil: true,
		computeGas: true,
		contactSource: 'inherit',
		reservoirType: 'auto',
		owc: -2500,
		goc: -2400,
		gwc: -2500,
		contactsByZone: {},
		saturationMethod: 'log_average',
		fwlUncertainty: 5.0,
		crestDepth: null,
		hypsometryModel: 'structural_dome',
		shfA: 0.35,
		shfB: -0.45,
		shfSwirr: 0.12,
		vshMax: 0.4,
		phitMin: 0.05,
		swtMax: 0.8,
		nSamples: 5000,
		seed: 42,
		activeTab: 'exceedance',
		rightTab: 'crossplot',
		selectedZone: 'ALL',
		fluidMetric: 'oil',
		zoneConfigs: {
			ALL: { ...DEFAULT_VOLUMETRIC_ZONE_CONFIG }
		}
	});

	// FDP Petrophysical Report Generation Workbench State
	reportConfig = $state<ReportConfigParams>({
		reportType: 'FDP_FULL',
		reportTitle: 'Field Development Plan (FDP) Petrophysical Evaluation Report',
		author: 'Petrophysics Team',
		company: 'Subsurface Asset Team',
		fieldName: '',
		includeAiSummary: true,
		cutoffs: { vsh: 0.4, phie: 0.08, sw: 0.65 },
		sections: [
			{
				section_id: 'exec_summary',
				title: '1. Executive Summary',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'data_inventory',
				title: '2. Well & Data Inventory',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'qaqc',
				title: '3. Log QAQC & Conditioning',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'litho_poro',
				title: '4. Lithology & Porosity Evaluation',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'rock_perm',
				title: '5. Rock Typing & Permeability Modeling',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'saturation',
				title: '6. Water Saturation & Fluid Contacts',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'shf_scal',
				title: '7. Saturation Height Function (SHF) & SCAL',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'ressum_uncert',
				title: '8. Reservoir Summary (Net Pay) & Uncertainty',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'upscaling',
				title: '9. Geocellular Model Handover & Upscaling',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'appendix_cpi',
				title: '10. Appendices (Well CPI Plots & Audit Trail)',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			}
		]
	});

	// Single-Well CPI Dossier Configuration (streamlined 6 chapters)
	singleWellReportConfig = $state<ReportConfigParams>({
		reportType: 'SINGLE_WELL_CPI',
		reportTitle: 'Single-Well Petrophysical CPI & Evaluation Dossier',
		author: 'Petrophysics Team',
		company: 'Subsurface Asset Team',
		fieldName: '',
		includeAiSummary: true,
		cutoffs: { vsh: 0.4, phie: 0.08, sw: 0.65 },
		sections: [
			{
				section_id: 'exec_summary',
				title: '1. Well Evaluation Summary',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'qaqc',
				title: '2. Borehole QA/QC & Log Conditioning',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'litho_poro',
				title: '3. Lithology, Porosity & Core Parity',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'saturation',
				title: '4. Water Saturation & Fluid Contacts',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'ressum_uncert',
				title: '5. Zonal Net Pay Summary & Cutoffs',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			},
			{
				section_id: 'appendix_cpi',
				title: '6. 6-Track Composite CPI Log',
				enabled: true,
				include_plots: true,
				include_tables: true,
				custom_notes: ''
			}
		]
	});

	singleWellReport = $state<ActiveReportState | null>(null);
	multiWellReport = $state<ActiveReportState | null>(null);

	get activeReport(): ActiveReportState | null {
		return this.scope === 'single-well' ? this.singleWellReport : this.multiWellReport;
	}

	set activeReport(val: ActiveReportState | null) {
		if (this.scope === 'single-well') {
			this.singleWellReport = val;
		} else {
			this.multiWellReport = val;
		}
	}

	reportViewMode = $state<'preview' | 'markdown' | 'split'>('preview');
	selectedReportSectionId = $state<string>('exec_summary');
	isGeneratingReport = $state<boolean>(false);

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

	updateSiltEndpointFromAngle() {
		const [siltNphi, siltRhob] = computeSiltIntersection(
			[this.lithoParams.drySandNphi, this.lithoParams.drySandRhob],
			[this.lithoParams.dryClayNphi, this.lithoParams.dryClayRhob],
			[this.lithoParams.fluidNphi, this.lithoParams.fluidRhob],
			this.lithoParams.siltLineAngle
		);
		this.lithoParams.drySiltNphi = Number(siltNphi.toFixed(3));
		this.lithoParams.drySiltRhob = Number(siltRhob.toFixed(3));
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
