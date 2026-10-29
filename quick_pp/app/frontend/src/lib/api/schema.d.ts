/**
 * Generated OpenAPI TypeScript definitions for quick_pp FastAPI backend.
 * Can be regenerated via `npm run gen:api`.
 */

export interface WellSummaryItem {
	well_id: number;
	name: string;
	uwi: string;
	depth_uom?: string;
	df_elev?: number | null;
	min_depth: number | null;
	max_depth: number | null;
	depth_count: number;
	curves: string[];
	curve_count: number;
	zones: string[];
	formation_tops_count: number;
	fluid_contacts_count: number;
	pressure_tests_count: number;
	core_samples_count: number;
	core_poro_count: number;
	core_perm_count: number;
	total_depth?: number;
	row_count?: number;
	available_curves?: string[];
	formation_tops?: Array<{
		name: string;
		depth: number | null;
		base_depth?: number | null;
		curves?: string[];
		computed_curves?: string[];
	}>;
	zone_curves?: Record<string, string[]>;
	formation_top_count?: number;
	fluid_contacts?: Array<{ type: string; depth: number | null }>;
	fluid_contact_count?: number;
	pressure_test_count?: number;
	survey_count?: number;
	core_sample_count?: number;
	core_availability_pct?: number;
	scal_pc_count?: number;
	scal_sample_count?: number;
	relperm_count?: number;
	relperm_sample_count?: number;
}

export interface ProjectSummaryResponse {
	project_id: number;
	name: string;
	description?: string;
	well_count: number;
	all_curves: string[];
	all_zones: string[];
	min_depth: number | null;
	max_depth: number | null;
	wells: WellSummaryItem[];
}

export interface CurveSummaryStats {
	count: number;
	top?: number | null;
	bottom?: number | null;
	min: number | null;
	max: number | null;
	mean: number | null;
	std?: number | null;
	p10: number | null;
	p50: number | null;
	p90: number | null;
}

export interface CoreComparisonMetrics {
	count: number;
	rmse: number | null;
	mae: number | null;
	r2: number | null;
	metric_space?: string;
}

export interface WellBundleResponse {
	project_id: number;
	well_name: string;
	row_count: number;
	columns: string[];
	curves: Record<string, (number | string | null)[]>;
	stats: Record<string, CurveSummaryStats>;
	tops: Array<{ id: string; name: string; depth: number }>;
	contacts: Array<{ id: string; type: string; name: string; depth: number }>;
	core_samples: Array<Record<string, any>>;
	pressure_tests: Array<Record<string, any>>;
	survey: Array<Record<string, any>>;
	config: Record<string, any>;
}

export interface paths {
	'/quick_pp/database/projects': {
		get: {
			responses: {
				200: {
					content: {
						'application/json': Array<{
							project_id: number;
							name: string;
							description?: string;
							created_at?: string;
						}>;
					};
				};
			};
		};
		post: {
			requestBody: {
				content: {
					'application/json': { name: string; description?: string };
				};
			};
			responses: {
				200: {
					content: {
						'application/json': { project_id: number; name: string; description?: string };
					};
				};
			};
		};
	};
	'/quick_pp/database/projects/{project_id}/summary': {
		get: {
			parameters: { path: { project_id: number | string } };
			responses: {
				200: {
					content: {
						'application/json': ProjectSummaryResponse;
					};
				};
			};
		};
	};
	'/quick_pp/workspace/projects/{project_id}/summary': {
		get: {
			parameters: { path: { project_id: number | string } };
			responses: {
				200: {
					content: {
						'application/json': ProjectSummaryResponse;
					};
				};
			};
		};
	};
	'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/bundle': {
		get: {
			parameters: {
				path: { project_id: number | string; well_name: string };
				query?: { min_depth?: number; max_depth?: number; zones?: string };
			};
			responses: {
				200: {
					content: {
						'application/json': WellBundleResponse;
					};
				};
			};
		};
	};
	'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/hc_correction': {
		post: {
			parameters: { path: { project_id: number | string; well_name: string } };
			requestBody: {
				content: {
					'application/json': {
						dry_sand_point?: [number, number];
						dry_clay_point?: [number, number];
						water_point?: [number, number];
						corr_angle?: number | null;
						rho_water?: number;
						rho_hc?: number;
						hi_hc?: number;
						buffer?: number;
						min_depth?: number | null;
						max_depth?: number | null;
						zones?: string[] | null;
						save?: boolean;
					};
				};
			};
			responses: {
				200: {
					content: {
						'application/json': {
							well_name: string;
							corr_angle: number;
							saved: boolean;
							saved_rows: number;
							stats: Record<string, CurveSummaryStats>;
							columns: string[];
							row_count: number;
							data: Record<string, (number | null)[]>;
						};
					};
				};
			};
		};
	};
	'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/lithology': {
		post: {
			parameters: { path: { project_id: number | string; well_name: string } };
			requestBody: {
				content: {
					'application/json': {
						model?: string;
						use_hc_corrected?: boolean;
						dry_sand_point?: [number, number];
						dry_silt_point?: [number | null, number | null];
						dry_clay_point?: [number | null, number | null];
						fluid_point?: [number, number];
						wet_clay_point?: [number | null, number | null];
						silt_line_angle?: number;
						minerals?: string[];
						porosity_method?: string;
						auto_scale?: boolean;
						min_depth?: number | null;
						max_depth?: number | null;
						zones?: string[] | null;
						save?: boolean;
					};
				};
			};
			responses: {
				200: {
					content: {
						'application/json': {
							well_name: string;
							model: string;
							saved: boolean;
							saved_rows: number;
							crossplot: Record<string, any>;
							stats: Record<string, CurveSummaryStats>;
							columns: string[];
							row_count: number;
							data: Record<string, (number | null)[]>;
						};
					};
				};
			};
		};
	};
	'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/porosity': {
		post: {
			parameters: { path: { project_id: number | string; well_name: string } };
			requestBody: {
				content: {
					'application/json': {
						method?: string;
						crossplot_model?: string;
						use_hc_corrected?: boolean;
						dry_sand_point?: [number, number];
						dry_silt_point?: [number | null, number | null];
						dry_clay_point?: [number | null, number | null];
						fluid_point?: [number, number];
						wet_clay_point?: [number | null, number | null];
						silt_line_angle?: number;
						min_depth?: number | null;
						max_depth?: number | null;
						zones?: string[] | null;
						save?: boolean;
					};
				};
			};
			responses: {
				200: {
					content: {
						'application/json': {
							well_name: string;
							method: string;
							saved: boolean;
							saved_rows: number;
							core_metrics: CoreComparisonMetrics;
							stats: Record<string, CurveSummaryStats>;
							columns: string[];
							row_count: number;
							data: Record<string, (number | null)[]>;
						};
					};
				};
			};
		};
	};
	'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/permeability': {
		post: {
			parameters: { path: { project_id: number | string; well_name: string } };
			requestBody: {
				content: {
					'application/json': {
						method?: string;
						m?: number;
						A?: number;
						B?: number;
						C?: number;
						s?: number;
						swirr?: number;
						min_depth?: number | null;
						max_depth?: number | null;
						zones?: string[] | null;
						save?: boolean;
					};
				};
			};
			responses: {
				200: {
					content: {
						'application/json': {
							well_name: string;
							method: string;
							saved: boolean;
							saved_rows: number;
							core_metrics: CoreComparisonMetrics;
							stats: Record<string, CurveSummaryStats>;
							columns: string[];
							row_count: number;
							data: Record<string, (number | null)[]>;
						};
					};
				};
			};
		};
	};
	'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/saturation': {
		post: {
			parameters: { path: { project_id: number | string; well_name: string } };
			requestBody: {
				content: {
					'application/json': {
						method?: string;
						meas_system?: string;
						water_salinity?: number;
						m?: number;
						a?: number;
						n?: number;
						temp_gradient?: number;
						surface_temp?: number;
						b_method?: string;
						qv_method?: string;
						cec_clay?: number;
						rho_dry_clay?: number;
						min_depth?: number | null;
						max_depth?: number | null;
						zones?: string[] | null;
						save?: boolean;
					};
				};
			};
			responses: {
				200: {
					content: {
						'application/json': {
							well_name: string;
							method: string;
							saved: boolean;
							saved_rows: number;
							stats: Record<string, CurveSummaryStats>;
							columns: string[];
							row_count: number;
							data: Record<string, (number | null)[]>;
						};
					};
				};
			};
		};
	};
	'/quick_pp/workspace/projects/{project_id}/wells/{well_name}/reservoir_summary': {
		post: {
			parameters: { path: { project_id: number | string; well_name: string } };
			requestBody: {
				content: {
					'application/json': {
						cut_offs?: { VSHALE: number; PHIT: number; SWT: number };
						Zone?: string | null;
						include_upscaling?: boolean;
						cell_size?: number;
						agg_methods?: Record<string, string> | null;
						min_depth?: number | null;
						max_depth?: number | null;
						zones?: string[] | null;
						save?: boolean;
					};
				};
			};
			responses: {
				200: {
					content: {
						'application/json': {
							well_name: string;
							cut_offs: { VSHALE: number; PHIT: number; SWT: number };
							saved: boolean;
							saved_rows: number;
							summary: Array<Record<string, any>>;
							flags: {
								columns: string[];
								row_count: number;
								data: Record<string, (number | string | null)[]>;
							};
							upscaling?: {
								cell_size: number;
								n_cells: number;
								columns: string[];
								row_count: number;
								data: Record<string, (number | null)[]>;
							};
						};
					};
				};
			};
		};
	};
	'/quick_pp/workspace/projects/{project_id}/rock_typing/compute': {
		post: {
			parameters: { path: { project_id: number | string } };
			requestBody: {
				content: {
					'application/json': {
						method?: 'fzi' | 'winland' | 'lucia';
						cutoffs: number[];
						higher_is_better?: boolean;
						auto_find_cutoffs?: boolean;
						no_of_cutoffs?: number;
						save?: boolean;
					};
				};
			};
			responses: {
				200: {
					content: {
						'application/json': {
							project_id: number;
							method?: 'fzi' | 'winland' | 'lucia';
							cutoffs: number[];
							higher_is_better?: boolean;
							saved: boolean;
							saved_wells: number;
							rock_type_counts: Record<string, number>;
							data: {
								well_names: string[];
								depths: number[];
								phit: number[];
								perm: number[];
								fzi: (number | null)[];
								r35?: (number | null)[];
								rfn?: (number | null)[];
								drt_value?: (number | null)[];
								rqi?: (number | null)[];
								phiz?: (number | null)[];
								rock_flags: (number | null)[];
							};
						};
					};
				};
			};
		};
	};
	'/quick_pp/workspace/projects/{project_id}/perm_transform/compute': {
		post: {
			parameters: { path: { project_id: number | string } };
			requestBody: {
				content: {
					'application/json': {
						cutoffs?: number[];
						fits?: Record<string, { a: number; b: number }> | null;
						save?: boolean;
					};
				};
			};
			responses: {
				200: {
					content: {
						'application/json': {
							project_id: number;
							fits: Record<string, { a: number; b: number; r2: number | null; count: number }>;
							saved: boolean;
							saved_wells: number;
							data: {
								well_names: string[];
								depths: number[];
								phit: number[];
								perm: number[];
								rock_flags: (number | null)[];
								perm_pred: (number | null)[];
							};
						};
					};
				};
			};
		};
	};
	'/quick_pp/workspace/projects/{project_id}/shf/compute': {
		post: {
			parameters: { path: { project_id: number | string } };
			requestBody: {
				content: {
					'application/json': {
						cutoffs?: number[];
						fwl?: number;
						ift?: number;
						theta?: number;
						gw?: number;
						ghc?: number;
						fits?: Record<string, { a: number; b: number }> | null;
						save?: boolean;
					};
				};
			};
			responses: {
				200: {
					content: {
						'application/json': {
							project_id: number;
							fits: Record<string, { a: number; b: number; rmse?: number }>;
							saved: boolean;
							saved_wells: number;
							data: {
								well_names: string[];
								depths: number[];
								phit: number[];
								perm: number[];
								rock_flags: (number | null)[];
								shf: (number | null)[];
							};
						};
					};
				};
			};
		};
	};
}

