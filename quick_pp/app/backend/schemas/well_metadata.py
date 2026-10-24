"""Pydantic schemas for Well Header Metadata and Two-Pass Import Workflows."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


def sanitize_json_primitives(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and NaN/Inf into JSON-safe Python primitives."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        fval = float(obj)
        return None if (np.isnan(fval) or np.isinf(fval)) else fval
    if isinstance(obj, np.ndarray):
        return [sanitize_json_primitives(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): sanitize_json_primitives(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_json_primitives(x) for x in obj]
    return obj


class WellHeaderMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    uwi: Optional[str] = None
    well_name: str = "UNKNOWN_WELL"
    field: Optional[str] = None
    company: Optional[str] = None
    country: Optional[str] = None
    spud_date: Optional[str] = None
    log_date: Optional[str] = None
    # Coordinates & Datum
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None
    crs: Optional[str] = None
    kb_elev: Optional[float] = Field(None, description="Kelly Bushing elevation")
    df_elev: Optional[float] = Field(None, description="Derrick Floor elevation")
    gl_elev: Optional[float] = Field(
        None, description="Ground Level / Seabed elevation"
    )
    water_depth: Optional[float] = None
    apd: Optional[float] = None
    datum_unit: str = "m"
    source_depth_unit: Optional[str] = None
    # Depth Info (~Well)
    start_depth: Optional[float] = None
    stop_depth: Optional[float] = None
    step: Optional[float] = None
    null_val: Optional[float] = -999.25
    # Logging Environment (~Parameter)
    bht: Optional[float] = None
    bht_unit: Optional[str] = "degC"
    surface_temp: Optional[float] = None
    bit_size: Optional[float] = None
    bs: Optional[float] = None
    rmf: Optional[float] = None
    rmf_temp: Optional[float] = None
    rmc: Optional[float] = None
    mud_density: Optional[float] = None
    mud_weight: Optional[float] = None
    # Raw fallback for unmapped headers
    raw_well_header: Dict[str, Any] = Field(default_factory=dict)
    raw_parameters: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases_and_numpy(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = sanitize_json_primitives(dict(data))
        if "kb_elevation" in d and d.get("kb_elev") is None:
            d["kb_elev"] = d["kb_elevation"]
        if "ground_elevation" in d and d.get("gl_elev") is None:
            d["gl_elev"] = d["ground_elevation"]
        if "dfe_elevation" in d and d.get("df_elev") is None:
            d["df_elev"] = d["dfe_elevation"]
        if "field_name" in d and d.get("field") is None:
            d["field"] = d["field_name"]
        if "depth_unit" in d and not d.get("datum_unit"):
            d["datum_unit"] = d["depth_unit"]
        return d

    @computed_field
    @property
    def kb_elevation(self) -> Optional[float]:
        return self.kb_elev

    @computed_field
    @property
    def ground_elevation(self) -> Optional[float]:
        return self.gl_elev

    @computed_field
    @property
    def dfe_elevation(self) -> Optional[float]:
        return self.df_elev

    @computed_field
    @property
    def field_name(self) -> Optional[str]:
        return self.field

    @computed_field
    @property
    def depth_unit(self) -> str:
        return self.datum_unit

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class CurveInspectionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    raw_mnemonic: str
    target_mnemonic: str
    raw_unit: str = ""
    target_unit: str = ""
    description: str = ""
    auto_conversion: Optional[str] = None
    is_array_channel: bool = False
    dimensions: List[int] = Field(default_factory=lambda: [1])
    include: bool = True
    # Optional QA/QC diagnostics if sample/full data inspected
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    completeness_pct: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _clean_numpy(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = sanitize_json_primitives(dict(data))
            if "mapped_mnemonic" in d and "target_mnemonic" not in d:
                d["target_mnemonic"] = d["mapped_mnemonic"]
            if "source_unit" in d and "raw_unit" not in d:
                d["raw_unit"] = d["source_unit"] or ""
            return d
        return data

    @computed_field
    @property
    def mapped_mnemonic(self) -> str:
        return self.target_mnemonic

    @computed_field
    @property
    def source_unit(self) -> str:
        return self.raw_unit

    @computed_field
    @property
    def unit_conversion_applied(self) -> bool:
        return bool(self.auto_conversion)


class DlisFrameInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    logical_file_index: int = 0
    frame_name: str
    index_type: str = "BOREHOLE-DEPTH"
    index_channel: Optional[str] = "DEPTH"
    start_depth: Optional[float] = None
    stop_depth: Optional[float] = None
    step: Optional[float] = None
    depth_unit: str = "m"
    scalar_channels: List[str] = Field(default_factory=list)
    array_channels: List[str] = Field(default_factory=list)
    selected: bool = True

    @model_validator(mode="before")
    @classmethod
    def _clean_numpy(cls, data: Any) -> Any:
        return sanitize_json_primitives(data) if isinstance(data, dict) else data


class LogInspectFileResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    file_id: str
    filename: str
    format: str = "LAS 2.0"
    file_size_bytes: int = 0
    metadata: WellHeaderMetadata
    curves: List[CurveInspectionItem] = Field(default_factory=list)
    dlis_frames: List[DlisFrameInfo] = Field(default_factory=list)
    has_embedded_core: bool = False
    has_embedded_tops: bool = False
    qaqc_warnings: List[str] = Field(default_factory=list)
    inspection_ms: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _clean_numpy(cls, data: Any) -> Any:
        return sanitize_json_primitives(data) if isinstance(data, dict) else data


class LogInspectResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    files: List[LogInspectFileResult] = Field(default_factory=list)
    total_inspection_ms: float = 0.0


class LogCommitFileConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    file_id: Optional[str] = None
    filename: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    metadata_overrides: Optional[Dict[str, Any]] = None
    mnemonic_map: Dict[str, str] = Field(default_factory=dict)
    curve_mappings: Dict[str, str] = Field(default_factory=dict)
    unit_map: Dict[str, str] = Field(default_factory=dict)
    unit_overrides: Dict[str, str] = Field(default_factory=dict)
    excluded_curves: List[str] = Field(default_factory=list)
    selected_dlis_frames: Optional[List[str]] = None
    selected_dlis_frame: Optional[str] = None
    skip_array_channels: bool = True
    import_embedded_ancillary: bool = True
    splice_overlap_mode: Optional[str] = None
    target_step: Optional[float] = None


class LogCommitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    project_id: Optional[int] = None
    target_depth_uom: str = "m"
    regularize_grid: bool = False
    regularize_step: float = 0.1524
    splicing_mode: str = Field(
        "keep_last", description="Overlap conflict resolution: keep_last, keep_first, mean"
    )
    splice_multi_runs: bool = True
    splice_overlap_mode: Optional[str] = None
    files: List[LogCommitFileConfig] = Field(default_factory=list)


class AncillaryColumnMapping(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    source_column: str
    cleaned_header: str = ""
    target_column: Optional[str] = None
    source_unit: Optional[str] = None
    detected_unit: Optional[str] = None
    target_unit: Optional[str] = None
    conversion_badge: Optional[str] = None
    confidence: float = 0.0
    match_stage: str = "unmapped"  # exact, alias, fuzzy, unmapped

    @model_validator(mode="before")
    @classmethod
    def _sync_units(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = sanitize_json_primitives(dict(data))
            if d.get("source_unit") is None and d.get("detected_unit") is not None:
                d["source_unit"] = d["detected_unit"]
            if d.get("detected_unit") is None and d.get("source_unit") is not None:
                d["detected_unit"] = d["source_unit"]
            return d
        return data


class AncillaryPreviewResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    file_id: str = ""
    filename: str
    data_type: str
    sheet_names: List[str] = Field(default_factory=list)
    active_sheet: Optional[str] = None
    detected_header_row: int = 0
    detected_unit_row: Optional[int] = None
    detected_well_name: Optional[str] = None
    total_rows: int = 0
    headers: List[str] = Field(default_factory=list)
    column_mappings: List[AncillaryColumnMapping] = Field(default_factory=list)
    mappings: List[AncillaryColumnMapping] = Field(default_factory=list)
    detected: Dict[str, Optional[str]] = Field(default_factory=dict)
    preview: List[Dict[str, Any]] = Field(default_factory=list)
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list)
    target_schema: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _sync_preview_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = sanitize_json_primitives(dict(data))
            if "column_mappings" in d and not d.get("mappings"):
                d["mappings"] = d["column_mappings"]
            elif "mappings" in d and not d.get("column_mappings"):
                d["column_mappings"] = d["mappings"]
            if "preview" in d and not d.get("preview_rows"):
                d["preview_rows"] = d["preview"]
            elif "preview_rows" in d and not d.get("preview"):
                d["preview"] = d["preview_rows"]
            return d
        return data


class AncillaryCommitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    project_id: int
    well_name: Optional[str] = None
    file_id: Optional[str] = None
    file_content_base64: Optional[str] = None
    filename: Optional[str] = None
    data_type: str = "core_samples"
    sheet_name: Optional[str] = None
    header_row: Optional[int] = None
    column_mappings: List[AncillaryColumnMapping] = Field(default_factory=list)
    column_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps source CSV/Excel column name to target quick_pp column name",
    )
    unit_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps source CSV/Excel column name to source unit",
    )
    rows: Optional[List[Dict[str, Any]]] = None
    # Survey options
    calculate_tvd: bool = True
    kb_elev_override: Optional[float] = None
    # Core depth alignment & shift options
    snap_to_log_grid: bool = True
    snap_tolerance: float = 0.08
    core_snap_tolerance_m: float = 0.08
    depth_shift_constant: float = 0.0
    depth_shift_table: Optional[List[Tuple[float, float]]] = None
    core_depth_shifts: Optional[List[Tuple[float, float]]] = None
