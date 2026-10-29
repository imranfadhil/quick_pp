"""Pydantic schemas for the FDP Petrophysical Report Generation Module."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

ReportTypeLiteral = Literal[
    "FDP_FULL",
    "SINGLE_WELL_CPI",
    "QAQC_SUMMARY",
    "RESSUM_ONLY",
]


DEFAULT_REPORT_SECTIONS: List[Dict[str, Any]] = [
    {
        "section_id": "01_executive_summary",
        "title": "1. Executive Summary",
        "enabled": True,
        "include_ai_commentary": True,
    },
    {
        "section_id": "02_data_inventory",
        "title": "2. Well Inventory & Data Availability",
        "enabled": True,
        "include_ai_commentary": True,
    },
    {
        "section_id": "03_qaqc",
        "title": "3. Data Quality Control (QA/QC) & Conditioning",
        "enabled": True,
        "include_ai_commentary": True,
    },
    {
        "section_id": "04_lithology_porosity",
        "title": "4. Lithology, Mineralogy & Porosity Evaluation",
        "enabled": True,
        "include_ai_commentary": True,
    },
    {
        "section_id": "05_rock_typing_perm",
        "title": "5. Rock Typing & Permeability Modeling",
        "enabled": True,
        "include_ai_commentary": True,
    },
    {
        "section_id": "06_saturation_contacts",
        "title": "6. Fluid Contacts & Water Saturation (Sw)",
        "enabled": True,
        "include_ai_commentary": True,
    },
    {
        "section_id": "07_shf_scal",
        "title": "7. Capillary Pressure, Saturation Height Modeling (SHF) & SCAL",
        "enabled": True,
        "include_ai_commentary": True,
    },
    {
        "section_id": "08_ressum_uncertainty",
        "title": "8. Cut-offs, Sums & Averages (Reservoir Summary) & Uncertainty",
        "enabled": True,
        "include_ai_commentary": True,
    },
    {
        "section_id": "09_upscaling",
        "title": "9. Upscaling, Log Blocking & Deliverables Handover",
        "enabled": True,
        "include_ai_commentary": True,
    },
    {
        "section_id": "10_appendix_cpi",
        "title": "Appendix A: Composite Petrophysical Evaluation Logs (CPI)",
        "enabled": True,
        "include_ai_commentary": False,
    },
]


class ReportSectionConfig(BaseModel):
    section_id: str = Field(..., description="Unique section identifier")
    title: str = Field(..., description="Section heading title")
    enabled: bool = Field(
        True, description="Whether to include this section in the report"
    )
    include_ai_commentary: bool = Field(
        True, description="Whether to include synthesized technical commentary"
    )
    custom_notes: Optional[str] = Field(
        default=None,
        description="Optional user or petrophysicist custom notes to inject into this section",
    )


class ReportGenerationRequest(BaseModel):
    project_id: Union[int, str] = Field(
        ..., description="Project ID in quick_pp database"
    )
    well_names: List[str] = Field(
        default_factory=lambda: ["ALL"],
        description="List of wells to include, or ['ALL']",
    )
    report_type: ReportTypeLiteral = Field(
        default="FDP_FULL",
        description="Preset report scope: FDP_FULL, SINGLE_WELL_CPI, QAQC_SUMMARY, or RESSUM_ONLY",
    )
    title: str = Field(
        default="Field Development Plan - Petrophysical Evaluation Report",
        description="Report document title",
    )
    report_title: Optional[str] = Field(
        default=None,
        description="Optional alias for title",
    )
    author: str = Field(
        default="quick_pp Petrophysical Team",
        description="Report author or team attribution",
    )
    company: Optional[str] = Field(
        default=None, description="Optional company or asset team"
    )
    field_name: Optional[str] = Field(
        default=None, description="Optional field name override"
    )
    sections: Optional[List[ReportSectionConfig]] = Field(
        default=None,
        description="Optional section configuration overrides",
    )
    cutoffs: Dict[str, float] = Field(
        default_factory=lambda: {"vsh": 0.40, "phie": 0.08, "sw": 0.65},
        description="Cut-offs used for Net Rock, Net Reservoir, and Net Pay",
    )
    zones: Optional[List[str]] = Field(
        default=None, description="Optional geological zone filter"
    )
    target_zones: Optional[List[str]] = Field(
        default=None,
        description="Optional target reservoir zones for Executive Summary KPIs and primary summary tables",
    )
    fwl_tvdss: Optional[float] = Field(
        default=None,
        description="Optional explicit Free Water Level (m TVDss)",
    )
    min_depth: Optional[float] = Field(
        default=None, description="Optional minimum depth filter (m)"
    )
    max_depth: Optional[float] = Field(
        default=None, description="Optional maximum depth filter (m)"
    )
    export_format: Literal["markdown", "pdf", "both"] = Field(
        default="markdown",
        description="Output format to compile on generation",
    )
    include_ai_commentary: bool = Field(
        default=True,
        description="Whether to populate technical interpretation narrative",
    )
    include_ai_summary: Optional[bool] = Field(
        default=None,
        description="Optional alias for include_ai_commentary",
    )
    provider: Optional[str] = Field(
        default="native_mcp",
        description="AI narrative provider ('native_mcp', 'openai', 'gemini', 'anthropic', 'ollama')",
    )
    model: Optional[str] = Field(
        default=None, description="Optional LLM model identifier"
    )
    api_key: Optional[str] = Field(default=None, description="Optional LLM API key")
    base_url: Optional[str] = Field(default=None, description="Optional LLM base URL")


class ReportSectionUpdateRequest(BaseModel):
    section_id: Optional[str] = Field(default=None, description="Section ID to update")
    markdown_content: str = Field(
        ..., description="Replacement Markdown content for the section"
    )
    title: Optional[str] = Field(
        default=None, description="Optional updated section title"
    )
    custom_notes: Optional[str] = Field(
        default=None, description="Optional updated custom notes"
    )


class ReportFullMarkdownUpdateRequest(BaseModel):
    markdown_content: str = Field(..., description="Updated full Markdown document")
    title: Optional[str] = Field(
        default=None, description="Optional updated report title"
    )
    author: Optional[str] = Field(default=None, description="Optional updated author")


class ReportAiEnhanceRequest(BaseModel):
    section_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific section IDs to enhance, or None for all enabled sections",
    )
    instruction: Optional[str] = Field(
        default=None,
        description="Optional domain instruction (e.g., 'Explain why Waxman-Smits was selected over Archie')",
    )
    provider: str = Field(
        default="native_mcp",
        description="LLM provider: 'native_mcp', 'openai', 'gemini', 'anthropic', or 'ollama'",
    )
    model: Optional[str] = Field(default=None, description="Model name")
    api_key: Optional[str] = Field(default=None, description="Optional API key")
    base_url: Optional[str] = Field(
        default=None, description="Optional custom base URL"
    )


class ReportGenerationResponse(BaseModel):
    report_id: str
    project_id: str
    project_name: str = ""
    title: str
    author: str
    report_type: str = "FDP_FULL"
    well_names: List[str] = Field(default_factory=list)
    sections: List[ReportSectionConfig] = Field(default_factory=list)
    section_contents: Dict[str, str] = Field(default_factory=dict)
    markdown_content: str
    html_content: str = ""
    asset_urls: Dict[str, Any] = Field(default_factory=dict)
    pdf_download_url: Optional[str] = None
    print_html_url: Optional[str] = None
    context_summary: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: Optional[str] = None
