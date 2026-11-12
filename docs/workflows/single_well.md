# Single-Well Petrophysical Workflow

The Single-Well space delivers deterministic, optimization-driven, and core-calibrated formation evaluation along a single wellbore trajectory. It is structured into **7 Primary Sequential Workflow Steps** culminating in a Computer Processed Interpretation (CPI) report, accompanied by a **Standalone Core & Pc Prep Lab Module**.

When importing LAS logs, check the depth curve's **Source Unit** in the uploader. Feet are converted to meters by multiplying the depth values by 0.3048; edit the source unit if the LAS curve header is incorrect. Previously imported wells are not automatically corrected, and importing into an existing well merges data rather than replacing it.

For LAS imports through **Review / Commit**, renamed log curves retain a separate copy of their source values and unit alongside the standardized curve. Name collisions receive a `RAW_` prefix or a numeric suffix. The import summary and Data & QAQC well header show each source file, mnemonic, stored curve, and raw copy; these records describe the import, not subsequent edits. Depth remains the shared, converted index rather than a separate raw curve. Optional grid regularization also applies to the raw copies. Existing imports are not backfilled.

The same log uploader also accepts DLIS and LIS79 (`.lis`) files. LIS Enhanced LIS/LIS84 variants are not supported by the current parser.

The **Log / interval** unit selector (m or ft) affects the single-well log plot and readouts, working interval and top-bar depth filter, and shared tables whose values are in meters. Table edits, pasted values, and ancillary quick-entry depths are converted back to meters for processing. The preference is saved in the browser; it does not change stored data. Other plots, reports, and exports retain their existing units.

Zone-filtered workspace calculations require resolved zone labels for the selected well, normally from its zone data or formation tops. If zone labels are unavailable, the calculation reports an error instead of evaluating the entire well.

In Data & QAQC, **Missing Data Imputation** saves selected logs as new `_IMP` curves by default. Alternatively, replace the selected curves while preserving a full-well `_ORI` backup; existing backups are never overwritten. Choose mean, median, most frequent, forward/backward fill, depth-based linear interpolation, KNN, or iterative imputation; multivariate methods require at least two curves. Linear interpolation fills only gaps between observed samples, without extrapolating end gaps. Directional fills follow increasing depth and may also leave end gaps unfilled. Active depth and zone filters apply; existing `_IMP` curves must be renamed or deleted before re-running in new-curve mode. The right-pane **Missing Matrix** shows binned missing fractions by curve, with deeper depths at the top and grey bins indicating no samples.

---

## 1. Workflow Architecture (Level 1 & Level 2) {#sw-level-1-2-diagrams}

### Level 1: Macro Workflow

```mermaid
flowchart LR
    subgraph PrimarySeq["Primary Sequential Workflow (Steps 1–7)"]
        A["1. Data & QA/QC\n(Bad-Hole, Repair, Normalization)"] --> B["2. Tops & Ancillary\n(Min. Curvature, Tops Ingestion)"]
        B --> C["3. Lithology & Porosity\n(Vsh, SSC, Multi-Mineral, Phie)"]
        C --> D["4. Permeability\n(Choo, Coates, Tixier, FZI)"]
        D --> E["5. Saturation\n(Archie, Waxman-Smits, Dual Water)"]
        E --> F["6. Reservoir Summary\n(Cutoffs, Net Pay, Log Blocking)"]
        F --> G["7. Well CPI Report\n(Automated Dossier, FDP Export)"]
    end

    subgraph StandaloneLab["Standalone Lab Module"]
        LAB["Core & Pc Prep\n(Depth Shift, Stress Corrections, Lab Pc)"]
    end

    LAB -.->|Depth-Aligned Core| C
    LAB -.->|Stress-Corrected Perm| D

    style A fill:#f0f9ff,stroke:#0284c7,stroke-width:1.5px
    style B fill:#f8fafc,stroke:#475569,stroke-width:1.5px
    style C fill:#f0fdf4,stroke:#16a34a,stroke-width:1.5px
    style D fill:#fefce8,stroke:#ca8a04,stroke-width:1.5px
    style E fill:#fff1f2,stroke:#e11d48,stroke-width:1.5px
    style F fill:#fef3c7,stroke:#d97706,stroke-width:1.5px
    style G fill:#eff6ff,stroke:#2563eb,stroke-width:2px
    style LAB fill:#faf5ff,stroke:#9333ea,stroke-width:1.5px,stroke-dasharray: 4 4
```

### Level 2: Sub-Process & Curve Handoff Architecture

```mermaid
flowchart TD
    subgraph S1["1. Data & QA/QC (InspectorQaqc)"]
        LAS["Raw LAS Curves\n(GR, RHOB, NPHI, RT, DTC, CALI)"]
        CALI["CALI & Bit Size (BS)"] --> BAD["Bad-Hole Flag (BADHOLE)\nDifferential Caliper & Rugosity σ_RHOB"]
        LAS --> BAD
        BAD --> REPAIR["Synthetic Curve Repair\nML / Multi-Log Reconstruction"]
    end

    subgraph S2["2. Tops & Ancillary Data (InspectorAncillary)"]
        SURV["Directional Survey\n(MD, INC, AZI)"] --> MINCURV["Minimum Curvature Solver\nTVD, TVDSS, Northing, Easting"]
        TOPS["Stratigraphic Tops & Zones"]
        MINCURV --> TVD_GRID["Continuous TVDSS Depth Grid"]
    end

    subgraph S3["3. Lithology & Porosity Evaluation (InspectorLithoPoro)"]
        REPAIR --> VSH_MOD["Shale Volume (VSH_GR)\nLinear, Larionov, Clavier, Stieber"]
        REPAIR & VSH_MOD --> LITHO_SOLVER{"Lithology Engine"}
        LITHO_SOLVER -->|Sand-Silt-Clay| SSC["SSC Ternary Model\n(VSAND, VSILT, VCLAY)"]
        LITHO_SOLVER -->|Thin Beds| TB["Thomas-Stieber Tensor Model\n(V_lam, V_disp, Sand Phie)"]
        LITHO_SOLVER -->|Carbonate| CARB["Carbonate Partitioning\n(VCALC, VDOLO, VANHY)"]
        LITHO_SOLVER -->|Multi-Mineral| MM["SLSQP Matrix Inversion\nMin ||Av - b||^2"]
        SSC & TB & CARB & MM --> PORO_CALC["Total & Effective Porosity\n(PHIT, PHIE) via Density/Neutron/Sonic"]
        PORO_CALC <-->|Gas / Condensate| HCCORR["Hydrocarbon Correction\nExcavation & Density Adjustment"]
    end

    subgraph S4["4. Permeability Estimation (InspectorPerm)"]
        PORO_CALC & VSH_MOD --> PERM_ENGINE{"Permeability Model"}
        PERM_ENGINE -->|Shaly Clastics| CHOO["Choo Model: f(PHIT, VCLAY, VSILT)"]
        PERM_ENGINE -->|Free Fluid NMR| COATES["Timur-Coates / Free Fluid Model"]
        PERM_ENGINE -->|Empirical Clastics| TIXIER["Tixier / Morris-Biggs Gas & Oil"]
        PERM_ENGINE -->|Hydraulic Units| HFU_PERM["Amaefule FZI / RQI Transform"]
    end

    subgraph S5["5. Fluid Saturation Evaluation (InspectorSaturation)"]
        TVD_GRID --> TEMP_GRAD["Geothermal Gradient & Rw(T)\nArps Temperature Conversion"]
        PORO_CALC & TEMP_GRAD --> SAT_ENGINE{"Saturation Model"}
        SAT_ENGINE -->|Clean / Carbonate| ARCHIE["Archie Equation\n(Swt, Swe via m, n, a, Rw)"]
        SAT_ENGINE -->|Dispersed Shaly Sand| WS["Waxman-Smits Model\n(CEC, Qv, B Counter-Ion Mobility)"]
        SAT_ENGINE -->|Dual Conductivity| DW["Dual Water Model\n(Bound Water Swb & Free Water Swf)"]
        SAT_ENGINE -->|Total Shale Clastic| SIM_IND["Simandoux / Indonesian Model"]
    end

    subgraph S6["6. Reservoir Summary & Log Blocking (InspectorResSum)"]
        PORO_CALC & PERM_ENGINE & SAT_ENGINE & TOPS --> CUTOFFS["Net Pay Cutoffs\n(Vsh_cut, Phie_cut, Sw_cut, Perm_cut)"]
        CUTOFFS --> SUMMARY["Zonal Net Reservoir / Net Pay Sum\n(Gross, Net, NTG, Phie_avg, Sw_avg, HCPV)"]
        CUTOFFS --> BLOCKING["Log Upscaling (Blocking)\nArithmetic, Geometric, Harmonic"]
    end

    subgraph S7["7. Well CPI Report (InspectorReport)"]
        SUMMARY & BLOCKING --> DOSSIER["CPI Composite Plot, Property Tables\n& Automated FDP Markdown/PDF"]
    end

    subgraph S_LAB["Standalone: Core & Pc Prep (InspectorScal)"]
        CORE_RAW["Routine Core Analysis (RCA)\n(CPOR, CPERM, Grain Density)"] --> SHIFT["Core-to-Log Warping\nInteractive Depth Alignment Δz(MD)"]
        CORE_RAW --> STRESS["Overburden & Klinkenberg Corrections\nNet Confining Stress Compaction"]
        PC_LAB["Lab Capillary Pressure (Pc)"] --> PC_SCALE["Lab-to-Reservoir Pc Scaling\nInterfacial Tension & Contact Angle"]
    end

    SHIFT -.->|Shifted Core Porosity| PORO_CALC
    STRESS -.->|Liquid-Equiv Perm| PERM_ENGINE
```

---

## 2. Primary Sequential Steps (1 to 7)

### Step 1: Data & Log QA/QC (`InspectorQaqc`) {#sw-step-1-qaqc}

* **Objective:** Identify intervals where borehole washout or severe rugosity invalidates shallow-reading pad tools ($\text{RHOB}, \text{NPHI}, \text{PEF}$), perform environmental corrections, and reconstruct missing or corrupted log intervals using multi-log regressions.

* **Technical Foundation:**

    * **Differential Caliper & Rugosity Flagging:**
      Differential caliper $\Delta \text{CALI} = \text{CALI} - \text{BS}$ measures borehole enlargement relative to bit size. A rolling window of standard deviation $\sigma_{\text{RHOB}}$ evaluates micro-rugosity. The boolean `BADHOLE` flag is triggered if:
      
      $$\Delta \text{CALI} > \tau_{\text{cali}} \quad \lor \quad \sigma_{\text{RHOB}} > \tau_{\sigma} \quad \lor \quad \rho_b < 1.95\,\text{g/cm}^3 \quad \lor \quad \rho_b > 3.05\,\text{g/cm}^3$$

    * **Borehole Environmental Corrections:**
      Mudcake buildup and borehole temperature/pressure effects are compensated analytically:
      
      $$\Delta \rho_{\text{mudcake}} = -0.015 \cdot (\text{CALI} - \text{BS}) \cdot \frac{\rho_m - 1.0}{\rho_m}$$

    * **Synthetic Curve Repair:**
      Corrupted pad readings are reconstructed via multivariate regression or Random Forest models trained on in-gauge intervals ($\text{BADHOLE} = 0$) using deep-reading, borehole-insensitive logs ($\text{GR}, \text{DTC}, \ln(\text{RT})$).

* **Input Curves:** `CALI`, `BS`, `GR`, `RHOB`, `NPHI`, `DTC`, `RT`  
* **Output Curves:** `BADHOLE` (flag), `GR_ENV_CORR`, `NPHI_ENV_CORR`, `RHOB_ENV_CORR`, `RT_ENV_CORR`.
  Environmental-correction outputs are preview curves; downstream calculations continue to use
  their configured raw or hydrocarbon-corrected inputs until explicitly selected.

* **Curve editing and splicing:** In Curve Editor & Management, select **View log vs depth in
  right pane** to compare the selected curve with its original backup or edit source. In **Splice
  Runs**, optionally enter a top and/or bottom depth in metres for each run; blank limits use the
  full run and both limits include their boundary samples. For example, set Run 1's bottom to the
  end of its good data and Run 2's top to the start of its good data. The overlap mode resolves
  samples covered by both intervals; uncovered depths stay empty unless gap interpolation is
  enabled. Auto-level uses only overlap within the selected intervals. Diagnostics compare the
  first two runs. Click **Apply Splice** after changing the settings to update the preview, then
  **Save** to persist it. The selected runs are shown alongside the applied splice result.
  The plot follows the active depth and zone filters and preserves gaps.
  Use the plot selector to return to the histogram or another view. Log-versus-depth plots lock
  the horizontal scale; the mouse wheel zooms depth, while selection tools remain available.

---

### Step 2: Tops & Ancillary Data (`InspectorAncillary`) {#sw-step-2-ancillary}

* **Objective:** Establish the True Vertical Depth Sub-Sea ($\text{TVDSS}$) coordinate frame, ingest stratigraphic formation tops, and manage wellbore survey geometry.

* **Formation-test mobility:** Enter or import optional drawdown mobility with formation pressure
  tests. The well plot displays positive mobility values as pink triangles at measured depth in
  Track 5, with a separate 0.01–10,000 mD/cP logarithmic scale and an FPT mobility toggle.
  Points outside that scale are counted as off-scale; depth and zone filters apply. Hover a
  point for its value, depth, and available fluid, quality, and remarks. Mobility is not
  permeability: matching numerical positions on the two scales do not imply equivalent
  properties, and no viscosity or permeability conversion is assumed.

* **Technical Foundation:**

    * **Minimum Curvature Directional Survey Inversion:**
      Converts directional survey stations $(\text{MD}_i, I_i, A_i)$ (Measured Depth, Inclination, Azimuth) into continuous spatial coordinates $(\text{TVD}, N, E)$ using the dogleg angle $\beta_i$ and ratio factor $RF_i$:
      
      $$\beta_i = \arccos\left(\cos I_{i-1} \cos I_i + \sin I_{i-1} \sin I_i \cos(A_i - A_{i-1})\right)$$
      
      $$RF_i = \frac{2}{\beta_i} \tan\left(\frac{\beta_i}{2}\right) \quad (\text{with } RF_i \to 1 \text{ as } \beta_i \to 0)$$
      
      $$\Delta \text{TVD}_i = \frac{\Delta \text{MD}_i}{2} (\cos I_{i-1} + \cos I_i) \cdot RF_i$$
      
      $$\text{TVDSS} = \text{TVD} - \text{KB}$$

* **Input Data:** Directional survey CSV, Kelly Bushing elevation ($\text{KB}$), Stratigraphic Tops  
* **Output Data:** Continuous `TVD`, `TVDSS`, `DX`, `DY`, Formation tops depth registration

---

### Step 3: Lithology & Porosity Evaluation (`InspectorLithoPoro`) {#sw-step-3-litho-poro}

* **Objective:** Partition the rock solid volume into mineral volumes ($V_{\text{sh}}, V_{\text{sand}}, V_{\text{silt}}, V_{\text{clay}}, V_{\text{calc}}, V_{\text{dolo}}$) and calculate Total Porosity ($\phi_t$) and Effective Porosity ($\phi_e$) corrected for shale and light hydrocarbon effects.

* **Technical Foundation:**

    * **Gamma Ray Index & Empirical $V_{\text{sh}}$ Models:**
      
      $$I_{GR} = \text{clip}\left(\frac{\text{GR} - \text{GR}_{\text{clean}}}{\text{GR}_{\text{shale}} - \text{GR}_{\text{clean}}}, 0, 1\right)$$
      
      $$\text{Linear: } V_{\text{sh}} = I_{GR}$$
      
      $$\text{Larionov (Tertiary): } V_{\text{sh}} = 0.083 \left( 2^{3.7 \cdot I_{GR}} - 1 \right)$$
      
      $$\text{Larionov (Older / Mesozoic): } V_{\text{sh}} = 0.33 \left( 2^{2 \cdot I_{GR}} - 1 \right)$$
      
      $$\text{Clavier (1971): } V_{\text{sh}} = 1.7 - \sqrt{3.38 - (I_{GR} + 0.7)^2}$$
      
      $$\text{Stieber (1970): } V_{\text{sh}} = \frac{I_{GR}}{3 - 2 \cdot I_{GR}}$$

    * **Sand-Silt-Clay (SSC) Ternary Model:**
      Partitioning resolves the barycentric triangle of Clean Quartz Sand $(\text{NPHI}_{\text{sd}}, \rho_{\text{sd}})$, Dry Silt $(\text{NPHI}_{\text{slt}}, \rho_{\text{slt}})$, and Dry Clay $(\text{NPHI}_{\text{cl}}, \rho_{\text{cl}})$:
      
      $$V_{\text{sand}} + V_{\text{silt}} + V_{\text{clay}} + \phi_t = 1$$

    * **Thin-Bed Thomas-Stieber & Tensor Resistivity:**
      Partitions shale into laminated ($V_{\text{lam}}$), dispersed ($V_{\text{disp}}$), and structural ($V_{\text{struct}}$) forms. On the $\phi_t$ vs. $V_{\text{sh}}$ crossplot, points below the laminated line $\phi_{\text{lam}} = \phi_{\text{sd}} + (\phi_{\text{sh}} - \phi_{\text{sd}}) V_{\text{sh}}$ fall in the dispersed triangle and split shale into $V_{\text{lam}} + V_{\text{disp}}$; points above it fall in the structural triangle bounded by the grain-replacing line $\phi_{\text{str}} = \phi_{\text{sd}} + \phi_{\text{sh}} V_{\text{sh}}$ and split shale into $V_{\text{lam}} + V_{\text{struct}}$:

      $$V_{\text{lam}} = \begin{cases} \dfrac{\phi_t - \phi_{\text{sd}} + V_{\text{sh}}(1-\phi_{\text{sh}})}{1-\phi_{\text{sd}}} & \phi_t \le \phi_{\text{lam}} \\[2ex] \dfrac{\phi_{\text{sd}} + V_{\text{sh}}\,\phi_{\text{sh}} - \phi_t}{\phi_{\text{sd}}} & \phi_t > \phi_{\text{lam}} \end{cases}, \quad V_{\text{disp}} + V_{\text{struct}} = V_{\text{sh}} - V_{\text{lam}}$$

      $$R_h^{-1} = \frac{1 - V_{\text{lam}}}{R_{\text{sand}}} + \frac{V_{\text{lam}}}{R_{\text{shale}}}, \qquad R_v = (1 - V_{\text{lam}}) R_{\text{sand}} + V_{\text{lam}} R_{\text{shale}}$$

    * **Multi-Mineral Constrained Optimization (SLSQP):**
      Given log measurements $\mathbf{b}$ and mineral response matrix $\mathbf{A}$:
      
      $$\min_{\mathbf{v}} \sum_{j=1}^N w_j \left( \frac{\sum_{i=1}^M A_{ji} v_i - b_j}{\sigma_j} \right)^2 \quad \text{s.t.} \quad \sum_{i=1}^{M+1} v_i = 1, \quad 0 \le v_i \le 1$$

    * **Total Porosity ($\phi_t$) & Effective Porosity ($\phi_e$):**
      
      $$\text{Density Porosity: } \phi_{D} = \frac{\rho_{\text{ma}} - \rho_b}{\rho_{\text{ma}} - \rho_{\text{fl}}}$$
      
      $$\text{Sonic Porosity (Wyllie): } \phi_S = \frac{\Delta t - \Delta t_{\text{ma}}}{\Delta t_{\text{fl}} - \Delta t_{\text{ma}}}$$
      
      $$\text{Hunt-Raymer: } \phi_S = -c - \sqrt{c^2 + \frac{\Delta t_{\text{ma}}}{\Delta t} - 1}, \quad c = \frac{\Delta t_{\text{ma}}}{2\Delta t_{\text{fl}}} - 1$$
      
      $$\phi_e = \phi_t - V_{\text{clay}} \cdot \phi_{\text{clay}} \quad \text{or} \quad \phi_e = \phi_t (1 - V_{\text{sh}})$$

* **Input Curves:** `GR`, `RHOB`, `NPHI`, `PEF`, `DTC`, `POTA`, `THOR`, `URAN`  
* **Output Curves:** `VSH`, `VSAND`, `VSILT`, `VCLAY`, `VCALC`, `VDOLO`, `PHIT`, `PHIE`

* **Selectable input logs:** The multi-mineral inversion accepts any subset of the
  available curves via `input_logs` (UI: *Input Logs* checkboxes). Leaving the selection
  empty auto-uses every loaded curve among `GR`, `NPHI`, `RHOB`, `PEF`, `DTC`, `POTA`,
  `THOR`, `URAN`. Exclude curves distorted by the mineralogy — for example drop `GR`,
  `NPHI`, and `RHOB` in heavy-mineral or hot-sand intervals and invert on `PEF`, `THOR`,
  and `POTA` instead. At least two curves are required, and each depth row must carry
  enough finite measurements to constrain the solve. The selection is persisted as
  `litho_input_logs` in the well config and reported in the generated report.

* **Porosity method:** Alongside density and the neutron-density crossplot, choose
  Wyllie or Hunt-Raymer sonic porosity, or a neutron-density weighted
  $(2\phi_D+\phi_N)/3$, arithmetic $(\phi_D+\phi_N)/2$, or RMS
  $\sqrt{(\phi_D^2+\phi_N^2)/2}$ average. The library's Gaymard and gas
  averaging options use the same RMS formula. Averages use the selected fluid
  density and lithology-derived matrix density; they are empirical corrections,
  not substitutes for calibration to local lithology and core.
  Sonic methods require `DTC` (or `DT`) as well as the neutron and density logs
  required by the lithology workflow. Matrix transit time is the matrix-volume
  weighted sum of configured mineral DTC values; silt uses quartz. The fluid
  transit time is adjustable (default 190 µs/ft). Samples with missing or
  non-positive sonic transit time, fluid transit time not exceeding matrix transit
  time, or an invalid Hunt-Raymer result are null. Badhole interpolation may
  interpolate flagged samples; coal and GR-based badhole handling retain their
  existing special treatment. Effective porosity uses the existing shale correction.
  The new methods are not available in batch evaluation.

* **Porosity comparison:** Run Lithology & Porosity, then select **Porosity Methods
  vs Depth + Core** from the Crossplot dropdown. Toggle the methods to overlay
  their total-porosity curves with core markers; the active method is emphasized.
  Calculations use the current parameters and most recently calculated lithology and
  corrected logs, but never overwrite the working or saved PHIT. Run the module
  again after editing inputs. The fixed porosity scale is 0–0.5 v/v; use the mouse
  wheel to zoom depth. RMSE and bias are computed
  from depth-matched core porosity and each method's valid results, excluding flagged
  coal and badhole samples. Core is matched to a log sample within 0.5 m, and markers
  appear at the matched log depth. Different log/core scales, depth alignment and
  laboratory conditions can affect the comparison. Depth and zone filters apply;
  brushed calculation selections are not supported.

* **Vsh from gamma:** Select GR, an available gamma curve such as SGR or CGR,
  or estimated `CGR_SYN = a_Th × THOR + a_K × POTA`. Verify thorium and potassium
  units (ppm and percent) and coefficients against the tool calibration. The
  provisional coefficients (4 gAPI/ppm and 16 gAPI/%) are not universal.
  Calibrate endpoints independently for each response; imported GR aliases may
  already refer to CGR. The gamma index is a shale proxy, not a mineralogical
  clay measurement.
  Select the response in **Vshale Calibration** and adjust its clean and shale
  sliders there. The calibration comparison, auto-calibration, and P10/P90 reset
  use the selected response. GR-based log correction and badhole fallback retain
  their separate GR endpoints.

  Saved lithology fractions are matrix-relative and are not scaled by PHIT.
  Multi-mineral inversion still jointly solves mineral and fluid volumes, then
  reports mineral fractions relative to their sum. PHIT is computed separately;
  the well plot scales mineral fractions to `1 − PHIT` for visualization only.
  Recalculate previously saved lithology before comparing results or applying
  cutoffs, because earlier versions saved PHIT-scaled volumes.

* **Spectral crossplots:** The right pane also offers thorium vs potassium, PEF vs potassium,
  and PEF vs Th/K. These use `THOR` in ppm, `POTA` in percent and `PEF` in b/e;
  verify the input units before interpreting them. Potassium values are used as
  stored, including values below 0.5%; no magnitude-based conversion is applied.
  Curves stored as fractions must be converted to percent before use. The Th/K plot uses a logarithmic
  ratio axis and excludes non-positive K or Th. Ratios are unstable at low K.
  Th/K guide lines and the scatter are interpretive aids, not mineral classifications.
  Diamond markers are illustrative configured mineral response endpoints, not
  calibrated field boundaries. Selected Th and K curves are shared with the
  derived-CGR controls.
  PEF plots initially display 0–20 b/e and report the number of paired samples
  above that range; zoom out to inspect them. Values are not clipped or modified.
  The plots do not alter the GR-based shale calculation.

---

### Step 4: Permeability Estimation (`InspectorPerm`) {#sw-step-4-perm}

* **Objective:** Generate continuous absolute permeability ($k$) profiles calibrated against core measurements.

* **Technical Foundation:**

    * **Choo Model (Shaly Clastics):**
      
      $$k_{\text{Choo}} = A \cdot \frac{\phi_t^B}{10^{\left(C \cdot V_{\text{clay}} + D \cdot V_{\text{silt}}\right)}}$$
      
      where $A \approx 0.125 \cdot r_{\text{grain}}^2 \times 10^8$, $B \in [5.5, 6.0]$, and $C, D$ weight the surface area tortuosity penalties of clay and silt.

    * **Timur-Coates Free-Fluid Model:**
      
      $$k_{\text{Coates}} = C_{\text{TC}}^2 \cdot \phi_e^4 \cdot \left( \frac{1 - S_{\text{wirr}}}{S_{\text{wirr}}} \right)^2$$

    * **Tixier & Morris-Biggs Models:**
      
      $$k_{\text{Tixier}} = \left( 250 \cdot \frac{\phi_e^3}{S_{\text{wirr}}} \right)^2, \qquad k_{\text{gas}} = \left( 62.5 \cdot \frac{\phi_e^3}{S_{\text{wirr}}} \right)^2$$

    * **Amaefule Hydraulic Flow Unit (FZI / RQI) Transform:**
      
      $$\text{RQI} = 0.0314 \sqrt{\frac{k}{\phi_e}}, \qquad \phi_z = \frac{\phi_e}{1 - \phi_e}, \qquad \text{FZI} = \frac{\text{RQI}}{\phi_z}$$
      
      $$k = 1014 \cdot \text{FZI}^2 \cdot \frac{\phi_e^3}{(1 - \phi_e)^2}$$

* **Input Curves:** `PHIE`, `PHIT`, `VCLAY`, `VSILT`, `SWIRR`  
* **Output Curves:** `PERM`, `PERM_CHOO`, `PERM_COATES`, `RQI`, `FZI`

* **Permeability comparison:** Run Permeability, then select **Permeability Methods
  vs Depth + Core** from the Crossplot dropdown to overlay Choo, Kozeny-Carman, Timur,
  Tixier and Coates-Dumanoir curves against `CPERM` on a logarithmic
  permeability axis (fixed at 0.1–100,000 mD). Use the mouse wheel to zoom depth.
  Toggle methods and use the emphasized active method as a
  reference. The comparison uses the current model parameters, porosity and
  lithology inputs, including Thomas-Stieber inputs when enabled, without
  changing working or saved `PERM`. Only positive values are displayed or
  included in the matched-core metrics. RMSE and bias are in log10 permeability
  units; bias is log10(PERM) − log10(CPERM). Coal and badhole samples are
  excluded from metrics. Core is matched to the nearest log sample within
  0.5 m, with markers at matched log depths. Depth and zone filters apply;
  brushed calculation selections are not supported. Core/log scale, depth
  alignment and measurement conditions can influence the comparison.

---

### Step 5: Fluid Saturation Evaluation (`InspectorSaturation`) {#sw-step-5-saturation}

* **Objective:** Quantify Total Water Saturation ($S_{wt}$) and Effective Water Saturation ($S_{we}$) across clean sands, carbonates, and conductive shaly formations.

* **Technical Foundation:**

    * **Geothermal Gradient & Arps Brine Resistivity ($R_w(T)$):**
      
      $$T(z) = T_{\text{surf}} + g_T \cdot \text{TVD}(z), \qquad R_w(T) = R_w(T_{\text{ref}}) \cdot \frac{T_{\text{ref}} + 21.5}{T(z) + 21.5} \quad (^\circ\text{C})$$

    * **Archie Equation (Clean Sands & Carbonates):**
      
      $$S_w = \left( \frac{a \cdot R_w}{\phi^m \cdot R_t} \right)^{\frac{1}{n}}$$

    * **Waxman-Smits Shaly Sand Model:**
      Solves for excess electrical conduction via Cation Exchange Capacity ($Q_v$, in $\text{meq/mL}$) and counter-ion mobility ($B$, in $\text{mho}\cdot\text{m}^2/\text{meq}$):
      
      $$Q_v = \frac{\text{CEC} \cdot (1 - \phi_t) \cdot \rho_{\text{ma,clay}} \cdot V_{\text{clay}}}{\phi_t}$$
      
      $$\frac{1}{R_t} = \frac{\phi_t^{m^*} S_{wt}^{n^*}}{a^*} \left( \frac{1}{R_w} + \frac{B \cdot Q_v}{S_{wt}} \right)$$
      
      Solved via Newton-Raphson iteration for $S_{wt}$, followed by clay-bound water stripping:
      
      $$S_{we} = \text{clip}\left(\frac{S_{wt} \cdot \phi_t - V_{\text{clay}} \phi_{\text{clay}}}{\phi_e}, 0, 1\right)$$

    * **Dual Water & Indonesian Models:**
      
      $$\text{Dual Water: } C_t = \phi_t^m S_{wt}^n \left( C_{wf} + \frac{S_{wb}}{S_{wt}} (C_{wb} - C_{wf}) \right)$$
      
      $$\text{Indonesian: } \frac{1}{\sqrt{R_t}} = \left( \sqrt{\frac{V_{\text{sh}}^{2 - V_{\text{sh}}}}{R_{\text{sh}}}} + \sqrt{\frac{\phi_e^m}{a \cdot R_w}} \right) \cdot S_w^{n/2}$$

* **Input Curves:** `RT`, `PHIT`, `PHIE`, `VSH`, `VCLAY`, `TVD`  
* **Output Curves:** `SWT`, `SWE`, `BVW` ($\phi_e \cdot S_{we}$), `BVW_IRR`

---

### Step 6: Reservoir Summary & Log Blocking (`InspectorResSum`) {#sw-step-6-ressum}

* **Objective:** Apply petrophysical cutoffs to delineate Net Reservoir and Net Pay, compute zonal property averages, and upscale fine logs into coarse simulation layers.

* **Technical Foundation:**

    * **Rock & Pay Delineation:**
      
      $$\text{RES\_FLAG} = \mathbb{I}\left(V_{\text{sh}} \le V_{\text{sh,cut}} \;\land\; \phi_e \ge \phi_{e,\text{cut}} \;\land\; k \ge k_{\text{cut}}\right)$$
      
      $$\text{PAY\_FLAG} = \text{RES\_FLAG} \;\land\; \mathbb{I}\left(S_{we} \le S_{w,\text{cut}}\right)$$

    * **Zonal Integration:**
      
      $$H_{\text{gross}} = \sum_{i \in \text{Zone}} \Delta \text{TVD}_i, \qquad H_{\text{net}} = \sum_{i \in \text{Pay}} \Delta \text{TVD}_i, \qquad \text{NTG} = \frac{H_{\text{net}}}{H_{\text{gross}}}$$
      
      $$\overline{\phi_e} = \frac{\sum_{i \in \text{Pay}} \phi_{e,i} \Delta \text{TVD}_i}{H_{\text{net}}}, \qquad \overline{S_w} = \frac{\sum_{i \in \text{Pay}} S_{w,i} \phi_{e,i} \Delta \text{TVD}_i}{\sum_{i \in \text{Pay}} \phi_{e,i} \Delta \text{TVD}_i}$$
      
      $$\text{HCPV} = \sum_{i \in \text{Pay}} \phi_{e,i} (1 - S_{w,i}) \Delta \text{TVD}_i$$

    * **Log Blocking / Upscaling Averaging Laws:**
      
      $$\text{Arithmetic (Porosity, Volume): } \bar{x}_A = \frac{1}{\Delta z}\int_{z_a}^{z_b} x(z) \, dz$$
      
      $$\text{Geometric (Heterogeneous Horizontal Perm): } \bar{k}_G = \exp\left( \frac{1}{\Delta z}\int_{z_a}^{z_b} \ln k(z) \, dz \right)$$
      
      $$\text{Harmonic (Layered Vertical Perm): } \bar{k}_H = \left( \frac{1}{\Delta z}\int_{z_a}^{z_b} \frac{1}{k(z)} \, dz \right)^{-1}$$

* **Input Data:** `VSH`, `PHIE`, `SWE`, `PERM`, `TVD`, Stratigraphic Tops  
* **Output Data:** `RES_FLAG`, `PAY_FLAG`, Zonal summary table, Blocked curves (`PHIE_BLK`, `PERM_BLK`, `SWE_BLK`)

---

### Step 7: Well CPI Report & Dossier Generation (`InspectorReport`) {#sw-step-7-report}

* **Objective:** Assemble evaluated curves, cutoffs, zonal summaries, composite log plots, and crossplots into a Computer Processed Interpretation (CPI) report in Markdown, PDF, and interactive HTML.

---

## 3. Standalone Lab Module: Core & Pc Prep (`InspectorScal`) {#sw-standalone-core-pc}

> [!NOTE]
> **Architecture Note:** In `quick_pp`, dynamic multiphase **Relative Permeability modeling and ECLIPSE `SWOF` deck generation have been consolidated into the Multi-Well Space** ([MultiWellSwirr](multi_well.md#mw-standalone-scal-relperm)), where rock-type partitioned $S_{\text{wirr}}$ fits and Corey/LET functions are calibrated consistently across the entire field. The Single-Well `InspectorScal` module serves as the **dedicated laboratory conditioning workbench** for core depth alignment, stress corrections, and lab capillary pressure scaling.

* **Objective:** Condition Routine Core Analysis ($\text{RCA}$) ground-truth samples, apply overburden stress and gas slippage corrections, and scale laboratory mercury injection / porous plate capillary pressure measurements to in-situ reservoir conditions.

* **Technical Foundation:**

    * **Interactive Core-to-Log Dynamic Depth Shifting:**
      Aligns core plugs ($\phi_{\text{core}}, k_{\text{core}}$) to wireline logs ($\rho_b, \text{GR}$) by applying depth warping shifts $\Delta z(\text{MD})$:
      
      $$\text{MD}_{\text{shifted}} = \text{MD}_{\text{core}} + \Delta z(\text{MD})$$

    * **Overburden Confining Stress Corrections:**
      Converts ambient laboratory porosity ($\phi_{\text{amb}}$) and permeability ($k_{\text{amb}}$) to effective reservoir confining stress $\sigma_{\text{net}}$:
      
      $$\phi_{\text{res}} = \phi_{\text{amb}} \cdot (1 - C_\phi \cdot \sigma_{\text{net}})$$
      
      $$k_{\text{res}} = k_{\text{amb}} \cdot \exp(-C_k \cdot \sigma_{\text{net}})$$
      
      where $C_\phi$ and $C_k$ are rock-specific compaction coefficients (sandstone, friable sandstone, carbonate).

      **Log-derived option:** Integrates bulk density over positive-down TVD to calculate vertical stress, estimates pore pressure from a user-selected gradient, and applies Biot's coefficient. When both DTC and DTS are available, dynamic Poisson's ratio is calculated from their velocity ratio; otherwise a configured fallback is used. Equivalent effective confining stress is the mean of vertical and two uniaxial-strain horizontal effective stresses. The profile assumes density at the shallowest valid sample continues to the depth datum; measured depth is only a TVD approximation when no TVD curve is available. It does not infer pore pressure from sonic compaction trends.

    * **Klinkenberg Gas Slippage Correction:**
      Corrects air permeability ($k_{\text{air}}$) to equivalent liquid permeability ($k_{\infty}$):
      
      $$k_{\infty} = \frac{k_{\text{air}}}{1 + \frac{b_K}{P_{\text{mean}}}}$$
      
      using either the Jones or Heithaus formulation.

    * **Lab-to-Reservoir Capillary Pressure ($P_c$) Scaling:**
      Rescales laboratory air-mercury or air-brine capillary pressures ($P_{c,\text{lab}}$) to reservoir fluid fluid system conditions ($P_{c,\text{res}}$):
      
      $$P_{c,\text{res}} = P_{c,\text{lab}} \cdot \frac{(\sigma \cos\theta)_{\text{res}}}{(\sigma \cos\theta)_{\text{lab}}}$$

* **Input Data:** Core plugs table (`DEPTH`, `CPORE`, `CPERM`), Overburden stress parameters, Lab IFT and contact angle  
* **Output Data:** `CPOR_SHIFTED`, `CPERM_SHIFTED`, `SV_EST`, `PP_EST`, `NU_DYN`, `NCS_EST`, Stress-corrected core properties, Scaled $P_c$ curves
