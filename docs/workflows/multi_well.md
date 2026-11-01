# Multi-Well Field Evaluation Workflow

The Multi-Well space harmonizes multi-vintage logging suites across the entire field, establishes consistent fluid contacts from Formation Pressure Tests ($\text{FPT}$), propagates core-calibrated Petrophysical Rock Types and permeability to uncored wells via Physics-Informed Machine Learning ($\text{PIML}$), models field-wide Saturation-Height Functions ($\text{SHF}$), and runs Monte Carlo volumetric uncertainty.

---

## 1. Workflow Architecture (Level 1 & Level 2) {#mw-level-1-2-diagrams}

### Level 1: Macro Workflow

High-level field evaluation progression from multi-well ingestion to probabilistic STOIIP/GIIP reserves.

```mermaid
flowchart LR
    M1["1. Field Inventory\n(Multi-LAS, Surveys, Tops)"] --> M2["2. Field QA/QC\n(Stratigraphic Normalization)"]
    M2 --> M3["3. FPT & Contacts\n(Gradients, FWL, GOC, OWC)"]
    M2 --> M4["4. Field Rock Typing\n(FZI, Winland, Lucia)"]
    M4 --> M5["5. PIML & RT Propagation\n(Classifier & Conformal Bounds)"]
    M5 --> M6["6. Field Perm & Swirr\n(RT Transforms & Swirr Model)"]
    M3 & M6 --> M7["7. Saturation-Height (SHF)\n(Leverett J, Skelt-Harrison, FWL)"]
    M7 --> M8["8. Probabilistic Volumetrics\n(Monte Carlo LHS STOIIP/GIIP)"]
    M8 --> M9["9. Field FDP Report\n(Field Synthesis Dossier)"]

    style M1 fill:#f8fafc,stroke:#475569,stroke-width:1.5px
    style M2 fill:#f0f9ff,stroke:#0284c7,stroke-width:1.5px
    style M3 fill:#fff1f2,stroke:#e11d48,stroke-width:1.5px
    style M4 fill:#fefce8,stroke:#ca8a04,stroke-width:1.5px
    style M5 fill:#faf5ff,stroke:#9333ea,stroke-width:1.5px
    style M6 fill:#f0fdf4,stroke:#16a34a,stroke-width:1.5px
    style M7 fill:#fff7ed,stroke:#ea580c,stroke-width:1.5px
    style M8 fill:#fef3c7,stroke:#d97706,stroke-width:1.5px
    style M9 fill:#eff6ff,stroke:#2563eb,stroke-width:2px
```

### Level 2: Sub-Process, Training Loops & Reconciliation Architecture

Detailed architecture showing how cored "Key Wells" anchor field-wide models, how pressure surveys define fluid contacts, and how capillary pressure reconciles with resistivity log saturation.

```mermaid
flowchart TD
    subgraph MW1["1. Multi-Well Inventory & Standardization (MultiWellInventory)"]
        WELLS["Multi-Well LAS Catalog\n(Cored Key Wells + Uncored Field Wells)"]
        ALIAS["Mnemonic Aliasing & Canonical Mapping\n(GR, RHOB, NPHI, RT, DTC, CALI, PEF)"]
        CORE_DB["Field RCA Core Database\n(Porosity, Permeability, Grain Density)"]
        PRESS_DB["FPT / MDT / RFT Pressure Database"]
        WELLS --> ALIAS
    end

    subgraph MW2["2. Field-Wide Stratigraphic Normalization (MultiWellQaqc)"]
        ALIAS --> MARKER["Select Regional Stratigraphic Marker\n(Marine Shale Anchor / Regional Aquifer)"]
        MARKER --> HIST["Histogram & Multi-Well Quantile Audit\n(Identify Calibration Drift & Tool Vintage Shifts)"]
        HIST --> NORM["Two-Point Quantile Rescaling\nX_norm = P_low^ref + (X - P_low) / (P_high - P_low) * ΔP^ref"]
    end

    subgraph MW3["3. Formation Pressure Testing (MultiWellFpt)"]
        PRESS_DB --> REGRESS["Phase Linear Regressions: P vs TVDSS\n(Gas, Oil, Water Gradients dp/dz)"]
        REGRESS --> EXCESS["Excess Pressure Analysis\nΔP = P - (0.4335 · ρ_w · z)"]
        REGRESS --> DENSITY["In-Situ Fluid Densities (ρ_g, ρ_o, ρ_w)\nρ = m / 0.433527"]
        REGRESS --> CONTACTS["Contact Intersections & Confidence Intervals\nFWL, GOC, OWC, GWC"]
    end

    subgraph MW4["4. Core Rock Typing (MultiWellRockTyping)"]
        CORE_DB --> RT_SELECT{"Rock Typing Scheme"}
        RT_SELECT -->|Clastics / Universal| FZI_FLOW["FZI / RQI Log-Normal CDF\n& Ward Hierarchical Clustering"]
        RT_SELECT -->|Pore-Throat Contours| WINLAND_FLOW["Winland r35 / Pittman\nPore-Throat Radii"]
        RT_SELECT -->|Carbonates| LUCIA_FLOW["Lucia Rock Fabric Number (RFN)\nInterparticle Porosity Classes"]
        FZI_FLOW & WINLAND_FLOW & LUCIA_FLOW --> CORE_RTS["Discrete Petrophysical Rock Types\n(RT_1 ... RT_K)"]
    end

    subgraph MW5["5. Supervised PIML & Conformal RT Propagation (MultiWellRtClassifierTrainer / MlTrainer)"]
        NORM --> ENG_FEAT["Feature Engineering\n(Normalized Logs, Ratios, Rolling Statistics)"]
        CORE_RTS & ENG_FEAT --> TRAIN_CLF["Rock Type Classifier (RF / XGBoost)"]
        TRAIN_CLF --> PROP_RTS["Propagated Rock Types on All Wells\n(RT_PRED with Class Probabilities)"]
        ENG_FEAT & CORE_DB --> TRAIN_REG["Continuous PIML Regressors\n(Physics Penalties & Monotonic Constraints)"]
        TRAIN_REG --> CONFORMAL["Split Conformal Prediction Engine\nDistribution-Free [y_low, y_high] Bounds"]
    end

    subgraph MW6["6. Field Permeability & Swirr Transforms (MultiWellPermTransform / MultiWellSwirr)"]
        PROP_RTS & CORE_DB --> RT_PERM["Rock-Type Partitioned Poro-Perm\nlog10(k) = a_k · φ_e + b_k"]
        RT_PERM & CONFORMAL --> FIELD_PERM["Continuous Permeability Across Field"]
        FIELD_PERM & PROP_RTS --> SWIRR_ENGINE["Continuous Irreducible Saturation\nSwirr = f(φ_e, k, FZI)"]
    end

    subgraph MW7["7. Saturation-Height Function (SHF) & Reconciliation (MultiWellShf)"]
        CONTACTS & DENSITY --> H_CAP["Height Above FWL: h = z_FWL - TVDSS\nCapillary Pressure Pc = Δρ · g · h"]
        H_CAP & FIELD_PERM & SWIRR_ENGINE --> SHF_CHOICE{"SHF Formulation"}
        SHF_CHOICE -->|Dimensionless Pc| LEV_J["Leverett J-Function: J = A_J · (Sw*)^(-B_J)"]
        SHF_CHOICE -->|Analytical Smooth| SKELT_H["Skelt-Harrison / Cuddy Model"]
        LEV_J & SKELT_H --> SW_SHF_CURVE["Capillary Saturation Curve (SW_SHF)"]
        SW_SHF_CURVE <-->|Crossplot & RMSE Audit| SW_RESISTIVITY["Resistivity Log Saturation (SW_LOG)"]
    end

    subgraph MW8["8. Probabilistic Volumetrics & Uncertainty (MultiWellVolumetrics)"]
        SW_SHF_CURVE & SW_RESISTIVITY --> ZONAL_FIELD["Multi-Well Zonal Net Pay Summaries"]
        ZONAL_FIELD & CONFORMAL --> LHS_ENGINE["Monte Carlo / Latin Hypercube Engine\n(Iman-Conover Parameter Correlation)"]
        LHS_ENGINE --> RESERVES["Field Reserves (P90 / P50 / P10)\nSTOIIP, GIIP, Recoverable Volumes"]
        LHS_ENGINE --> TORNADO["Tornado Sensitivity Ranking\n(Spearman Rank Correlation ρ_s)"]
    end

    subgraph MW9["9. Comprehensive Field Report (InspectorReport)"]
        RESERVES & TORNADO & CONTACTS --> FDP_DOC["Automated Field Development Plan (FDP)\nSynthesis Dossier (MD / PDF)"]
    end
```

---

## 2. Step-by-Step Technical Foundations

### Step 1: Multi-Well Inventory & Mnemonic Standardization (`MultiWellInventory`) {#mw-step-1-inventory}

* **Objective:** Ingest heterogeneous well databases, resolve vendor curve mnemonics into canonical families, harmonize physical units, and perform spatial audit of survey, core, top, and pressure tables.

* **Technical Foundation:**
  Enforces deterministic mapping into `quick_pp` canonical curve families:
  
  $$\text{Gamma Ray: } \{\text{GR, CGR, SGR, RGR, GRR}\} \to \text{GR (gAPI)}$$
  
  $$\text{Bulk Density: } \{\text{RHOB, RHOZ, ZDEN, DEN, RHO8}\} \to \text{RHOB (g/cm}^3)$$
  
  $$\text{Neutron Porosity: } \{\text{NPHI, TNPH, NPOR, CNC, NEU}\} \to \text{NPHI (dec or v/v)}$$
  
  $$\text{Deep Resistivity: } \{\text{RT, RD, ILD, RDEP, AT90, HRLD}\} \to \text{RT (ohm}\cdot\text{m)}$$
  
  $$\text{Compressional Sonic: } \{\text{DTC, DT, DTCO, AC, ITT}\} \to \text{DTC (}\mu\text{s/ft)}$$

---

### Step 2: Field-Wide QA/QC & Stratigraphic Normalization (`MultiWellQaqc`) {#mw-step-2-qaqc}

* **Objective:** Eliminate tool calibration drift, mud salinity variations, and service vendor biases across multi-decade drilling campaigns prior to field-wide rock typing and ML modeling.

* **Technical Foundation:**

    * **Key-Well Reference Marker Distribution:**
      A cored, high-integrity reference well ("Key Well") is selected within a regionally extensive, lithologically stable stratigraphic marker zone (e.g. marine condensed section shale or regional clean aquifer sand) to establish target percentiles $(P_{\text{low}}^{\text{ref}}, P_{\text{high}}^{\text{ref}})$, typically $(P_{05}, P_{95})$ or $(P_{10}, P_{90})$.

    * **Two-Point Quantile Linear Rescaling:**
      For each subject well $w$ with marker percentiles $(P_{\text{low}}^{(w)}, P_{\text{high}}^{(w)})$:
      
      $$X_{w,\text{norm}} = P_{\text{low}}^{\text{ref}} + \left( \frac{X_w - P_{\text{low}}^{(w)}}{P_{\text{high}}^{(w)} - P_{\text{low}}^{(w)}} \right) \cdot \left( P_{\text{high}}^{\text{ref}} - P_{\text{low}}^{\text{ref}} \right)$$

* **Input Data:** Multi-well raw curves, Reference key-well selection, Marker top/base boundaries  
* **Output Data:** `GR_NORM`, `NPHI_NORM`, `RHOB_NORM`, Quantile diagnostic histograms

---

### Step 3: Formation Pressure Testing (FPT) & Fluid Contacts (`MultiWellFpt`) {#mw-step-3-fpt}

* **Objective:** Determine in-situ fluid phase pressure gradients, calculate in-situ fluid densities ($\rho_g, \rho_o, \rho_w$), identify hydraulic communication across fault blocks, and calculate the Free Water Level ($\text{FWL}$), Gas-Oil Contact ($\text{GOC}$), and producible contacts ($\text{OWC}, \text{GWC}$) with uncertainty bounds.

* **Technical Foundation:**

    * **Hydrostatic Phase Regression:**
      In each mobile phase $p \in \{\text{gas, oil, water}\}$, pressure $P$ varies linearly with depth $z = \text{TVDSS}$:
      
      $$P_p(z) = m_p \cdot z + c_p$$
      
      where the gradient $m_p = \frac{dP_p}{dz}$ (in $\text{psi/ft}$) is converted to in-situ fluid density $\rho_p$ (in $\text{g/cm}^3$):
      
      $$\rho_p = \frac{m_p}{0.433527}$$

    * **Diagnostic Excess Pressure:**
      Identifies pressure regime anomalies (supercharging, drawdown, depleted reservoirs):
      
      $$P_{\text{excess}}(z) = P(z) - \left( 0.433527 \cdot \rho_{\text{water,datum}} \cdot z \right)$$

    * **Equilibrium Fluid Contact Intersection Mathematics:**
      Phase contacts occur at the exact intersection of phase lines ($P_c = P_{\text{non-wet}} - P_{\text{wet}} = 0$):
      
      $$\text{FWL (Oil-Water): } z_{\text{FWL}} = \frac{c_w - c_o}{m_o - m_w}$$
      
      $$\text{GOC (Gas-Oil): } z_{\text{GOC}} = \frac{c_o - c_g}{m_g - m_o}$$
      
      $$\text{GWC (Gas-Water): } z_{\text{GWC}} = \frac{c_w - c_g}{m_g - m_w}$$

    * **Contact Uncertainty Propagation:**
      Using the covariance matrix $\boldsymbol{\Sigma}_{(m, c)}$ from least-squares regression, the contact depth variance is:
      
      $$\sigma_{z_{\text{contact}}}^2 = \mathbf{J} \cdot \boldsymbol{\Sigma} \cdot \mathbf{J}^T \quad \text{where } \mathbf{J} = \left[ \frac{\partial z}{\partial m_1}, \frac{\partial z}{\partial c_1}, \frac{\partial z}{\partial m_2}, \frac{\partial z}{\partial c_2} \right]$$

    * **Displacement Threshold Offset (OWC vs FWL):**
      In rocks with non-zero entry capillary pressure $P_d$, the producible Oil-Water Contact ($\text{OWC}$) sits above the $\text{FWL}$ by:
      
      $$z_{\text{OWC}} = z_{\text{FWL}} - \frac{P_{d,\text{res}}}{0.433527 \cdot (\rho_w - \rho_o)}$$

* **Input Data:** MDT / RFT / RCI pressure survey table $(\text{TVDSS}, P_{\text{form}}, \text{QualityFlag})$  
* **Output Data:** Fluid gradients ($m_g, m_o, m_w$), In-situ densities ($\rho_g, \rho_o, \rho_w$), Contact depths ($z_{\text{FWL}}, z_{\text{GOC}}, z_{\text{OWC}}$) $\pm \sigma$

---

### Step 4: Field Rock Typing (`MultiWellRockTyping`) {#mw-step-4-rock-typing}

* **Objective:** Partition core plug measurements into discrete Petrophysical Rock Types ($RT_1, \dots, RT_K$) that share consistent pore-throat size distributions, flow geometry, and capillary behavior.

* **Technical Foundation:**

    * **Hydraulic Flow Units (Amaefule FZI):**
      
      $$\text{RQI} = 0.0314 \sqrt{\frac{k}{\phi_e}}, \qquad \phi_z = \frac{\phi_e}{1 - \phi_e}, \qquad \text{FZI} = \frac{\text{RQI}}{\phi_z}$$
      
      Rock types appear as distinct normal distributions on probability plots of $\log_{10}(\text{FZI})$ or clusters identified via Ward's hierarchical agglomerative clustering.

    * **Winland $r_{35}$ & Pittman Pore-Throat Radii:**
      Pore-throat radius at $35\%$ mercury saturation (apical pore opening):
      
      $$\log_{10}(r_{35}) = 0.732 + 0.588 \log_{10}(k) - 0.864 \log_{10}(\phi_{\%})$$
      
      * **Megaporous:** $r_{35} > 10\,\mu\text{m}$
      * **Macroporous:** $r_{35} \in [2, 10]\,\mu\text{m}$
      * **Mesoporous:** $r_{35} \in [0.5, 2]\,\mu\text{m}$
      * **Microporous:** $r_{35} \in [0.1, 0.5]\,\mu\text{m}$
      * **Nanoporous:** $r_{35} < 0.1\,\mu\text{m}$

    * **Lucia Rock Fabric Number (RFN, Carbonates):**
      Links interparticle porosity ($\phi_{ip}$) and permeability to grain-dominated vs mud-dominated fabrics:
      
      $$\log_{10}(k) = (9.7982 - 12.0838 \log_{10} \text{RFN}) + (8.6711 - 8.2965 \log_{10} \text{RFN}) \log_{10}(\phi_{ip})$$

    * **Modified Lorenz Plot ($F_m$ vs $\Phi_m$):**
      Plots cumulative flow capacity $F_m = \frac{\sum (k \cdot h)_i}{\sum k \cdot h}$ against cumulative storage capacity $\Phi_m = \frac{\sum (\phi \cdot h)_i}{\sum \phi \cdot h}$ to diagnose reservoir heterogeneity and flow units.

* **Input Data:** Core porosity ($\phi_{\text{core}}$), Permeability ($k_{\text{core}}$), Grain density  
* **Output Data:** Core rock type labels ($RT_1, \dots, RT_K$), FZI cutoffs, Winland classes

---

### Step 5: Supervised PIML & Conformal RT Propagation (`MultiWellRtClassifierTrainer`, `MultiWellMlTrainer`) {#mw-step-5-ml-training}

* **Objective:** Train supervised classifiers and regressors on cored wells to propagate discrete Rock Types ($RT$) and continuous petrophysical properties ($\phi_e, k, S_w$) to uncored wells with guaranteed uncertainty bounds.

* **Technical Foundation:**

    * **Domain-Specific Feature Engineering:**
      Constructs multi-scale features from normalized logs:
      * Apparent acoustic impedance surrogate: $\text{AI} = \text{RHOB\_NORM} \cdot (10^6 / \text{DTC})$
      * Neutron-density separation: $\Delta \phi_{ND} = \text{NPHI\_NORM} - \phi_D$
      * Multi-scale rolling standard deviations and gradients ($\pm 1\,\text{m}, \pm 3\,\text{m}$).

    * **Physics-Informed Machine Learning (PIML) Regularization:**
      Enforces physical laws during tree boosting (XGBoost / LightGBM) or neural training:
      * Enforces strict monotonicity: $\frac{\partial k}{\partial \phi_e} \ge 0$, $\frac{\partial k}{\partial V_{\text{clay}}} \le 0$
      * Bounded predictions: $\phi_e \in [0, \phi_{\max}]$, $S_w \in [S_{\text{wirr}}, 1.0]$.

    * **Split Conformal Prediction (Distribution-Free Coverage Guarantee):**
      Given calibration samples $\mathcal{D}_{\text{cal}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, computes absolute non-conformity scores:
      
      $$s_i = |y_i - \hat{f}(\mathbf{x}_i)|$$
      
      For target error rate $\alpha$ (e.g. $\alpha = 0.10$ for $90\%$ confidence), the empirical quantile $\hat{q}_{1-\alpha}$ defines prediction intervals on uncored wells:
      
      $$\hat{q}_{1-\alpha} = \text{Quantile}\left( \{s_i\}_{i=1}^n, \;\; \frac{\lceil(n+1)(1-\alpha)\rceil}{n} \right)$$
      
      $$\mathcal{C}(\mathbf{x}_{\text{new}}) = \left[ \hat{f}(\mathbf{x}_{\text{new}}) - \hat{q}_{1-\alpha}, \;\; \hat{f}(\mathbf{x}_{\text{new}}) + \hat{q}_{1-\alpha} \right]$$
      
      Guarantees marginal coverage $\mathbb{P}\left(Y_{\text{new}} \in \mathcal{C}(\mathbf{X}_{\text{new}})\right) \ge 1 - \alpha$ without parametric Gaussian assumptions.

* **Input Data:** Normalized logs on all wells, Cored well training labels  
* **Output Data:** `RT_PRED`, Class probabilities, Continuous predictions with Conformal $[\text{P10}, \text{P90}]$ bounds

---

### Step 6: Field Permeability Transforms & Irreducible Water Saturation (`MultiWellPermTransform`, `MultiWellSwirr`) {#mw-step-6-perm-swirr}

* **Objective:** Establish Rock-Type-specific porosity-permeability transforms and generate continuous Irreducible Water Saturation ($S_{\text{wirr}}$) profiles across all field wells.

* **Technical Foundation:**

    * **Rock-Type Partitioned Poro-Perm Regressions:**
      Within each propagated Rock Type $RT_k$:
      
      $$\log_{10}(k) = a_k \cdot \phi_e + b_k \quad \lor \quad k = 1014 \cdot \overline{\text{FZI}}_k^2 \cdot \frac{\phi_e^3}{(1 - \phi_e)^2}$$

    * **Continuous Irreducible Saturation ($S_{\text{wirr}}$):**
      Models high-capillary-pressure asymptotic water saturation as an inverse function of rock quality:
      
      $$S_{\text{wirr}} = \text{clip}\left( c_1 \cdot \phi_e^{c_2} \cdot k^{c_3}, \;\; S_{\text{wirr,min}}, \;\; 1.0 \right) \quad \lor \quad S_{\text{wirr}} = 1 - \frac{1}{a_{\text{FZI}} + b_{\text{FZI}} \cdot \text{FZI}^{-c_{\text{FZI}}}}$$

* **Input Curves:** `RT_PRED`, `PHIE`  
* **Output Curves:** `PERM_FIELD`, `SWIRR`

---

### Step 7: Saturation-Height Modelling (SHF) & Reconciliation (`MultiWellShf`) {#mw-step-7-shf}

* **Objective:** Construct a capillary-equilibrium water saturation profile $S_{w,\text{SHF}}(h)$ above the Free Water Level ($\text{FWL}$) and reconcile it with resistivity log saturation ($S_{w,\text{log}}$).

* **Technical Foundation:**

    * **Height Above FWL & Capillary Pressure:**
      
      $$h = \max\left(0, \;\; z_{\text{FWL}} - \text{TVDSS}\right)$$
      
      $$P_{c,\text{res}}(h) = 0.433527 \cdot (\rho_w - \rho_{\text{hc}}) \cdot h \quad (\text{psi})$$

    * **Leverett $J$-Function Model:**
      Collapses multi-rock-type capillary pressure curves into a dimensionless function:
      
      $$J(S_w) = \frac{0.2166 \cdot P_{c,\text{res}}}{\sigma_{\text{res}} \cos\theta_{\text{res}}} \sqrt{\frac{k}{\phi_e}}$$
      
      Fitting the power-law $J = A_J \cdot (S_w^*)^{-B_J}$ (where $S_w^* = \frac{S_w - S_{\text{wirr}}}{1 - S_{\text{wirr}}}$) yields saturation at any elevation:
      
      $$S_{w,\text{SHF}}(h, k, \phi_e) = S_{\text{wirr}} + (1 - S_{\text{wirr}}) \cdot \left( \frac{J(h, k, \phi_e)}{A_J} \right)^{-\frac{1}{B_J}}$$

    * **Skelt-Harrison Analytical Model:**
      Provides a smooth 4-parameter continuous saturation profile:
      
      $$S_{w,\text{SHF}}(h) = 1 - A \cdot \exp\left( - \left( \frac{B}{h + D} \right)^C \right)$$

    * **FWL Inverse Optimization:**
      When pressure survey points are ambiguous, $z_{\text{FWL}}$ and SHF parameters are inverted simultaneously by minimizing the residual against deep resistivity log saturation across all wells:
      
      $$\min_{z_{\text{FWL}}, \mathbf{\theta}_{\text{SHF}}} \sum_{w=1}^{N_w} \sum_{i=1}^{M_w} \left( S_{w,\text{SHF}}(z_{w,i}; z_{\text{FWL}}, \mathbf{\theta}) - S_{w,\text{log},i} \right)^2$$

* **Input Data:** `TVDSS`, `PERM_FIELD`, `PHIE`, $z_{\text{FWL}}$, In-situ fluid densities  
* **Output Data:** `SW_SHF`, Reconciliation crossplots, Residual error statistics ($\text{RMSE}, R^2$)

---

### Step 8: Probabilistic Field Volumetrics & Uncertainty (`MultiWellVolumetrics`) {#mw-step-8-volumetrics}

* **Objective:** Combine multi-well net-pay distributions, structural geometry, and fluid properties to evaluate P90 (1P/Proven), P50 (2P), and P10 (3P) Hydrocarbons Initially In Place (STOIIP / GIIP) and identify primary subsurface risks.

* **Technical Foundation:**

    * **Governing Volumetric Equations:**
      
      $$\text{STOIIP (bbl)} = \frac{7758 \cdot A \cdot H_{\text{gross}} \cdot \text{NTG} \cdot \phi_e \cdot (1 - S_w)}{B_o}$$
      
      $$\text{GIIP (scf)} = \frac{43560 \cdot A \cdot H_{\text{gross}} \cdot \text{NTG} \cdot \phi_e \cdot (1 - S_w)}{B_g}$$
      
      $$\text{Metric (m}^3\text{): } V = \frac{10^6 \cdot A_{\text{km}^2} \cdot H_{\text{net}} \cdot \phi_e \cdot (1 - S_w)}{B}$$

    * **Monte Carlo / Latin Hypercube Sampling (LHS):**
      Draws $N_{\text{sim}} \ge 10,000$ stratifying samples across distributions (Normal, Log-Normal, Triangular, PERT) for Area, Gross Thickness, NTG, $\phi_e$, $S_w$, and $B_o/B_g$.

    * **Iman-Conover Rank Correlation Induction:**
      Prevents unphysical realizations by enforcing the empirical rank correlation matrix $\mathbf{C}$ (e.g. negative correlation between $\phi_e$ and $S_w$) on the sampled matrix:
      
      $$\mathbf{X}^* = \mathbf{X} \cdot \mathbf{L}^T \quad \text{where } \mathbf{C} = \mathbf{L}\mathbf{L}^T$$

    * **PRMS Oilfield Percentiles & Sensitivity Tornado:**
      Extracts cumulative exceedance percentiles:
      * **P90 (Proven / 1P):** $90\%$ probability that actual volume equals or exceeds this estimate ($Q_{0.10}$)
      * **P50 (Probable / 2P):** Median expectation ($Q_{0.50}$)
      * **P10 (Possible / 3P):** $10\%$ upside expectation ($Q_{0.90}$)
      
      Computes Spearman rank correlation coefficients $\rho_s(\theta_j, \text{STOIIP})$ to generate the Tornado chart ranking uncertainty drivers.

* **Input Data:** Zonal summaries across all wells, Structural area distributions, Fluid PVT ($B_o, B_g$)  
* **Output Data:** P90/P50/P10 STOIIP & GIIP, Cumulative exceedance curves, Tornado sensitivity chart

---

### Step 9: Comprehensive Field Report (`InspectorReport` / Multi-Well Dossier) {#mw-step-9-report}

* **Objective:** Synthesize multi-well data inventory, stratigraphic normalization results, pressure gradients, rock typing distributions, SHF models, and probabilistic volumetric reserve estimates into an executive Field Development Plan (FDP) report.

* **Technical Foundation:**
  Automated multi-well dossier compilation in Markdown and PDF featuring executive summaries, zonal net pay rollups, fluid contact determinations, and volumetric risk distributions.

