# Domain Workflows & Technical Foundations

`quick_pp` provides a dual-space architecture that bridges single-well deterministic petrophysics with field-scale multi-well reservoir characterization. Both spaces distinguish between **Primary Sequential Workflow Steps** (numbered 1 through 7) and **Standalone Specialized Modules**.

---

## Architectural Overview

```mermaid
flowchart TD
    subgraph SingleWellSpace["Single-Well Space (Trajectory / Log Scale)"]
        direction TB
        subgraph SW_Seq["Primary Sequential Workflow (Steps 1–7)"]
            SW1["1. Data & QA/QC"] --> SW2["2. Tops & Ancillary"]
            SW2 --> SW3["3. Lithology & Porosity"]
            SW3 --> SW4["4. Permeability"]
            SW4 --> SW5["5. Saturation"]
            SW5 --> SW6["6. Reservoir Summary"]
            SW6 --> SW7["7. Well CPI Report"]
        end
        subgraph SW_Stand["Standalone Lab Module"]
            SW_LAB["Core & Pc Prep\n(Depth Shift, Stress Corrections, Lab Pc)"]
        end
        SW_LAB -.->|Calibrated Core & Pc| SW3
        SW_LAB -.->|Calibrated Core Perm| SW4
    end

    subgraph MultiWellSpace["Multi-Well Space (Field Scale)"]
        direction TB
        subgraph MW_Seq["Primary Sequential Workflow (Steps 1–7)"]
            MW1["1. Field Inventory"] --> MW2["2. Data & Field QA/QC (Norm)"]
            MW2 --> MW3["3. Rock Typing (FZI)"]
            MW3 --> MW4["4. Perm Transform"]
            MW4 --> MW5["5. Saturation Height (SHF)"]
            MW5 --> MW6["6. Volumetric Assessment"]
            MW6 --> MW7["7. FDP Report"]
        end
        subgraph MW_Stand["Standalone Specialized Modules"]
            MW_SCAL["SCAL & Rel-Perm\n(Swirr, Corey, LET, SWOF Deck)"]
            MW_FPT["Pressure & Contacts\n(MDT/RFT Gradients, FWL/OWC/GOC)"]
            MW_RT["RT Classifier\n(Discrete Rock Type Predictor)"]
            MW_PIML["Conformal ML Trainer\n(P10–P90 Physics ML Bounds)"]
        end
        MW_FPT -.->|FWL & Fluid Densities| MW5
        MW_SCAL -.->|Swirr & Capillary Parameters| MW5
        MW_RT -.->|Predicted RTs| MW4
        MW_PIML -.->|Conformal Bounds| MW6
    end

    SingleWellSpace -.->|Calibrated Key Wells & Parameters| MultiWellSpace
    MultiWellSpace -.->|Field Normalization & Fluid Contacts| SingleWellSpace

    style SingleWellSpace fill:#f8fafc,stroke:#0284c7,stroke-width:2px
    style MultiWellSpace fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
    style SW_Seq fill:#ffffff,stroke:#0284c7,stroke-width:1px
    style SW_Stand fill:#f0f9ff,stroke:#0284c7,stroke-width:1px,stroke-dasharray: 4 4
    style MW_Seq fill:#ffffff,stroke:#16a34a,stroke-width:1px
    style MW_Stand fill:#f0fdf4,stroke:#16a34a,stroke-width:1px,stroke-dasharray: 4 4
```

---

## Workflow Step Mapping Matrix

The tables below define the 1-to-1 correspondence between the SvelteKit workbench UI (`WorkflowStepper.svelte`), backend modules, and technical documentation anchors.

### Single-Well Space

| Step Category | Stepper ID | UI Component | Markdown Anchor | Primary Petrophysical Function |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1** | `qaqc` | `InspectorQaqc.svelte` | [`#sw-step-1-qaqc`](single_well.md#sw-step-1-qaqc) | Outer limits, bad-hole flagging & curve repair |
| **Step 2** | `ancillary` | `InspectorAncillary.svelte` | [`#sw-step-2-ancillary`](single_well.md#sw-step-2-ancillary) | Minimum curvature trajectory & tops ingestion |
| **Step 3** | `litho-poro` | `InspectorLithoPoro.svelte` | [`#sw-step-3-litho-poro`](single_well.md#sw-step-3-litho-poro) | $V_{\text{sh}}$, SSC ternary, multi-mineral SLSQP, $\phi_e$ & $\phi_t$ |
| **Step 4** | `perm` | `InspectorPerm.svelte` | [`#sw-step-4-perm`](single_well.md#sw-step-4-perm) | Choo, Timur-Coates, Tixier & Morris-Biggs permeability |
| **Step 5** | `saturation` | `InspectorSaturation.svelte` | [`#sw-step-5-saturation`](single_well.md#sw-step-5-saturation) | Archie, Waxman-Smits ($Q_v, B$), Dual Water & Indonesian |
| **Step 6** | `ressum` | `InspectorResSum.svelte` | [`#sw-step-6-ressum`](single_well.md#sw-step-6-ressum) | Net Pay cutoffs, zonal summation & log blocking upscaling |
| **Step 7** | `report` | `InspectorReport.svelte` | [`#sw-step-7-report`](single_well.md#sw-step-7-report) | Automated CPI well report compilation |
| **Standalone** | `scal` | `InspectorScal.svelte` | [`#sw-standalone-core-pc`](single_well.md#sw-standalone-core-pc) | **Core & Pc Prep**: Depth shifting, stress corrections & lab-to-res $P_c$ |

### Multi-Well Space

| Step Category | Stepper ID | UI Component | Markdown Anchor | Primary Field Petrophysical Function |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1** | `inventory` | `MultiWellInventory.svelte` | [`#mw-step-1-inventory`](multi_well.md#mw-step-1-inventory) | Multi-well completeness audit & canonical mnemonic aliasing |
| **Step 2** | `qaqc` | `MultiWellQaqc.svelte` | [`#mw-step-2-qaqc`](multi_well.md#mw-step-2-qaqc) | Key-well designation & stratigraphic quantile normalization |
| **Step 3** | `rock-typing` | `MultiWellRockTyping.svelte` | [`#mw-step-3-rock-typing`](multi_well.md#mw-step-3-rock-typing) | Core Hydraulic Flow Units (FZI), Winland $r_{35}$ & Lucia RFN |
| **Step 4** | `perm-transform` | `MultiWellPermTransform.svelte` | [`#mw-step-4-perm-transform`](multi_well.md#mw-step-4-perm-transform) | Rock-type partitioned poro-perm regression ($k = a \cdot e^{b\phi}$) |
| **Step 5** | `shf` | `MultiWellShf.svelte` | [`#mw-step-5-shf`](multi_well.md#mw-step-5-shf) | Saturation-Height Modeling (Leverett $J$ / Skelt-Harrison) & log reconciliation |
| **Step 6** | `volumetrics` | `MultiWellVolumetrics.svelte` | [`#mw-step-6-volumetrics`](multi_well.md#mw-step-6-volumetrics) | Probabilistic Monte Carlo LHS STOIIP/GIIP (P90/P50/P10) & Tornado |
| **Step 7** | `report` | `InspectorReport.svelte` | [`#mw-step-7-report`](multi_well.md#mw-step-7-report) | Field Development Plan (FDP) comprehensive report generation |
| **Standalone** | `swirr-corr` | `MultiWellSwirr.svelte` | [`#mw-standalone-scal-relperm`](multi_well.md#mw-standalone-scal-relperm) | **SCAL & Rel-Perm**: Field Swirr fit, Brooks-Corey, LET & SWOF decks |
| **Standalone** | `fpt` | `MultiWellFpt.svelte` | [`#mw-standalone-fpt`](multi_well.md#mw-standalone-fpt) | **Pressure & Contacts**: MDT/RFT gradients, excess pressure, FWL/OWC/GOC |
| **Standalone** | `rt-classifier` | `MultiWellRtClassifierTrainer.svelte` | [`#mw-standalone-rt-classifier`](multi_well.md#mw-standalone-rt-classifier) | **RT Classifier**: Supervised classification (Random Forest / XGBoost) |
| **Standalone** | `ml-training` | `MultiWellMlTrainer.svelte` | [`#mw-standalone-ml-training`](multi_well.md#mw-standalone-ml-training) | **Conformal ML Trainer**: Physics-informed continuous ML with P10–P90 bounds |

---

## Detailed Technical References

- [**Single-Well Petrophysical Workflow**](single_well.md):
  Covers the 7-step deterministic evaluation pipeline and the standalone Core & Pc Prep module.

- [**Multi-Well Field Evaluation Workflow**](multi_well.md):
  Covers the 7-step field characterization pipeline and the 4 standalone specialized modules (SCAL & Rel-Perm, Formation Pressure & Contacts, RT Classifier, Conformal PIML).
