# Multi-Well QA/QC & Stratigraphic Normalization (`quick_pp.qaqc`)

Borehole environmental corrections, neutron-density hydrocarbon corrections, bad-hole flagging, and multi-well zone-by-zone quantile/histogram normalization.

## Invaded-zone hydrocarbon correction

The workspace HC-correction endpoint offers an optional **Standard invaded-zone**
method alongside the existing crossplot correction. It searches invaded-zone
hydrocarbon saturation `S_hi` from 0 to 1 in 0.01 increments and selects the
candidate whose corrected density-neutron shale volume is closest to `VSH_GR`.
The density and neutron shifts, light/heavy hydrocarbon coefficients, mud-filtrate
hydrogen index, and lithology-scaled excavation term follow the supplied method
equations. The selected saturation is returned as `SHC_INV`.

The method requires well-header `mud_filtrate_density` (g/cc) and
`mud_filtrate_salinity` (ppm), editable in the QAQC Well Header panel. Lithology
coefficient `K` defaults to 1. Density porosity is estimated from the dry-sand
density endpoint and mud-filtrate density; corrected `VSH_DN` is evaluated in the
matrix/clay/mud-filtrate crossplot triangle. These are explicit implementation
conventions where the supplied excerpts did not define `phi_D` or the exact
`VSH_DN` calculation.

This is an effective invaded-zone saturation correction, not a radial invasion
simulator: separate neutron and density investigation radii and radial saturation
profiles are not modeled. The separate “Default method” in the supplied screenshots
is not implemented because its `NFAC` and `HCORR` equations were not provided.

::: quick_pp.qaqc
