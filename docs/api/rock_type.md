# Rock Typing & Formation Pressure Fluid Analysis (`quick_pp.rock_type`, `quick_pp.fluid_type`)

Hydraulic Flow Unit (FZI/RQI/GHE) classification, Winland/Pittman pore throat estimation, supervised multi-well `RandomForestClassifier` rock-type training & inference (`train_and_validate_rock_type_classifier`, `predict_with_saved_rt_classifier`, `compute_rock_flag_comparison`), hydrocarbon composition fluid typing, and multi-well Formation Pressure Test (FPT / MDT / RFT) gradient regression & Monte Carlo fluid contact intersection (`FormationPressureAnalyzer`, `FluidGradientResult`, `FluidContactResult`).

## Rock Typing (`quick_pp.rock_type`)

::: quick_pp.rock_type

## Fluid Typing & Formation Pressure Analysis (`quick_pp.fluid_type`)

::: quick_pp.fluid_type

### Dynamic P-T Fluid Properties

`estimate_fluid_properties()` estimates depth-aligned brine, live-oil, and gas
density and hydrogen-index curves from TVD, temperature/pressure profiles, and
fluid composition. Supply `temp_curve` and/or `pres_curve` to use measured well
logs instead of the synthesized gradients. Returned densities are in g/cc,
pressure is in MPa, and temperature is in °C.
