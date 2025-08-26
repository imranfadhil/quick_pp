def contact_model(K_matrix, G_matrix, K_grain, G_grain, porosity):
    """
    Contact model (Hertz-Mindlin) for dry rock frame moduli.
    """
    phi_c = 0.36  # Critical porosity
    C = (1 - porosity / phi_c)
    K_dry = (C ** 2) * K_grain
    G_dry = (C ** 2) * G_grain
    return K_dry, G_dry


def inclusion_model(K_matrix, G_matrix, K_incl, G_incl, inclusion_fraction):
    """
    Inclusion model (Self-consistent approximation) for effective moduli.
    """
    K_eff = K_matrix + inclusion_fraction * (K_incl - K_matrix)
    G_eff = G_matrix + inclusion_fraction * (G_incl - G_matrix)
    return K_eff, G_eff


def rock_physics_model(K_matrix, G_matrix, K_grain, G_grain, K_fluid, G_fluid, porosity, inclusion_fraction):
    """
    Combines contact and inclusion models to estimate saturated rock moduli.
    """
    # Dry frame moduli from contact model
    K_dry, G_dry = contact_model(K_matrix, G_matrix, K_grain, G_grain, porosity)
    # Effective moduli from inclusion model
    K_eff, G_eff = inclusion_model(K_dry, G_dry, K_fluid, G_fluid, inclusion_fraction)
    return K_eff, G_eff


# Example usage
if __name__ == "__main__":
    # Matrix (quartz) properties
    K_matrix = 37.0  # GPa
    G_matrix = 44.0  # GPa
    # Grain properties
    K_grain = 36.6   # GPa
    G_grain = 45.0   # GPa
    # Fluid properties
    K_fluid = 2.2    # GPa
    G_fluid = 0.0    # GPa
    porosity = 0.25
    inclusion_fraction = 0.1

    K_sat, G_sat = rock_physics_model(
        K_matrix, G_matrix, K_grain, G_grain, K_fluid, G_fluid, porosity, inclusion_fraction
    )
    print(f"Saturated Bulk Modulus: {K_sat:.2f} GPa")
    print(f"Saturated Shear Modulus: {G_sat:.2f} GPa")
