from __future__ import annotations


DIRECT_FEATURE_LABELS = {
    "Li_channel_size": "n_Li × r⁻ (pm)",
    "Li_channels_size": "n_Li × r⁻ (pm)",
    "n_Li × r⁻ (pm)": "n_Li × r⁻ (pm)",
    "chi_range_x_r_avg": r"$\Delta\chi$ × $r_{all}$",
    "ir_std_nonLi": "Ionic radius std. (non-Li)",
    "ir_mean_nonLi": "Ionic radius mean (non-Li)",
    "d_electrons_std_nonLi": "d-electron std. (non-Li)",
    "r_ratio_x_chi_diff": r"$(r^+_{(\mathrm{incl\ Li}^{+})} \,/\, r^-) \times (\chi^+_{(\mathrm{incl\ Li}^{+})} - \chi^-)$",
    "log_rho_ratio": "log charge-density ratio",
    "field_x_r_diff": r"$\Phi^+_{(\mathrm{incl\ Li}^{+})}$ × ($r^+_{(\mathrm{incl\ Li}^{+})} - r^-$)",
    "χₐₗₗ": r"$\chi_{all}$",
    "χ⁻": r"$\chi^-$",
    "χₘₐₓ - χₘᵢₙ": r"$\chi_{max} - \chi_{min}$",
    "rₐₗₗ (pm)": r"$r_{all}$",
    "r⁻ (pm)": r"$r^-$",
    "n_Li": r"$n_{Li}$",
    "nₕₐₗᵢdₑ": r"$n_{\mathrm{halide}}$",
    "nₕₒₛₜ cₐₜᵢₒₙ": r"$n_{\mathrm{host\ cation}}$",
    "nₐₙᵢₒₙ": r"$n_{\mathrm{anion}}$",
    "χ⁺(incl Li⁺)": r"$\chi^+_{(\mathrm{incl\ Li}^{+})}$",
    "χ⁺(excl Li⁺)": r"$\chi^+_{(\mathrm{excl\ Li}^{+})}$",
    "χ⁺(incl Li⁺) - χ⁻": r"$\chi^+_{(\mathrm{incl\ Li}^{+})} - \chi^-$",
    "χ⁺(excl Li⁺) - χ⁻": r"$\chi^+_{(\mathrm{excl\ Li}^{+})} - \chi^-$",
    "r⁺(incl Li⁺) (pm)": r"$r^+_{(\mathrm{incl\ Li}^{+})}$",
    "r⁺(excl Li⁺) (pm)": r"$r^+_{(\mathrm{excl\ Li}^{+})}$",
    "r⁺(incl Li⁺) - r⁻": r"$r^+_{(\mathrm{incl\ Li}^{+})} - r^-$",
    "r⁺(excl Li⁺) - r⁻": r"$r^+_{(\mathrm{excl\ Li}^{+})} - r^-$",
    "r⁺(incl Li⁺) / r⁻": r"$r^+_{(\mathrm{incl\ Li}^{+})} \,/\, r^-$",
    "r⁺(excl Li⁺) / r⁻": r"$r^+_{(\mathrm{excl\ Li}^{+})} \,/\, r^-$",
    "ρₐₗₗ (C m⁻³)": r"$\rho_{all}$",
    "ρ⁻ (C m⁻³)": r"$\rho^-$",
    "ρ⁺(incl Li⁺) (C m⁻³)": r"$\rho^+_{(\mathrm{incl\ Li}^{+})}$",
    "ρ⁺(excl Li⁺) (C m⁻³)": r"$\rho^+_{(\mathrm{excl\ Li}^{+})}$",
    "ρ⁺(incl Li⁺) / ρ⁻": r"$\rho^+_{(\mathrm{incl\ Li}^{+})} \,/\, \rho^-$",
    "ρ⁺(excl Li⁺) / ρ⁻": r"$\rho^+_{(\mathrm{excl\ Li}^{+})} \,/\, \rho^-$",
    "ρ⁺(incl Li⁺) - ρ⁻": r"$\rho^+_{(\mathrm{incl\ Li}^{+})} - \rho^-$",
    "ρ⁺(excl Li⁺) - ρ⁻": r"$\rho^+_{(\mathrm{excl\ Li}^{+})} - \rho^-$",
    "Φ⁺(incl Li⁺) (|Z| pm⁻¹)": r"$\Phi^+_{(\mathrm{incl\ Li}^{+})}$",
    "Φ⁺(excl Li⁺) (|Z| pm⁻¹)": r"$\Phi^+_{(\mathrm{excl\ Li}^{+})}$",
}


def display_feature_label(feature: str) -> str:
    if feature in DIRECT_FEATURE_LABELS:
        return DIRECT_FEATURE_LABELS[feature]

    replacements = {
        "chi": "χ",
        "rho": "ρ",
        "avg": "avg.",
        "std": "std.",
        "mean": "mean",
        "diff": "difference",
        "nonLi": "non-Li",
        "Li": "Li",
        "x": "×",
        "r": "radius",
    }
    words = feature.replace("_", " ").split()
    return " ".join(replacements.get(word, word) for word in words)
