"""Köppen climate zones for the 16 benchmark cities + a numeric climate-distance
from the DE source baseline (Cfb). Used as Task 3 x-axis (transfer decay) and
Task 2d SHAP-similarity clustering ordering.

climate_dist: 0 = identical to DE (Cfb temperate oceanic); larger = more dissimilar.
Group hierarchy: same-letter (C/A/B/D) and same-subgroup get small distances;
cross-group (temperate→arid/tropical) get large distances.
"""
# city -> (koppen, group_name, climate_dist_from_DE)
CITIES = {
    # DE 8 (source for Task 3) — all Cfb temperate oceanic
    "berlin":      ("Cfb", "temperate-oceanic",   0.0),
    "hamburg":     ("Cfb", "temperate-oceanic",   0.0),
    "munich":      ("Cfb", "temperate-oceanic",   0.0),
    "cologne":     ("Cfb", "temperate-oceanic",   0.0),
    "dortmund":    ("Cfb", "temperate-oceanic",   0.0),
    "dusseldorf":  ("Cfb", "temperate-oceanic",   0.0),
    "frankfurt":   ("Cfb", "temperate-oceanic",   0.0),
    "stuttgart":   ("Cfb", "temperate-oceanic",   0.0),
    # Intl 8 (Task 3 targets)
    "warsaw":      ("Dfb", "continental-warm-summer", 0.35),
    "bucharest":   ("Cfa", "humid-subtropical",   0.5),
    "sao_paulo":   ("Cfa", "humid-subtropical",   0.6),   # + SHem season flip
    "buenos_aires":("Cfa", "humid-subtropical",   0.6),
    "johannesburg":("Cwb", "highland-subtropical",0.7),
    "lagos":       ("Aw",  "tropical-savanna",    1.0),
    "cairo":       ("BWh", "arid-desert",         1.2),
    "riyadh":      ("BWh", "arid-desert",         1.3),
    # OOD 4 (no Ta-UHI; Task 3 excluded, listed for completeness)
    "tehran":      ("BSh", "semi-arid",           1.1),
    "istanbul":    ("Csa", "mediterranean",       0.7),
    "casablanca":  ("Csa", "mediterranean",       0.8),
    "khartoum":    ("BWh", "arid-desert",         1.4),
}

DE_SOURCE = [c for c, (k, g, d) in CITIES.items() if k == "Cfb" and c in
            {"berlin", "hamburg", "munich", "cologne", "dortmund",
             "dusseldorf", "frankfurt", "stuttgart"}]
INTL_TARGET = ["warsaw", "bucharest", "sao_paulo", "buenos_aires",
               "johannesburg", "lagos", "cairo", "riyadh"]


def dist(a, b):
    """Pairwise climate distance in [0,2]: |Δdist_from_DE| + penalty if different group."""
    ka, ga, da = CITIES[a]; kb, gb, db = CITIES[b]
    pen = 0.0 if ga == gb else 0.3
    return abs(da - db) + pen


if __name__ == "__main__":
    print(f"DE source (8): {DE_SOURCE}")
    print(f"Intl target (8): {INTL_TARGET}")
    print("\nTarget climate distance from DE baseline:")
    for c in INTL_TARGET:
        k, g, d = CITIES[c]
        print(f"  {c:<14} {k}  {g:<20} dist={d}")
