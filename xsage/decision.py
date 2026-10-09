"""Decision rule of the paper: reliable improvement (*) and confirmed equivalence (+, printed as a dagger)."""

MARGIN = 0.005
LOWER_IS_BETTER = {"Gini"}
RANK = {"*": 2, "+": 1, "": 0}


def mark(c, met):
    """'*' reliable improvement, '+' confirmed equivalence, '' neither; c has d, lo, hi, seeds."""
    if met in LOWER_IS_BETTER:
        star = c["d"] < 0 and c["hi"] < 0 and c["seeds"] == 0
    else:
        star = c["d"] > 0 and c["lo"] > 0 and c["seeds"] == 5
    if star:
        return "*"
    if c["lo"] > -MARGIN and c["hi"] < MARGIN:
        return "+"
    return ""


def weaker(*marks):
    """Weaker of the marks of several instances (Foursquare: New York and Sao Paulo)."""
    return min(marks, key=lambda x: RANK[x])
