BEGIN {
    FS = OFS = ","
}

function conductivity(value) {
    sub(/^[[:space:]]*[≈~]/, "", value)
    return value + 0
}

ARGIND == 1 {
    if (FNR > 1 && !(conductivity($5) < 1e-6 && conductivity($6) < 1e-6)) {
        keep[$2] = 1
    }
    next
}

FNR == 1 || ($1 in keep) {
    print > output
}
