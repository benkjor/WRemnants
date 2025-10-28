axis_labels = {
    "time": {"label": r"Time", "unit": "h"},
}

process_colors = {
    "stat": "red",
    "linearity": "blue",
    "stability": "orange",
    "prefiring_syst": "purple",
    "prefiring_stat": "grey",
}


# nuisance_grouping = {
#         "unfolding":
#         [
#             "stat",
#             "linearity",
#             "stability"
#             ]
#         }

nuisance_grouping = {
    "liv_unc": [
        "stat",
        "linearity",
        "stability",
        "prefiring_syst",
        "prefiring_stat",
        "bkg",
    ]
}

# nuisance_grouping = {
#         "noBinByBin":
#         [
#             "stat",
#             "linearity",
#             "cross_detector_stability",
#             "prefiring_syst",
#             "prefiring_stat",
#             "bkg",
#             "nz",
#             "eff_1",
#             "eff_2"
#             ]
#         }
