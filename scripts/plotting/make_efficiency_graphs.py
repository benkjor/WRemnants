import re

import matplotlib.pyplot as plt

# Read your input from a text file
with open("efficiency_graph_fit.txt", "r") as f:
    input_text = f.read()

# Extract parameter names and constraints using regular expressions
pattern = re.compile(
    r"^\s*(\S+)\s+([-+]?\d*\.?\d+)\s+\+/-\s+([-+]?\d*\.?\d+)", re.MULTILINE
)
matches = pattern.findall(input_text)
parameters = [param for param, _, _ in matches]
param_constraints = [constraint for _, _, constraint in matches]

params_hlt = [param for param in parameters if "hlt_" in param]
params_id = [param for param in parameters if "id_" in param]
params_num = [param for param in parameters if "n_" in param]


def extract_mll_and_time(text):
    # Match digits after 'mll' and 'time'
    mll_match = re.search(r"mll(\d+)", text)
    time_match = re.search(r"time(\d+)", text)

    mll = int(mll_match.group(1)) if mll_match else None
    time = int(time_match.group(1)) if time_match else None
    return mll, time


colors = ["C0", "C1", "C2"]
check_color = [0, 0, 0]
legend_entries = ["m = 89.3 - 90.1 GeV", "m = 90.1 - 90.8 GeV", "m = 90.8 - 91.4 GeV"]
low_mll_hlt = []
med_mll_hlt = []
high_mll_hlt = []
for name in params_hlt:
    i = parameters.index(name)
    value = param_constraints[i]
    mll, time = extract_mll_and_time(name)

    if mll == 3:
        low_mll_hlt.append(float(value) * 0.01)
    elif mll == 4:
        med_mll_hlt.append(float(value) * 0.01)
    elif mll == 5:
        high_mll_hlt.append(float(value) * 0.01)

plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

plt.plot(low_mll_hlt, color="C0", label="m = 89.3 - 90.1 GeV")
plt.plot(med_mll_hlt, color="C1", label="m = 90.1 - 90.8 GeV")
plt.plot(high_mll_hlt, color="C2", label="m = 90.8 - 91.4 GeV")

plt.xlabel("Sidereal time [hrs]")
plt.ylabel("HLT efficiency uncertainty")
plt.legend(loc="center right")
plt.savefig(
    "/home/submit/jbenke/public_html/liv_model_fits/2025-07-28/hlt_efficiencies.png"
)
plt.clf()


low_mll_id = []
med_mll_id = []
high_mll_id = []
for name in params_id:
    i = parameters.index(name)
    value = param_constraints[i]
    mll, time = extract_mll_and_time(name)

    if mll == 3:
        low_mll_id.append(float(value) * 0.01)
    elif mll == 4:
        med_mll_id.append(float(value) * 0.01)
    elif mll == 5:
        high_mll_id.append(float(value) * 0.01)

plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

plt.plot(low_mll_id, color="C0", label="m = 89.3 - 90.1 GeV")
plt.plot(med_mll_id, color="C1", label="m = 90.1 - 90.8 GeV")
plt.plot(high_mll_id, color="C2", label="m = 90.8 - 91.4 GeV")

plt.xlabel("Sidereal time [hrs]")
plt.ylabel("ID efficiency uncertainty")
plt.legend(loc="center right")
plt.savefig(
    "/home/submit/jbenke/public_html/liv_model_fits/2025-07-28/id_efficiencies.png"
)


# pdb.set_trace()

# print(output)

# Prepare CSV data
# csv_data = [("Parameter", "Constraint")]
# csv_data.extend([(param, constraint) for param, _, constraint in matches])

# # Save to a CSV file
# with open("efficiency_output.csv", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(csv_data)
