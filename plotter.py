import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Helvetica"

# Define a list of known abbreviations
abbreviations = ["cc", "nih", "uspto"]
# Function to calculate average excluding coding datasets
def average_no_coding(values):
    coding_datasets = ['github', 'stackexchange']
    non_coding_values = [v for k, v in values.items() if k not in coding_datasets]
    return np.mean(non_coding_values)

# Convert labels to sentence case or uppercase if an abbreviation
def format_label(label):
    if label in abbreviations:
        return label.upper()
    return label.replace('_', ' ').capitalize()

# Provided data
data = {
    "facebook/opt-1.3B": {
        "stackexchange": 22.867662240982057,
        "wikipedia": 6.06385578250885,
        "cc": 16.272702721357344,
        "github": 24.378156826019286,
        "pubmed_abstracts": 20.81046000289917,
        "openwebtext2": 13.443036963701248,
        "freelaw": 19.30048088979721,
        "math": 10.347422858715058,
        "nih": 15.406369069099426,
        "uspto": 7.22795484495163,
        "hackernews": 38.18226307868957,
        "enron": 20.49165243244171,
        "books3": 22.468206367888357,
        "pubmed_central": 43.289554561138154,
        "gutenberg": 29.35105010131737,
        "arxiv": 29.46771403503418,
        "bookcorpus2": 19.61259437228647,
        "opensubtitles": 18.23072248363495,
        "youtubesubtitles": 20.738288586581948,
        "ubuntu": 54.47045308997832,
        "europarl": 18.470123628173212,
        "philpapers": 34.07191749710903
    },
    "microsoft/phi-1_5": {
        "stackexchange": 13.306405331134796,
        "wikipedia": 24.43848635816574,
        "cc": 45.18441801929474,
        "github": 22.34434101510048,
        "pubmed_abstracts": 39.45042000007629,
        "openwebtext2": 46.6698781042099,
        "freelaw": 37.0780347366333,
        "math": 44.496000124931335,
        "nih": 26.842753434181212,
        "uspto": 20.371497662067412,
        "hackernews": 62.6556734790802,
        "enron": 78.1529347114563,
        "books3": 39.561276884829965,
        "pubmed_central": 37.22445684576034,
        "gutenberg": 69.75787701359043,
        "arxiv": 40.60544632673263,
        "bookcorpus2": 36.07749028702901,
        "opensubtitles": 40.4382812538147,
        "youtubesubtitles": 53.461380879762075,
        "ubuntu": 135.5466095568186,
        "europarl": 89.16952714936099,
        "philpapers": 74.93117472167327
    },
    "NousResearch/Llama-2-7b-hf": {
        "stackexchange": 17.882692171096803,
        "wikipedia": 5.3967883477211,
        "cc": 11.792089898586273,
        "github": 8.042657522201537,
        "pubmed_abstracts": 7.880014232635498,
        "openwebtext2": 10.291916546344757,
        "freelaw": 6.428656887054443,
        "math": 18.886051611900328,
        "nih": 7.706928003311157,
        "uspto": 6.648721785068512,
        "hackernews": 21.15103667640686,
        "enron": 24.680454632282256,
        "books3": 12.348250312284328,
        "pubmed_central": 11.609683396816253,
        "gutenberg": 17.307688935391315,
        "arxiv": 11.446100658416748,
        "bookcorpus2": 13.496133859848744,
        "opensubtitles": 14.319103336334228,
        "youtubesubtitles": 14.966658102169728,
        "ubuntu": 35.92732435536672,
        "europarl": 13.757284387594924,
        "philpapers": 16.62703704165521
   }
}

# Format labels
formatted_labels = ["Average"] + [format_label(label) for label in data["facebook/opt-1.3B"].keys()]
facebook_values = [np.mean(list(data["facebook/opt-1.3B"].values())) ]  + [data["facebook/opt-1.3B"][label] for label in data["facebook/opt-1.3B"].keys()]
microsoft_values = [np.mean(list(data["microsoft/phi-1_5"].values())) ] + [data["microsoft/phi-1_5"][label] for label in data["microsoft/phi-1_5"].keys()]
nousresearch_values = [np.mean(list(data["NousResearch/Llama-2-7b-hf"].values())) ]  + [data["NousResearch/Llama-2-7b-hf"][label] for label in data["NousResearch/Llama-2-7b-hf"].keys()]

x = range(len(formatted_labels))
width = 0.28  # bar width

fig, ax = plt.subplots(figsize=(17, 10))

# Refined colors for the bars
colors = ['#E63946', '#A8DADC', '#457B9D']

rects1 = ax.bar([i - width for i in x], facebook_values, width, label='OPT-1.3B', color=colors[0], hatch='/')
rects3 = ax.bar(x, nousresearch_values, width, label='Llama2-7B', color=colors[2], hatch='.')
rects2 = ax.bar([i + width for i in x], microsoft_values, width, label='Phi-1.5', color=colors[1], hatch='')

# Enhance font settings
ax.set_ylabel('Mean Perplexity', fontsize=24)
ax.set_title('Mean Perplexity by Model and Task', fontsize=24)
ax.set_xticks(x)
ax.set_xticklabels(formatted_labels, rotation=90, fontsize=20)
ax.legend(fontsize=20, loc='upper left', )

# Improve y-tick labels font
ax.tick_params(axis="y", labelsize=20)

# Calculate the increase factor from OPT to PHI1.5 for the "Average" value
increase_factor = microsoft_values[0] / facebook_values[0]
increase_factor_microsoft_llama = microsoft_values[0] / nousresearch_values[0] 
print(f"OPT-1.3B to Phi-1.5 increase factor: {increase_factor:.1f}x")
print(f"OPT-1.3B to Llama2-7B increase factor: {increase_factor_microsoft_llama:.1f}x")
increase_text = f"{increase_factor:.1f}x higher average PPL"

# Add an arrow annotation
arrowprops = dict(arrowstyle="->", linewidth=4, color="red")
ax.annotate("", xy=(0.3, microsoft_values[0]), xytext=(-0.3, (facebook_values[0])),
            arrowprops=arrowprops, ha='center', va='bottom', fontsize=14, color="red")

ax.text(1.8, microsoft_values[0] + 5, increase_text, ha='center', va='bottom', fontsize=24, color="red", fontweight=1000)


fig.tight_layout()

plt.savefig("perplexity.png")
