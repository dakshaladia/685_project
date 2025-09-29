import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

file_paths = {
    "GPT-4o": "gpt/output_Candidate_Name_Job_Title_gpt.csv",
    "Qwen 2.5": "qwen/actual/output_api_qwen_TEST_100rows_use_Candidate_Name_Job_Title.csv",
    "LLaMA 3.3": "llama33_output_Candidate_Name_Job_Title_Results.csv"
}
file_paths_flipped = {
    "GPT-4o": "gpt/output_position_flipped_Candidate_Name_Job_Title_gpt.csv",
    "Qwen 2.5": "qwen/actual/output_api_qwen_TEST_100rows_use_flipped_Candidate_Name_Job_Title.csv",
    "LLaMA 3.3": "llama33_output_Candidate_Name_Job_Title_Flipped.csv"
}

custom_palette = {
    "he/him": "#8ecae6",     # soft blue
    "she/her": "#f4a261",    # soft orange
    "they/them": "#90be6d"   # soft green
}

def occupation_visualizations(selected_occupations, file_paths, flipped=False):
    pronoun_data = []

    for model_name, file_path in file_paths.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = pd.read_csv(file_path)
            df = df[df["Job Title"].isin(selected_occupations)]
            df["LLM"] = model_name
            trial_cols = [col for col in df.columns if col.startswith("trial_")]

            # Keep Job Title to preserve identity across files
            melted = df.melt(id_vars=["Job Title", "LLM"], value_vars=trial_cols,
                            var_name="Trial", value_name="Pronoun")
            melted["LLM"] = model_name
            counts = (
                melted.groupby(["LLM", "Job Title"])["Pronoun"]
                .value_counts(normalize=True)
                .rename("Proportion")
                .reset_index()
            )
            pronoun_data.append(counts)

    # Combine all models' data
    final_df = pd.concat(pronoun_data, ignore_index=True)
    sns.set_theme(style="whitegrid", palette="pastel")

    # Generate separate plots per occupation
    for occupation in selected_occupations:
        plt.figure(figsize=(10, 5))
        subset = final_df[final_df["Job Title"] == occupation]
        sns.barplot(
        data=subset,
        x="LLM",
        y="Proportion",
        hue="Pronoun",
        palette=custom_palette,
        edgecolor="gray"
    )
        SMALL  = 10   # main tick-label size
        MEDIUM = 11   # axis labels
        LARGE  = 12   # title and legend
        # sns.barplot(data=subset, x="LLM", y="Proportion", hue="Pronoun", palette=custom_palette)
        # sns.barplot(data=subset, x="LLM", y="Proportion", hue="Pronoun", palette="Set2")
        plt.title(f"{occupation} (they/them as second candidate)", fontsize=LARGE, weight='bold') if not flipped else plt.title(f"{occupation} (they/them as first candidate)", fontsize=20, weight='bold')
        plt.ylabel("Probability", fontsize=MEDIUM, weight='bold')
        plt.xlabel("LLM", fontsize=MEDIUM, weight='bold')
        plt.ylim(0, 1)
        plt.xticks(fontsize=SMALL)
        plt.yticks(fontsize=SMALL)
        plt.legend(title="Pronoun", title_fontsize=MEDIUM, fontsize=SMALL)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{occupation}_flipped.png", bbox_inches="tight") if flipped else plt.savefig(f"{occupation}.png", bbox_inches="tight")
        

        # plt.title(..., fontsize=LARGE)
        # plt.xlabel("LLM", fontsize=MEDIUM)
        # plt.ylabel("Proportion", fontsize=MEDIUM)
        # plt.xticks(fontsize=SMALL)
        # plt.yticks(fontsize=SMALL)
        # plt.legend(title="Pronoun", title_fontsize=MEDIUM, fontsize=SMALL)


if __name__ == '__main__':
    occupations = ['Fashion Designer', 'Daycare Worker', 'Research Scientist', 'Preschool Teacher', 'Surf Instructor']
    occupation_visualizations(occupations, file_paths)
    occupation_visualizations(occupations, file_paths_flipped, True)
