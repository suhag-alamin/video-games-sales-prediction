import gradio as gr
import pandas as pd
import pickle

with open("video_games_sales.pkl", "rb") as f:
    model = pickle.load(f)

feature_names = list(getattr(model, "feature_names_in_", []))

platform_choices = sorted({c.split("Platform_", 1)[1]
                          for c in feature_names if c.startswith("Platform_")}) or ["PS2"]
genre_choices = sorted({c.split("Genre_", 1)[1]
                       for c in feature_names if c.startswith("Genre_")}) or ["Action"]
publisher_choices = sorted({c.split("Publisher_", 1)[1]
                           for c in feature_names if c.startswith("Publisher_")}) or ["Nintendo"]

# Prediction logic


def predict_sales(rank, platform, year, genre, publisher, na_sales, eu_sales, jp_sales, other_sales):
    row = {col: 0 for col in feature_names}

    for col, val in [("Rank", rank), ("Year", year), ("NA_Sales", na_sales), ("EU_Sales", eu_sales), ("JP_Sales", jp_sales), ("Other_Sales", other_sales)]:
        if col in row:
            row[col] = val

    if "Game_Age" in row and year is not None:
        row["Game_Age"] = 2026 - year

    for prefix, choice in [("Platform_", platform), ("Genre_", genre), ("Publisher_", publisher)]:
        col_name = f"{prefix}{choice}"
        if col_name in row:
            row[col_name] = 1

    input_df = pd.DataFrame([row], columns=feature_names)

    prediction = model.predict(input_df)[0]
    return f"Predicted Global Sales: {prediction:.2f} Million"


# Define Gradio interface inputs
inputs = [
    gr.Number(label="Rank", value=1, interactive=True),
    gr.Dropdown(label="Platform", choices=platform_choices,
                value="PS2", interactive=True),
    gr.Number(label="Year", value=2006, interactive=True),
    gr.Dropdown(label="Genre", choices=genre_choices,
                value="Action", interactive=True),
    gr.Dropdown(label="Publisher", choices=publisher_choices,
                value="Nintendo", interactive=True),
    gr.Number(label="NA_Sales (Millions)", value=0.5, interactive=True),
    gr.Number(label="EU_Sales (Millions)", value=0.2, interactive=True),
    gr.Number(label="JP_Sales (Millions)", value=0.1, interactive=True),
    gr.Number(label="Other_Sales (Millions)", value=0.05, interactive=True)
]

# Define Gradio interface output
outputs = gr.Textbox(label="Predicted Global Sales")

# Create and launch the interface
print("Launching Gradio Interface...")
app = gr.Interface(fn=predict_sales, inputs=inputs, outputs=outputs,
                   title="Video Game Sales Predictor",
                   description="Enter the features of a video game to predict its global sales using a trained RandomForest model.")

app.launch(share=True)
