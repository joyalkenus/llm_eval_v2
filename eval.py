import os
import pandas as pd
import json
import math
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from openai import OpenAI


client = OpenAI()

# 1. Load full dataset of 15,000 samples (as per original instructions)
# The user code loads "expanded_categorized_dataset.csv" - ensuring it has 15,000 samples is user's responsibility.
df = pd.read_csv("expanded_categorized_dataset.csv")
print("Total samples:", len(df))  # Expecting 15,000

# 2. Split into 1600 (train) and 400 (validation)
# The provided code shows a different indexing (200 train, 200 validation),
# but the instructions said 1600 train and 400 validation. Let's correct that.
train_df = df.iloc[:400].copy()
val_df = df.iloc[400:800].copy()  # 400 samples for validation
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

# Ensure preference is integer
train_df["Preference"] = train_df["Preference"].astype(int)
val_df["Preference"] = val_df["Preference"].astype(int)

# Extract only the necessary columns for training
train_data = train_df[["title", "topic", "subtopic", "Preference"]].to_dict(orient="records")

# 3. Create a single prompt with training data: system + user message
system_message = {
    "role": "system",
    "content": (
        "You are a helpful assistant that learns user preferences from provided training data. "
        "We have a dataset of user interactions with posts. Each sample includes a title, a topic, a subtopic, "
        "and a binary preference (0 or 1), where 1 means the user likes that kind of content and 0 means they do not.\n"
        "After seeing the training data, when asked, you will predict the preference for new unseen titles based on the user's preferences from the training data.\n"
        "You must rely solely on the provided training data distribution and patterns.\n"
        "DO NOT produce predictions until you are explicitly asked. When asked for predictions, you will receive a set of titles and must output predictions in JSON format.\n"
    )
}

# Convert training data to JSON string
training_json = json.dumps(train_data, ensure_ascii=False)

training_data_message = {
    "role": "user",
    "content": (
        "Below is the entire training dataset as a JSON array. Each element has "
        "title, topic, subtopic, and Preference.\n"
        f"{training_json}\n"
        "Wait until I ask you to predict before responding."
    )
}

# We'll gather the validation data
val_data = val_df[["title", "Preference"]].to_dict(orient="records")

# We'll predict the validation set in batches of 10
batch_size = 10
val_batches = [val_data[i:i+batch_size] for i in range(0, len(val_data), batch_size)]

# JSON schema for structured output
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "predictions_schema",
        "schema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "predicted_preference": {"type": "number"}
                        },
                        "required": ["title", "predicted_preference"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["predictions"],
            "additionalProperties": False
        },
        "strict": True
    }
}

predictions_all = []

# 4. For each batch of 10 validation titles, we ask the model for predictions
# We must send the system_message and training_data_message each time along with a new prediction request.
for batch_i, batch in enumerate(val_batches):
    titles_for_prediction = [{"title": b["title"]} for b in batch]
    prediction_user_message = {
        "role": "user",
        "content": (
            "Now I will provide new titles for which you must predict the user's preference (0 or 1). "
            "Only respond in the requested JSON format. Do not include explanations.\n"
            f"Here are the titles:\n{json.dumps(titles_for_prediction, ensure_ascii=False)}\n"
            "Respond with a JSON following the schema:\n"
            "{\n"
            "  \"predictions\": [\n"
            "    {\"title\": \"string\", \"predicted_preference\": 0 or 1},\n"
            "    ...\n"
            "  ]\n"
            "}\n"
        )
    }

    # Make the API call
    messages = [system_message, training_data_message, prediction_user_message]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format=response_format,
        temperature=0
    )

    response_content = completion.choices[0].message.content
    batch_predictions = json.loads(response_content)

    predictions_all.extend(batch_predictions["predictions"])

# 5. Compute metrics
pred_df = pd.DataFrame(predictions_all)
merged = pd.merge(val_df, pred_df, on="title", how="inner")

y_true = merged["Preference"].values
y_pred = merged["predicted_preference"].values

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)



print("Validation Accuracy:", accuracy)
print("Validation F1:", f1)
