import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm

# Define the available topics
TOPICS = [
    "Trump’s Legal Convictions and Felony Charges",
    "Biden vs. Trump Presidential Debates",
    "Israel-Hamas Conflict and Biden’s Ceasefire Proposal",
    "Hunter Biden’s Legal Troubles (e.g., Gun Charges)",
    "U.S. Policy on Ukraine and Russia",
    "Trump’s Tax Promises and Election Campaign",
    "Biden’s Immigration Policies and Executive Orders",
    "Legal Proceedings in Georgia’s 2020 Election Case Against Trump",
    "Trump’s Rallies and Live Events Coverage",
    "Celebrations of Trump (e.g., Birthdays and Tributes)",
    "Pro-Trump and MAGA Advocacy",
    "Nonsense",
]


model_id = "meta-llama/Llama-3.3-70B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)


# Define the prompt template
PROMPT_TEMPLATE = """
You are an AI assistant that evaluates whether a given post aligns with its assigned topic. 
Please follow these steps:
1. Evaluate whether the post aligns with the assigned topic. Use a loose standard for this judgment—if any part of the post contains context related to the topic, consider it aligned. If the post is aligned, respond with: "Labeled correctly: True". 
2. If not, respond with "Labeled correctly: False" and suggest the correct topic from the provided list.
3. If none of the topics fit, generate a new topic and respond accordingly.

Format your response strictly as follows:
Response:
Labeled correctly: [True/False]
If not, the correct label is: [Correct Topic or "Generated Topic: XYZ"]

Post: "{post}"

Assigned Topic: "{assigned_topic}"

List of Topics:
{topics}

Your Response:
"""


def evaluate_post(post, assigned_topic):
    """
    Uses the Transformers model to generate an evaluation response for the given post.
    Returns the generated text (the LLM's response).
    """
    prompt = PROMPT_TEMPLATE.format(
        post=post, assigned_topic=assigned_topic, topics="\n".join(TOPICS)
    )

    # Tokenize the prompt and move inputs to the same device as the model.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    # Generate up to 200 new tokens. Adjust parameters as needed.
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,  # or set to True with sampling parameters if desired
    )

    # Extract only the generated text (exclude the prompt)
    generated_ids = output_ids[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def parse_response(response):
    """
    Parses the LLM response to extract whether the post was correctly labeled and, if not,
    what the correct topic is.
    Returns a dictionary with alignment status and suggested topic.
    """
    # Check for misclassification by looking for "False" in the response
    labeled_correctly = "False" not in response

    if labeled_correctly:
        return {"Labeled correctly": True, "New Topic": None}

    # Extract the suggested topic using a regular expression.
    match = re.search(r"If not, the correct label is: (.+)", response)
    new_topic = match.group(1).strip() if match else None
    return {"Labeled correctly": False, "New Topic": new_topic}


if __name__ == "__main__":
    # Read the CSV files containing the posts and assigned topics
    bsky_df = pd.read_csv(
        "/nfs/turbo/isr-fconrad1/Mao/projects/information-diffusion/data/topic_eval_bsky_sample.csv"
    )
    ts_df = pd.read_csv(
        "/nfs/turbo/isr-fconrad1/Mao/projects/information-diffusion/data/topic_eval_ts_sample.csv"
    )

    # Process the Bluesky data
    for i, row in tqdm(bsky_df.iterrows()):
        sample_post = row["post"]
        assigned_topic = row["topic_label"]
        raw_result = evaluate_post(sample_post, assigned_topic)
        parsed_result = parse_response(raw_result)
        bsky_df.loc[i, "llm_response"] = raw_result
        bsky_df.loc[i, "parsed_judgement"] = parsed_result["Labeled correctly"]
        bsky_df.loc[i, "parsed_topic"] = parsed_result["New Topic"]

    # Process the TS data
    for i, row in tqdm(ts_df.iterrows()):
        sample_post = row["post"]
        assigned_topic = row["topic_label"]
        raw_result = evaluate_post(sample_post, assigned_topic)
        parsed_result = parse_response(raw_result)
        ts_df.loc[i, "llm_response"] = raw_result
        ts_df.loc[i, "parsed_judgement"] = parsed_result["Labeled correctly"]
        ts_df.loc[i, "parsed_topic"] = parsed_result["New Topic"]

    # Save the updated Bluesky dataframe to CSV
    bsky_df.to_csv(
        "/nfs/turbo/isr-fconrad1/Mao/projects/information-diffusion/data/topic_eval_bsky_sample_llama3_70b.csv",
        index=False,
    )
    ts_df.to_csv(
        "/nfs/turbo/isr-fconrad1/Mao/projects/information-diffusion/data/topic_eval_ts_sample_llama3_70b.csv",
        index=False,
    )
