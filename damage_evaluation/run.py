# =============================================
# Import Statements
# =============================================
import os
import json
import pandas as pd
from PIL import Image
import argparse
import sys
import base64
from datetime import datetime
import re
import torch

# =============================================
# Global Constants
# =============================================
JSON_DIR = "2019-ridgecrest_filtered"
IMAGE_DIR = "2019-ridgecrest_filtered_images"

# =============================================
# Data Processing Functions
# =============================================
def get_tweet_content(dir):
    files = os.listdir(dir)
    files = [f for f in files if f.endswith(".json")]
    files = sorted(files)
    dfs = []

    for file in files:
        print(f"Processing: {file}")
        file_path = os.path.join(JSON_DIR, file)
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            records = []
            for line in lines:
                try:
                    record = json.loads(line)
                    records.append(record)
                except (ValueError, OverflowError) as e:
                    print(f"Skipping bad line in {file}: {e}")

            if records:
                df = pd.json_normalize(records)
                dfs.append(df)
            else:
                print(f"No valid records found in file: {file}")

        except Exception as e:
            print(f"Failed to read/process file {file}: {e}")

    df = pd.concat(dfs)
    df = df.drop_duplicates(subset=['id', 'text'])
    print(df.head(5))
    return df

# =============================================
# Prompt Generation Functions
# =============================================
def get_location_prompt(tweet, location, place, longitude, latitude):
    with open("prompt/prompt_location.txt", "r") as f:
        prompt = f.read()
    return eval(prompt)

def get_event_related_prompt(tweet):
    with open("prompt/prompt_event.txt", "r") as f:
        prompt = f.read()
    return eval(prompt)

def get_damage_prompt(tweet, damage_prompt_file):
    with open(damage_prompt_file, "r") as f:
        prompt = f.read()
    return eval(prompt)

def get_damage_prompt_without_text():
    with open("prompt/prompt_imageonly.txt", "r") as f:
        prompt = f.read()
    return eval(prompt)

def get_final_prompt(tweet, text_response, image_response, text_image_response):
    with open("prompt/prompt_vote.txt", "r") as f:
        prompt = f.read()
    return eval(prompt)
    
# =============================================
# Model Initialization Functions
# =============================================
def init_gpt(secret_key_file):
    with open(secret_key_file) as f:
        lines = f.readlines()
        for line in lines:
            if line.split(',')[0].strip() == "openai_key":
                openai_key = line.split(',')[1].strip()
                break

    from openai import OpenAI
    
    global openai_client
    openai_client = OpenAI(api_key=openai_key)

def init_gemini(secret_key_file):
    with open(secret_key_file) as f:
        lines = f.readlines()
        for line in lines:
            if line.split(',')[0].strip() == "gemini_key":
                gemini_key = line.split(',')[1].strip()
                break
    from google import genai
    global gemini_client
    gemini_client = genai.Client(api_key=gemini_key)

def init_qwen():
    global qwen_model, qwen_processor
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor
    )
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    qwen_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct"
    )

    
def init_llava():
    global llava_model, llava_processor
    from transformers import (
        LlavaNextProcessor,
        LlavaNextForConditionalGeneration
    )
    
    llava_processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llama3-llava-next-8b-hf"
    )
    llava_model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llama3-llava-next-8b-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    ) 
    
# =============================================
# Model Calling Functions
# =============================================
def encode_image64(image_path):
    """Encode an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_gpt4o(model, text, images):
    """Call the GPT-4o model for image information and return the response."""
    if model not in {"gpt-4o", "gpt-4o-mini"}:
        raise ValueError(
            f"Invalid model: {model}. Must be 'gpt-4o' or 'gpt-4o-mini'"
        )
    
    if not images:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=1024
        )
        result = response.choices[0].message.content
        cleaned_string = result.strip('`').replace('json\n', '', 1)
        return cleaned_string
    
    base64_images = []
    for image in images:
        base64_image = encode_image64(image)
        base64_images.append(base64_image)
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"}
                        }
                        for img in base64_images
                    ]
                ],
            }
        ],
        temperature=0.0,
        max_tokens=1024
    )
    result = response.choices[0].message.content
    cleaned_string = result.strip('`').replace('json\n', '', 1)
    return cleaned_string

def call_gemini(text, image_paths):
    images = [Image.open(p) for p in image_paths]
    response = gemini_client.models.generate_content(
        model= "gemini-2.0-flash", #"gemini-2.5-pro-exp-03-25",
        contents=[text] + images)
    
    cleaned_string = response.text.strip('`').replace('json\n', '', 1)
    return cleaned_string

def call_qwen(text, image_paths):
    content = [{"type": "image", "image": f"{path}"} for path in image_paths]
    content.append({"type": "text", "text": text})
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]   
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    match = re.search(r"```json\s*(.*?)\s*```", output_text[0], re.DOTALL)
    if match:
        json_str = match.group(1)
        print(f"{json_str=}")
        return json_str
    else:
        print("No valid JSON block found.")
        return output_text[0]
    
def call_llava(text, image_paths):
    image_contents = [{"type": "image"} for _ in range(len(image_paths))]
    
    images = [Image.open(p) for p in image_paths]
    conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": text},
        ] + image_contents,
    },
]
    prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    if len(images) == 0:
        inputs = llava_processor(text=prompt, return_tensors="pt").to(llava_model.device)
    else:
        inputs = llava_processor(images=images, text=prompt, return_tensors="pt").to(llava_model.device)
    output = llava_model.generate(**inputs, max_new_tokens=1000)
    decoded_output = llava_processor.decode(output[0], skip_special_tokens=True)
    response = decoded_output.split('assistant', 1)[1].strip()
    
    cleaned_string = response.strip('`').replace('json\n', '', 1)
    return cleaned_string

def call_model(model_name, text, image_paths):
    if model_name == "gpt-4o" or model_name == "gpt-4o-mini":
        return call_gpt4o(model_name, text, image_paths)
    elif model_name == "gemini":
        return call_gemini(text, image_paths)
    elif model_name == "llava":
        return call_llava(text, image_paths)
    elif model_name == "qwen":
        return call_qwen(text, image_paths)
    
        

# =============================================
# Response Processing Functions
# =============================================
def get_location_from_json(s):

    try:
        data = json.loads(s)
        return data.get("location", None)
    except json.JSONDecodeError:
        return None
    
def get_event_related_from_json(s):
    try:
        data = json.loads(s)
        return data.get("is_event_related", None)
    except json.JSONDecodeError:
        return None
    


def generate_response(model, index, df):
    print(index)
    text = df['text'].iloc[index]
    location_prompt = df['location_prompt'].iloc[index]
    event_related_prompt = df['event_related_prompt'].iloc[index]
    damage_prompt = df['damage_prompt'].iloc[index]
    damage_prompt_without_tweet_text = df['damage_prompt_without_tweet_text'].iloc[index]
    image_urls = df['unique_image_urls'].iloc[index]
    
    save_paths = []
    for image_url in image_urls:
        path = os.path.join(IMAGE_DIR, image_url.split('/')[-1])
        if not os.path.exists(path):
            print(f"Image dir doesn't exit: {image_url}")
            continue
        save_paths.append(path)
    location_response = call_model(model,location_prompt, [])
    print(f"{location_response=}")
    
    df.at[index, 'location_response_json'] = location_response 
    
    location = get_location_from_json(location_response)
    
    print(f"location: {location}")
    
    df.at[index, 'location_response'] = location
    
    if not location or 'No' in location:
        df.at[index, 'model_response'] = "Not in U.S" 
        return True
    
    # step 2: check if it's event related
        
    event_related_response = call_model(model, event_related_prompt, save_paths) 
    print(f"{event_related_response=}")
    event_related = get_event_related_from_json(event_related_response)
    
    print(f"event_related_reponse: {event_related}")
    
    df.at[index, 'event_related_response_json'] = event_related_response
    
    df.at[index, 'event_related_response'] = event_related
    
    if not event_related or 'No' in event_related:
        df.at[index, 'model_response'] = "Not related to earthquake" 
        return True
    
    # step 3: 4-layer response
    
    df.at[index, 'text_only_response'] = None
    df.at[index, 'image_only_response'] = None
    df.at[index, 'text_image_response'] = None
    
    if not text:
        # Only images
        image_only_response = call_model(model, damage_prompt_without_tweet_text, save_paths)
        print(f"{image_only_response=}")
        df.at[index, 'image_only_response'] = image_only_response
        df.at[index, 'text_only_response'] = None
        df.at[index, 'model_response'] = image_only_response
    elif not image_urls:
        # Only text
        text_only_response = call_model(model, damage_prompt, [])
        print(f"{text_only_response=}")
        df.at[index, 'text_only_response'] = text_only_response
        df.at[index, 'image_only_response'] = None
        df.at[index, 'model_response'] = text_only_response
    else:
        # Contain both text and images    
        text_only_response = call_model(model, damage_prompt, [])
        image_only_response = call_model(model, damage_prompt_without_tweet_text, save_paths)
        text_image_response = call_model(model, damage_prompt, save_paths)
        
        final_prompt = get_final_prompt(text, text_only_response, image_only_response, text_image_response)
        df.at[index, 'final_prompt'] = final_prompt
        
        final_response = call_model(model, final_prompt, save_paths)
        
        df.at[index, 'text_only_response'] = text_only_response
        df.at[index, 'image_only_response'] = image_only_response
        df.at[index, 'text_image_response'] = text_image_response
        
        
        df.at[index, 'model_response'] = final_response

# Define a function to extract human impact, damage type and level only if both exist
def extract_damage_info(s):
    try:
        data = json.loads(s)
        return [data.get("human_impact", None), data.get("damage_type", None), data.get("damage_level", None), data.get("reasoning", None), data.get("confidence", None)]
    except json.JSONDecodeError:
        return [None, None, None, None, None]


# =============================================
# Main Function
# =============================================
def main():
    parser = argparse.ArgumentParser(
        description="LLaVA or GPT-4 model runner"
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['llava', 'qwen', 'gpt-4o', 'gpt-4o-mini', 'gemini'],
        default='llava',
        help='Model to use: llava or gpt4'
    )
    
    parser.add_argument(
        '--key_file',
        type=str,
        default=None,
        help='Path to GPT-4/Gemini API key file (required if model is gpt4/Gemini)'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start index, default 0'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=-1,
        help='End index, default -1'
    )
    
    parser.add_argument(
        '--damage_prompt_file',
        type=str,
        default='prompts_bundle/v1.txt',
        help='Damage evaluation prompt file.'      
    )

    args = parser.parse_args()

    if args.model in ['gpt-4o', 'gpt-4o-mini', 'gemini']:
        if not args.key_file:
            print(
                "Error: --key_file must be provided when using model gpt4 or gemini"
            )
            sys.exit(1)
        if not os.path.exists(args.key_file):
            print(f"Error: The key file '{args.key_file}' does not exist.")
            sys.exit(1)
    
    print(f"Init {args.model}!")
    if args.model == 'gpt-4o' or args.model == 'gpt-4o-mini':
        init_gpt(args.key_file)
    elif args.model == 'gemini':
        init_gemini(args.key_file)
    elif args.model == 'llava':
        init_llava()
    elif args.model == 'qwen':
        init_qwen()
        
    # data processing
    df = get_tweet_content(JSON_DIR)
    
    print('=========')
    print(df.shape)
    
    # create prompts
    df['location_prompt'] = df.apply(
        lambda x: get_location_prompt(
            tweet=x['text'],
            location=x['location'],
            place=x['place'],
            longitude=x['longitude'],
            latitude=x['latitude']
        ),
        axis=1
    )
    
    df['event_related_prompt'] = df.apply(
        lambda x: get_event_related_prompt(tweet=x['text']),
        axis=1
    )
    
    df['damage_prompt'] = df.apply(
        lambda x: get_damage_prompt(tweet=x['text'], prompt_file=args.prompt_file),
        axis=1
    )
    
    df['damage_prompt_without_tweet_text'] = df.apply(
        lambda x: get_damage_prompt_without_text(),
        axis=1
    )
    
    df = df.reset_index(drop=True)
    
    # Define columns to save
    columns_to_save = [
        'user', 'user_name', 'description', 'verified', 'created',
        'time', 'location', 'latitude', 'longitude', 'place',
        'text', 'unique_image_urls', 'location_response_json',
        'location_response', 'event_related_response_json',
        'event_related_response', 'text_only_response',
        'image_only_response', 'text_image_response',
        'model_response', 'human_impact', 'damage_type',
        'damage_level', 'reasoning', 'confidence'
    ]
    
    for col in columns_to_save:
        if col not in df.columns:
            df[col] = None 
    
    print(f"{df.shape=}")
    save_interval = 100  # Save every 100 rows
    length = df.shape[0]
    print(f"=================df size: {length}=======================")
    
    start = args.start 
    end = args.end
    if end == -1:
        end = length
        
    print(f"{start=}")
    print(f"{end=}")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame index range: {df.index.min()} to {df.index.max()}")
    
    if start >= length or end > length:
        print(
            f"Error: Start ({start}) or end ({end}) index is out of bounds. "
            f"DataFrame length is {length}"
        )
        sys.exit(1)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output_{args.model}_{timestamp}_{start}_{end}.csv"
    print(f"{output_path=}")
    
    count = 0
    for i in range(start, end):
        print(f"index: {i}")
        generate_response(args.model, i, df)
        count += 1
        
        human_impact, damage_type, damage_level, reasoning, confidence = (
            extract_damage_info(df.at[i, 'model_response'])
        )
        
        df.at[i, 'human_impact'] = human_impact
        df.at[i, 'damage_type'] = damage_type
        df.at[i, 'damage_level'] = damage_level
        df.at[i, 'reasoning'] = reasoning
        df.at[i, 'confidence'] = confidence
        
        # Save every {save_interval} iterations
        if (i + 1) % save_interval == 0:
            df.reset_index(drop=True).to_csv(
                output_path,
                columns=columns_to_save,
                index=True
            )
            print(f"Saved progress to {output_path} at row {i + 1}")
            
    print(
        f"========There are {df.shape[0]} lines of data in total. "
        f"Generated {count} responses!======="
    )
    
    df.reset_index(drop=True).to_csv(
        output_path,
        columns=columns_to_save,
        index=True
    )
    print(f"Saved final results to {output_path}.")

if __name__ == "__main__":
    main()

