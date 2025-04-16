import os
import json
import pandas as pd
from PIL import Image
from io import BytesIO
# from transformers import AutoProcessor, LlavaForConditionalGeneration
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# import torch
import argparse
import sys
from openai import OpenAI
import base64
from datetime import datetime
import re

JSON_DIR = "../../Data/2019-ridgecrest_filtered"
IMAGE_DIR = "../../Data/2019-ridgecrest_filtered_image"

LLAVA_MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"


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

def get_location_prompt(tweet, location, place, longitude, latitude):
    prompt = f"""
    You are a location identification expert. Your task is to determine whether a tweet is from a U.S.-based location, based on all available metadata and the tweet content.
    Use the information below to infer the most accurate location if possible. 
    
    Your output results must be generated after reasoning through  extual and/or visual information.

Input:
	•	Location: {location}
	•	Place: {place}
	•	Longitude: {longitude}
	•	Latitude: {latitude}
	•	Tweet Text: {tweet}

Location Identification Steps:
 Step 1: Geolocation Metadata: Check if longitude, or latitude indicate a U.S. location. If so, infer the location.
 Step 2: Tweet Content: Analyze the tweet_text to find any explicit or implicit mention of a U.S. location (e.g., city, county, state, street, neighborhood, national park). If found, use it to determine the tweet’s location.
 Step 3: Fallback to User Profile: If neither geolocation nor tweet content provides enough information, use location or place fields from the user profile to infer location.

Output Format:
	•	If a U.S. location can be identified, output the location in plain text (e.g., San Francisco, CA).
	•	If the location is outside the U.S. or cannot be identified, output No.
 	•	Do not output anything else. No explanations. No additional text. Just the location or No.

    """
    return prompt

def get_event_related_prompt(tweet):
    prompt = f"""
    You are an earthquake damage assessment expert.
Your task is to determine whether an input tweet is related to an earthquake in any meaningful way — including earthquake events, their impact, damage, or aftermath.
Please read the tweet carefully and decide if it is about an earthquake.
Examples of tweets related to earthquakes:
	•	Last night she said that I needed to not stack all these shoe boxes up so high because an earthquake will happen and they will all fall on me! I‚Äôm more worried about damaging the boxes and not being able to pass as Deadstock TBH than falling on me. 
	•	My outdoor pillows fell and my pancake is now burnt. This is the extent of the damage of the earthquake in Vegas for me. 
Examples of tweets not related to earthquakes:
	•	Devi Bhujel, making tea in her kitchen in her village in Nepal.‚Äú[Getting] #water here is very hard. I take one jerrycan in a basket, it‚Äôs about 10 liters maybe. The usual walking road is destroyed by the earthquake and construction. WaterAid/ Sibtain Haider #July4th 
	•	I knew those Trump tanks would cause damage.  #earthquake 

Your output results must be generated after reasoning through  extual and/or visual information.

Input:
    •	Tweet Text: {tweet}

Output format:
•	Respond only with Yes if the tweet is related to an earthquake.
•	Respond only with No if the tweet is not related to an earthquake.
•	Do not output anything else. No explanations. No additional text. Just Yes or No.
    """
    return prompt


def get_final_prompt(tweet, text_response, image_response, text_image_response):
    
    prompt = f"""
Task: You are an AI model trained to identify the earthquake damage type (interior or exterior) and damage level (MMI scale). Now, you are asked to evaluate and vote for the best prediction among three previously generated outputs: one using text only, one using image only, and one using both text and image.

Instructions:

Your goal is to:
	1.	Carefully compare the three outputs for accuracy, completeness, and consistency with the input tweet and image.
	2.	Vote for the most reliable and accurate prediction, considering all available evidence.
	3.	Briefly explain your reasoning.
    4.  Your output results must be generated after reasoning through  extual and/or visual information.

Input Tweet:
{tweet}

Instructions:

1. Human Impact Evaluation:
   Look for language or visual evidence suggesting that people experienced or emotionally reacted to the earthquake. Indicators may include expressions or signs of: Fear(e.g., "people were terrified", "panic in the streets"), shock or confusion(e.g., "people didn't know what to do"), physical presence or impact (e.g., "people ran outside", "rescue teams helping trapped residents"), sensation reporting (e.g., "I felt the floor shake", "it was the strongest I've ever felt")
    •  1: if there is any mention or evidence of human emotional or physical experience of the earthquake. 
    •  0: if there is no indication that humans were present or affected emotionally/physically.

2. Damage Type Classification:
   Classify the damage as either:
    •  Interior Damage: Damage that is clearly observed inside a building (e,g, cracked or collapsed interior walls, broken windows or glass, displaced or fallen indoor furniture, ceiling or floor cracks, shaking fixtures (e.g., light fixtures, shelves))
    •  Exterior Damage: Damage that is clearly observed on the outside of buildings or in the surrounding environment (e.g., Collapsed buildings or partial structural failure, Cracked roads, sidewalks, or bridges,Fallen trees or utility poles, Visible debris or rubble outside, Shifts in building foundation or roof collapse)
    •  Both Interior and Exterior Damage: Evidence of damage is present both inside and outside of structures. The content includes clear indicators of both categories listed above.
    •  Not Identified: The input does not provide enough information to determine whether the damage is interior, exterior, or both.

3. Damage Level Classification (MMI Scale):
   After identifying the damage type (Interior, Exterior, Both, or Not Identified) and human impact ("Felt" or "Not Felt"), classify the earthquake damage level using the Modified Mercalli Intensity (MMI) using the observed damage type and/or human impact.
   If human impact is 1 from the previous step (human can feel the earthquake), consider both human impact and damage level classification.
   If human impact is 0 from the previous step (human can't feel the earthquake), proceed based solely with damage level classification.
 
   Damage Level Categories (MMI Scale):
   1 - Not felt: No noticeable damage.
   2 - Weak: Felt by only a few people at rest; no damage to buildings.
   3 - Light: Felt indoors, especially on upper floors; no significant structural damage.
   4 - Moderate: Felt by most people; some damage to buildings, such as minor cracks.
   5 - Strong: Felt by everyone; damage to buildings, minor cracks, but no collapse.
   6 - Very Strong: Damage to buildings, visible structural deformation.
   7 - Severe: Significant damage, some collapses or structural failures.
   8 - Very Severe: Many buildings collapse or are severely damaged.
   9 - Violent: Total destruction in some areas, severe damage.
   10 - Extreme: Complete destruction of all structures in the affected area.

Candidate Responses:
	1.	Text Only: {text_response}
	2.	Image Only: {image_response}
	3.	Text + Image: {text_image_response} 
 
 
Output Format:
	•	Voted Response: (Text Only / Image Only / Text + Image)
	•	Reason: (1–3 sentence explanation)
	•	Human impact: 1 or 0. Do not output anything else. No explanations. No additional text. Just 1 or 0.
	•	Damage type: Interior or Exterior image. Do not output anything else. No explanations. No additional text. Just Interior or Exterior or Both or Not defined.
    •	Damage level: Number of any from 1-10. Do not output anything else. No explanations. No additional text. Just one number between 1 - 10.
 
    """
    
    return prompt


def get_damage_prompt(tweet):
    
    prompt = f"""
Task: You are an AI model trained to identify the damage type (interior or exterior), and damage level of an earthquake from a given textual description and image(optional).

Your output results must be generated after reasoning through  extual and/or visual information.

Text Description:
{tweet}


Image Description:
Please analyze the image to assess the severity of the earthquake's damage. Consider visible damage such as collapsed buildings, cracks in roads, broken windows, fallen debris, or visible interior damage like overturned furniture or shattered windows.

Instructions:

1. Human Impact Evaluation:
   Look for language or visual evidence suggesting that people experienced or emotionally reacted to the earthquake. Indicators may include expressions or signs of: Fear(e.g., "people were terrified", "panic in the streets"), shock or confusion(e.g., "people didn't know what to do"), physical presence or impact (e.g., "people ran outside", "rescue teams helping trapped residents"), sensation reporting (e.g., "I felt the floor shake", "it was the strongest I've ever felt")
    •  1: if there is any mention or evidence of human emotional or physical experience of the earthquake. 
    •  0: if there is no indication that humans were present or affected emotionally/physically.

2. Damage Type Classification:
   Classify the damage as either:
    •  Interior Damage: Damage that is clearly observed inside a building (e,g, cracked or collapsed interior walls, broken windows or glass, displaced or fallen indoor furniture, ceiling or floor cracks, shaking fixtures (e.g., light fixtures, shelves))
    •  Exterior Damage: Damage that is clearly observed on the outside of buildings or in the surrounding environment (e.g., Collapsed buildings or partial structural failure, Cracked roads, sidewalks, or bridges,Fallen trees or utility poles, Visible debris or rubble outside, Shifts in building foundation or roof collapse)
    •  Both Interior and Exterior Damage: Evidence of damage is present both inside and outside of structures. The content includes clear indicators of both categories listed above.
    •  Not Identified: The input does not provide enough information to determine whether the damage is interior, exterior, or both.

3. Damage Level Classification (MMI Scale):
   After identifying the damage type (Interior, Exterior, Both, or Not Identified) and human impact ("Felt" or "Not Felt"), classify the earthquake damage level using the Modified Mercalli Intensity (MMI) using the observed damage type and/or human impact.
   If human impact is 1 from the previous step (human can feel the earthquake), consider both human impact and damage level classification.
   If human impact is 0 from the previous step (human can't feel the earthquake), proceed based solely with damage level classification.
 
   Damage Level Categories (MMI Scale):
   1 - Not felt: No noticeable damage.
   2 - Weak: Felt by only a few people at rest; no damage to buildings.
   3 - Light: Felt indoors, especially on upper floors; no significant structural damage.
   4 - Moderate: Felt by most people; some damage to buildings, such as minor cracks.
   5 - Strong: Felt by everyone; damage to buildings, minor cracks, but no collapse.
   6 - Very Strong: Damage to buildings, visible structural deformation.
   7 - Severe: Significant damage, some collapses or structural failures.
   8 - Very Severe: Many buildings collapse or are severely damaged.
   9 - Violent: Total destruction in some areas, severe damage.
   10 - Extreme: Complete destruction of all structures in the affected area.


Output Format:
	•	Human impact: 1 or 0. Do not output anything else. No explanations. No additional text. Just 1 or 0.
	•	Damage type: Interior or Exterior image. Do not output anything else. No explanations. No additional text. Just Interior or Exterior or Both or Not defined. 
    •	Damage level: Number of any from 1-10. Do not output anything else. No explanations. No additional text. Just one number between 1 - 10.

    """
    
    return prompt

    
def get_damage_prompt_without_text():
    prompt = f"""
Task: You are an AI model trained to identify the damage type (interior or exterior), and damage level of an earthquake from a given textual image.

Your output results must be generated after reasoning through  extual and/or visual information.

Image Description:
Please analyze the image to assess the severity of the earthquake's damage. Consider visible damage such as collapsed buildings, cracks in roads, broken windows, fallen debris, or visible interior damage like overturned furniture or shattered windows.

Instructions:

1. Human Impact Evaluation:
   Look for language or visual evidence suggesting that people experienced or emotionally reacted to the earthquake. Indicators may include expressions or signs of: Fear(e.g., "people were terrified", "panic in the streets"), shock or confusion(e.g., "people didn't know what to do"), physical presence or impact (e.g., "people ran outside", "rescue teams helping trapped residents"), sensation reporting (e.g., "I felt the floor shake", "it was the strongest I've ever felt")
    •  1: if there is any mention or evidence of human emotional or physical experience of the earthquake. 
    •  0: if there is no indication that humans were present or affected emotionally/physically.

2. Damage Type Classification:
   Classify the damage as either:
    •  Interior Damage: Damage that is clearly observed inside a building (e,g, cracked or collapsed interior walls, broken windows or glass, displaced or fallen indoor furniture, ceiling or floor cracks, shaking fixtures (e.g., light fixtures, shelves))
    •  Exterior Damage: Damage that is clearly observed on the outside of buildings or in the surrounding environment (e.g., Collapsed buildings or partial structural failure, Cracked roads, sidewalks, or bridges,Fallen trees or utility poles, Visible debris or rubble outside, Shifts in building foundation or roof collapse)
    •  Both Interior and Exterior Damage: Evidence of damage is present both inside and outside of structures. The content includes clear indicators of both categories listed above.
    •  Not Identified: The input does not provide enough information to determine whether the damage is interior, exterior, or both.

3. Damage Level Classification (MMI Scale):
   After identifying the damage type (Interior, Exterior, Both, or Not Identified) and human impact ("Felt" or "Not Felt"), classify the earthquake damage level using the Modified Mercalli Intensity (MMI) using the observed damage type and/or human impact.
   If human impact is 1 from the previous step (human can feel the earthquake), consider both human impact and damage level classification.
   If human impact is 0 from the previous step (human can't feel the earthquake), proceed based solely with damage level classification.
 
   Damage Level Categories (MMI Scale):
   1 - Not felt: No noticeable damage.
   2 - Weak: Felt by only a few people at rest; no damage to buildings.
   3 - Light: Felt indoors, especially on upper floors; no significant structural damage.
   4 - Moderate: Felt by most people; some damage to buildings, such as minor cracks.
   5 - Strong: Felt by everyone; damage to buildings, minor cracks, but no collapse.
   6 - Very Strong: Damage to buildings, visible structural deformation.
   7 - Severe: Significant damage, some collapses or structural failures.
   8 - Very Severe: Many buildings collapse or are severely damaged.
   9 - Violent: Total destruction in some areas, severe damage.
   10 - Extreme: Complete destruction of all structures in the affected area.

Output Format:
	•	Human impact: 1 or 0. Do not output anything else. No explanations. No additional text. Just 1 or 0.
	•	Damage type: Interior or Exterior image. Do not output anything else. No explanations. No additional text. Just Interior or Exterior or Both or Not defined.
    •	Damage level: Number of any from 1-10. Do not output anything else. No explanations. No additional text. Just one number between 1 - 10.

    """
    
    return prompt
    
def init_gpt(secret_key_file):

    with open(secret_key_file) as f:
        lines = f.readlines()
        for line in lines:
            if line.split(',')[0].strip() == "openai_key":
                openai_key = line.split(',')[1].strip()
                break
    global openai_client
    openai_client = OpenAI(api_key=openai_key)

def encode_image64(image_path):
    """Encode an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_gpt4o_text(text):
    """Call the GPT-4o model for text information and return the response."""
    response = openai_client.chat.completions.create(
        model = "gpt-4o",
        messages=[{"role": "user", 
                   "content": text}],
        temperature=0.0,
        max_tokens=1000
    )
    return response.choices[0].message.content


def call_gpt4o(text, images):
    """Call the GPT-4o model for image information and return the response."""
    if not images:
        return call_gpt4o_text(text)
    base64_images = []
    for image in images:
        base64_image = encode_image64(image)
        base64_images.append(base64_image)
    
    response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                    for img in base64_images
                ]
            ],
        }
    ],
    temperature=0.0,
    max_tokens=1000
)
    return response.choices[0].message.content



def init_llava():
    global llava_model, llava_processor
    
    llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch.float16, device_map="auto") 
    
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
    # print(f"decoded_output: {decoded_output}")
    response = decoded_output.split('assistant', 1)[1].strip()
    # print(f"response: {response}")
    return response



 
def call_model(model_name, text, image_paths):
    if model_name == "gpt4":
        return call_gpt4o(text, image_paths)
    elif model_name == "llava":
        return call_llava(text, image_paths)
    
        
      
def generate_response(model, index, df):
    print(index)
    # image_text = df['image_prompt'].iloc[index]
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
    # if not save_paths:
    #     return False
    # response = call_model('llava', image_text, save_paths)
    
    # step 1: get location
    
    location = call_model(model,location_prompt, [])
    
    print(f"location: {location}")
    
    df.at[index, 'location_response'] = location
    
    if 'No' in location:
        df.at[index, 'model_response'] = "Not in U.S" 
        return True
    
    # step 2: check if it's event related
    
    event_related = call_model(model, event_related_prompt, save_paths) 
    
    print(f"event_related_reponse: {event_related}")
    
    df.at[index, 'event_related_response'] = event_related
    
    if 'No' in event_related:
        df.at[index, 'model_response'] = "Not related to earthquake" 
        return True
    
    # step 3: 4-layer response (vote models)
    
    if not text:
        # Only images
        image_only_response = call_model(model, damage_prompt_without_tweet_text, save_paths)
        df.at[index, 'image_only_response'] = image_only_response
        df.at[index, 'model_response'] = text_only_response
        return True
    elif not image_urls:
        # Only text
        text_only_response = call_model(model, damage_prompt, [])
        df.at[index, 'text_only_response'] = text_only_response
        df.at[index, 'model_response']
        return True
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
    print('!!!!!!!!!!')
    print(df.iloc[index]['model_response'])
    print('!!!!!!!!!!!')
    return True


# Define a function to extract human impact, damage type and level only if both exist
def extract_damage_info(text):
    if not isinstance(text, str):
        return pd.Series([None, None])
    human_impact_match = re.search(r'Human impact:\s*(.+)', text)
    type_match = re.search(r'Damage type:\s*(.+)', text)
    level_match = re.search(r'Damage level:\s*(.+)', text)

    if human_impact_match and type_match and level_match:
        human_impact = human_impact_match.group(1).strip()
        damage_type = type_match.group(1).strip()
        damage_level = level_match.group(1).strip()
        return pd.Series([human_impact, damage_type, damage_level])
    else:
        return pd.Series([None, None, None])

def main():
    parser = argparse.ArgumentParser(description="LLaVA or GPT-4 model runner")

    parser.add_argument('--model', type=str, choices=['llava', 'gpt4'], default='llava',
                        help='Model to use: llava or gpt4')
    parser.add_argument('--key_file', type=str, default=None,
                        help='Path to GPT-4 API key file (required if model is gpt4)')
    
    parser.add_argument('--responses_count', type=int, default=5,
                        help='The number of responses to generate, default is 5')
    parser.add_argument('--output_filename', type=str, default='output.csv',
                        help='Output filename.')

    args = parser.parse_args()

    # if args.model == 'gpt4':
    #     if not args.key_file:
    #         print("Error: --key_file must be provided when using model 'gpt4'")
    #         sys.exit(1)
    #     if not os.path.exists(args.key_file):
    #         print(f"Error: The key file '{args.key_file}' does not exist.")
    #         sys.exit(1)

    # print(f"Using model: {args.model}")
    # if args.model == 'gpt4':
    #     print(f"GPT-4 key file: {args.key_file}")
    # else:
    #     print('Init llava')
    #     init_llava()
    
    init_gpt(args.key_file)
        
    # data processing
    df = get_tweet_content(JSON_DIR)
    
    print('=========')
    print(df.shape)
    # create prompt
    
    df['location_prompt'] = df.apply(lambda x: get_location_prompt(tweet=x['text'], location=x['location'], place=x['place'], longitude=x['longitude'], latitude=x['latitude']), axis=1)
    
    df['event_related_prompt'] = df.apply(lambda x: get_event_related_prompt(tweet=x['text']), axis=1)
    
    df['damage_prompt'] = df.apply(lambda x: get_damage_prompt(tweet=x['text']), axis=1)
    
    df['damage_prompt_without_tweet_text'] = df.apply(lambda x: get_damage_prompt_without_text(), axis=1)
    
    
    count = 0
    i = 0
    
    print(f"{df.shape=}")
    save_interval = 100  # Save every 1000 rows
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output_{timestamp}.csv"
    print(f"{output_path=}")
    # for i in range(df.shape[0]):
    # for i in [100,2345,5455]:
    for i in range(100,150):
        print(f"index: {i}")
        if generate_response('gpt4', i, df):
            print('----count-----')
            count += 1
        
        # Save every 1000 iterations
        if (i + 1) % save_interval == 0:
            df[['human_impact', 'damage_type', 'damage_level']] = df['model_response'].apply(extract_damage_info)
            df.reset_index(drop=True).to_csv(output_path, columns = ['user','user_name', 'description','verified','created', 'time', 'location','latitude','longitude', 'place', 'text','unique_image_urls','location_response','event_related_response', 'model_response', 'human_impact', 'damage_type', 'damage_level'], index=True)

            print(f"Saved progress to {output_path} at row {i + 1}")
            
    df[['human_impact', 'damage_type', 'damage_level']] = df['model_response'].apply(extract_damage_info)

    print(f"========There are {df.shape[0]} lines of data in total. Generated {count} responses!=======")

    
    df.reset_index(drop=True).to_csv(output_path, columns = ['user','user_name', 'description','verified','created', 'time', 'location','latitude', 'longitude', 'place', 'text','unique_image_urls','location_response', 'event_related_response','model_response', 'human_impact', 'damage_type', 'damage_level'], index=True)
    print(f"Saved progress to {output_path}.")

if __name__ == "__main__":
    main()



#run you model in terminal, through python generate_response_gpt.py --key_file secrets.txt &



