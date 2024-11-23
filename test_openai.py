import os
from openai import OpenAI
import yaml

# Load the openai_token from config.yaml
def load_openai_key_from_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
        return config.get("openai_token")
    


if __name__=="__main__":
    # Path to your config.yaml
    config_path = "config.yaml"

    # Set the OpenAI key
    openai_token = load_openai_key_from_config(config_path)
    # # # Configure the credentials
    # # if openai_token:
    # #     SKLLMConfig.set_openai_key(openai_token)
    # # else:
    # #     raise ValueError("OpenAI token not found in the config file!")
    
    # print(openai_token)
    client = OpenAI(
        api_key=openai_token,  # This is the default and can be omitted
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="llama3-8b-8192",
    )