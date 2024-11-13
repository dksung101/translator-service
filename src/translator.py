import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# load_dotenv()
# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key = os.getenv("API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint="https://ellia-yang-openai.openai.azure.com/"
)

def translate_content(content: str) -> tuple[bool, str]:
    context = (
        "Check if this text is written in English or not and translate it to English. "
        "If the text is unintelligible or malformed, return '0 Error processing post'. "
        "If the text is in English, return '1 {Original Text}'. "
        "If the text is not in English, return '0 {Text Translated to English}'. Here is the text:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": context},
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        output = response.choices[0].message.content
        # print(output)

        if output.startswith("1 ") or output.startswith("0 "):
            return (output[0] == '1', output[2:])
        else:
            return (False, "Unexpected translation error.")
    except Exception as e:
        print(f"Error in query_llm_robust: {e}")
        return (False, "Error processing the request.")
