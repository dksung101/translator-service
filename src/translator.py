import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
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
        print(output)

        if output.startswith("1 ") or output.startswith("0 "):
            return (output[0] == '1', output[2:])
        else:
            return (False, "Unexpected translation error.")
    except Exception as e:
        print(f"Error in query_llm_robust: {e}")
        return (False, "Error processing the request.")

    # if content == "这是一条中文消息":
    #     return False, "This is a Chinese message"
    # if content == "Ceci est un message en français":
    #     return False, "This is a French message"
    # if content == "Esta es un mensaje en español":
    #     return False, "This is a Spanish message"
    # if content == "Esta é uma mensagem em português":
    #     return False, "This is a Portuguese message"
    # if content  == "これは日本語のメッセージです":
    #     return False, "This is a Japanese message"
    # if content == "이것은 한국어 메시지입니다":
    #     return False, "This is a Korean message"
    # if content == "Dies ist eine Nachricht auf Deutsch":
    #     return False, "This is a German message"
    # if content == "Questo è un messaggio in italiano":
    #     return False, "This is an Italian message"
    # if content == "Это сообщение на русском":
    #     return False, "This is a Russian message"
    # if content == "هذه رسالة باللغة العربية":
    #     return False, "This is an Arabic message"
    # if content == "यह हिंदी में संदेश है":
    #     return False, "This is a Hindi message"
    # if content == "นี่คือข้อความภาษาไทย":
    #     return False, "This is a Thai message"
    # if content == "Bu bir Türkçe mesajdır":
    #     return False, "This is a Turkish message"
    # if content == "Đây là một tin nhắn bằng tiếng Việt":
    #     return False, "This is a Vietnamese message"
    # if content == "Esto es un mensaje en catalán":
    #     return False, "This is a Catalan message"
    # if content == "This is an English message":
    #     return True, "This is an English message"
    # return True, content
