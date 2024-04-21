print('!!!!! make sure server.py has started !!!!')

# Set OpenAI's API key and API base to use vLLM's API server.
import openai
import time

openai_api_key = "EMPTY"
openai_api_base = "http://u.local:8000/v1"

model = "meta-llama/Meta-Llama-3-8B-Instruct"


client = openai.OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base
)

def step(prompt):
  chat_response = client.chat.completions.create(
    model=model,
    messages=[
        #{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    stream=False,
    extra_body={"stop_token_ids":[128009]}
  )
  #print("Chat response:", chat_response[0].text)
  return chat_response.choices[0].message.content


def run():
    while True:
        text = input('Q: ')
        if not text:
            break
        t1 = time.time()
        r = step(text)
        t2 = time.time()
        print('A:', r)
        print('TIME:', round(t2 - t1, 2))


if __name__ == "__main__":
    run()
