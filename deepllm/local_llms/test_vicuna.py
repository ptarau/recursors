print('!!!!! make sure server.py has started !!!!')
print('info at: https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md')
import openai
import time

openai.api_key = "EMPTY"  # Not support yet
openai.api_base = "http://u.local:8000/v1"

model = "vicuna-7b-v1.5"


def complete(prompt="Once upon a time"):
    # create a completion
    completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
    # print the completion
    print(prompt + completion.choices[0].text)


def step(text):
    # create a chat completion
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": text}]
    )
    # return the completion
    return completion.choices[0].message.content


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
