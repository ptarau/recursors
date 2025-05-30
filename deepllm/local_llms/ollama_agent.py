import time
import ollama
from deepllm.params import LOCAL_PARAMS, tprint


def ask_ollama(query, model=None):
    tprint("\nENTER OLLAMA -------------------")
    tprint("QUERY:", query)
    if model is None:
        model = LOCAL_PARAMS["model"]

    t1 = time.time()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Return short and precise answers!"},
            {
                "role": "user",
                "content": query,
            },
        ],
        options={"seed": 42, "temperature": 0},
    )
    t2 = time.time()
    # print(response)

    answer = response["message"]["content"]
    tprint("\nOLLAMA ANSWER:", answer)
    tprint("OLLAMA TIME:", round(t2 - t1, 2))
    tprint("EXIT OLLAMA -------------------")
    return [answer], 0, 0, 0


def test_local_agents():
    print(ask_ollama("What is generative AI?"))
    print(ask_ollama("How are transformers used in LLMs?"))
    print(ask_ollama("What is the twin primes conjecture?"))
    print(ask_ollama("Why is the Riemann conjecture hard to prove?"))


if __name__ == "__main__":
    test_local_agents()
