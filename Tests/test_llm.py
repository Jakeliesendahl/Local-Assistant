# tests/test_llm.py
from core.llm import LLMClient

if __name__ == "__main__":
    llm = LLMClient(model="llama3")  # or "mistral", etc.
    text = llm.chat("list the animals in the zoo", stream=False)
    print("MODEL SAID:", text)
