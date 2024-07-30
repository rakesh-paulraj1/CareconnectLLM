from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceLLM

model_name = "ruslanmv/Medical-Llama3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = PromptTemplate(template="This is a demo statement from our patient: {statement}")
llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)
chain = LLMChain(prompt=prompt, llm=llm)

statement = "I have been experiencing severe headaches and dizziness."
result = chain.run(statement=statement)

print(result)
