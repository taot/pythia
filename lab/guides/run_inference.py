from transformers import GPTNeoXForCausalLM, AutoTokenizer

from lab import utils


MODEL_NAME = "EleutherAI/pythia-160m-deduped"
REVISION = "step143000"

cache_dir = utils.get_cache_dir(MODEL_NAME, REVISION)

model = GPTNeoXForCausalLM.from_pretrained(
  MODEL_NAME,
  revision=REVISION,
  cache_dir=cache_dir,
)

tokenizer = AutoTokenizer.from_pretrained(
  MODEL_NAME,
  revision=REVISION,
  cache_dir=cache_dir,
)

inputs = tokenizer("The capital of France is", return_tensors="pt")
print(inputs)
tokens = model.generate(**inputs)
print(tokenizer.decode(tokens[0]))
