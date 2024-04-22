# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.5")