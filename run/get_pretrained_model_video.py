# Load model directly
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("YanweiLi/llama-vid-13b-pretrain-224-video-fps-1")
model = AutoModelForCausalLM.from_pretrained("YanweiLi/llama-vid-13b-pretrain-224-video-fps-1")