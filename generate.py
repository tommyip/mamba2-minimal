import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mamba2 import Mamba2LMHeadModel

torch.manual_seed(42)

device = torch.device("mps")  # cpu, cuda, mps

PROMPT = "My model cutoff date is"
MAX_LEN = 50
TOP_K = 1

model = Mamba2LMHeadModel.from_pretrained("state-spaces/mamba2-130m", device=device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id

tokens = tokenizer(PROMPT, return_tensors="pt")
input_ids = tokens.input_ids.to(device)
print(tokenizer.decode(input_ids.tolist()[0]), end="")

for _ in range(MAX_LEN):
    logits = model(input_ids)[0, -1]
    probs = F.softmax(logits, dim=0)

    probs, indices = torch.topk(probs, k=TOP_K)
    probs /= probs.sum()
    next_index_topk = torch.multinomial(probs, num_samples=1)
    next_index = indices[next_index_topk]

    if next_index == tokenizer.eos_token_id:
        break

    input_ids = torch.cat((input_ids, next_index.unsqueeze(0)), dim=-1)

    output = tokenizer.decode(next_index.tolist())
    print(output, end="", flush=True)

print()
