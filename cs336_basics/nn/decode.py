import torch

from .config import Config
from .util import softmax, load_checkpoint
from .transformer import TransformerLM
from cs336_basics.tokenizer import Tokenizer

class Decoder():
    def __init__(
        self,
        config: Config | str,       # model config(config.json)
        ckpt_path: str,           # model parameters(checkpoint)
        vocab_path: str,
        merges_path: str
    ) -> None:
        if isinstance(config, str):
            config = Config.from_json(config)
        mc = config.model

        self.model = TransformerLM(
            vocab_size=mc.vocab_size,
            context_length=mc.context_length,
            num_layers=mc.num_layers,
            d_model=mc.d_model,
            num_heads=mc.num_head,
            d_ff=mc.d_ff,
            theta=mc.theta,
            device=torch.device(mc.device),
            dtype=getattr(torch, mc.dtype)
        )
        load_checkpoint(ckpt_path, self.model)  # load parameters
        self.model.eval()
        self.device = mc.device 
        self.model.to(self.device)

        self.context_length = mc.context_length
        self.tokenizer = Tokenizer.from_files(vocab_path, merges_path)
        self.endtoken = "<|endoftext|>"

    @torch.no_grad()
    def decode(
        self,
        prompt: str, 
        max_length: int=50,
        temperature: float=0.9, 
        p:float=0.9
    ) -> str:
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        end_token_id =  self.tokenizer.encode(self.endtoken)[0]

        max_length = min(max_length, self.context_length)
        while len(prompt_ids) < max_length:
            logits = self.model(prompt_tensor)
            next_token_logits = logits[0, -1, :]
            next_token_logits = next_token_logits.flatten()

            # temperature
            if temperature > 0.0:
                next_token_logits /= temperature
            else:
                next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
                prompt_ids.append(next_token_id)
                if next_token_id == end_token_id:
                    break
                prompt_tensor = torch.tensor([prompt_ids], device=self.device)
                continue
            
            # top-p
            next_token_logits = softmax(next_token_logits, dim=-1)
            sorted_probs, index = next_token_logits.sort(dim=-1, descending=True)

            cum_prob = 0.0
            count = 0
            while cum_prob < p:
                cum_prob += sorted_probs[count].item()
                count += 1
            
            top_probs = sorted_probs[:count]
            top_probs = top_probs / top_probs.sum()
            top_indices = index[:count]

            # get next token
            sampled_id = int(torch.multinomial(top_probs, 1).item())
            next_token_id = int(top_indices[sampled_id].item())
            
            prompt_ids.append(next_token_id)
            if next_token_id == end_token_id:
                break
            prompt_tensor = torch.tensor([prompt_ids], device=self.device)

        return self.tokenizer.decode(prompt_ids)


# ============== test ====================
# python -m cs336_basics.nn.decode
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    def get_abs_path(rel_path: str) -> str:
        return os.path.normpath(os.path.join(current_dir, rel_path))

    config = get_abs_path("../../experiments/config.json")
    ckpt_path = get_abs_path("../../experiments/TinyStories/lr_0.001/checkpoint_5000.pt")
    vocab_path = get_abs_path("../../results/TinyStories/vocab.json")
    merges_path = get_abs_path("../../results/TinyStories/merges.txt")

    # decoder
    decoder = Decoder(config, ckpt_path, vocab_path, merges_path)
    prompt = "Once upon a time"
    output = decoder.decode(prompt, 256, temperature=1.2, p=0.9)
    print(output)