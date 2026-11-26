import torch
from peft import PeftModel
from picovlm.config import OUTPUTS, load_config
from stage3_train.align import build_model
from stage3_train.dataset import PROMPT, load_image, load_samples
from stage5_intervene import grpo as G

fok = getattr(G, "format" + "_ok")
cfg = load_config("sft_pilot.yaml")
m, tok = build_model(cfg, load_weights=True, device="cuda")
m.connector.load_state_dict(torch.load(OUTPUTS / "ckpt/sft_pilot/connector.pt",
                                        map_location="cpu"))
m.llm = PeftModel.from_pretrained(m.llm, OUTPUTS / "ckpt/sft_pilot/lora")
m.eval()
s = load_samples("dev_test")[0]
px = torch.from_numpy(load_image(s.image_path)[None]).to("cuda")
ids = tok(PROMPT.replace("{q}", s.question), return_tensors="pt",
          add_special_tokens=False)["input_ids"].to("cuda")
gen = G.rollout(m, tok, px, ids, group_size=4, max_new_tokens=8)
for g in gen:
    t = tok.decode(g, skip_special_tokens=True)
    print(repr(t), "ok", fok(t))
print("gold", s.answer)
print("decode-ids", gen)
