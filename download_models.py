"""
Pre-download all models needed for the 2x2 ATR experiment.
Downloads are cached by HuggingFace, so notebooks will use the cache.
Run: python download_models.py
"""
from transformer_lens import HookedTransformer

models = ["pythia-160m", "gpt2-medium", "pythia-410m"]

for name in models:
    print(f"\n{'='*50}")
    print(f"  Downloading: {name}")
    print(f"{'='*50}")
    model = HookedTransformer.from_pretrained(name, device="cpu")
    print(f"  ✓ {name}: {model.cfg.n_layers}L, {model.cfg.n_heads}H, d={model.cfg.d_model}")
    del model  # Free memory before loading next

print(f"\n✅ All models cached. Notebooks will load instantly.")
