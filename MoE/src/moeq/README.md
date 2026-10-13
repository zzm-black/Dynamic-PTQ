# Vendored quantization code

These files are vendored from the official repository of
*Efficient Quantization of Mixture-of-Experts with Theoretical Generalization
Guarantees* (ICLR 2026, arXiv:2604.06515):
https://github.com/nowazrabbani/moe_quantization

They implement the GPTQ quantizer (1/2/3/4-bit paths, per-group MSE grid
search), the router-norm / MaxVar / combined expert orderings, and the
three-level bit allocator.

Local changes:

- `modelutils.py`: load the model from a caller-supplied path.
- `eval_ppl_utils.py`: capture and pass `position_ids` to decoder layers
  (required by transformers >= 4.4x) and return the perplexity.
- `datautils.py`: trimmed to the WikiText2 loader.
- Unused loaders for other model families were removed.

No quantization semantics were changed.
