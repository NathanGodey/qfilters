# Q-Filters: Leveraging Query-Key Geometry for Efficient Key-Value Cache Compression

[![arXiv](https://img.shields.io/badge/arXiv-2503.02812-b31b1b.svg)](https://arxiv.org/abs/2503.02812)

<p align="center">
  <img width=50% height=auto src="qfilters_demo.gif" />
</p>


> **Abstract**: Autoregressive language models rely on a Key-Value (KV) Cache, which avoids re-computing past hidden states during generation, making it faster. As model sizes and context lengths grow, the KV cache becomes a significant memory bottleneck, which calls for compression methods that limit its size during generation. In this paper, we discover surprising properties of Query (Q) and Key (K) vectors that allow us to efficiently approximate attention scores without computing the attention maps. We propose Q-Filters, a training-free KV cache compression method that filters out less crucial Key-Value pairs based on a single context-agnostic projection. Contrarily to many alternatives, Q-Filters is compatible with FlashAttention, as it does not require direct access to attention weights. Experimental results in long-context settings demonstrate that Q-Filters is competitive with attention-based compression methods such as SnapKV in retrieval tasks while consistently outperforming efficient compression schemes such as Streaming-LLM in generation setups. Notably, Q-Filters achieves a 99% accuracy in the needle-in-a-haystack task with a x32 compression level while reducing the generation perplexity drop by up to 65% in text generation compared to Streaming-LLM.


## Setup
1. Install required libraries in a virtual environment:
```bash
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
````
2. Configure HuggingFace\'s environment:
```bash
export HF_DATASETS_CACHE=<path_to_hf_cache>
export HF_HOME=<path_to_hf_cache>
export HF_TOKEN=<hf_token>
```

## Generate with Q-Filters
Here is an example of how to use Q-Filters in a generation setup:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from src.hf_cache import QFiltersCache
from datasets import load_dataset

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype="bfloat16"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

question = """What is the probability of two integers selected at random having a greatest common divisor of 1."""
input_text = f"<|User|>{question}<|Assistant|><think>\n"

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

past_key_values = QFiltersCache(
    window_length=64,
    max_length=128, 
    model_name=model_name
)

out = model.generate(
    **inputs,
    do_sample=True, 
    temperature=0.5, 
    max_new_tokens=4096, 
    past_key_values=past_key_values, 
    streamer=streamer
)
```

## Compute Q-Filters for a new model
1. Verify that the target model does not already have [pre-computed Q-Filters](https://huggingface.co/collections/nthngdy/q-filters-67a4994dcb302a3d37f3d119).
2. Use the `make_filters.py` script to generate the filters. For instance:
```bash
python make_filters.py \
--model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--model_cls Qwen2ForCausalLM \
--max_seq_len 2048 \
--num_sequences 10 \
--num_svd_samples 3000 \
--dataset_name PatrickHaller/fineweb-1B \
--save_mode disk \
# --save_mode hub \
# --save_mode hub+disk \
# --hf_user_id nthngdy \
--save_dir ../filters
```
3. For Q-Filters saved on disk, you can upload them later using this command:
```bash
huggingface-cli upload path_to_hf_repo path_to_local_qfilters .
```

## Citation
```bibtex
@misc{godey2025qfiltersleveragingqkgeometry,
      title={Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression}, 
      author={Nathan Godey and Alessio Devoto and Yu Zhao and Simone Scardapane and Pasquale Minervini and Éric de la Clergerie and Benoît Sagot},
      year={2025},
      eprint={2503.02812},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.02812}, 
}
```
