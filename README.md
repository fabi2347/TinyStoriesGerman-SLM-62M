# 🧸 TinyStoriesGerman-SLM (~62M parameters)

A lightweight **German small language model (~62M parameters)** trained on the [TinyStoriesGerman dataset](https://huggingface.co/datasets/fabi2347/TinyStoriesGerman).  
The model is designed to generate **simple children’s stories in German** with limited vocabulary and short context windows.

---

## 📌 Model Details

- **Architecture:** LLaMA-based Causal Language Model (~62M parameters)  
- **Vocabulary Size:** 32,000 (custom ByteLevel BPE tokenizer)  
- **Context Length:** 256 tokens  
- **Training Objective:** Next-token prediction (Causal LM)  
- **Language:** German (`de`)  
- **Dataset:** [fabi2347/TinyStoriesGerman](https://huggingface.co/datasets/fabi2347/TinyStoriesGerman)  
- **License:** CDLA-Sharing-1.0  

The model was built as a **tiny-scale experiment** to train a German LLaMA-like model for story generation tasks.

---

## 🚀 How to Use

You can use the model directly with Hugging Face `transformers`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

repo_id = "fabi2347/TinyStoriesGerman-62M"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id)

gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

prompt = "Es war einmal"

result = gen_pipeline(
    prompt,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    early_stopping=True,
    no_repeat_ngram_size=2
)
generated_text = result[0]["generated_text"]
new_text = generated_text[len(prompt):].strip()
print(f"{prompt} {new_text}")
```

---

## 🛠 Training Setup

- **Tokenizer:** Custom ByteLevel BPE, vocab size = 32k  
- **Sequence Length:** 256 tokens  
- **Training Samples:** ~1.9M German stories  
- **Optimizer:** AdamW (fused)  
- **Precision:** FP16 mixed precision  
- **Effective Batch Size:** 24 × 2 (grad accumulation)  
- **Learning Rate:** 4e-4 (with warmup)  
- **Epochs:** 1  

**Model configuration:**
- Hidden size: 384  
- Layers: 16  
- Attention heads: 8  
- Intermediate size: 1536  

📊 **Parameters:** ~62M

---

## ✨ Example Outputs

Prompts and sample generations (temperature=0.7, top_p=0.9):

- **Prompt:** *"Es war einmal"*  

Es war einmal ein kleiner Junge namens Timmy. Timmy liebte es, mit seinen Spielsachen zu spielen, besonders mit seinem Spielzeugauto. Eines Tages ging Timmys Spielzeugauto kaputt und er war sehr traurig.

Timmys Mutter sah ihn weinen und fragte: „Was ist los, Timmy?“
„Mein Auto ist kaputt“, sagte Timmy traurig und zeigte auf sein kaputtes Auto. „Es ist ein kaputtes Teil.“
Seine Mutter sagte: „Mach dir keine Sorgen, wir können es reparieren.“ Sie nahm das kaputte Teil und reparierte es für Timmy, sodass es wieder wie neu war. Als Timmy sein Spielzeugauto sah, war er überglücklich und umarmte seine Mutter fest.

- **Prompt:** *"In einem magischen Wald"*  

In einem magischen Wald lebte ein kleiner Bär. Er war sehr einsam und wünschte sich einen Freund.

Eines Tages kam ein freundlicher Fuchs zu dem Bären. Der Fuchs sagte: „Hallo, Bär! Willst du mein Freund sein?“ Der Bär war so glücklich und sagte ja!
Der Fuchs und der Bär spielten den ganzen Tag zusammen. Sie rannten, sprangen und lachten. Als die Sonne unterging, verabschiedeten sie sich. Dann gingen sie nach Hause und schliefen.

---

## 🙏 Acknowledgements

- Original [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan & Yuanzhi Li  
- Translation to German: [fabi2347/TinyStoriesGerman](https://huggingface.co/datasets/fabi2347/TinyStoriesGerman)  
- Model architecture inspired by [LLaMA](https://arxiv.org/abs/2302.13971)  