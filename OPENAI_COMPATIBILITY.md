# MM-Rec OpenAI Uyumluluk Rehberi

**Versiyon**: 1.0  
**Tarih**: 2025-12-08  
**Hedef**: OpenAI Chat API uyumlu eğitim ve inference

---

## ✅ Mevcut Özellikler

### 1. Chat Format Support ✅
- **Dosya**: `mm_rec/data/chat_format.py`
- **Özellikler**:
  - System/User/Assistant mesaj formatı
  - OpenAI Chat API format parsing
  - Training pair generation (input/target)
  - Special token support (`<|system|>`, `<|user|>`, `<|assistant|>`)

### 2. OpenAI Tokenizer Support ✅
- **Dosya**: `mm_rec/tokenizers/openai_tokenizer.py`
- **Özellikler**:
  - tiktoken entegrasyonu (GPT-4, GPT-3.5)
  - Fallback simple tokenizer
  - Vocabulary size auto-detection
  - Special tokens (EOS, PAD)

### 3. SFT Training ✅
- **Dosya**: `mm_rec/training/sft_trainer.py`
- **Özellikler**:
  - Supervised Fine-Tuning trainer
  - Loss masking (sadece assistant responses predict)
  - Chat format input preparation
  - OpenAI-compatible training loop

### 4. Chat Completion API ✅
- **Dosya**: `mm_rec/training/sft_trainer.py` (ChatCompletionAPI class)
- **Özellikler**:
  - OpenAI Chat Completion API format
  - Temperature/top_p sampling
  - Message formatting
  - Response extraction

---

## 🚀 Kullanım

### 1. OpenAI Tokenizer Kurulumu

```bash
pip install tiktoken
```

### 2. Chat Format Dataset Oluşturma

```python
from mm_rec.data.chat_format import ChatFormatter, create_chat_example
import json

# Örnek OpenAI format
example = create_chat_example()
# {
#   "messages": [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is Python?"},
#     {"role": "assistant", "content": "Python is a programming language."}
#   ]
# }

# JSONL dosyasına kaydet
with open("chat_data.jsonl", "w") as f:
    f.write(json.dumps(example) + "\n")
```

### 3. OpenAI Tokenizer ile Eğitim

```python
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig
from mm_rec.models.mmrec_100m import MMRec100M
import torch

# Model ve tokenizer
model = MMRec100M(vocab_size=100256)  # GPT-4 vocab size
tokenizer = get_tokenizer(model_name="gpt-4")

# SFT Trainer
config = SFTConfig(
    model_name="gpt-4",
    max_length=2048,
    only_predict_assistant=True
)
trainer = SFTTrainer(model, tokenizer, config)

# Eğitim
messages = [
    ChatMessage(role="system", content="You are helpful."),
    ChatMessage(role="user", content="Hello!"),
    ChatMessage(role="assistant", content="Hi! How can I help?")
]

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
metrics = trainer.train_step(messages, optimizer, device=torch.device('cpu'))
print(f"Loss: {metrics['loss']}, Perplexity: {metrics['perplexity']}")
```

### 4. OpenAI Chat Completion API

```python
from mm_rec.training.sft_trainer import ChatCompletionAPI

# API instance
api = ChatCompletionAPI(model, tokenizer)

# Chat completion
response = api.create(
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is Python?"}
    ],
    max_tokens=100,
    temperature=0.7,
    device=torch.device('cpu')
)

print(response["choices"][0]["message"]["content"])
```

---

## 📋 OpenAI Uyumluluk Checklist

### ✅ Tamamlananlar

- [x] Chat format support (system/user/assistant)
- [x] OpenAI tokenizer (tiktoken)
- [x] SFT training format
- [x] Loss masking (only assistant responses)
- [x] Chat Completion API format
- [x] Message formatting
- [x] Special tokens

### ⚠️ Eksikler / İyileştirmeler

- [ ] **Function Calling**: OpenAI function calling API desteği
- [ ] **Streaming**: Streaming response support
- [ ] **Tool Use**: Tool/function calling support
- [ ] **System Message Handling**: Daha gelişmiş system message işleme
- [ ] **Multi-turn Conversations**: Uzun konuşmalar için optimizasyon
- [ ] **Fine-tuning API**: OpenAI Fine-tuning API uyumlu endpoint
- [ ] **Evaluation Metrics**: Chat-specific metrics (BLEU, ROUGE, etc.)

---

## 🔧 Teknik Detaylar

### Chat Format

**Format**:
```
<|system|>
You are a helpful assistant.
<|endoftext|>
<|user|>
What is Python?
<|endoftext|>
<|assistant|>
Python is a programming language.
<|endoftext|>
```

**Training**:
- Input: System + User messages
- Target: Assistant response only
- Loss: Only computed on assistant tokens

### Tokenizer

**OpenAI Models**:
- `gpt-4`: cl100k_base (100256 tokens)
- `gpt-3.5-turbo`: cl100k_base (100256 tokens)
- `text-davinci-003`: p50k_base (50257 tokens)

**Fallback**:
- SimpleTokenizer: Character-level hashing
- Vocab size: 32000 (default)

### Loss Masking

**Strategy**:
- Only predict assistant responses
- Ignore system/user tokens (label = -100)
- Compute loss only on assistant tokens

**Implementation**:
```python
labels = [-100] * len(input_ids)  # Ignore all
labels[assistant_start:] = target_ids  # Predict assistant
```

---

## 📊 Örnek Eğitim Script

```python
import torch
from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig
from mm_rec.data.chat_format import ChatDataset, ChatMessage

# Model
model = MMRec100M(vocab_size=100256)
tokenizer = get_tokenizer("gpt-4")

# Dataset
dataset = ChatDataset("chat_data.jsonl")

# Trainer
config = SFTConfig(max_length=2048, only_predict_assistant=True)
trainer = SFTTrainer(model, tokenizer, config)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(10):
    for item in dataset:
        messages = item['messages']
        metrics = trainer.train_step(messages, optimizer, device)
        print(f"Epoch {epoch}, Loss: {metrics['loss']}")
```

---

## 🎯 OpenAI API Uyumluluğu

### Chat Completion Endpoint

**Request**:
```python
POST /v1/chat/completions
{
  "model": "mm-rec-100m",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Response**:
```python
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hi! How can I help?"
    },
    "finish_reason": "stop"
  }]
}
```

### Fine-tuning Format

**Training Data** (JSONL):
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

---

## ⚠️ Önemli Notlar

1. **Vocabulary Size**: OpenAI tokenizer (cl100k_base) 100256 token kullanır. Model vocab_size'ı buna göre ayarlanmalı.

2. **Loss Masking**: Sadece assistant responses predict edilir. System/user tokens ignore edilir.

3. **Special Tokens**: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|endoftext|>` tokenizer tarafından handle edilir.

4. **Memory**: Chat format uzun konuşmalar için memory-efficient olmalı (MM-Rec'in avantajı).

---

## 📖 Referanslar

- [OpenAI Chat API](https://platform.openai.com/docs/api-reference/chat)
- [OpenAI Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
- [tiktoken](https://github.com/openai/tiktoken)

---

**Son Güncelleme**: 2025-12-08  
**Hazırlayan**: MM-Rec Development Team

