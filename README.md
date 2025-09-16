# ChatTTS
[**English**](./README.md) | [**中文简体**](./README_CN.md)

ChatTTS is a text-to-speech model designed specifically for dialogue scenarios, such as LLM assistants. It supports both English and Chinese languages. Our model is trained with over 100,000 hours of Chinese and English data. The open-source version on HuggingFace is a 40,000-hour pretrained model without SFT.

For formal inquiries about the model and roadmap, please contact us at open-source@2noise.com. You can also join our QQ group: 808364215 for discussions. GitHub issues are always welcome.

---

## Table of Contents
- [Highlights](#highlights)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Advanced Usage](#advanced-usage)
- [API Documentation](#api-documentation)
- [Roadmap](#roadmap)
- [FAQ](#faq)
- [Disclaimer](#disclaimer)
- [Acknowledgements](#acknowledgements)
- [Special Appreciation](#special-appreciation)

---

## Highlights
1.  **Conversational TTS**: ChatTTS is optimized for dialogue-based tasks, enabling natural and expressive speech synthesis. It supports multiple speakers, facilitating interactive conversations.
2.  **Fine-grained Control**: The model can predict and control fine-grained prosodic features, including laughter, pauses, and interjections.
3.  **Better Prosody**: ChatTTS surpasses most open-source TTS models in terms of prosody. We provide pretrained models to support further research and development.

For a detailed description of the model, you can refer to the [video on Bilibili](https://www.bilibili.com/video/BV1zn4y1o7iV).

---

## Project Structure
The repository is organized as follows:
```
.
├── ChatTTS/
│   ├── __init__.py
│   ├── core.py           # Main Chat class for inference
│   ├── experimental/
│   │   └── llm.py        # Experimental LLM API wrapper
│   ├── infer/
│   │   └── api.py        # Inference API for text refinement and code generation
│   ├── model/
│   │   ├── dvae.py       # Discrete VAE model
│   │   └── gpt.py        # GPT-style model for text and audio generation
│   └── utils/
│       ├── gpu_utils.py  # GPU selection utilities
│       ├── infer_utils.py# Inference utilities
│       └── io_utils.py   # I/O utilities
├── README.md             # This file
├── requirements.txt      # Project dependencies
└── setup.py              # Setup script for installation
```

---

## Installation
You can install ChatTTS directly from the GitHub repository:
```bash
pip install git+https://github.com/2noise/ChatTTS.git
```
Alternatively, you can clone the repository and install it in editable mode:
```bash
git clone https://github.com/2noise/ChatTTS.git
cd ChatTTS
pip install -e .
```

---

## Usage

### Basic Usage
Here is a simple example of how to use ChatTTS to synthesize speech:
```python
import ChatTTS
from IPython.display import Audio

chat = ChatTTS.Chat()
chat.load_models()

texts = ["<PUT YOUR TEXT HERE>",]

wavs = chat.infer(texts)
Audio(wavs[0], rate=24_000, autoplay=True)
```

### Advanced Usage

#### Sample a Speaker from a Gaussian Distribution
You can sample a random speaker embedding to generate speech with different voices:
```python
import torch

# Load speaker statistics
std, mean = torch.load('ChatTTS/asset/spk_stat.pt').chunk(2)
# Sample a random speaker
rand_spk = torch.randn(768) * std + mean

params_infer_code = {
  'spk_emb': rand_spk, # Add sampled speaker
  'temperature': .3,   # Use custom temperature
  'top_P': 0.7,        # Top-P decoding
  'top_K': 20,         # Top-K decoding
}

wav = chat.infer(texts, params_infer_code=params_infer_code)
```

#### Sentence-Level Manual Control
You can control prosodic features at the sentence level using special tokens:
```python
# Use oral_(0-9), laugh_(0-2), break_(0-7) to generate special tokens
params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_6]'
} 

wav = chat.infer(texts, params_refine_text=params_refine_text)
```

#### Word-Level Manual Control
You can also control prosody at the word level by inserting special tokens directly into the text:
```python
text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
wav = chat.infer([text], skip_refine_text=True)
```

---

## API Documentation
The main entry point for using ChatTTS is the `Chat` class in `ChatTTS.core`.

### `Chat()`
The main class for text-to-speech synthesis.

#### `Chat.load_models(source='huggingface', force_redownload=False, local_path='<LOCAL_PATH>')`
Loads the pretrained models from the specified source.
-   **`source`**: `'huggingface'` or `'local'`.
-   **`force_redownload`**: If `True`, forces redownload from Hugging Face.
-   **`local_path`**: The local path to the models if `source` is `'local'`.

#### `Chat.infer(text, skip_refine_text=False, refine_text_only=False, params_refine_text={}, params_infer_code={}, use_decoder=False)`
Synthesizes speech from text.
-   **`text`**: A list of text strings to synthesize.
-   **`skip_refine_text`**: If `True`, skips the text refinement step.
-   **`refine_text_only`**: If `True`, only performs text refinement and returns the refined text.
-   **`params_refine_text`**: Parameters for the text refinement step.
-   **`params_infer_code`**: Parameters for the code inference step.
-   **`use_decoder`**: If `True`, uses the decoder for generating audio.

#### `Chat.sample_random_speaker()`
Samples a random speaker embedding.

For more details, please refer to the docstrings in the source code.

---

## Roadmap
- [x] Open-source the 40k hour base model and spk_stats file
- [ ] Open-source VQ encoder and Lora training code
- [ ] Streaming audio generation without refining the text*
- [ ] Open-source the 40k hour version with multi-emotion control
- [ ] ChatTTS.cpp maybe? (PR or new repo are welcomed.)

---

## FAQ

##### How much VRAM do I need? How about infer speed?
For a 30-second audio clip, at least 4GB of GPU memory is required. For the 4090D GPU, it can generate audio corresponding to approximately 7 semantic tokens per second. The Real-Time Factor (RTF) is around 0.65.

##### Model stability is not good enough, with issues such as multi speakers or poor audio quality.
This is a problem that typically occurs with autoregressive models (e.g., Bark, VALL-E). It's generally difficult to avoid. One can try multiple samples to find a suitable result.

##### Besides laughter, can we control anything else? Can we control other emotions?
In the current released model, the only token-level control units are `[laugh]`, `[uv_break]`, and `[lbreak]`. In future versions, we may open-source models with additional emotional control capabilities.

---

## Disclaimer

This repo is for academic purposes only. It is intended for educational and research use and should not be used for any commercial or legal purposes. The authors do not guarantee the accuracy, completeness, or reliability of the information. The information and data used in this repo are for academic and research purposes only. The data is obtained from publicly available sources, and the authors do not claim any ownership or copyright over the data.

ChatTTS is a powerful text-to-speech system. However, it is very important to utilize this technology responsibly and ethically. To limit the use of ChatTTS, we added a small amount of high-frequency noise during the training of the 40,000-hour model and compressed the audio quality as much as possible using MP3 format to prevent malicious actors from potentially using it for criminal purposes. At the same time, we have internally trained a detection model and plan to open-source it in the future.

---

## Acknowledgements
-   [bark](https://github.com/suno-ai/bark), [XTTSv2](https://github.com/coqui-ai/TTS), and [valle](https://arxiv.org/abs/2301.02111) demonstrate a remarkable TTS result by an autoregressive-style system.
-   [fish-speech](https://github.com/fishaudio/fish-speech) reveals the capability of GVQ as an audio tokenizer for LLM modeling.
-   [vocos](https://github.com/gemelo-ai/vocos) which is used as a pretrained vocoder.

---

## Special Appreciation
-   [wlu-audio lab](https://audio.westlake.edu.cn/) for early algorithm experiments.
