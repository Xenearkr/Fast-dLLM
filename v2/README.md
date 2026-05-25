# Fast-dLLM v2: Efficient Block-Diffusion Large Language Model

## Progress 2025.05.25 更新说明

### 改进内容

1. 在v2/base_models/Model-Qwen-3-8B/下添加了原生Qwen3版本的modeling.py和configuration.py

2. 完全修复了mbpp和humaneval评测逻辑（注意：如果运行eval_humaneval_fast.sh或eval_mbpp_fast.sh报错第x行发现未知符号，只要再bash ?.sh一次就行了）（对mbpp，由于清洗逻辑相对宽松，表现略偏高，但基本符合原文数据）

3. 添加inspect_humaneval_samples.py和inspect_mbpp_samples.py，用于检查coding任务的生成代码是否正确（简要说明为什么要自己写humaneval和mbpp的评测逻辑：似乎原生评测逻辑的接口和Fast类接口不对齐？）

4. 修复LoRA脚本，并最大限度与full脚本对齐【当前状况：能运行、能merge，结果不理想，正在控制变量排查是由于LoRA方法本身不行还是训练方法不对】

### TBD

1. LoRA效果：目前测试效果对比全量微调不佳。若感兴趣，可以在本地自己跑脚本 + coding task评测看效果如何。【注意：当前LoRA设定是不自动合并adapters，需要手动运行merge_lora代码合并】

2. 前期准备结束，正式开始修缮Qwen3-8B子仓库

### 评测结果

```
# MBPP
# 论文结果为：0.630, 0.523

# original mbpp
mbpp (base tests)
pass@1: 0.683
mbpp+ (base + extra tests)
pass@1: 0.582

# qwen2.5_1000 mbpp
mbpp (base tests)
pass@1: 0.505
mbpp+ (base + extra tests)
pass@1: 0.418


# HumanEval
# 论文结果为：0.634, 0.585

# original humaneval
humaneval (base tests)
pass@1: 0.579
humaneval+ (base + extra tests)
pass@1: 0.549

# qwen2.5_1000 humaneval
humaneval (base tests)
pass@1: 0.439
humaneval+ (base + extra tests)
pass@1: 0.384
```


### 其他

训练需要时间。新一轮mini-batch检测与LoRA结果对比最早明天中午才能拿到。

---

## Progress 2026.05.24-2 更新说明

### 改进内容

1. 添加了可以上传的base_models权重：Qwen2.5是改装好的，Qwen3是原先的。注意.gitignore规则，必须忽略跟踪大文件。【这个文件夹的配置是否正确有待进一步检验】

2. 修复了mbpp评测逻辑，复现原文结果，除了一点瑕疵之外基本修正成功；humaneval评测逻辑有待改进。支持一键评测。

3. 修复了本README文件，读起来更美观。

4. 添加了LoRA训练脚本。问题：合并逻辑出错、训练效果未知。

### TBD

1. 修复Humaneval评测逻辑，通过Fast-dLLM/v2/inspect_evalplus_samples.py文件的检查

2. 改良Qwen3.8仓库，引入Qwen3自带modeling等文件作为基线仓库


---

## Progress 2026.05.24-1 使用须知

对本仓库添加的架构说明如下：

### `v2/base_models/ref`

- 包含了对Qwen2.5-7B-Instruct进行Fast包装的重要代码组件，以防误删，上传

### `v2/base_models`

- 提醒：关于这个文件夹内部哪些文件要跟踪，请关注 Fast-dLLM/.gitignore 最后部分。总体而言，一方面，大文件不宜加入跟踪，即使在历史状态中出现，也可能导致无法上传的问题；另一方面，大文件都是不怎么改的，可以直接从HF上下载，重点都在其中的代码文件部分。
- 使用的AR模型基座。
- 目前包含：Fast-dLLM-v2, Qwen2.5-7B-Instruct, Qwen3-8B
- 其中Fast-dLLM-v2和Qwen3-8B保留了原始形态（即跟直接从Hugging Face上下载的没区别），Qwen2.5-7B-Instruct的配置已经包装成了Fast类模型
- 具体怎么包装的呢？见下文，里面有👀符号的，文件内容请参考v2/base_model_ref里面的文件

#### 符号体系

- copy-paste：直接把Fast里面的这个文件复制粘贴/覆盖过去
- keep：保留Qwen里面的这个文件
- delete：不用管的文件

#### 下面是改后的Qwen2.5目录

```text
权重区
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
添加区：这些文件是只有Fast里面有的
├── modeling.py ✅️copy-paste
├── configuration.py ✅️copy-paste
├── added_tokens.json ✅️copy-paste
├── chat_template.jinja ✅️copy-paste
├── latest ❌️delete
├── special_tokens_map.json ✅️copy-paste
修改区：这些文件是两个仓库中都有的
├── config.json 👀 大改
├── generation_config.json 👀 change 4.53.1/4.37.0
├── merges.txt ✅️keep
├── model.safetensors.index.json 👀 modify 1 line
├── tokenizer_config.json✅️ copy-paste
├── tokenizer.json✅️copy-paste
└── vocab.json ✅️keep
```

### `v2/train_scripts`

- 添加finetune_a2d_v0.sh，主要训练脚本
- 已有finetune脚本产生的文件需转换呈safetensors，添加convert_model.sh和convert_inplace_to_safetensors.py

### `v2/utils`

- 一些辅助函数，主要用于将原始Llama-Nemotron转换成可以套用finetune脚本的格式
- 当然很奇怪的一点是：这个工作理论上开发者做过一次，为啥不直接提供复现方法？

### `v2/data`

- Llama-Nemotron数据集放在这里，太大了我就不上传了，我的组织结构如下：

```text
.
├── alpaca
│   ├── test
│   │   └── test_252.json
│   ├── test_conversation
│   │   └── test_252.json
│   ├── train
│   │   └── train_52002.json
│   └── train_conversation
│       └── train_52002.json
├── download.sh
└── Llama-Nemotron-code-v1.1
    ├── README.md
    ├── SFT
    │   └── code
    │       └── code_v1.1.jsonl
    ├── train_conversation # 这里放的是使用utils中代码转换好的文件
    │   ├── train-00001.json
    │   ├── ......
    │   └── train-00049.json
    └── use # 我第一次只用了转换好的1/50，所以单独拎出来了第一个分片
        └── train-00000.json
```

### 评测脚本

- v2/eval_codetask.sh 评测代码任务，需要额外evalplus支持，效果不理想
- v2/eval_tmp.sh 评测mmlu，本质就是eval_script.sh的一部分

### 其他说明

1. 训练所得模型自动放在v2/output_models/下
2. 训练调用的部分代码涉及Fast-dLLM/third_party/lmflow等文件夹，不在Fast-dLLM/v2目录下

---

## Progress 2026.05.23 进度说明

### Target 1 直接运行

- 目前都是全量微调，未适配LoRA
- GPU配置不同，导致：1. 容易报错OOM，2. loss波动大；建议：再试试调整参数？
- 除了按照下面的要求安装requirements之外，还要安装flash-attn 2.8.3

### Target 2 换 Fast 为 Qwen 2.5

- 已经把包装套好了，套的对不对还不好说...但就loss趋势来看似乎可以...

### Target 3 换 Alpaca 为 Llama-Nemotron

- 放在v2/data/底下，只选了一部分，并做了格式调整

### Target 4 评测

- 评测MMLU对比如下：

#### qwen 2.5 + llama-nemotron 训了1000 steps

| Groups | Version | Filter | n-shot | Metric |  | Value |  | Stderr |
|---|---:|---|---|---|---|---:|---|---:|
| mmlu | 2 | none |  | acc | ↑ | 0.6716 | ± | 0.0037 |
| - humanities | 2 | none |  | acc | ↑ | 0.5911 | ± | 0.0067 |
| - other | 2 | none |  | acc | ↑ | 0.7184 | ± | 0.0078 |
| - social sciences | 2 | none |  | acc | ↑ | 0.7836 | ± | 0.0072 |
| - stem | 2 | none |  | acc | ↑ | 0.6362 | ± | 0.0083 |

#### 直接用Fast-dllm-v2-7B

| Groups | Version | Filter | n-shot | Metric |  | Value |  | Stderr |
|---|---:|---|---|---|---|---:|---|---:|
| mmlu | 2 | none |  | acc | ↑ | 0.6797 | ± | 0.0037 |
| - humanities | 2 | none |  | acc | ↑ | 0.5951 | ± | 0.0066 |
| - other | 2 | none |  | acc | ↑ | 0.7377 | ± | 0.0076 |
| - social sciences | 2 | none |  | acc | ↑ | 0.7930 | ± | 0.0072 |
| - stem | 2 | none |  | acc | ↑ | 0.6384 | ± | 0.0083 |

- 评测coding任务：
  - 需要额外下载evalplus支持？直接在eval脚本里把task-name改成mbpp，会跑出0.0？见仓库issues
  - 添加了evalplus支持之后，似乎结果不太理想？【以下都是直接用Fast-dllm-v2-7B测的】

    ```text
    mbpp (base tests)
    pass@1: 0.325
    mbpp+ (base + extra tests)
    pass@1: 0.265
    ```

  - humaneval需要后训练吗？（可以肯定的是mbpp不需要）


---

## 下面是原来的内容

[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://nvlabs.github.io/Fast-dLLM/v2)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2509.26328)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B)

Fast-dLLM v2 is a carefully designed block diffusion language model (dLLM) that efficiently adapts pretrained autoregressive (AR) models into dLLMs for parallel text generation, requiring only approximately 1B tokens of fine-tuning. This represents a **500x reduction** in training data compared to full-attention diffusion LLMs while preserving the original model's performance.

## 🎬 Demo
https://github.com/user-attachments/assets/f2e055f5-3a44-41ca-9ef8-c84cf3ac2951

## 🎯 Key Features

### 1. **Block Diffusion Mechanism**
- Novel training recipe combining block diffusion with complementary attention masks
- Enables blockwise bidirectional context modeling 
- Token shift mechanism to retain autoregressive characteristics

<div align="center">
  <img src="asset/training_recipe.png" alt="Training Recipe" width="700"/>
  <p><em>Block-wise causal attention mask and complementary training strategy</em></p>
</div>

### 2. **Hierarchical Caching System**
- **Block-level cache**: Stores historical context representations across blocks
- **Sub-block cache**: Enables efficient parallel generation within partially decoded blocks

### 3. **Parallel Decoding Pipeline**
- Achieves up to **2.5x speedup** over standard AR decoding
- Real-time visualization of the denoising process
- Maintains generation quality while delivering state-of-the-art efficiency

<div align="center">
  <img src="asset/visualization_animation.gif" alt="Generation Process Visualization" width="700"/>
  <p><em>Block-level autoregressive generation with sub-block parallelization</em></p>
</div>

## 🚀 Performance

### Throughput Comparison
Fast-dLLM v2 significantly outperforms baselines in both efficiency and accuracy:
- **2.54× higher throughput** than Qwen2.5-7B-Instruct
- **5.2% accuracy improvement** over Fast-dLLM-LLaDA

<div align="center">
  <img src="asset/throughput.png" alt="Throughput Comparison" width="700"/>
  <p><em>Throughput and accuracy comparison across different model variants</em></p>
</div>

### Benchmark Results
Comprehensive evaluation across diverse tasks:

| Model Size | Model | HumanEval-Base | HumanEval-Plus| MBPP-Base | MBPP-Plus |  GSM8K | Math | IFEval | MMLU | GPQA | Average |
|------------|-------|-----------|--|---|---|-------|------|--------|------|------|---------|
| **1B-scale** | Fast-dLLM v2 (1.5B) | 43.9 | 40.2 | 50.0 | 41.3 | 62.0 | 38.1  | 47.0 | 55.1 | 27.7 | **45.0** |
| **7B+ scale** | Fast-dLLM v2 (7B) | 63.4 | 58.5 | 63.0 | 52.3 | 83.7 | 61.6 | 61.4 | 66.6 | 31.9 | **60.3** |

<div align="center">
  <img src="asset/benchmark_results.png" alt="Benchmark Results" width="800"/>
  <p><em>Comprehensive benchmark comparison across diverse tasks</em></p>
</div>


## 🏋️ Training

### Environment Setup
First, create and activate a conda environment:

```bash
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
```

### Installation
Install the package in development mode:

```bash
pip install -e .
```

### Data Preparation
Download the training data (e.g., Alpaca dataset):

```bash
cd data
bash download.sh alpaca
```

### Fine-tuning
Run the fine-tuning script:

```bash
bash train_scripts/finetune_alpaca.sh
```

This will start the training process using the Alpaca dataset with the optimized block diffusion training recipe.

## 🎮 Quick Start

### Interactive Chatbot
Launch the Gradio-based web interface:

```bash
python app.py
```

This will start a web server at `http://localhost:10086` with:
- Real-time conversation interface
- Live visualization of the denoising process
- Adjustable generation parameters (block size, temperature, threshold)
- Performance metrics display

### Command Line Chat
For a simple command-line interface:

```bash
python run_chatbot.py
```

Commands:
- Type your message and press Enter
- `clear` - Clear conversation history
- `exit` - Quit the chatbot


## 📊 Evaluation

### Run Benchmark Evaluation
Execute the evaluation script for comprehensive benchmarking:

```bash
bash eval_script.sh
```

This script evaluates the model on:
- **MMLU**: Massive Multitask Language Understanding
- **GPQA**: Graduate-level Google-Proof Q&A
- **GSM8K**: Grade School Math 8K
- **Minerva Math**: Mathematical reasoning
- **IFEval**: Instruction following evaluation

### Custom Evaluation
For custom evaluation with specific parameters:

```bash
accelerate launch eval.py \
    --tasks gsm8k \
    --batch_size 32 \
    --num_fewshot 0 \
    --model fast_dllm_v2 \
    --model_args model_path=Efficient-Large-Model/Fast_dLLM_v2_7B,threshold=0.9
```

## 🏗️ Architecture

### Training Recipe
- **Token Shift Mechanism**: Each masked token is predicted using the logit of its preceding token
- **Block-wise Causal Attention**: Access to all clean tokens from previous blocks and noisy tokens within current block
- **Complementary Masks**: Alternate masking patterns ensure every token position is learned

### Generation Process
1. **Block-level Generation**: Autoregressive at the block level
2. **Sub-block Parallelization**: Parallel decoding within blocks for efficiency
3. **Hierarchical Caching**: Block and sub-block level caching for speed optimization

## 📁 File Structure

```
v2/
├── app.py                    # Gradio web interface
├── run_chatbot.py           # Command-line chatbot
├── eval.py                  # Evaluation harness integration
├── eval_script.sh           # Benchmark evaluation script
├── generation_functions.py  # Core generation algorithms
├── index.html              # Project webpage
├── asset/                  # Visual assets
│   ├── demo.mp4
│   ├── benchmark_results.png
│   ├── throughput.png
│   ├── training_recipe.png
│   └── visualization_animation.gif
└── README.md               # This file
```

## 🎨 Visualization Features

The web interface provides real-time visualization of:
- **Denoising Process**: Watch tokens being unmasked in real-time
- **Generation Progress**: Visual feedback of the generation pipeline
- **Performance Metrics**: Live throughput and timing information
- **Slow Motion Replay**: Detailed step-by-step visualization

## 🔬 Technical Details

### Model Architecture
- Based on Qwen2.5 architecture with block diffusion modifications
- 7B parameter model with efficient parallel decoding capabilities
- Custom attention mechanisms for block-wise processing

### Optimization Techniques
- Block-level KV caching for reduced computation
- Sub-block parallel processing for improved throughput
- Confidence-aware token unmasking for quality preservation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](../LICENSE) file for details.

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{wu2025fastdllmv2efficientblockdiffusion,
      title={Fast-dLLM v2: Efficient Block-Diffusion LLM}, 
      author={Chengyue Wu and Hao Zhang and Shuchen Xue and Shizhe Diao and Yonggan Fu and Zhijian Liu and Pavlo Molchanov and Ping Luo and Song Han and Enze Xie},
      year={2025},
      eprint={2509.26328},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.26328}, 
}
```

## 🙏 Acknowledgements

We thank [Qwen2.5](https://github.com/QwenLM/Qwen2.5) for the base model architecture
