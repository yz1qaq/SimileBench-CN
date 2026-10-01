# SimileBench-CN

SimileBench-CN contains four Chinese simile understanding tasks. `data/SimileSet-CN/` contains the SimileSet-CN dataset. `models/symbolic_simcot/model.py` contains the main graph-based sentiment classifier described for Task 4.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/SimileSet-CN/train.json`, `test.json` | Canonical train/test records: sentence, eight-class emotion label, and gold tenor–vehicle annotations. |
| `data/SimileSet-CN/task1/` … `task4/` | Gold, task-specific views of the same train/test records. These are labels, not model predictions. |
| `models/symbolic_simcot/model.py` | Main SymbolicSimCoT classifier, graph input adapter, batch collation, and class-weighted loss. |
| `extraction_results/` | **Empty placeholder.** Put your own upstream extraction and cultural-interpretation outputs here as `train.json` and `test.json`. This directory is ignored by Git. |

No predicted tenor–vehicle extraction results, cultural interpretations, model checkpoints, or API credentials are included in this repository.

## Data

SimileSet-CN has 4,000 training sentences with 7,205 gold tenor–vehicle pairs and 1,000 test sentences with 1,844 gold pairs. The eight labels are `好`, `乐`, `怒`, `哀`, `惧`, `恶`, `惊`, and `无情绪`. The task-specific files preserve the same sentence IDs and sentences.

`gold_pairs` in the dataset are human-checked reference annotations. They must **not** be passed off as predictions from the upstream extraction model. For a paper-faithful Task 4 run, produce extraction results separately for each split and place them in `extraction_results/`.

## Where to put upstream results

Create these local files before using the classifier:

```text
extraction_results/
├── train.json
└── test.json
```

Each file is a JSON list with exactly one entry per sentence in the corresponding canonical split. Match by `sentence_id` and copy the original `sentence` exactly. Each entry needs a `pair_analyses` list. An empty list is allowed when the extractor finds no pairs in that sentence; a completely empty split is rejected. Each listed pair needs all four nonempty string fields: `tenor`, `vehicle`, `cultural_implication`, and `combined_meaning`. The last two fields are the text used to initialize each graph node. `load_split` checks coverage, IDs, sentence text, and required fields; it fails if these inputs are missing.

Minimal input shape (placeholders, not extraction results):

```json
[{"sentence_id":"1","sentence":"<exact sentence from the dataset>","pair_analyses":[{"tenor":"<text>","vehicle":"<text>","cultural_implication":"<text>","combined_meaning":"<text>"}]}]
```

The upstream step described in the paper uses Qwen3-8B to extract tenor–vehicle pairs and obtains their cultural implications and combined meanings. That step and its outputs are not included here. Never use test gold pairs as a replacement for missing extraction outputs.

## Main model

Install PyTorch and Transformers, then supply a 768-dimensional Chinese RoBERTa encoder checkpoint through `encoder_name_or_path`. The main components in `models/symbolic_simcot/model.py` are:

1. Encode the sentence and each pair's `cultural_implication` plus `combined_meaning` text with the same encoder.
2. Connect pairs sharing a tenor, add self-loops, and aggregate with **two GAT layers, four attention heads, and 0.2 dropout**.
3. Mean-pool valid graph nodes, inject graph information into the sentence representation, and predict one of eight emotions.
4. Use class-weighted cross-entropy with weights `M / (8 × M_c)` computed from the training split.

The paper writes a softmax over one global graph vector. A one-element softmax is always 1 and cannot learn an attention weight. This implementation includes an explicit null option in that softmax, so the graph injection depends on the sentence and graph. This is a functional implementation of the stated text–graph attention intent; it is a small mathematical clarification of the paper's equation.

`GraphCollator` pads to the largest number of pairs in each batch. It does not silently drop sentences containing more than eight pairs. The model module supplies the architecture, data validation, collation, and loss, but **does not include** the upstream extraction pipeline, a training runner, trained weights, or a claim that the paper's reported scores have been reproduced. The paper's remaining training settings are batch size 32, 15 epochs, AdamW learning rate `2e-5`, weight decay `0.01`, warm-up ratio `0.1`, and gradient clipping at `1.0`.

Minimal integration after providing `extraction_results/train.json`:

```python
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.symbolic_simcot.model import (
    GraphCollator, SimileGraphDataset, SymbolicSimCoT,
    class_weights_from_training, load_split,
)

encoder_path = "<local or downloaded 768-dimensional Chinese RoBERTa checkpoint>"
train = load_split("data/SimileSet-CN/train.json", "extraction_results/train.json")
tokenizer = AutoTokenizer.from_pretrained(encoder_path)
loader = DataLoader(SimileGraphDataset(train), batch_size=32,
                    collate_fn=GraphCollator(tokenizer))
model = SymbolicSimCoT(encoder_path, class_weights_from_training(train))
batch = next(iter(loader))
outputs = model(**{key: value for key, value in batch.items() if key != "sentence_id"})
loss = outputs["loss"]
```

## 中文说明

论文数据集位于 `data/SimileSet-CN/`，Task 4 主模型位于 `models/symbolic_simcot/model.py`。请把自行生成的本体—喻体抽取与文化寓意分析结果分别放在 `extraction_results/train.json` 和 `extraction_results/test.json`。这两个结果文件不会上传到 GitHub。数据集中的 `gold_pairs` 是参考标注，不能代替模型抽取结果。
