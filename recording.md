# 实验记录

## 实验结果

条数：500

数据集：![alt text](image.png)

sc采样：5条

| Llama-3.1-8B-Instruct | commonsenseqa| gsm8k | strategyqa|
| :----- | :------: | -----: | -----: |
| standard |  68.23| 43.8 | 68.6|
| self-consistency|77.4|63.2|62.6|
|ccot|64.4|87.0|64.4|
|dpro|79.8|78.2|86.2|
|Qwen3-8B|commonsenseqa| gsm8k | strategyqa|
|standard|75.6|90.4|66.8|
|self-consistency|86.98|90.65|76.6|
|ccot|68.2|75.6|62.7|
|dpro|87.71|94.15|72.6|