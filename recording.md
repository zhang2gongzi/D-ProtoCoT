# 实验记录

## 实验结果

条数：500

数据集：![alt text](image.png)

sc采样：5条

| Llama-3.1-8B-Instruct | commonsenseqa| gsm8k | strategyqa|
| :----- | :------: | -----: | -----: |
| standard |  68.23| 43.8 | 68.6|
| self-consistency|62.0|63.2|62.6|
|ccot|64.4|54.3|51.2|
|dpro||74.2(78.2)||