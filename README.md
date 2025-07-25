# 文本分析工具

一个功能全面的命令行工具，用于分析文本中的词频、情感倾向和提取关键词。

## 功能特点

- **词频分析**：统计文本中出现频率最高的词汇
- **情感分析**：判断文本的情感倾向（积极、消极或中性）
- **关键词提取**：利用TF-IDF算法提取文本中的重要关键词
- 支持从文件读取文本或直接输入文本内容
- 可自定义显示结果的数量

## 安装要求

- Python 3.6 或更高版本
- 所需Python库：nltk, scikit-learn

## 安装步骤

1. 克隆或下载本项目到本地
   ```bash
   git clone https://github.com/ChidcGithub/Words_analyzer.git
   cd text-analyzer
   ```

2. 安装依赖库
   ```bash
   pip install -r requirements.txt
   ```
   或者直接安装所需库：
   ```bash
   pip install nltk scikit-learn
   ```

## 使用方法

### 基本用法

#### 分析文本文件python text_analyzer.py -f path/to/your/file.txt
#### 直接分析文本内容python text_analyzer.py -t "这是一段需要分析的文本内容..."
### 自定义选项

#### 更改显示结果数量
默认情况下，工具会显示前10个词频和关键词，你可以通过`-n`或`--num`参数自定义数量：python text_analyzer.py -f example.txt -n 20
## 示例输出
===== 文本分析结果 =====

1. 词频分析（前10个）:
  人工智能: 15次
  技术: 12次
  学习: 9次
  ...

2. 情感分析:
  主要情感: 积极
  积极分数: 0.3520
  消极分数: 0.0840
  中性分数: 0.5640
  综合分数: 0.8765

3. 关键词提取（前10个）:
  人工智能, 技术, 学习, 应用, 未来, ...

========================
## 工作原理

1. **文本预处理**：将文本转换为小写、去除标点符号、分词、移除停用词并进行词形还原
2. **词频分析**：使用计数器统计处理后词汇的出现频率
3. **情感分析**：使用NLTK的VADER模型计算文本的情感分数
4. **关键词提取**：使用scikit-learn的TF-IDF向量器提取重要关键词

## 许可证

本项目采用GNU3.0许可证 - 详见LICENSE文件    
