# 中文文本分析工具

这个项目是一个高级中文文本分析工具，提供词频分析、情感分析、关键词提取和文本摘要生成等功能。它使用jieba分词库和自定义词典，特别适用于新闻、评论等中文文本的分析任务。

## 主要功能

1. **词频分析**：统计文本中词语出现的频率
2. **情感分析**：识别文本的情感倾向（积极/消极/中性）
3. **关键词提取**：使用TF-IDF或TextRank算法提取关键词语
4. **文本摘要**：自动生成文本的核心内容摘要

## 快速开始

### 安装依赖
```bash
pip install jieba
```

### 运行分析
```bash
# 使用示例文件进行分析
python main.py --file input_file.txt

# 直接分析文本内容
python main.py --text "您的文本内容"
```

### 可选参数
| 参数 | 描述 | 默认值 |
|------|------|--------|
| `-n NUM` | 显示的词频和关键词数量 | 10 |
| `--algorithm ALG` | 关键词提取算法(tfidf/textrank) | tfidf |
| `--summary` | 生成文本摘要 | 关闭 |
| `--sentiment-details` | 显示详细情感分析数据 | 关闭 |

## 文件说明

- `main.py`：主程序，实现所有分析功能
- `chinese_stopwords.txt`：中文停用词表
- `input_file.txt`：示例输入文本
- `run_analyze.bat`：Windows批处理运行脚本
- `download_necessary_nltk.py`：NLTK下载脚本（备用）

## 使用示例

```bash
python main.py --file input_file.txt --sentiment-details --summary --n 15
```

### 示例输出：
```
===== 文本分析结果 =====

1. 词频分析（前15个）:
  ...

2. 情感分析:
  ...

3. 关键词提取（使用TFIDF，前15个）:
  ...

4. 文本摘要:
  ...
```

## 自定义配置

程序内置了多个领域的专业词汇，包括：
- 军事领域术语
- 科技领域术语
- 经济金融术语
- 医疗健康术语
- 教育文化术语
- 生活消费术语
- 环境能源术语
- 综合词汇

您可以在`main.py`中的`setup_jieba()`函数中添加自定义词汇以优化分词效果。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！