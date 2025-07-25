#!/usr/bin/env python3
import argparse
import string
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# 确保所需的NLTK数据已下载
def download_nltk_data():
    required_corpora = [
        'stopwords',
        'vader_lexicon',
        'wordnet',
        'punkt'
    ]
    for corpus in required_corpora:
        try:
            nltk.data.find(f'corpora/{corpus}')
        except LookupError:
            print(f"下载必要的NLTK数据: {corpus}")
            nltk.download(corpus)


# 文本预处理
def preprocess_text(text):
    # 转换为小写
    text = text.lower()

    # 移除标点符号和特殊字符
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 分词
    words = nltk.word_tokenize(text)

    # 移除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words, ' '.join(words)


# 分析词频
def analyze_word_frequency(words, top_n=10):
    word_counts = Counter(words)
    return word_counts.most_common(top_n)


# 分析情感倾向
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    # 确定主要情感
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment = "积极"
    elif compound_score <= -0.05:
        sentiment = "消极"
    else:
        sentiment = "中性"

    return sentiment_scores, sentiment


# 提取关键词
def extract_keywords(processed_text, top_n=10):
    if not processed_text:
        return []

    vectorizer = TfidfVectorizer(max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    feature_names = vectorizer.get_feature_names_out()

    # 获取TF-IDF分数并排序
    tfidf_scores = tfidf_matrix.toarray()[0]
    keywords_with_scores = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)

    return [keyword for keyword, _ in keywords_with_scores]


# 从文件读取文本
def read_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        exit(1)
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        exit(1)


# 主函数
def main():
    # 下载必要的NLTK数据
    download_nltk_data()

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='文本分析工具：分析词频、情感倾向和提取关键词')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help='要分析的文本文件路径')
    group.add_argument('-t', '--text', help='要分析的文本内容')

    parser.add_argument('-n', '--num', type=int, default=10,
                        help='显示的词频和关键词数量（默认：10）')

    args = parser.parse_args()

    # 获取文本内容
    if args.file:
        text = read_text_from_file(args.file)
    else:
        text = args.text

    # 预处理文本
    words, processed_text = preprocess_text(text)

    if not words:
        print("没有可分析的有效文本内容。")
        return

    # 执行分析
    word_freq = analyze_word_frequency(words, args.num)
    sentiment_scores, sentiment = analyze_sentiment(processed_text)
    keywords = extract_keywords(processed_text, args.num)

    # 显示结果
    print("\n===== 文本分析结果 =====")

    print("\n1. 词频分析（前{}个）:".format(args.num))
    for word, count in word_freq:
        print(f"  {word}: {count}次")

    print("\n2. 情感分析:")
    print(f"  主要情感: {sentiment}")
    print(f"  积极分数: {sentiment_scores['pos']:.4f}")
    print(f"  消极分数: {sentiment_scores['neg']:.4f}")
    print(f"  中性分数: {sentiment_scores['neu']:.4f}")
    print(f"  综合分数: {sentiment_scores['compound']:.4f}")

    print("\n3. 关键词提取（前{}个）:".format(args.num))
    if keywords:
        print("  " + ", ".join(keywords))
    else:
        print("  无法提取关键词")

    print("\n========================")


if __name__ == "__main__":
    main()
