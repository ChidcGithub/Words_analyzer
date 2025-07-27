#!/usr/bin/env python3
import argparse
import re
import os
import math
import jieba
import jieba.analyse
from collections import Counter


# 初始化jieba分词器
def setup_jieba():
    # 军事领域
    military_terms = [
        '铸魂育人', '实战化训练', '战斗精神', '野外驻训', '教育质效',
        '政治教员', '训练标兵', '战法检验', '攻坚克难', '忠诚信仰',
        '使命担当', '心理素质', '意志品质', '畏难情绪', '攻坚劲头',
        '士气高昂', '新传人', '一线带头', '战斗力建设', '思想摸排',
        '舍生忘死', '英勇战斗', '重走进藏', '教育鼓点', '精准把脉',
        '多维宣贯', '触及灵魂', '拨动心弦', '战斗精神培育', '教育授课',
        '金句', '历史资料', '革命先辈', '忠诚信仰', '使命担当', '心理素质'
    ]

    # 科技领域
    tech_terms = [
        '人工智能', '机器学习', '深度学习', '神经网络', '大数据',
        '云计算', '区块链', '物联网', '量子计算', '5G通信',
        '半导体', '集成电路', '虚拟现实', '增强现实', '自动驾驶',
        '无人机', '机器人', '生物识别', '基因编辑', '纳米技术',
        '数字化转型', '智能制造', '工业互联网', '智慧城市', '算法优化'
    ]

    # 经济金融
    finance_terms = [
        '宏观经济', '微观经济', '货币政策', '财政政策', '供给侧改革',
        '消费升级', '产业升级', '资本市场', '股票市场', '债券市场',
        '风险投资', '私募股权', '区块链金融', '数字货币', '跨境支付',
        '供应链金融', '绿色金融', '普惠金融', '金融科技', '信用体系',
        '通货膨胀', '通货紧缩', '经济周期', '市场波动', '量化宽松'
    ]

    # 医疗健康
    medical_terms = [
        '公共卫生', '疫情防控', '疫苗接种', '基因测序', '精准医疗',
        '远程医疗', '智慧医疗', '中医药学', '中西医结合', '预防医学',
        '康复治疗', '心理健康', '慢性病管理', '健康管理', '医疗资源',
        '分级诊疗', '医疗改革', '药品集采', '医疗器械', '临床试验',
        '生物制药', '细胞治疗', '免疫疗法', '健康中国', '医养结合'
    ]

    # 教育文化
    education_terms = [
        '素质教育', '职业教育', '高等教育', '终身学习', '教育公平',
        '在线教育', '混合式教学', '课程改革', '教育评价', '师资建设',
        '文化自信', '文化遗产', '非物质文化遗产', '文化创新', '文化传播',
        '文化认同', '文化产业', '文化软实力', '文化多样性', '文化传承',
        '艺术教育', '美育教育', '体育教育', '劳动教育', '教育现代化'
    ]

    # 生活消费
    lifestyle_terms = [
        '消费升级', '新零售', '社交电商', '直播带货', '国潮品牌',
        '智能家居', '健康饮食', '运动健身', '旅游休闲', '绿色出行',
        '共享经济', '社区团购', '预制菜', '轻食主义', '宠物经济',
        '银发经济', '单身经济', '夜间经济', '体验经济', '悦己消费',
        '宅经济', '懒人经济', '颜值经济', '知识付费', '精神消费'
    ]

    # 环境能源
    environment_terms = [
        '碳中和', '碳达峰', '绿色发展', '生态文明', '污染防治',
        '清洁能源', '可再生能源', '光伏发电', '风力发电', '氢能源',
        '节能减排', '循环经济', '垃圾分类', '生态保护', '生物多样性',
        '气候变化', '碳交易', '绿色金融', '可持续交通', '环保材料',
        '零碳建筑', '生态修复', '海绵城市', '绿色制造', '环境监测'
    ]

    # 综合词汇
    general_terms = [
        '数字化转型', '创新发展', '高质量发展', '区域协调', '乡村振兴',
        '社会治理', '公共服务', '营商环境', '科技创新', '人才战略',
        '网络安全', '数据安全', '隐私保护', '知识产权', '反垄断',
        '共同富裕', '社会保障', '养老服务', '儿童友好', '青年发展'
    ]

    # 合并所有词汇
    all_terms = military_terms + tech_terms + finance_terms + medical_terms + \
                education_terms + lifestyle_terms + environment_terms + general_terms

    # 添加到jieba词典
    for term in all_terms:
        jieba.add_word(term)



# 加载自定义情感词典
def load_sentiment_dict():
    """返回包含积极词、消极词、程度副词和否定词的情感词典"""
    return {
        # 积极情感词及强度
        'positives': {
            '提升': 1, '有效': 1.2, '热情': 1.5, '丰富多彩': 1.3, '生动活泼': 1.3,
            '精准': 1, '创新': 1.2, '灵魂': 1.5, '心弦': 1.5, '针对性': 1,
            '深情': 1.5, '感人': 1.8, '提振': 1.5, '士气': 1.2, '热议': 1,
            '反思': 1, '忠诚': 2, '使命': 1.5, '担当': 1.5, '高昂': 1.5,
            '攻坚克难': 1.8, '圆满完成': 1.5, '有的放矢': 1, '实打实': 1.2,
            '铸魂育人': 1.5, '奋楫': 1.2, '传承': 1, '忠诚信仰': 2, '成功': 1.5,
            '成就': 1.4, '突破': 1.3, '卓越': 1.6, '优秀': 1.4, '先进': 1.3,
            '典范': 1.5, '榜样': 1.4, '领先': 1.3, '高效': 1.2, '优质': 1.3,
            '喜悦': 1.7, '满意': 1.4, '赞赏': 1.6, '认同': 1.3, '支持': 1.2,
            '发展': 1.1, '进步': 1.2, '改善': 1.2, '增强': 1.2, '优化': 1.2,
            '胜利': 1.8, '成就': 1.5, '荣誉': 1.7, '表彰': 1.5, '奖励': 1.4,
            '创新': 1.5, '突破': 1.4, '领先': 1.3, '高效': 1.2, '优质': 1.3,
            '可靠': 1.3, '稳定': 1.2, '安全': 1.3, '健康': 1.2, '和谐': 1.4
        },

        # 消极情感词及强度
        'negatives': {
            '畏难': 1.5, '不足': 1, '考验': 0.8, '冲动': 1.2, '退缩': 1.5,
            '问题': 0.8, '短板': 1, '倔犟': 1, '我行我素': 1.2, '老生常谈': 0.5,
            '失败': 1.8, '错误': 1.5, '缺陷': 1.4, '不足': 1.2, '困难': 1.1,
            '挑战': 1.0, '压力': 1.1, '矛盾': 1.3, '冲突': 1.6, '危机': 1.7,
            '损失': 1.5, '破坏': 1.6, '威胁': 1.7, '风险': 1.3, '隐患': 1.4,
            '衰退': 1.6, '下降': 1.2, '恶化': 1.5, '落后': 1.3, '批评': 1.4,
            '反对': 1.3, '抗议': 1.7, '抵制': 1.6, '抱怨': 1.3, '失望': 1.5,
            '担忧': 1.4, '恐惧': 1.8, '痛苦': 1.7, '悲伤': 1.8, '愤怒': 1.7,
            '挫折': 1.5, '障碍': 1.3, '负担': 1.2, '压力': 1.3, '疲劳': 1.2
        },

        # 程度副词（增强或减弱情感强度）
        'intensifiers': {
            '极其': 2.5, '非常': 2.0, '很': 1.8, '太': 1.8, '十分': 1.8,
            '格外': 1.8, '相当': 1.5, '颇为': 1.5, '比较': 1.2, '较为': 1.2,
            '有点': 0.8, '稍微': 0.8, '略微': 0.8, '几乎不': 0.3, '完全不': 0.1,
            '极度': 2.8, '格外': 2.2, '异常': 2.0, '特别': 1.9, '异常': 1.9,
            '稍': 0.7, '略': 0.6, '轻微': 0.5, '丝毫': 0.2, '几乎': 0.3
        },

        # 否定词（反转情感）
        'negations': {
            '不', '没', '非', '无', '未', '莫', '勿', '别', '否',
            '没有', '不会', '不能', '不可', '不必', '未曾', '无须', '不要',
            '从未', '毫无', '绝非', '并非', '不足以', '不太', '不够', '不受'
        },

        # 双重否定词（增强情感）
        'double_negations': {
            '不得不', '不能不', '不可不', '不会不', '不是不', '无不', '非不',
            '未必不', '不能不', '不可不', '没有不', '不会没有'
        }
    }


# 中文文本预处理
def preprocess_chinese_text(text):
    # 移除特殊字符和标点（包括中文标点）
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 使用jieba分词
    words = jieba.lcut(text)

    # 加载中文停用词表
    stopwords_path = "chinese_stopwords.txt"
    if not os.path.exists(stopwords_path):
        generate_stopwords_file(stopwords_path)

    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stop_words = set([line.strip() for line in f])

    # 过滤停用词和单字
    words = [word for word in words if word not in stop_words and len(word) > 1]

    return words, text


# 生成中文停用词文件
def generate_stopwords_file(file_path):
    chinese_stopwords = """
    的 一 不 在 人 有 是 为 以 于 上 他 而 后 之 来 及 了 因 下 可 到 由
    这 与 也 此 但 并 个 其 已 无 小 我 们 起 最 再 今 去 好 只 又 或
    很 亦 某 把 那 你 乃 它 吧 被 比 别 趁 当 从 到 得 打 凡 儿 尔
    该 各 给 跟 和 何 还 即 几 既 看 据 可 啦 了 另 么 每 们 嘛 拿
    哪 您 凭 且 却 让 仍 啥 是 使 谁 虽 同 哇 往 向 沿 哟 用 于 咱
    则 怎 曾 至 致 着 诸 自
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(chinese_stopwords)


# 分析词频
def analyze_word_frequency(words, top_n=10):
    word_counts = Counter(words)
    return word_counts.most_common(top_n)


# 增强版中文情感分析（已修复双重否定处理）
def analyze_chinese_sentiment(text):
    # 加载情感词典
    sentiment_dict = load_sentiment_dict()

    words = jieba.lcut(text)
    positive_score = 0
    negative_score = 0
    modifier = 1  # 程度修饰系数
    negation_flag = False  # 否定标志
    double_negation_flag = False  # 双重否定标志

    # 情感分析规则引擎
    i = 0
    while i < len(words):
        word = words[i]
        skip_next = False  # 标记是否跳过下一个词

        # 检查双重否定词（两词组合）
        if i < len(words) - 1:
            double_word = word + words[i + 1]
            if double_word in sentiment_dict['double_negations']:
                double_negation_flag = True
                i += 1  # 跳过下一个词
                skip_next = True

        # 检查程度副词
        if not skip_next and word in sentiment_dict['intensifiers']:
            modifier = sentiment_dict['intensifiers'][word]

        # 检查否定词
        elif not skip_next and word in sentiment_dict['negations']:
            negation_flag = not negation_flag  # 切换否定状态

        # 检查积极词
        elif not skip_next and word in sentiment_dict['positives']:
            base_score = sentiment_dict['positives'][word] * modifier

            # 处理否定和双重否定
            if negation_flag:
                base_score *= -1  # 否定反转
            if double_negation_flag:
                base_score *= -1  # 双重否定再次反转（负负得正）
                base_score *= 1.5  # 双重否定增强效果

            if base_score > 0:
                positive_score += base_score
            else:
                negative_score += abs(base_score)

            # 重置标记
            modifier = 1
            negation_flag = False
            double_negation_flag = False

        # 检查消极词
        elif not skip_next and word in sentiment_dict['negatives']:
            base_score = sentiment_dict['negatives'][word] * modifier

            # 处理否定和双重否定
            if negation_flag:
                base_score *= -1  # 否定反转（消极变积极）
            if double_negation_flag:
                base_score *= -1  # 双重否定再次反转
                base_score *= 1.5  # 双重否定增强效果

            if base_score > 0:
                negative_score += base_score
            else:
                positive_score += abs(base_score)

            # 重置标记
            modifier = 1
            negation_flag = False
            double_negation_flag = False

        i += 1

    # 计算综合情感分数
    total_score = positive_score - negative_score

    # 使用sigmoid函数标准化到0-1范围
    normalized_score = 1 / (1 + math.exp(-total_score / 3))

    # 确定情感标签
    if normalized_score > 0.7:
        sentiment = "积极"
    elif normalized_score < 0.3:
        sentiment = "消极"
    else:
        sentiment = "中性"

    # 返回详细分数和情感
    return {
        'score': normalized_score,
        'sentiment': sentiment,
        'positive': positive_score,
        'negative': negative_score,
        'total': total_score
    }


# 提取中文关键词（使用TF-IDF算法）
def extract_keywords_tfidf(text, topK=10):
    return jieba.analyse.extract_tags(text, topK=topK, withWeight=False)


# 提取中文关键词（使用TextRank算法）
def extract_keywords_textrank(text, topK=10):
    return jieba.analyse.textrank(text, topK=topK, withWeight=False)


# 文本摘要生成
def generate_summary(text, ratio=0.2):
    from jieba.analyse import set_stop_words
    from jieba.analyse import extract_tags

    # 设置停用词
    stopwords_path = "chinese_stopwords.txt"
    if os.path.exists(stopwords_path):
        set_stop_words(stopwords_path)

    # 提取关键词
    keywords = extract_tags(text, withWeight=True)

    # 计算句子权重
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    sentence_scores = {}

    for i, sentence in enumerate(sentences):
        score = 0
        for word, weight in keywords:
            if word in sentence:
                score += weight
        sentence_scores[i] = score

    # 选择最重要的句子
    num_sentences = max(1, int(len(sentences) * ratio))
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    selected_indices = [idx for idx, _ in sorted_sentences]
    selected_indices.sort()

    # 生成摘要
    summary = ' '.join([sentences[i] for i in selected_indices])
    return summary


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
    # 设置jieba分词
    setup_jieba()

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='高级中文文本分析工具')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help='要分析的文本文件路径')
    group.add_argument('-t', '--text', help='要分析的文本内容')

    parser.add_argument('-n', '--num', type=int, default=10,
                        help='显示的词频和关键词数量（默认：10）')
    parser.add_argument('--algorithm', choices=['tfidf', 'textrank'], default='tfidf',
                        help='关键词提取算法（tfidf或textrank，默认：tfidf）')
    parser.add_argument('--summary', action='store_true',
                        help='生成文本摘要')
    parser.add_argument('--sentiment-details', action='store_true',
                        help='显示详细情感分析数据')

    args = parser.parse_args()

    # 获取文本内容
    if args.file:
        text = read_text_from_file(args.file)
    else:
        text = args.text

    # 预处理中文文本
    words, processed_text = preprocess_chinese_text(text)

    if not words:
        print("没有可分析的有效文本内容。")
        return

    # 执行分析
    word_freq = analyze_word_frequency(words, args.num)
    sentiment_result = analyze_chinese_sentiment(processed_text)

    # 选择关键词提取算法
    if args.algorithm == 'textrank':
        keywords = extract_keywords_textrank(processed_text, args.num)
    else:
        keywords = extract_keywords_tfidf(processed_text, args.num)

    # 生成摘要（如果需要）
    summary = ""
    if args.summary:
        summary = generate_summary(text)

    # 显示结果
    print("\n===== 文本分析结果 =====")

    print("\n1. 词频分析（前{}个）:".format(args.num))
    for word, count in word_freq:
        print(f"  {word}: {count}次")

    print("\n2. 情感分析:")
    print(f"  主要情感: {sentiment_result['sentiment']}")
    print(f"  情感分数: {sentiment_result['score']:.4f} (0-1, 越接近1越积极)")

    if args.sentiment_details:
        print(f"  积极分数: {sentiment_result['positive']:.2f}")
        print(f"  消极分数: {sentiment_result['negative']:.2f}")
        print(f"  综合分数: {sentiment_result['total']:.2f}")

    print("\n3. 关键词提取（使用{}，前{}个）:".format(args.algorithm.upper(), args.num))
    if keywords:
        print("  " + ", ".join(keywords))
    else:
        print("  无法提取关键词")

    if args.summary:
        print("\n4. 文本摘要:")
        print(f"  {summary}")

    print("\n========================")


if __name__ == "__main__":
    main()