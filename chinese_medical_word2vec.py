import os
import re
import jieba
import logging
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from gensim.models import word2vec
from gensim.models import KeyedVectors

# 设置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练中文医学词向量")
    parser.add_argument('--corpus_dir', type=str, default='./corpus',
                        help='语料库文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出文件夹路径')
    parser.add_argument('--med_dict', type=str, default='./med_dict.txt',
                        help='医学词典路径')
    parser.add_argument('--vector_size', type=int, default=512,
                        help='词向量维度')
    parser.add_argument('--window', type=int, default=5,
                        help='上下文窗口大小')
    parser.add_argument('--min_count', type=int, default=4,
                        help='最小词频')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                        help='使用的CPU核心数')
    parser.add_argument('--sg', type=int, default=0,
                        choices=[0, 1], help='0表示CBOW模型，1表示Skip-gram模型')
    parser.add_argument('--epochs', type=int, default=8,
                        help='训练轮数')
    parser.add_argument('--negative', type=int, default=3,
                        help='负采样样本数')
    parser.add_argument('--sample', type=float, default=0.001,
                        help='高频词降采样')
    parser.add_argument('--hs', type=int, default=1,
                        choices=[0, 1], help='0表示不使用层次softmax，1表示使用')

    return parser.parse_args()


def load_medical_dict(dict_path):
    """加载医学词典"""
    if not os.path.exists(dict_path):
        logging.warning(f"医学词典 {dict_path} 不存在，将使用jieba默认词典")
        return False

    logging.info(f"加载医学词典: {dict_path}")
    jieba.load_userdict(dict_path)
    return True


def clean_text(text):
    """清洗文本"""
    if not isinstance(text, str):
        return ""

    # 移除URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 移除非中文、数字、字母、常见标点之外的字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,;:!?，。；：！？、]', ' ', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def segment_text(text):
    """分词"""
    return jieba.lcut(text)


def collect_corpus_files(corpus_dir):
    """收集所有语料文件"""
    corpus_files = []
    corpus_dir = Path(corpus_dir)

    if not corpus_dir.exists():
        logging.error(f"语料库路径 {corpus_dir} 不存在!")
        return []

    for file_path in corpus_dir.glob('**/*'):
        if file_path.is_file():
            if file_path.suffix.lower() in ['.txt', '.csv', '.json']:
                corpus_files.append(file_path)

    logging.info(f"找到 {len(corpus_files)} 个语料文件")
    return corpus_files


def process_txt_file(file_path):
    """处理TXT文件"""
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = clean_text(line)
                if line:
                    sentences.append(segment_text(line))
    except Exception as e:
        logging.error(f"处理文件 {file_path} 时出错: {e}")

    return sentences


def process_csv_file(file_path):
    """处理CSV文件"""
    sentences = []
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        for column in df.columns:
            if df[column].dtype == 'object':  # 只处理文本列
                for text in df[column].dropna():
                    text = clean_text(text)
                    if text:
                        sentences.append(segment_text(text))
    except Exception as e:
        logging.error(f"处理文件 {file_path} 时出错: {e}")

    return sentences


def process_json_file(file_path):
    """处理JSON文件"""
    sentences = []
    try:
        df = pd.read_json(file_path, encoding='utf-8')
        for column in df.columns:
            if df[column].dtype == 'object':  # 只处理文本列
                for text in df[column].dropna():
                    if isinstance(text, str):
                        text = clean_text(text)
                        if text:
                            sentences.append(segment_text(text))
    except Exception as e:
        logging.error(f"处理文件 {file_path} 时出错: {e}")

    return sentences


def prepare_training_data(corpus_files):
    """准备训练数据"""
    all_sentences = []

    for file_path in tqdm(corpus_files, desc="处理语料文件"):
        suffix = file_path.suffix.lower()

        if suffix == '.txt':
            sentences = process_txt_file(file_path)
        elif suffix == '.csv':
            sentences = process_csv_file(file_path)
        elif suffix == '.json':
            sentences = process_json_file(file_path)
        else:
            continue

        all_sentences.extend(sentences)

    logging.info(f"总共处理了 {len(all_sentences)} 个句子")
    return all_sentences


def train_word2vec_model(sentences, args):
    """训练Word2Vec模型"""
    logging.info("开始训练Word2Vec模型...")

    model = word2vec.Word2Vec(
        sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,  # 0 = CBOW, 1 = Skip-gram
        epochs=args.epochs,  # 训练轮数
        negative=args.negative,  # 负采样
        sample=args.sample,  # 高频词降采样
        hs=args.hs  # 是否使用层次softmax
    )

    logging.info("模型训练完成!")
    return model


def save_model(model, output_dir):
    """保存模型"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整模型
    model_path = os.path.join(output_dir, 'medical_word2vec.model')
    model.save(model_path)
    logging.info(f"完整模型已保存到: {model_path}")

    # 保存词向量
    vector_path = os.path.join(output_dir, 'Medical.txt')
    model.wv.save_word2vec_format(vector_path, binary=False)
    logging.info(f"词向量已保存到: {vector_path}")

    # 保存词汇表
    vocab_path = os.path.join(output_dir, 'vocabulary.txt')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for word in model.wv.index_to_key:
            f.write(f"{word}\n")
    logging.info(f"词汇表已保存到: {vocab_path}")

    # 保存高频医学词汇表
    high_freq_words = [(word, model.wv.get_vecattr(word, "count"))
                       for word in model.wv.index_to_key]
    high_freq_words.sort(key=lambda x: x[1], reverse=True)

    med_words_path = os.path.join(output_dir, 'med_word.txt')
    with open(med_words_path, 'w', encoding='utf-8') as f:
        count = 0
        for word, freq in high_freq_words:
            if freq >= 200:  # 筛选出现次数大于200的词
                f.write(f"{word}\t{freq}\n")
                count += 1
                if count >= 5000:  # 最多保存5000个高频词
                    break
    logging.info(f"医学高频词表已保存到: {med_words_path}")


def extract_high_freq_medical_terms(model, output_dir, top_n=5000, min_freq=200):
    """提取高频医学术语"""
    word_freqs = [(word, model.wv.get_vecattr(word, "count")) for word in model.wv.index_to_key]
    word_freqs.sort(key=lambda x: x[1], reverse=True)

    med_words_path = os.path.join(output_dir, 'med_word.txt')
    with open(med_words_path, 'w', encoding='utf-8') as f:
        count = 0
        for word, freq in word_freqs:
            if freq >= min_freq:
                f.write(f"{word}\t{freq}\n")
                count += 1
                if count >= top_n:
                    break

    logging.info(f"已保存 {count} 个高频医学词汇到 {med_words_path}")


def test_model(model):
    """测试模型效果"""
    logging.info("测试模型效果...")

    test_words = ['海马', '头孢', '肝炎', '心脏', '糖尿病']
    for word in test_words:
        if word in model.wv:
            logging.info(f"\n{word} 的最相似词:")
            for similar_word, similarity in model.wv.most_similar(word):
                logging.info(f"  {similar_word}: {similarity:.4f}")
        else:
            logging.warning(f"{word} 不在词汇表中")


def main():
    """主函数"""
    args = parse_args()

    # 加载医学词典
    load_medical_dict(args.med_dict)

    # 收集语料文件
    corpus_files = collect_corpus_files(args.corpus_dir)
    if not corpus_files:
        logging.error("没有找到语料文件，程序退出")
        return

    # 准备训练数据
    sentences = prepare_training_data(corpus_files)
    if not sentences:
        logging.error("处理后没有可用的训练数据，程序退出")
        return

    # 训练模型
    model = train_word2vec_model(sentences, args)

    # 测试模型
    test_model(model)

    # 保存模型
    save_model(model, args.output_dir)

    logging.info("所有操作完成!")


if __name__ == "__main__":
    main()