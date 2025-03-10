# Chinese-Word2vec-Medicine



#### 中文生物医学词向量，可能是目前唯一一个医学领域的中文大型开源词向量。



之前需要用医学相关的词向量，可惜一直找不到，只好自己来做了。
除了这个词向量，还另外整理了一份五千词的生物医学高频词表，通过对医学词汇进行统计，取出现次数在200次以上的医学词汇构建而成，如有需要可直接github下载med_word.txt。

为了完成这个医学词向量，花了几天时间找了各种医学生物语料库和相关数据集并对其整理。整体语料库包括医学文献，医患对话，维基百科百度知道等医学相关语料，整体语料库共计1.6G,总共7052948句子，仅为生物医学领域相关语料。
使用专业医学类词汇进行分词（词汇表详见http://thuocl.thunlp.org/）
注意，因为部分语料来自网络医患对话，导致错别字的出现，比如‘身体’的最近词向量竟是‘生体’，因此请小心使用。
词向量已上传至百度网盘  欢迎使用

### https://pan.baidu.com/s/1YqTOlDqZ3bTzGYAGMxW2Cw 

### 提取码：**8888** 






## 训练与使用
```
python chinese-medical-word2vec.py --corpus_dir ./corpus --output_dir ./output --med_dict ./med_dict.txt
```


## 测试
```
python usage-example.py
```

## 词向量对比

### 医学词向量
```
#### wv1.most_similar('海马')
 #### Out[30]: 
 
 [('额叶', 0.4515002965927124),
  ('颞叶', 0.4498691260814667),
  ('枕叶', 0.38755619525909424),
  ('顶叶', 0.386254221200943),
  ('基底节', 0.381935179233551),
  ('岛叶', 0.35826876759529114),
  ('苍白球', 0.33769935369491577),
  ('尾状核', 0.33755943179130554),
  ('大脑半球', 0.33359262347221375),
  ('额页', 0.32096001505851746)]

#### wv1.most_similar('头孢丙烯片')
#### Out[32]: 

[('头孢地尼', 0.5654973387718201),
 ('阿莫西林', 0.5394408106803894),
 ('头孢地尼胶囊', 0.5379139184951782),
 ('妇乐片', 0.5260443091392517),
 ('头孢地尼分散片', 0.5213251709938049),
 ('康妇炎胶囊', 0.5203120708465576),
 ('裸花紫珠胶囊', 0.5182883143424988),
 ('头孢克洛缓释片', 0.5178096294403076),
 ('头胞克洛', 0.5159974098205566),
 ('罗红霉素', 0.5115748643875122)]
 ```
 
### 通用词向量(https://github.com/Embedding/Chinese-Word-Vectors)
```
 #### wv2.most_similar('海马')
 #### Out[31]: 

 [('海马牌', 0.6078361868858337), 
 ('海马齿', 0.5532827377319336), 
 ('普力马', 0.5418268442153931),
  ('马自达', 0.5407805442810059),
  ('东南汽车', 0.5387718677520752),
  ('000572', 0.5375587344169617), 
 ('宝骏', 0.5361850261688232),
  ('海马回', 0.5352568030357361), 
 ('北汽', 0.5325318574905396), 
 ('小海马', 0.5315144062042236)]
 #### wv2.most_similar('头孢')
 #### Out[33]: 
[('头孢拉定', 0.7558029294013977), 
('头孢菌素', 0.7490127086639404), 
('头孢类', 0.7476578950881958), ('头孢氨苄', 0.7415952682495117), ('头孢曲松钠', 0.7406224608421326), ('头孢哌酮', 0.7398018836975098), ('头孢噻肟钠', 0.7393568158149719), ('头孢噻吩', 0.7348008751869202), ('头孢哌酮钠', 0.729317843914032), ('氨苄', 0.7292327284812927)]
```
### 此医学词向量含278256个生物医学相关词汇，维度512，使用gensim训练。

```python
model = word2vec.Word2Vec(sent, sg=0, epochs=8,vector_size=512,  window=5,  min_count=4,  negative=3, sample=0.001, hs=1, workers=16)
```

想要使用只需要通过 gensim.models.KeyedVectors加载使用即可。

```python
model = KeyedVectors.load_word2vec_format('Medical.txt', binary=False)
```

### 训练语料
可以从本人发布的另一个医疗数据集中，进行访问，(https://github.com/WENGSYX/CMCQA)

### 引用
如果我的词向量能帮助您，欢迎引用：
```
@article{li2024distinct,
  title={Distinct but correct: generating diversified and entity-revised medical response},
  author={Li, Bin and Sun, Bin and Li, Shutao and Chen, Encheng and Liu, Hongru and Weng, Yixuan and Bai, Yongping and Hu, Meiling},
  journal={Science China Information Sciences},
  volume={67},
  number={3},
  pages={1--20},
  year={2024},
  publisher={Springer}
}

@article{weng2023large,
  title={Large Language Models Need Holistically Thought in Medical Conversational QA},
  author={Weng, Yixuan and Li, Bin and Xia, Fei and Zhu, Minjun and Sun, Bin and He, Shizhu and Liu, Kang and Zhao, Jun},
  journal={arXiv preprint arXiv:2305.05410},
  year={2023}
}

```
