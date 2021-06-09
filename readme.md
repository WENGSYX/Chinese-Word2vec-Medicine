# Chinese-Word2vec-Medicine



#### 中文生物医学词向量，可能是目前唯一一个医学领域的中文词向量。



##### 之前需要做医学相关的课题，要用医学相关的词向量，可惜一直找不到，只好自己来做了。

##### 为了完成这个医学词向量，花了几天时间找了各种医学生物语料库和相关数据集并对其整理。整体语料库包括医学文献，医患对话，维基百科百度知道等医学相关问题，整体语料库共计1.6G,总共7052948句子，仅为生物医学领域相关语料。



### 已上传至百度网盘  欢迎使用

#### https://pan.baidu.com/s/1YqTOlDqZ3bTzGYAGMxW2Cw 

### 提取码：**8888** 

### 





## 词向量对比

### 医学词向量

> wv1.most_similar('海马')
> Out[30]: 
> [('额叶', 0.4515002965927124),
>  ('颞叶', 0.4498691260814667),
>  ('枕叶', 0.38755619525909424),
>  ('顶叶', 0.386254221200943),
>  ('基底节', 0.381935179233551),
>  ('岛叶', 0.35826876759529114),
>  ('苍白球', 0.33769935369491577),
>  ('尾状核', 0.33755943179130554),
>  ('大脑半球', 0.33359262347221375),
>  ('额页', 0.32096001505851746)]

### 通用词向量(https://github.com/Embedding/Chinese-Word-Vectors)

> wv2.most_similar('海马')
> Out[31]: 
>
> [('海马牌', 0.6078361868858337), 
>
> ('海马齿', 0.5532827377319336), 
>
> ('普力马', 0.5418268442153931),
>
>  ('马自达', 0.5407805442810059),
>
>  ('东南汽车', 0.5387718677520752),
>
>  ('000572', 0.5375587344169617), 
>
> ('宝骏', 0.5361850261688232),
>
>  ('海马回', 0.5352568030357361), 
>
> ('北汽', 0.5325318574905396), 
>
> ('小海马', 0.5315144062042236)]



此医学词向量含278256个生物医学相关词汇，维度512，使用gensim训练。

```python
model = word2vec.Word2Vec(sent, sg=0, epochs=8,vector_size=512,  window=5,  min_count=4,  negative=3, sample=0.001, hs=1, workers=16)
```

想要使用只需要通过 gensim.models.KeyedVectors加载使用即可。

```python
model = KeyedVectors.load_word2vec_format('Medical.txt', binary=False)
```

### 

