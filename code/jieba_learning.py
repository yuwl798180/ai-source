import time
import jieba
import jieba.posseg as pseg
import jieba.analyse




# 添加自定义词典
# 词典格式为：一个词占一行，每一行分三部分：词语、词频（可省略）、词性（可省略）。
# lg_x5
# lgx5
# ---（这种无效）---
# lg x5
# lg-x5
jieba.load_userdict('data/userdict.txt')

# 自定义添加删除词语
jieba.add_word('石墨烯')
jieba.del_word('自定义词')

# 开启并行，不支持 window 系统
# 需要在自定义词典后
# 但是后面的 suggest_freq 仍然无效
## jieba.enable_parallel()


sent = (
'我来到北京清华大学,'
'lg_x5是我最喜欢的一款手机.\n'
'蔡徐坤著名舞蹈鸡你太美。'
'如果放到post中将出错。'
)

# cut_all=True 时为全切，自动去标点
# cut_for_search 粒度更细腻
# lcut 和 lcut_for_search 直接生成 list
time.sleep(0.1)
print("Default Mode: \n" + "/ ".join(jieba.cut(sent)))            # 精确模式
print("Full Mode: \n" + "/ ".join(jieba.cut(sent, cut_all=True))) # 全模式
print("Search Mode: \n" + "/ ".join(jieba.cut_for_search(sent)))  # 搜索引擎模式

# 动态调整词频，使其能（或不能）被分出来。此时最好 HMM=False
jieba.suggest_freq(('中', '将'), True)
print("\nsuggest_freq Mode: \n" + "/ ".join(jieba.cut(sent, HMM=False)))


# 词性标注
result = pseg.cut(sent)
print('\njieba.posseg: ')
for word, flag in result:
    print(word, "/", flag, ", ", end=' ')
print('\n')

# 词语在原文的起止位置
result = jieba.tokenize(sent)
for tk in result:
    print("word: %-10s start: %d, end:%d" % (tk[0],tk[1],tk[2]))


# 基于 tf-idf 的关键词抽取，可以使用自定义词典（IDF / Stop Words）
# jieba.analyse.set_stop_words("../extra_dict/stop_words.txt")
# jieba.analyse.set_idf_path("../extra_dict/idf.txt.big");
withWeight = True
print('\ntf-idf for keys: ')
tags = jieba.analyse.extract_tags(sent, topK=10, withWeight=withWeight)
if withWeight:
    for tag in tags:
        print("tag: %s\t weight: %f" % (tag[0],tag[1]))
else:
    print(tags)

# 基于 TextRank 算法的关键词抽取
print('\ntextrank for keys: ')
tags_other = jieba.analyse.textrank(sent, withWeight=withWeight)
if withWeight:
    for tag in tags_other:
        print("tag: %s\t weight: %f" % (tag[0],tag[1]))
else:
    print(tags_other)
