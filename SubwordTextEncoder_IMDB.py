import tensorflow_datasets as tfds
import urllib.request
import pandas as pd
from pprint import pprint 

"""
BPE의 변형 알고리즘인 Wordpiece Model 사용 -> 코퍼스의 우도(가능도(확률?)) 이 높은 쌍을 병합시킴 , _ 문장 복원을 함.
학습을 시키는게 아닌 단순한 사전을 이용하는것임.
"""

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
train_df = pd.read_csv('IMDb_Reviews.csv')

pprint(train_df['review'])

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    train_df['review'], target_vocab_size=2**13)

print(tokenizer.subwords[:100])

print(train_df['review'][20])

print('Tokenized sample question: {}'.format(tokenizer.encode(train_df['review'][20])))

# train_df에 존재하는 문장 중 일부를 발췌
sample_string = "It's mind-blowing to me that this film was even made."

# 인코딩한 결과를 tokenized_string에 저장
tokenized_string = tokenizer.encode(sample_string)
print ('정수 인코딩 후의 문장 {}'.format(tokenized_string))

# 이를 다시 디코딩
original_string = tokenizer.decode(tokenized_string)
print ('기존 문장: {}'.format(original_string))

print('단어 집합의 크기(Vocab size) :', tokenizer.vocab_size)

for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))


# 앞서 실습한 문장에 even 뒤에 임의로 xyz 추가\
# OOV 문제 발생!!
sample_string = "It's mind-blowing to me that this film was evenxyz made."

# 인코딩한 결과를 tokenized_string에 저장
tokenized_string = tokenizer.encode(sample_string)
print ('정수 인코딩 후의 문장 {}'.format(tokenized_string))

# 이를 다시 디코딩
original_string = tokenizer.decode(tokenized_string)
print ('기존 문장: {}'.format(original_string))


for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))