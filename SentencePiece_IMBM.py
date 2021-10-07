import sentencepiece as spm
import pandas as pd
import urllib.request
import csv

"""
구글의 센턴스피스 (내부단어분리)

먼저 토큰화를 진행해야지 단어 분리 알고리즘 적용 가능
하지만 센턴스피스는 사전 토큰화 없이 단어분리 토큰화를 알아서 해줌

Unigram Language Model Tokenizer 사용해
코퍼스의 우도(Likelihood)가 감소하는 정도를 정렬하여 최악의 영향을 주는 10%~20% 토큰을 제거함.
"""

#urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

train_df = pd.read_csv('IMDb_Reviews.csv')
train_df['review']

print('리뷰 개수 :',len(train_df)) # 리뷰 개수 출력

with open('imdb_review.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(train_df['review']))

"""
input : 학습시킬 파일
model_prefix : 만들어질 모델 이름
vocab_size : 단어 집합의 크기
model_type : 사용할 모델 (unigram(default), bpe, char, word)
max_sentence_length: 문장의 최대 길이
pad_id, pad_piece: pad token id, 값
unk_id, unk_piece: unknown token id, 값
bos_id, bos_piece: begin of sentence token id, 값
eos_id, eos_piece: end of sequence token id, 값
user_defined_symbols: 사용자 정의 토큰
"""
spm.SentencePieceTrainer.Train('--input=imdb_review.txt --model_prefix=imdb --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')

vocab_list = pd.read_csv('imdb.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
vocab_list.sample(10)

print(vocab_list)


sp = spm.SentencePieceProcessor()
vocab_file = "imdb.model"
sp.load(vocab_file)


lines = [
  "I didn't at all think of it this way.",
  "I have waited a long time for someone to film"
]
for line in lines:
  print(line)
  print(sp.encode_as_pieces(line))
  print(sp.encode_as_ids(line))
  print()


print(sp.GetPieceSize())

print(sp.IdToPiece(430))

print(sp.PieceToId('▁character'))

print(sp.DecodeIds([41, 141, 1364, 1120, 4, 666, 285, 92, 1078, 33, 91]))

print(sp.DecodePieces(['▁I', '▁have', '▁wa', 'ited', '▁a', '▁long', '▁time', '▁for', '▁someone', '▁to', '▁film']))

print(sp.encode('I have waited a long time for someone to film', out_type=str))
print(sp.encode('I have waited a long time for someone to film', out_type=int))






