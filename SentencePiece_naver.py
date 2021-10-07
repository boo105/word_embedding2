import pandas as pd
import sentencepiece as spm
import urllib.request
import csv
from pprint import pprint

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

naver_df = pd.read_table('ratings.txt')
pprint(naver_df[:5])

print('리뷰 개수 :',len(naver_df)) # 리뷰 개수 출력

print(naver_df.isnull().values.any())

naver_df = naver_df.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(naver_df.isnull().values.any()) # Null 값이 존재하는지 확인

print('리뷰 개수 :',len(naver_df)) # 리뷰 개수 출력

with open('naver_review.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(naver_df['document']))

spm.SentencePieceTrainer.Train('--input=naver_review.txt --model_prefix=naver --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')

vocab_list = pd.read_csv('naver.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
pprint(vocab_list[:10])

print(len(vocab_list))


sp = spm.SentencePieceProcessor()
vocab_file = "naver.model"
sp.load(vocab_file)


lines = [
  "뭐 이딴 것도 영화냐.",
  "진짜 최고의 영화입니다 ㅋㅋ",
]
for line in lines:
  print(line)
  print(sp.encode_as_pieces(line))
  print(sp.encode_as_ids(line))
  print()

print(sp.GetPieceSize())

print(sp.IdToPiece(4))

print(sp.PieceToId('영화'))

print(sp.DecodeIds([54, 200, 821, 85]))

print(sp.DecodePieces(['▁진짜', '▁최고의', '▁영화입니다', '▁ᄏᄏ']))

print(sp.encode('진짜 최고의 영화입니다 ㅋㅋ', out_type=str))
print(sp.encode('진짜 최고의 영화입니다 ㅋㅋ', out_type=int))
