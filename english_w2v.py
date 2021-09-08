import nltk
import zipfile
import load_data as ld
#nltk.download('punkt')


# result = ld.process_data()

# print('총 샘플의 개수 : {}'.format(len(result)))

# # 샘플 3개만 출력
# for line in result[:3]:
#     print(line)

# from gensim.models import Word2Vec
# model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)
# # 만약 TypeError: __init__() got an unexpected keyword argument 'size' 라는 에러 발생 시에는
# # size 대신 vector_size로 바꿔서 적어주세요.

# model_result = model.wv.most_similar("man")
# print(model_result)

from gensim.models import KeyedVectors
# model.wv.save_word2vec_format('eng_w2v') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") # 모델 로드

model_result = loaded_model.most_similar("man")
model_result = loaded_model.most_similar("electrofishing")

print(model_result)