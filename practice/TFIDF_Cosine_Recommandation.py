import json
import time
from math import log
import numpy as np
import pandas as pd
from numpy.dual import norm
from eunjeon import Mecab


# 특정문서 d에서의 특정단어 t의 등장 횟수
def tf(t, d):
    return d.count(t)

# df(t)에 반비례하는 수, df(t)=특정 단어 t가 등장한 문서의 수.
def idf(t, data):
    df = 0
    for d in data:
        df += t in d['content']
    return log(len(data) / (df + 1))

def tfidf(t, d, data):
    return tf(t, d) * idf(t, data)

# 코사인 유사도 계산
def cos_sim(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))

def practice(target_data):
    # data를 불러온다.
    with open('data.json') as file:
        data = json.load(file)

    data.append(target_data)

    # okt 토크나이저를 이용해 영화 시놉시스를 토큰화한다.
    mecab = Mecab()
    contents = list(map(lambda x: x['content'], data))
    # 명사 단어들을 추출합니다.
    vocab = []
    for content in contents:
        vocab += mecab.nouns(content)

    # 중복을 제거합니다.
    vocab = list(set(vocab))

    # 불용어를 제거합니다. (stop_words에 포함되거나 1글자짜리는 제거)
    stop_words = ['이제', '인물', '동안', '단번', '스무', '사이', '순간', '과연', '마저', '만큼', '누구', '주변', '소유자', '오늘']
    vocab = list(filter(lambda x: len(x) > 1 and x not in stop_words, vocab))

data = []
# 데이터 laad
# data를 불러온다.
with open('./practice/data/data.json', "r", encoding="UTF8") as file:
    data = json.load(file)

# data.append(target_data)
# okt 토크나이저를 이용해 영화 시놉시스를 토큰화한다.
mecab = Mecab()
contents = list(map(lambda x: x['content'], data))
# 명사 단어들을 추출합니다.
vocab = []
for content in contents:
    vocab += mecab.nouns(content)

# TF-IDF 구하기
result = []
for i in range(len(data)):
    result.append([])
    d = data[i]['content']
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t, d, data))

    # 만들어진 TF-IDF DTM 출력
    v = pd.DataFrame(result, columns=vocab, index=list(map(lambda x: x['name'], data)))
    print(f'* 영화 수: {v.shape[0]}, 단어 수: ${v.shape[1]}')
    print(v.to_string())

    # target_data와 코사인 유사도를 구해서 가장 높은 순으로 정렬
    sim_scores = []
    for i in range(len(data)):
        name = data[i]['name']
        if name != target_data['name']:
            sim_scores.append({
                'name': name,
                'score': cos_sim(v.loc[target_data['name']], v.loc[name])
            })

    print('* 추천 순위')
    sim_scores = sorted(sim_scores, key=lambda x: x['score'], reverse=True)
    print(pd.DataFrame(sim_scores).to_string())

if __name__ == '__main__':
    start = time.time()
    practice({
        'name': '베테랑',
        'content': "한 번 꽂힌 것은 무조건 끝을 보는 행동파 ‘서도철’(황정민), 20년 경력의 승부사 ‘오팀장’(오달수), 위장 전문 홍일점 ‘미스봉’(장윤주), 육체파 ‘왕형사’(오대환), "
                   "막내 ‘윤형사’(김시후)까지 겁 없고, 못 잡는 것 없고, 봐 주는 것 없는 특수 강력사건 담당 광역수사대. 오랫동안 쫓던 대형 범죄를 해결한 후 숨을 돌리려는 찰나, "
                   "서도철은 재벌 3세 ‘조태오’(유아인)를 만나게 된다. 세상 무서울 것 없는 안하무인의 조태오와 언제나 그의 곁을 지키는 오른팔 ‘최상무’(유해진). 서도철은 의문의 사건을 "
                   "쫓던 중 그들이 사건의 배후에 있음을 직감한다. 건들면 다친다는 충고에도 불구하고 포기하지 않는 서도철의 집념에 판은 걷잡을 수 없이 커져가고 조태오는 이를 비웃기라도 하듯 "
                   "유유히 포위망을 빠져 나가는데… 베테랑 광역수사대 VS 유아독존 재벌 3세 2015년 여름, 자존심을 건 한판 대결이 시작된다! "
    })
    end = time.time()
    print(f'* 총 실행 시간: {end - start}s')