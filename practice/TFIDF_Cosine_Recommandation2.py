from bs4 import BeautifulSoup  
import pandas as pd
from tqdm import tqdm_notebook
import nltk
import re
from urllib.request import urlopen

domain='https://movie.naver.com'
story=[]
title=[]
genre=[]
for i in tqdm_notebook(range (1,11)):
    
    url="https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=pnt&date=20191201&page="+str(i)
    html = urlopen(url)
    soup = BeautifulSoup(html,"html.parser")
    titles=soup.find_all('div',class_='tit5')
    hype=[]
    href=[]
  
    try:
        
        for each in titles:
           
                hype=each.find_all('a')
                for link in hype:
                    href.append(link['href'])
        for j in tqdm_notebook(range(len(href))):
                domain='https://movie.naver.com'
                domain=domain+href[j]
                html=urlopen(domain)
                soup=BeautifulSoup(html,"html.parser")
                story.append(soup.find('p',class_="con_tx").get_text())
            
                title_tag=soup.find('h3',class_='h_movie')
                title.append(title_tag.find('a').get_text())
            
                genre_tag=soup.find('p')
                genre.append(genre_tag.find('a').get_text())
    except:
        pass
                  

#스토리 정규화 처리
import re

for i in tqdm_notebook(range(len(story))):

    story[i] = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》“”’]','',story[i] )
    story[i] = re.sub('\r\xa0','',story[i] )


print(title[0])

Nmovie=pd.DataFrame(data={'제목':title,'줄거리':story,'장르':genre})

print(Nmovie.head())

Nmovie['합침'] = (Nmovie['제목']) + Nmovie['줄거리'] + (Nmovie['장르'])
print(Nmovie['합침'][0])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
Nmovie['합침'] = Nmovie['합침'].fillna('')

print(Nmovie)

tfidf_matrix = tfidf.fit_transform(Nmovie["줄거리"])
# overview에 대해서 tf-idf 수행
print(tfidf_matrix.shape)

#코사인유사도
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(Nmovie.index, index=Nmovie['제목']).drop_duplicates()
print(indices.head())
#영화의 타이틀과 인덱스를 가진 테이블을 만듬
#영화 타이틀을 입력하면 인덱스를 리턴하려고 만듬

def get_recomm(title, cosine_sim=cosine_sim):
    choice = []
    # 선택한 영화의 타이틀로부터 해당되는 인덱스를 받아옵니다. 이제 선택한 영화를 가지고 연산할 수 있습니다.
    idx = indices[title]

    # 모든 영화에 대해서 해당 영화와의 유사도를 구합니다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아옵니다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 영화의 인덱스를 받아옵니다.
    movie_indices = [i[0] for i in sim_scores]
    
    for i in range(10):
        choice.append(Nmovie['제목'][movie_indices[i]])
    # 가장 유사한 10개의 영화의 제목을 리턴합니다.
    print('***영화 추천 순위***')
    for i in range(10):
        print(str(i+1) + '순위 : ' + choice[i])

get_recomm('토이 스토리')
