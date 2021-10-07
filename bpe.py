import re, collections

"""
1994년에 제안된 알고리즘

OOV 문제를 완화 시키기 위해(희귀 단어 , 신조어)
서브워드 분리 알고리즘을 사용함.

BPE는 서브워드 분리 알고리즘 중 하나임.
"""

# BPE 수행할 횟수
num_merges = 10

# 훈련데이터에 있는 단어와 등장 빈도수라 가정
dictionary = {'l o w </w>' : 5,
         'l o w e r </w>' : 2,
         'n e w e s t </w>':6,
         'w i d e s t </w>':3
         }

"""
BPE의 특징은 알고리즘의 동작을 몇번 반복 할건지 사용자가 결정함.

가장 빈도수가 높은 유니그램의 쌍을 하나의 유니그램으로 통합하는 과정을 반복함.
"""
def get_stats(dictionary):
    # 유니그램의 pair들의 빈도수를 카운트
    pairs = collections.defaultdict(int)
    for word, freq in dictionary.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    print('현재 pair들의 빈도수 :', dict(pairs))
    return pairs

def merge_dictionary(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

bpe_codes = {}
bpe_codes_reverse = {}

for i in range(num_merges):
    print("### Iteration {}".format(i + 1))
    pairs = get_stats(dictionary)
    best = max(pairs, key=pairs.get)
    dictionary = merge_dictionary(best, dictionary)

    bpe_codes[best] = i
    bpe_codes_reverse[best[0] + best[1]] = best

    print("new merge: {}".format(best))
    print("dictionary: {}".format(dictionary))


print()
print("==== 단어 집합 사전 ====")
print(bpe_codes)
print("\n")

"""
단어 집합에 없는 단어를 대처하는 방법
"""

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as a tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def encode(orig):
    """Encode word based on list of BPE merge operations, which are applied consecutively"""

    word = tuple(orig) + ('</w>',)
    print("__word split into characters:__ <tt>{}</tt>".format(word))

    pairs = get_pairs(word)    

    if not pairs:
        return orig

    iteration = 0
    while True:
        iteration += 1
        print("__Iteration {}:__".format(iteration))

        print("bigrams in the word: {}".format(pairs))
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        print("candidate for merging: {}".format(bigram))
        if bigram not in bpe_codes:
            print("__Candidate not in BPE merges, algorithm stops.__")
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):

            # bigram의 first 단어를 찾아 new_word에 추가해줌
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break
            
            # bigram은 당연히 연속쌍이기 떄문에 i+1이 second인지 확인함
            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        print("word after merging: {}".format(word))
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # 특별 토큰인 </w>는 출력하지 않는다.
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>',''),)

    return word

encode("loki")
encode("lowest")
encode("lowing")
encode("highing")
