from gensim.models import FastText
import load_data as ld

# result = ld.process_data()

# model = FastText(result, vector_size=100, window=5, min_count=5, workers=4, sg=1)


# model.save("fast_text")
loaded_model = FastText.load("fast_text")
print(loaded_model.wv.most_similar("electrofishing"))

#print(model.wv.most_similar("electrofishing"))
