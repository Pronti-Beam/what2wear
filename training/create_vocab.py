#from nltk.book import text7, text8 # Wall Street Journal, Personals corpus
from collections import Counter, OrderedDict
from nltk import FreqDist, word_tokenize
import json

x = Counter()
with open("corpus.txt", "r") as fin:
    x.update(word_tokenize(fin.read()))
text_set = set(x.keys())
# add degree sign
text_set.add(u'\u2103')
text_set.add(u'\u00b0')

#old_dict = json.load(open('data/vocab.json'))
#for k,v in old_dict.items():
    #text_set.add(k)

# add numbers for temperature -- enough to encapsulate celsius and fahrenheit
for j in range(-50,125):
    text_set.add(str(j))

f = open('final_word_dict.txt', 'r')
lines = f.readlines()
for line in lines:
    word = line.split(' ')[0]
    text_set.add(word)
    if len(text_set) >= 2480:
        break

dic = OrderedDict()
text_set = sorted(list(text_set))
print(len(text_set))
for i, word in enumerate(text_set):
    dic[word] = i


#print(dic)
with open('./data/vocab.json', 'w') as outfile:
    json.dump(dic,outfile)
