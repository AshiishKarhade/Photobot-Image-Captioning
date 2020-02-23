from train import encode
from train import greedySearch

test = 'test.jpg'
pic = encode(test)
pic = pic.reshape((1, 2048))

text = greedySearch(pic)
# text is the output
