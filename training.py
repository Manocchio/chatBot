import random
import json
import numpy as np

import nltk
import codecs
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
lematizer = WordNetLemmatizer()

intents = json.loads(codecs.open('intents.json', 'r', 'utf-8-sig').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lematizer.lemmatize(word)
         for word in words if word not in ignoreLetters]

words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('classes.pkl', 'wb'))


training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lematizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append([bag, outputRow])


random.shuffle(training)
training = np.array(training)


trainX = list(training[':', 0])
trainY = list(training[':', 1])

model = Sequential()

model.add(Dense(128, input_shape=len(trainX[0]),), activation='relu'))
model.add(Dropout(0, 5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0, 5))
odel.add(Dense(len(trainY[0]), activation='softmax'))

sgd=SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = "categorical crossentropy",
              optimizer = sgd, metrics = ['accuracy'])

model.fit(np.array(trainX), np.array(trainY),
          epochs = 200, batch_size = 5, verbose =1)

mode.save("chatbot model.model")
