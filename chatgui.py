import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('D:/chatbox/chat1/chatbot_model.keras')
import random
import json
with open('D:/chatbox/chat1/intents.json', mode='r', encoding='utf-8') as file:
    intents = json.load(file)
words = pickle.load(open('D:/chatbox/chat1/words.pkl','rb'))
classes = pickle.load(open('D:/chatbox/chat1/classes.pkl','rb'))

from forward_chaining import ForwardChaining
from class_all import *
from class_all import ConvertData

db = ConvertData()
db.convertbenh()  # bang benh
db.converttrieuchung()  # bang trieu chung
db.getfc()
db.getbc()
luat_tien = db.groupfc()

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    # list_of_intents = intents_json['intents']  
    
    # for i in list_of_intents:
    #     if(i['tag']== tag):
    #         result = random.choice(i['responses'])
    #         break
    return tag

#suy diễn tiến
def forward_chaining(rule, fact, goal, file_name):
    fc = ForwardChaining(rule, fact, None, file_name)

    list_predicted_disease = [i for i in fc.facts if i[0] == "D"]
    # print(f'-->Chatbot: Chúng tôi dự đoán có thể bị bệnh :', end=" ")
    for i in list_predicted_disease:
        temp = db.get_benh_by_id(i)
        print(temp['tenBenh'], end=', ')
    # print()
    # print(f'-->Chatbot: Trên đây là chuẩn đoán sơ bộ của chúng tôi. Tiếp theo, chúng tôi sẽ hỏi {person.name} một số câu hỏi để đưa ra kết quả chính xác.', end=" ")
    return list_predicted_disease   
    
print("|============= Welcome to College Equiry Chatbot System! =============|")
print("|============================== Feel Free ============================|")
print("|================================== To ===============================|")
print("|=============== Ask your any query about our college ================|")
# while True:
#     message = input("| You: ")
#     if message == "bye" or message == "Goodbye":
#         ints = predict_class(message, model)
#         res = getResponse(ints, intents)
#         print("| Bot:", res)
#         print("|===================== The Program End here! =====================|")
#         exit()

#     else:
#         ints = predict_class(message, model)
#         res = getResponse(ints, intents)
#         print(ints[0]['intent'])
#         print("| Bot:", res)
list_symptom_of_person_id = []
message = input("| You: ")
ints = predict_class(message, model)
res = getResponse(ints, intents)
print(res)
list_symptom_of_person_id.append(res)
list_predicted_disease = forward_chaining(luat_tien, list_symptom_of_person_id, None, 'ex')
print(list_predicted_disease)