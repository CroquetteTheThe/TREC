import torch

import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from torch import tensor

class QuestionDataset(Dataset):
    
    # Special constructor
    # | nb_most_commons can either be the number of most common words you
    # | want to work with, OR a list of word you want to work with
    # If nb_most_commons == -1, then all word will count
    
    def __init__(self, train_data, nb_most_commons=-1):
        questions = []
        labels = []

        # Black list
        black_list = '\'`[@_!#$%^&*()<>?/\|}{~:]'
        
        for string in train_data:
            question_str = []
            for x in string.split()[1:]:
                s = ""
                for c in x:
                    if not c in black_list:
                        s += c
                if not s == "":
                    question_str.append(s.lower())
                        
            labels.append(string.split()[0])
            questions.append(question_str)

        
        if isinstance(nb_most_commons, int):
            # Vocabulary of unique words
            data = []
            for q in questions:
                for w in q:
                    data.append(w)
            self.reparti_word = Counter(data)
            
            if nb_most_commons < 0:
                most_commons_words = self.reparti_word.most_common(len(data))
            else:
                most_commons_words = self.reparti_word.most_common(nb_most_commons)
            
            self.word_list = list([x[0] for x in most_commons_words])
            self.word_list.append('<bos>')
            self.word_list.append('<eos>')
            self.word_list.append('<unk>')
        elif isinstance(nb_most_commons, list):
            self.word_list = nb_most_commons
        else:
            print("ERROR: second arg is neither an int, nor a list")
            
        words_array = np.array(self.word_list)
        
        # Add tags <bos> and <eos> to questions
        for q in questions:
            if q[0] != '<bos>' :
                q.insert(0, '<bos>')
                q.append('<eos>')

        # Integer encoding with OneHotEncoder
        words_tre = words_array.reshape(len(words_array),1)
        one_hot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = one_hot_encoder.fit_transform(words_tre)
        # Creating a dictionnary of word and its one hot array
        self.words_onehoted = {}
        for i in range(0, len(words_array)):
            self.words_onehoted[self.word_list[i]] = onehot_encoded[i]

        # One hot categories
        self.categories_num = {}
        self.categories_num['ABBR'] = 0 # Abbreviation
        self.categories_num['ENTY'] = 1 # Entity
        self.categories_num['DESC'] = 2 # Description
        self.categories_num['HUM']  = 3 # Human
        self.categories_num['LOC']  = 4 # Location
        self.categories_num['NUM']  = 5 # Numeric

        self.batch_data = []
        for num_question in range(len(questions)):
            # Construction of question_onehot list.
            question_onehot = [self.get_onehot_word(word) for word in questions[num_question]]

            # Construction of category_onehot.
            category = labels[num_question].partition(':')[0]
            category_onehot = self.get_num_category(category)
            self.batch_data.append([(question_onehot), (category_onehot)])
        
    
    # Function to get the corresponding one hot list for a category.
    def get_num_category(self, category):
        return self.categories_num[category]


    # Function to get the corresponding one hot list for a word.
    def get_onehot_word(self, word):
        if word in self.words_onehoted:
            return list(self.words_onehoted[word])
        else:
            return list(self.words_onehoted['<unk>'])

                
    def __len__(self):
        return len(self.batch_data)

    def __getitem__(self, idx):
        return self.batch_data[idx]
    
def pad_collate(batch):
    max_length = max([len(q[0]) for q in batch])

    inputs = torch.FloatTensor([[[0. for _ in range(len(x[0][0]))] for i in range(max_length-len(x[0]))]+x[0] for x in batch])
    outputs = torch.LongTensor([x[1] for x in batch])
    
    return inputs, outputs
    
