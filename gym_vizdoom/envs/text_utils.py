import numpy as np

import nltk
import string
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle as pkl

import json
import os

import random

class TokenizeStemSentence(object):

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.trans_table = str.maketrans(dict.fromkeys(string.punctuation))
        self.stop_word = nltk.corpus.stopwords.words('english')

    def tokenize_and_stem(self, sentence):
        sentence_wordlist = []
        sentence_wo_punct = str.translate(sentence, self.trans_table)
        words = nltk.tokenize.word_tokenize(sentence_wo_punct)

        for raw_word in words:
            word = raw_word.lower()
            if word not in self.stop_word:
                sentence_wordlist.append(self.stemmer.stem(word))

        return sentence_wordlist

    def __call__(self, sentence):
        return self.tokenize_and_stem(sentence=sentence)


class TextToIds(object):
    def __init__(self, max_sentence_length, path_to_vocab="data/text/vocabulary.pkl", onehot=False):

        self.path_to_vocabulary = path_to_vocab
        vocab_dict = pkl.load(open(self.path_to_vocabulary, 'rb'))
        self.all_words, self.max_sentence_length = vocab_dict['all_words'], vocab_dict['max_sentence_length']

        self.max_sentence_length = max_sentence_length
        self.onehot = onehot

        self.tokenizer = TokenizeStemSentence()
        self.word_to_id = LabelEncoder()

        all_words_encoded = self.word_to_id.fit_transform(np.array(list(self.all_words)))

        self.eos_id = int(self.word_to_id.transform(['eos'])[0])

        self.id_to_one_hot = OneHotEncoder(sparse=False)
        self.id_to_one_hot.fit(all_words_encoded.reshape(len(self.all_words),1))

    def pad_encode(self, sentence):
        sentence = self._sentence_to_matrix(sentence)
        sentence = self._pad(sentence=sentence)
        return sentence

    def _sentence_to_matrix(self, sentence):
        sentence_array = np.array(self.tokenizer(sentence))
        sentence_encoded = self.word_to_id.transform(sentence_array)

        if self.onehot :
            sentence_encoded = self.id_to_one_hot.transform(sentence_encoded.reshape(len(sentence_encoded), 1))

        return sentence_encoded

    def _pad(self, sentence):
        n_padding = self.max_sentence_length - len(sentence)
        if n_padding != 0:
            padding = np.ones(n_padding)*self.eos_id
            sentence = np.concatenate((sentence, padding), axis=0)
        return sentence

class TextObjectiveGenerator(object):

    def __init__(self, env_specific_vocab,
                 path_to_text="gym_vizdoom/envs/data/Basic",
                 sentence_file="sentences.json",
                 mode="simple",
                 onehot=False):
        """
        :param env_specific_vocab: a list of word that are being used in the env calling this Objective Generator.
        :param mode : can be simple, medium, hard
        """

        self.path_to_text = path_to_text

        self.path_to_sentences = os.path.join(self.path_to_text, sentence_file)
        self.path_to_vocabulary = os.path.join(self.path_to_text, 'vocabulary.pkl')

        # can be color, name of object etc ..., will be used to fill template
        self.env_specific_vocab = env_specific_vocab

        self.tokenize_sentence = TokenizeStemSentence()

        # Token like : begin of sentence, end of sentence etc ...
        self.special_tokens = ['eos']

        self.all_sentences_template = {} # keys are : sentence_color, absolute_position_sentence, relative_position_sentence
        self.load_sentences()

        if os.path.exists(self.path_to_vocabulary):
            vocab = pkl.load(open(self.path_to_vocabulary, 'rb'))
            self.all_words = vocab["all_words"]
            self.max_sentence_length = vocab["max_sentence_length"]
        else:
            self.build_vocabulary()

        self.voc_size = len(self.all_words)

        # Regarding difficulty of the task
        self.keys = ["sentence_color", "absolute_position_sentence", "relative_position_sentence"]
        self.mode = mode

        self.text_to_id = TextToIds(max_sentence_length=self.max_sentence_length,
                                    path_to_vocab=self.path_to_vocabulary,
                                    onehot=onehot)

        if onehot:
            self.text_shape = (self.voc_size, self.max_sentence_length)
        else:
            self.text_shape = (self.max_sentence_length)



    def sample(self, color, position, other_color):
        def _choice_color(sentences_template, color, position, other_color):
            random_sentence = random.choice(sentences_template["sentence_color"])
            random_sentence = random_sentence.format(color=color.lower())
            print(random_sentence)
            return random_sentence

        def _choice_abs_position(sentences_template, color, position, other_color):
            random_sentence = random.choice(sentences_template["absolute_position_sentence"][str(position)])
            return random_sentence

        def _choice_rel_position(sentences_template, color, position, other_color):
            assert NotImplementedError("Relative position not implemented yet")

        all_choices_function = [_choice_color, _choice_abs_position, _choice_rel_position]

        if self.mode == "simple" :
            possible_function = [all_choices_function[0]]
        elif self.mode == "medium" :
            possible_function = all_choices_function[:2]
        elif self.mode == "hard":
            possible_function = all_choices_function[:3]
        else:
            assert False, "Wrong difficulty parameters, should be simple, medium, hard. Not '{}'".format(self.mode)

        random_selector = random.randint(0, len(possible_function)-1) # -1 because randint include upper bound
        # call a random sampler based on all objective possibility
        random_sentence = possible_function[random_selector](self.all_sentences_template, color, position, other_color)

        return self.text_to_id.pad_encode(random_sentence)

    def load_sentences(self):

        self.all_sentences_template = json.load(open(self.path_to_sentences))

    def build_vocabulary(self):

        def _recurse_vocab_builder(sentences, all_word_list, maximum_sent_length):
            """
            Build vocabulary from nested dictionnary
            :param sentences: dictionnary to retrieve vocabulary from
            :param all_word_list: should be [] when called for first time
            :return: list of all word in nested dictionnary
            """

            if type(sentences) is list:
                new_words = []
                for sentence in sentences:
                    sentence.replace('{color}', '')
                    temp_words = self.tokenize_sentence(sentence)
                    maximum_sent_length = max(len(temp_words), maximum_sent_length)

                    new_words.extend(temp_words)

            elif type(sentences) is dict:

                new_words = []
                for key, item in sentences.items():
                    more_words, sub_max_length = _recurse_vocab_builder(item, all_word_list, maximum_sent_length=maximum_sent_length)
                    new_words.extend(more_words)
                    maximum_sent_length = max(maximum_sent_length, sub_max_length)

            else:
                assert False, "Should be list or dict, no way around. Is {}".format(type(sentences))

            all_word_list.extend(new_words)

            return all_word_list, maximum_sent_length

        all_words = set([self.tokenize_sentence.stemmer.stem(word.lower()) for word in self.env_specific_vocab])

        # Retrieve vocab from templates and add it to vocabulary
        words_from_sentences, self.max_sentence_length = _recurse_vocab_builder(sentences=self.all_sentences_template,
                                                                                all_word_list=[],
                                                                                maximum_sent_length=0)

        words_from_sentences = set(words_from_sentences)
        all_words = all_words.union(words_from_sentences)
        self.all_words = all_words.union(set(self.special_tokens))

        # Save vocabulary for later usage in model
        print("Saving vocab, vocab size is {}, max length is {}".format(len(self.all_words), self.max_sentence_length))
        pkl.dump({"all_words" : self.all_words, "max_sentence_length": self.max_sentence_length}, open(self.path_to_vocabulary, 'wb'))