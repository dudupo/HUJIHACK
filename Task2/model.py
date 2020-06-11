"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Auther(s):

===================================================
"""
import nltk
import numpy as np
import random
import string

import bs4 as bs
import urllib.request
import re
import sklearn
import pandas as pd

from sklearn.model_selection import train_test_split

words_to_remove_list = ["def", "import", "if", "else", "self", "from", "main", "in", "return", "and", "not", "in"]


def create_samples():
    samples_array = []
    # for index, fileName in enumerate(['building_tool_all_data.txt', 'espnet_all_data.txt', 'horovod_all_data.txt',
    #                                   'jina_all_data.txt', 'PaddleHub_all_data.txt', 'PySolFC_all_data.txt',
    #                                   'pytorch_geometric_all_data.txt']):
    file = open("building_tool_all_data.txt", encoding="utf8")
    dataFile = file.readlines()
    i = 0
    sample = ""
    rand = random.randint(1, 5)
    for line in dataFile:
        sample += line
        if i % rand == rand - 1:
            new_sample = [sample, "building_tool_all_data.txt"]
            samples_array.append(new_sample)
            sample = ""
            rand = random.randint(1, 5)
        i += 1
    file.close()
    x_train, x_test, y_train, y_test = train_test_split(samples_array, 0.2)
    return x_train, x_test, y_train, y_test


def tokenizer():
    wordfreq = {}
    file = open('building_tool_all_data.txt', encoding="utf8")
    dataFile = file.readlines()
    for sentence in dataFile:
        tokens = re.split(' |\n|\t|\(|\)', sentence)
        for token in tokens:
            if token not in words_to_remove_list:
                if token not in wordfreq.keys():
                    wordfreq[token] = 1
                else:
                    wordfreq[token] += 1
    return wordfreq


#
# class GitHubClassifier:
#
#     def classify(self, X):
#         """
#         Receives a list of m unclassified pieces of code, and predicts for each
#         one the Github project it belongs to.
#         :param X: a numpy array of shape (m,) containing the code segments (strings)
#         :return: y_hat - a numpy array of shape (m,) where each entry is a number between 0 and 6
#         0 - building_tool
#         1 - espnet
#         2 - horovod
#         3 - jina
#         4 - PuddleHub
#         5 - PySolFC
#         6 - pytorch_geometric
#         """
#
#     raise NotImplementedError("TODO: Implement this method by 11pm Friday!")

if __name__ == '__main__':
    # a = create_samples()
    # print(a.shape)
    # print(a)
    print(tokenizer())
