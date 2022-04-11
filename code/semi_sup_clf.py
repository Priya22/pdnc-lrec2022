from feature_extractors import *
import os, re, sys, json, csv, string, gzip
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import pickle
import nltk
import random

import joblib
from optparse import OptionParser
import sys
from time import time

import argparse

from itertools import chain
import logging
from tqdm import tqdm

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.manifold import MDS

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--novel', help='Name of the novel', type=str, required=True)
parser.add_argument('--save_path', metavar='FOLDER', required=True)

def get_clf_pipeline():
    clf_pipe = Pipeline([
        ('smote', RandomOverSampler(random_state=42)),
        ('norm', StandardScaler(with_mean=False)),
        ('varianceThresh', VarianceThreshold()),
        ('clf', LogisticRegression())
    ])
    
    return clf_pipe

clf_params = {'clf__C': [0.01, 0.1, 0.5, 1.0]}

def classify_round(traindf, testdf, train_y, test_y):
    curCorp = CorpusFeatureBuilder(traindf, testdf)
    
    train_feats = curCorp.fit_features()
    test_feats = curCorp.transform_features()

    clf_pipe = get_clf_pipeline()
    
    grid_search = GridSearchCV(clf_pipe, clf_params, n_jobs=-1, verbose=1, cv=3, scoring='f1_micro')
    
    grid_search.fit(train_feats, train_y)
    y_pred = grid_search.predict(test_feats)
#     print(y_pred[:2])
    print("Test performance")
    print(classification_report(test_y,y_pred))
    print(grid_search.best_params_)
    
    f1score_test = f1_score(test_y, y_pred, average='micro')
    
    cf1 = confusion_matrix(test_y, y_pred)
    lbls = sorted(list(set(list(test_y) + list(y_pred))))
    # cfd1 = ConfusionMatrixDisplay(cf1, display_labels=lbls)
    # cfd1.plot(xticks_rotation=90)
    # plt.show()
#     best_clf = grid_search.best_estimator_.named_steps['clf']
    
    #all prediction
    alldf = pd.concat([traindf, testdf])
    ally = list(train_y) + list(test_y)
    
    allfeats = curCorp.transform_features(alldf)
    allpred = grid_search.predict(allfeats)
    
    print("All eval performance")
    print(classification_report(ally,allpred))
    
#     all_pred_names = [best_clf.classes_[i] for i in allpred]
    
    pred_probs = grid_search.predict_proba(allfeats)

    max_pred_probs = [max(v) for x in pred_probs]
    
    f1score = f1_score(ally, allpred, average='micro')
    
    acc = accuracy_score(ally, allpred)
    
    #threshold method #1: mean of correct train predictions
    correct_probs = []
    wrong_probs = []

    breakInd = len(train_y)
    for yt, yp, ypp in zip(ally[:breakInd], allpred[:breakInd], pred_probs[:breakInd]):
        if yt == yp:
            correct_probs.append(max(ypp))
        else:
            wrong_probs.append(max(ypp))    

    thresh = np.mean(correct_probs)
    print("Correct/wrong predictions: ", len(correct_probs), len(wrong_probs))
    print("Correct/wrong, median probs: ", np.median(correct_probs), np.median(wrong_probs))
    print("Correct/wrong, mean probs: ", np.mean(correct_probs), np.mean(wrong_probs))
    
    add_inds = [i for i,v in enumerate(pred_probs) if max(v)>=thresh]
    notaddinds = [i for i,v in enumerate(pred_probs) if max(v)<thresh]

    #alternative ways of setting threholds:
    
    # (2) set threshold to median 
    # thresh = np.median(correct_probs)

    # (3) mean of train and test predictions (requires access to test labels; use only for comparision)
    # correct_probs = []
    # wrong_probs = []
    # for yt, yp, ypp in zip(ally, allpred, pred_probs):
    #     if yt == yp:
    #         correct_probs.append(max(ypp))
    #     else:
    #         wrong_probs.append(max(ypp))
    #thresh = np.mean(correct_probs)

    # (4) using gold labels, add only correct predictions to train set (requires access to test labels; use only for comparision)
    # add_inds = []
    # notaddinds = []

    # ind = 0
    # for yt, yp, ypp in zip(ally, allpred, pred_probs):
    #     if yt == yp:
    #        
    #         add_inds.append(ind)
    #     else:
    #         
    #         notaddinds.append(ind)
    #     ind += 1
    
    #create new dataframes
    newtrainrows = [alldf.iloc[i].tolist() for i in add_inds]
    newtestrows = [alldf.iloc[i].tolist() for i in notaddinds]
    
    newtrainy = [ally[i] for i in add_inds]
    newtesty = [ally[i] for i in notaddinds]
    
    newtraindf = pd.DataFrame(newtrainrows, columns=traindf.columns)
    newtestdf = pd.DataFrame(newtestrows, columns=testdf.columns)
    
    return grid_search, y_pred, max_pred_probs, thresh, newtraindf, newtestdf, newtrainy, newtesty, f1score_test, f1score, acc

def rearrange_dfs(traindf, testdf, train_y, test_y):
    traincounter = Counter(train_y)
    testcounter = Counter(test_y)
    
    add_train_inds = []
    
    for s in testcounter:
        if s not in traincounter or traincounter[s] < 5:
            diff = 5 - traincounter[s]
            sinds = [ind for ind, val in enumerate(test_y) if val==s]
            reps = random.sample(sinds, diff)
            add_train_inds.extend(reps)
    
    newtrainrows = traindf.values.tolist()
    newtestrows = []
    
    newtrainy = train_y[:]
    newtesty = []
    
    for ai in add_train_inds:
        newtrainrows.append(testdf.iloc[ai].tolist())
        newtrainy.append(test_y[ai])
    
    newtestrows = [testdf.iloc[i] for i in range(len(test_y)) if i not in add_train_inds]
    newtesty = [test_y[i] for i in range(len(test_y)) if i not in add_train_inds]
    
    newtraindf = pd.DataFrame(newtrainrows, columns=traindf.columns)
    newtestdf = pd.DataFrame(newtestrows, columns=testdf.columns)
    
    return newtraindf, newtestdf, newtrainy, newtesty

def iter_flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def add_men_str(row):
    ments = eval(row['menEnts'])
    ments = [x for sublist in ments for x in sublist]
    mstr = '_'.join(ments)
    row['menStr'] = mstr
    return row

ROOT = '/h/vkpriya/pdnc-lrec/'
DATA_ROOT = os.path.join(ROOT, 'data')
CODE_ROOT = os.path.join(ROOT, 'code')

def run_for(novel, save_path):  

    novel2evalsize = {}
    novel2f1scores_eval = {}
    novel2f1sores_test = {}
    novel2accuracies = {}
    novel2finaldata = {}
    novel2grid = {}

    print(novel)
    df = pd.read_csv(os.path.join(DATA_ROOT, novel, 'quotations.csv'), index_col=0, keep_default_na=False, dtype=str)

    # qindLists = {}

    df = df.apply(add_men_str, axis=1)

    speaker_counter = Counter(df['speaker'])

    remove_speakers = [x for x in speaker_counter if len(x) == 0 or x[0]=='_' or speaker_counter[x]<10]

    print(remove_speakers)

    df = df[~df['speaker'].isin(remove_speakers)]
    df = df[df['text']!='']

    print(Counter(df['qType']))
    novel2evalsize[novel] = len(df)

    novel2finaldata = {}
    
    traindf_init = df[df['qType']=='Explicit']
    testdf_init = df[df['qType']!='Explicit']
    trainy_init = traindf_init['speaker'].tolist()
    testy_init = testdf_init['speaker'].tolist()

    print("Loaded init train/test: ", len(traindf_init), len(testdf_init), len(trainy_init), len(testy_init))

    train_df, test_df, train_y, test_y = rearrange_dfs(traindf_init, testdf_init, trainy_init, testy_init)

    print("Round 0 train/test:", len(train_df), len(test_df), len(train_y), len(test_y))

    dropcount = 0
    zerocount = 0
    f1scores_test = []
    f1scores_eval = []
    accuracies = []

    for round_num in range(20):
        print("Round: ", round_num)
        print("Start train/test: ", len(train_df), len(test_df), len(train_y), len(test_y))

        #RETURNS: grid_search, y_pred, max_pred_probs, thresh, newtraindf, newtestdf, newtrainy, newtesty, f1score_test, f1score, acc

        grid_search, y_pred, all_pred_probs, thresh, \
            ntrain_df, ntest_df, ny_train, ny_test, f1score_test, f1score_eval, acc = classify_round(train_df, test_df, train_y, test_y)

        print("Test F1/ Eval F1: ", f1score_test, f1score_eval)

        print("End train/test: ", len(ntrain_df), len(ntest_df), len(ny_train), len(ny_test))

        novel2finaldata[round_num] = {
            'traindf': train_df,
            'testdf': test_df,
            'trainy': train_y,
            'testy': test_y,
            'testpredy': y_pred,
            'allpredprobs': all_pred_probs,
            'thresh': thresh,
            'testf1': f1score_test,
            'evalf1': f1score_eval,
            'evalacc': acc,
            'ntraindf': ntrain_df,
            'ntestdf': ntest_df,
            'ntrainy': ny_train,
            'ntesty': ny_test
        }

        if len(f1scores_eval)>0 and f1score_eval<f1scores_eval[-1]:
            dropcount += 1

        if f1score_test == 0.:
            zerocount += 1

        train_df, test_df, train_y, test_y = rearrange_dfs(ntrain_df, ntest_df, ny_train, ny_test)
        f1scores_eval.append(f1score_eval)
        f1scores_test.append(f1score_test)
        accuracies.append(acc)

        print("--"*10)

        # qindLists[round_num + 1] = {'train': train_df['QuoteID'].tolist(), 'test': test_df['QuoteID'].tolist()}

        if dropcount ==5:
            print("Max dropcount: exiting")
            break

        if zerocount ==3:
            print("Max zerocount: exiting")
            break


    #save 
    save_dir = os.path.join(save_path, novel)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    pickle.dump(novel2finaldata, open(os.path.join(save_dir, 'roundData.pkl'), 'wb'))
    

    for novel in novel2accuracies:
        print(novel, round(max(novel2accuracies[novel]), 3), np.argmax(novel2accuracies[novel]))

if __name__=='__main__':
    args = parser.parse_args()
    novel = args.novel
    save_path = args.save_path
    run_for(novel, save_path)