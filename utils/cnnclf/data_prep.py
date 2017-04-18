#!/usr/bin/env python3
"""
file: data_prep.py
author: Jiho Noh (jiho@cs.uky.edu)

This script is helper functions for data preparation part in the pipeline. It
examines all the xml documents (PMC document pool from 2015 TREC CDS data), 
extracts corresponding PubMed document id. By the given id numbers, it reaches 
NCBI database and retrieves document components of interest (abstract, title, 
and MeSH terms). Then it determines the document class (diagnosis, test, 
treatment, and others) based on the frequencies of categorized MeSH terms. 
Finally it prepares datasets for training and testing purposes. 
"""

from __future__ import print_function

import os
import sys
import glob
import re
import argparse
from lxml import etree as et
import csv
import time
import numpy as np
import logging
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import ShuffleSplit

DATA_ROOT = '/root/projects/TREC/data/TREC'
CONF = {
    'doc_path': DATA_ROOT + '/2015',
    # 'doc_path': '../toy_data/pmc-text-00/00',
    'entrez_email': "bornoriginal1@gmail.com",
    'file_desc2017': DATA_ROOT + '/MeSH/desc2017.xml',  # MeSH descriptors
    'file_qrels': DATA_ROOT + '/2015/eval2015/qrels-sampleeval-2015.txt',
    'file_topicsA': DATA_ROOT + '/2015/eval2015/topics2015A.xml',
    'file_doc_index': "data/pm.idx",
    'file_ui2tn': 'data/ui2tn.csv',
    'file_dist': 'data/prop_dist.csv',
    'batchsize': 1000,
    'datasize': 200000,
}

mesh_heading = {
    'diagnosis': ['C', 'F', 'G', 'D'],
    'test': ['E01', 'E05'],
    'treatment': ['E02', 'E03', 'E04']
}

mesh_stat = dict()
labels = ['diagnosis', 'test', 'treatment', 'others']
ui2tn = dict()  # dict which maps a mesh ui to tree number
ui2tn_stats = None  # MeSH terms distribution info of each doc classes
wv = None


def init():
    # check if necessary files exist

    # prepare a map file of MeSH unique id ("D000001") to
    #   Tree Number ("D03.633.100.221.173")
    if not os.path.isfile(CONF['file_ui2tn']):
        cnt = 0;
        # open a file to write
        with open(CONF['file_ui2tn'], 'w') as f_out:
            # read descriptor xml file
            with open(CONF['file_desc2017']) as f_desc:
                logging.info("parsing descriptor xml file")
                desc = et.parse(f_desc)
                for elm in desc.iter("DescriptorRecord"):
                    ui = elm.find("DescriptorUI").text
                    tn = '|'.join([x.text for x
                                   in elm.iterfind(".//TreeNumber")])
                    f_out.write("{},\"{}\"\n".format(ui, tn))
                    cnt += 1
                    print("Writing {:>10} [cnt={}]\r".format(ui, cnt), end="")
        logging.info("{} descriptors found and written on {}".
                     format(cnt, CONF['file_ui2tn']))


def run_getMesh(doc_path=CONF['doc_path']):
    """ 
    procedure:
        - It reads index file which contains file path and pubmed id. If the 
            file does not exist, create one.
        - Given pubmed id, it reaches to NCBI and retrieves MeSH terms (unique 
            ids) and abstract.
        - analyze the MeSH terms by its corresponding Tree Numbers, and decide 
            the document class ['diagnosis', 'test', 'treatment', 'others']
        - write the results on a file, which will be used to train a neural 
            network. The format as below:
              [doctype:int] [abstract:string]
    """
    print("=== Retrieving MeSH Codes and Title/Abstracts===")
    doc2retrieve = list()
    subdirs_pattern = re.compile(r".*(pmc-text-\d+/\d+).*")

    if not os.path.isfile(CONF['file_doc_index']):
        logging.info("index file does not exist. Reading files in {}".
                     format(doc_path))
        f_idx = open(CONF['file_doc_index'], 'w')
        for root, subdirs, files in os.walk(doc_path):
            m = subdirs_pattern.match(root)
            if not m:
                continue
            for f in files:
                filename, ext = os.path.splitext(f)
                if ext != '.nxml':
                    continue
                target = os.path.join(root, f)
                try:
                    doc = et.parse(target)
                    id = doc.find("//article-id[@pub-id-type='pmid']")
                    docid = id.text if id is not None else '-1'
                except et.XMLSyntaxError as e:
                    e = sys.exc_info()[0]
                    logging.warning("unable to parse doc and retrieve id [{}]".
                                    format(target))
                    logging.warning(e)
                else:
                    f_idx.write("{}\t{}\t{}\n".format(root, f, docid))
        f_idx.close()

    with open(CONF['file_doc_index']) as f_idx:
        for line in f_idx:
            doc2retrieve.append(line.strip().split('\t'))

    doc_cnt = 0
    for i in range(0, len(doc2retrieve), CONF['batchsize']):
        file_out = "data/doctype_{}.dat".format(i)
        if os.path.isfile(file_out):
            logging.warning("doctype file already exists. Skipping batch #{}".
                            format(i))
            continue
        # batch_path = [x[0] for x in doc2retrieve[i:i+CONF['batchsize']]]
        # batch_file = [x[1] for x in doc2retrieve[i:i+CONF['batchsize']]]
        batch_pmid = [x[2] for x in doc2retrieve[i:i+CONF['batchsize']]
                      if x[2] != '-1']
        meta = get_pubmed_meta(batch_pmid)

        with open(file_out, 'w') as f_out:
            logging.info("writing " + file_out)
            writer = csv.writer(f_out)
            for rec in meta:
                writer.writerow(rec)
        doc_cnt += len(meta)
        if doc_cnt > CONF['datasize']:
            logging.info("datasize {} reached. terminating".
                         format(CONF['datasize']))
            return

        # to avoid excessive queries
        time.sleep(10)  # 20 seconds delay


def run_getDist(y=False):
    """This reads all the datafiles (doctype_#.dat) and analyze proportional 
    distributions of each document types"""
    print("=== Analyzing MeSH Distribution ===")
    if os.path.isfile(CONF['file_dist']):
        if not y:
            ans = input("distribution stat file already exist. overwrite [y/N]? ")
        else:
            ans = 'n'
        if ans.lower() != 'y':
            with open(CONF['file_dist']) as f:
                for line in f:
                    d = line.split(',')
                    mesh_stat[d[0]] = [float(x) for x in d[1:]]
            for x in labels:
                print("[{:>10}]: mean={:.6f}, stdev={:.6f}".
                      format(x, mesh_stat[x][0], mesh_stat[x][1]))
            return

    # read ui2tn, and obtain stats
    logging.info("reading MeSH UI to Tree Number mapping file")
    with open(CONF['file_ui2tn']) as f_ui2tn:
        reader = csv.reader(f_ui2tn)
        for row in reader:
            # use just the first code for now
            ui2tn[row[0]] = row[1].split('|')[0]
    if len(ui2tn) <= 0:
        logging.error('ui2tn mapper does not exist')
        return

    counts = list()
    datafiles = glob.glob('data/doctype_*.dat')
    for file in datafiles:
        logging.info("reading datafile: {}".format(file))

        with open(file) as f:
            reader = csv.reader(f)
            for rec in reader:
                # if any of the fields is empty, pass
                if any(len(x) == 0 for x in rec):
                    continue
                cnt = count_codes(rec[0])
                counts.append(cnt)
    npcounts = np.asarray(counts)
    mean = npcounts.mean(axis=0)
    stdev = npcounts.std(axis=0)
    # report
    with open(CONF['file_dist'], 'w') as f_out:
        logging.info("writing on stat file: {}".format(CONF['file_dist']))
        for i, x in enumerate(labels):
            mesh_stat[x] = [mean[i], stdev[i]]
            f_out.write("{},{:.6f},{:.6f}\n".format(x, mean[i], stdev[i]))
            print("[{:>10}]: mean={:.6f}, stdev={:.6f}".
                  format(x, mean[i], stdev[i]))


def get_pubmed_meta(idlist):
    from Bio import Entrez
    # http://biopython.org/DIST/docs/api/Bio.Entrez-module.html

    ret = list()
    Entrez.email = CONF['entrez_email']
    ent_hd = Entrez.efetch("pubmed", id=','.join(idlist), retmode='xml')
    records = Entrez.read(ent_hd)
    records = records['PubmedArticle']
    for record in records:
        # retrieve MeSH ids
        uids = list()
        try:
            record['MedlineCitation']['MeshHeadingList']
        except KeyError:
            pass
        else:
            for mesh in record['MedlineCitation']['MeshHeadingList']:
                uids.append(mesh['DescriptorName'].attributes['UI'])
        # retrieve Abstract
        snippet = ''
        try:
            title = record['MedlineCitation']['Article']['ArticleTitle']
            abstract = \
                record['MedlineCitation']['Article']['Abstract']['AbstractText']
            snippet = title + ' '
            snippet += ' '.join(abstract)
        except KeyError:
            pass

        ret.append(['|'.join(uids), snippet])
    ent_hd.close()
    return ret


def count_codes(codelist):
    codes = codelist.split('|')
    cnt = [0, 0, 0, 0]
    for code in codes:
        other = True
        try:
            treenumber = ui2tn[code]
        except KeyError:
            print(codes, len(codes))
        for md in mesh_heading['diagnosis']:
            if treenumber.startswith(md):
                cnt[0] += 1
                other = False
        for mt in mesh_heading['test']:
            if treenumber.startswith(mt):
                cnt[1] += 1
                other = False
        for mr in mesh_heading['treatment']:
            if treenumber.startswith(mr):
                cnt[2] += 1
                other = False
        if other:
            cnt[3] += 1
    if sum(cnt) > 0:
        cnt = [i / sum(cnt) for i in cnt]
    return cnt


def run_getDatasets():
    """
    determine doctype by the given mesh ids, tokenize and clean the sentences.
    split dataset into train and test sets.
    """
    test_prop = 0.1
    max_num_sents = 0
    print("=== Cleaning Data ===")
    # check if data files exist
    datafiles = glob.glob('data/doctype_*.dat')
    if len(datafiles) <= 0:
        logging.error("datafiles do not exist")
        return
    # read distribution file
    print("Loading MeSH distribution")
    run_getDist(y=True)

    # read all data files, process and put in an array per sentences
    rs = ShuffleSplit(n_splits=1, test_size=test_prop)
    train_index, test_index = next(rs.split(datafiles))
    logging.info("{} files for train, {} files for test".
                 format(len(train_index), len(test_index)))

    for key, filter in {'train': train_index, 'test': test_index}.items():
        p_data = list()
        logging.info("reading files to create {} dataset".format(key))
        for idx, file in enumerate(datafiles):
            if idx not in filter:
                continue
            with open(file) as f:
                print("{:<30}\r".format(file), end="")
                reader = csv.reader(f)
                for rec in reader:
                    # if any of the fields is empty, pass
                    if any(len(x) == 0 for x in rec):
                        continue
                    cnt = count_codes(rec[0])
                    z_score = [0, 0, 0, 0]
                    for i, x in enumerate(labels):
                        z_score[i] = (cnt[i] - mesh_stat[x][0]) / mesh_stat[x][1]
                    y = z_score.index(max(z_score))

                    # x
                    sents = sent_tokenize(rec[1])
                    if len(sents) > max_num_sents:
                        max_num_sents = len(sents)
                    p_data.append("{} {}".format(y, rec[1]))

        logging.info("{} sentences cleaned".format(len(p_data)))
        logging.info("sanitize completed. writing output files.")
        logging.info("=== Following info may be needed later while building "
                     "NN model ===")
        logging.info("Max number of sentences: {}".format(max_num_sents))

        # just writing on one file
        with open("data/pdata." + key, 'w') as f:
            for line in p_data:
                f.write("{}\n".format(line))


def run_getDatasets_depr():
    """
    determine doctype by the given mesh ids, tokenize and clean the sentences.
    split dataset into train and test sets.
    """
    test_prop = 0.1
    print("=== Cleaning Data ===")
    # check if data files exist
    datafiles = glob.glob('data/doctype_*.dat')
    if len(datafiles) <= 0:
        logging.error("datafiles do not exist")
        return
    # read distribution file
    run_getDist(y=True)

    p_data = list()
    # read all data files, process and put in an array per sentences
    rs = ShuffleSplit(n_splits=1, test_size=test_prop)
    train_index, test_index = next(rs.split(datafiles))
    logging.info("{} files for train, {} files for test".
                 format(len(train_index), len(test_index)))
    logging.info("reading files of train datatset and clean")
    for idx, file in enumerate(datafiles):
        if idx not in train_index:
            continue
        with open(file) as f:
            print("{:<30}\r".format(file), end="")
            reader = csv.reader(f)
            for rec in reader:
                # if any of the fields is empty, pass
                if any(len(x) == 0 for x in rec):
                    continue
                cnt = count_codes(rec[0])
                z_score = [0, 0, 0, 0]
                for i, x in enumerate(labels):
                    z_score[i] = (cnt[i] - mesh_stat[x][0]) / mesh_stat[x][1]
                y = z_score.index(max(z_score))

                # string cleaning
                sentences = [clean_str(s) for s in sent_tokenize(rec[1])]
                for s in sentences:
                    p_data.append("{} {}".format(y, s))

    logging.info("{} sentences cleaned".format(len(p_data)))
    logging.info("sanitize completed. writing output files.")

    # just writing on one file
    with open("data/pdata.train", 'w') as f:
        for line in p_data:
            f.write("{}\n".format(line))

    # test data is not in sentence based. It is tested by the entire set of
    # sentences in order to predict the document type
    p_data_test = list()
    logging.info("reading files of test datatset and clean")
    for idx, file in enumerate(datafiles):
        if idx not in test_index:
            continue
        with open(file) as f:
            print("{:<30}\r".format(file), end="")
            reader = csv.reader(f)
            for rec in reader:
                # if any of the fields is empty, pass
                if any(len(x) == 0 for x in rec):
                    continue
                cnt = count_codes(rec[0])
                z_score = [0, 0, 0, 0]
                for i, x in enumerate(labels):
                    z_score[i] = (cnt[i] - mesh_stat[x][0]) / mesh_stat[x][1]
                y = z_score.index(max(z_score))

                # for test data, strings won't be cleaned until inference proc.
                p_data_test.append("{} {}".format(y, rec[1]))

    logging.info("{} sentences cleaned".format(len(p_data_test)))
    logging.info("sanitize completed. writing output files.")

    # just writing on one file
    with open("data/pdata.test", 'w') as f:
        for line in p_data_test: \
            f.write("{}\n".format(line))


def gen_qtype_score():
    # read pm.idx to read doc files
    doc_list = dict()
    with open(CONF['file_doc_index']) as f_idx:
        for line in f_idx:
            file_path, pmcid, pmid = line.split()
            pmcid = pmcid.split('.')[0]
            doc_list[pmcid] = [file_path, pmid]

    # read qrels file and for each line,
    with open(CONF['file_qrels']) as f_qrels:
        for line in f_qrels:
            qid, _, pmcid, cat, rel = line.split()
    # parse document get title and abstract
    # run tensorflow, restore trained parameters
    # make inference and get softmax cross-entrophy
    # append the qtype_score and write on a new file


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def sent_embedding(file_w2v, sents, seq_len, emb_len=200):
    global wv

    if wv is None:
        from gensim.models import KeyedVectors
        print("- loading pre-trained PMC word2vec file")
        sys.stdout.flush()
        wv = KeyedVectors.load_word2vec_format(file_w2v, binary=True)

    sents = [clean_str(s) for s in sent_tokenize(sents)]
    x = np.zeros((seq_len, emb_len))
    for i, s in enumerate(sents):
        if i >= seq_len:
            continue
        words = s.split()
        num_words = len(words)
        wv_avg = np.zeros(emb_len)
        if num_words > 0:
            for w in words:
                try:
                    wv_avg += wv[w]
                except KeyError:
                    num_words -= 1
                    pass
            wv_avg /= num_words
        x[i, :] = wv_avg
    return x


def load_data_and_labels(datafile, seq_len, emb_len):
    """using pre-trained word2vec embedding, build sentence embeddings by 
    averaging the word vectors which compose the sentence"""
    x_data = list()
    y = list()
    global wv

    if wv is None:
        from gensim.models import KeyedVectors
        print("loading pre-trained PMC word2vec file")
        sys.stdout.flush()
        wv = KeyedVectors.load_word2vec_format('./data/PMC-w2v.bin', binary=True)

    print("reading data file {}".format(datafile))
    with open(datafile, 'r') as f:
        cnt = 1
        for line in f:
            if cnt % 5000 == 0:
                print("{} docs parsed".format(cnt))
                sys.stdout.flush()
            cnt += 1

            # x
            sents = [clean_str(s) for s in sent_tokenize(line[2:])]
            x = np.zeros((seq_len, emb_len))
            for i, s in enumerate(sents):
                words = s.split()
                words_len = len(words)
                wv_avg = np.zeros(emb_len)
                if words_len > 0:
                    for w in words:
                        try:
                            wv_avg += wv[w]
                        except KeyError:
                            words_len -= 1
                            pass
                    if words_len > 0:
                        wv_avg /= words_len
                x[i,:] = wv_avg
            x_data.append(x)
            # y
            label = [0] * 4
            label[int(line[0])] = 1
            y.append(label)
    assert len(x_data) == len(y), "length of x_data is different than the " \
                                  "labels"
    return [np.array(x_data), np.array(y)]


def batch_iter(indices, batch_size, num_epochs):
    """
    returns the shuffled indices of batch_size for train step and k-fold 
    partition range for validation
    """
    num_batches_per_epoch = int((len(indices) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start = batch_num * batch_size
            end = min((batch_num + 1) * batch_size, len(indices))
            yield epoch, indices[start:end]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="run different subtasks",
                        choices=['getMesh', 'getDist', 'getDatasets',
                                 'addQtypeScore', 'test'])
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s:: %(message)s')
    log_hd = logging.getLogger()

    init()
    if args.command == 'getMesh':
        run_getMesh()
    elif args.command == 'getDist':
        run_getDist()
    elif args.command == 'getDatasets':
        run_getDatasets()
    elif args.command == 'addQtypeScore':
        gen_qtype_score()
    elif args.command == 'test':
        pass

