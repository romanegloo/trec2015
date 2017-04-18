#!/usr/bin/env python3
"""
examples of running on docker container

- parsing topics (queries)
    docker exec jiho-trec02 bash -c './runner.py parseQuery'

"""
from __future__ import print_function

import os
import copy
import shutil
import glob
from lxml import etree as et
from nltk.corpus import stopwords as stop
from nltk.stem.porter import *
import string
import argparse
import pickle
import logging
from collections import Counter
import subprocess
import datetime

dt = datetime.datetime.now().strftime("%m%d-%I%M")
data_path = '../data/TREC/2015/'
CONF = {
    # original document pool
    'dir_collection': data_path,
    # processed document pool
    'dir_pool': data_path + 'pool',
    # log file of runner script
    'file_logfile': 'var/log_runner_' + dt + '.log',
    # original queries
    'file_topicA': data_path + 'eval2015/topics2015A.xml',
    'file_topicB': data_path + '/eval2015/topics2015B.xml',
    'file_topicD': data_path + '/eval2015/topics_dev.xml',  # for dev
    # preprocessed queries
    'file_topic_p': data_path + '/eval2015/topics_pp.xml',
    # relevance judgement file
    'file_qrel': data_path + '/eval2015/qrels-treceval-2015.txt',
    'file_qrel_inf': data_path + '/eval2015/qrels-sampleval-2015.txt',
    # n-gram distribution file
    'dir_ngram': '../data/n-gram',
    'file_udist_gb': '../data/n-gram/googlebooks/1gram/cnt_gb.obj',
    'file_udist_pmc': '../data/n-gram/pmc/1gram/cnt_pmc.obj',
    # stopwords list
    'file_stopwords': 'share/stopword-list.txt'
}

CONF_QUERY = {  # default configuration for parsing queries
    'mode': 'single',  # sgml mode can be implemented if necessary
    'run': 'Aa',  # A|B: type A, a|m: automatic or manual
    'enable-weighting': True,  # enable n-gram weighting
    'enable-cui-expansion': False,  # query expansion with CUI preferred names
    'weight-factor': 3.5,  # emphasize terms that occur more often
    # in pmc documents
    'noweight_common_perc': 0,  # don't apply weights on common
    # terms
    'pipeline': ['stopwords'],
    # {stopwords, stemmer}
    'tags': ['description'],
    # {summary, description, diagnosis, etc.}
    'punctuation': '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~',
    # punctuation to be removed from original text
    # query expansion
    'expansion.terms': 20,  # default 10
    'expansion.documents': 5,  # default 3
}

terrier_path = '/opt/terrier/terrier-core-4.2/'
TERRIER = {  # information relevant to Terrier system
    'dir_var': terrier_path + 'var',
    'bin_setup': terrier_path + 'bin/trec_setup.sh',
    'bin_run': terrier_path + 'bin/trec_terrier.sh',
    'bin_eval': terrier_path + 'bin/trec_eval.sh',
    'matching_model': 'BM25',
    'enable_qe': True,  # enable default system query expansion
}

"""
available matching models:
(from http://terrier.org/docs/v4.2/configure_retrieval.html)

BB2, BM25, DFR_BM25, DLH, DLH13, DPH, DFRee, Hiemstra_LM, IFB2, In_expB2,
In_expC2, InL2, LemurTF_IDF, LGD, PL2, TF_IDF, DFRWeightingModel
 """

# these might be used multiple times while testing
p_gb = dict()
p_pmc = dict()


def _xml2simple(file, pool_dir):
    """reads nxml file, extracts selected nodes, and re-formats the document
    into a simpler xml file for IR indexing"""

    xpath_docid = "//article-id[@pub-id-type='pmc']"
    xpath_terms = {
        'TITLE': '//front//article-title',
        'ABSTRACT': '//abstract',
        'KEYWORDS': '//kwd-group',
        'BODY': '//body',
        'REF': '//back//article-title'
    }

    try:
        doc_from = et.parse(file)
    except et.XMLSyntaxError as e:
        logging.warning("unable to parse document file [{}]".format(file))
        return 1

    # create a new xml tree for output
    doc = et.Element("DOC")

    # read nodes and attach to the new tree
    for tag, path in xpath_terms.items():
        text = ''
        elms = doc_from.xpath(path)
        for elm in elms:
            text += "".join([x for x in elm.itertext()])
        ch = et.SubElement(doc, tag)
        ch.text = text
    docid = doc_from.xpath(xpath_docid)[0]
    docid.tag = 'DOCNO'
    del docid.attrib['pub-id-type']
    doc.append(docid)

    with open(os.path.join(pool_dir, docid.text + '.xml'), 'wb') as out:
        out.write(et.tostring(doc, pretty_print=True))
    return 0


def _run_parse_doc():
    """reads the given pmc documents which are in nxml format. Parser outputs
    an xml file containing the content of selected set of nodes such as
    [docid], [body], [additional info]"""

    logging.info("start/ parsing documents")
    # subdirs_pattern = re.compile(r".*(pmc-text-\d+/\d+).*")
    subdirs_pattern = re.compile(r".*(pmc-text-\d+/\d+).*")
    document_ext = '.nxml'

    counter = [0, 0, 0]  # [total, failed, skipped]
    debug_num_files = 10000  # in debug mode, just parse 10 documents

    # if pool directory is not empty, ask if the user wants to delete it and
    # re-create it for fresh start. Otherwise, continue parsing from where it is
    if os.path.exists(CONF['dir_pool']):
        ans = input("pool directory exists. {}\ndelete and start fresh [Y/n]?".
                    format(CONF['dir_pool']))
        if ans.lower().startswith('n'):
            pass  # continue
        else:
            logging.info("removing pool directory: {}".
                         format(CONF['dir_pool']))
            shutil.rmtree(CONF['dir_pool'])
    else:
        os.makedirs(CONF['dir_pool'])

    # traverse document directory
    for root, subdirs, files in os.walk(CONF['dir_collection']):
        m = subdirs_pattern.match(root)
        if not m:
            continue
        # check if pool subdir exists
        pool_sub = os.path.join(CONF['dir_pool'], m.group(1))
        if not os.path.exists(pool_sub):
            os.makedirs(pool_sub)

        for f in files:
            filename, ext = os.path.splitext(f)
            if ext != document_ext:
                continue
            counter[0] += 1
            target = os.path.join(CONF['dir_pool'], f)
            if os.path.isfile(target):
                # target file exists and skipping
                counter[2] += 1
                print("skipped {} [total {cnt[0]}, failed {cnt[1]} "
                      "skipped {cnt[2]}]\r".format(f, cnt=counter), end="")
                continue
            if _xml2simple(os.path.join(root, f), pool_sub):
                counter[1] += 1
                logging.warning("failed {} [total {cnt[0]}, failed {cnt[1]} "
                                "skipped {cnt[2]}]".format(f, cnt=counter))
            else:
                if args.debug:
                    if debug_num_files > 0:
                        debug_num_files -= 1
                    else:
                        break
                print("parsed {} [total {cnt[0]}, failed {cnt[1]} "
                      "skipped {cnt[2]}]\r".format(f, cnt=counter), end="")
        else:
            continue
        break  # break if the above for loop interrupted
    print()
    logging.info("completed parsing document [total {cnt[0]}, failed {cnt[1]} "
                 "skipped {cnt[2]}]".format(cnt=counter))


def _run_parse_query(conf=CONF_QUERY):
    """preprocess queries (topics) with the given configuration"""
    logging.info("start/ parsing queries")

    if conf['enable-cui-expansion']:
        from pymetamap import MetaMap
        mm = MetaMap.get_instance('/opt/public_mm/bin/metamap')

    stopwords = None
    global p_gb, p_pmc

    # read topics
    try:
        tp_tree = et.parse(CONF['file_topic' + conf['run'][0]])
    except et.XMLSyntaxError as e:
        logging.error("couldn't parse topic file")
        log = e.error_log.filter_from_level(et.ErrorLevels.FATAL)
        logging.error(log)
        raise SystemExit()

    # topics = [topic1]
    #
    # concepts, error = mm.extract_concepts(topics)
    # additional_terms = list()
    # for concept in concepts:
    #     pn = concept.preferred_name
    #     if pn.lower() not in topic1.lower():
    #         additional_terms.append(pn)
    #
    # print(topic1 + ' '.join(additional_terms))


    # read stopwords
    if 'stopwords' in conf['pipeline']:
        if os.path.isfile(CONF['file_stopwords']):
            with open(CONF['file_stopwords']) as f:
                stopwords = f.read().splitlines()
        else:
            stopwords = set(stop.words('english'))
        logging.debug("stopword list [len: {}] created".format(len(stopwords)))

    f_out = open(CONF['file_topic_p'], 'w')
    for topic in tp_tree.findall('./topic'):
        query_str = ''
        for tag in conf['tags']:
            query_str += topic.find(tag).text.lower()

        # cui query expansion
        if conf['enable-cui-expansion']:
            concepts, error = mm.extract_concepts([query_str])
            additional_terms = list()
            for concept in concepts:
                if concept.__class__.__name__ == "ConceptMMI":
                    pn = concept.preferred_name
                else:
                    pn = concept.long_form
                if pn.lower() not in query_str:
                    additional_terms.append(pn)
            query_str += ' ' + ' '.join(additional_terms)

        # remove punctuations
        query_str = ''.join(ch for ch in query_str
                            if ch not in conf['punctuation'])
        q = query_str.split()

        # remove stopwords
        if 'stopwords' in conf['pipeline']:
            len_q = len(q)
            q = [t for t in q if t not in stopwords]
            logging.debug(str(len_q - len(q)) + " stopwords removed")


        if conf['enable-weighting']:
            ## compute ngram weights
            q_weights = [0] * len(q)
            unigram_dist_gb = None
            unigram_dist_pmc = None
            # check if distribution files exist. If not, create them.
            if not os.path.isfile(CONF['file_udist_gb']):
                logging.warning("unigram distribution [gb] not found.")
                unigram_dist_gb = _create_ngram_dist('gb')
            if not os.path.isfile(CONF['file_udist_pmc']):
                logging.warning("unigram distribution [pmc] not found.")
                unigram_dist_pmc = _create_ngram_dist('pmc')

            if len(p_gb) == 0:
                logging.info("loading 1-gram distribution file (gb)")
                unigram_dist_gb = \
                    pickle.load(open(CONF['file_udist_gb'], 'rb'),
                                encoding='utf-8')
                total_freq_gb = sum(unigram_dist_gb.values())  # total frequency
                acc_gb = 0
                for t, v in unigram_dist_gb.most_common():  # compute proportion
                    acc_gb += v
                    p_gb[t] = 1.0 * acc_gb / total_freq_gb
                del unigram_dist_gb

            if len(p_pmc) == 0:
                logging.info("loading unigram distribution file (pmc)")
                unigram_dist_pmc = \
                    pickle.load(open(CONF['file_udist_pmc'], 'rb'),
                                encoding='utf-8')
                total_freq_pmc = sum(unigram_dist_pmc.values())
                acc_pmc = 0
                for t, v in unigram_dist_pmc.most_common():
                    acc_pmc += v
                    p_pmc[t] = 1.0 * acc_pmc / total_freq_pmc
                del unigram_dist_pmc

            for i, w in enumerate(q):
                if w not in p_gb or w not in p_pmc:
                    continue
                if p_gb[w] < conf['noweight_common_perc']:
                    continue
                q_weights[i] = p_gb[w] - p_pmc[w]

            # combine weights, while ignoring negative values
            for i, w in enumerate(q):
                if q_weights[i] > 0:
                    q[i] += '^' + str(1 + conf['weight-factor'] * q_weights[i])

        # add topic id number
        tp_id = topic.attrib['number']
        q.insert(0, tp_id)

        # write out
        f_out.write(' '.join(q) + "\n")
        logging.info('topic {} processed'.format(tp_id))


def _create_ngram_dist(d):
    """create unigram distribution of two corpora: Google Books and PMC"""

    gb_path = os.path.join(CONF['dir_ngram'], 'googlebooks/1gram')
    pmc_path = os.path.join(CONF['dir_ngram'], 'pmc/1gram')
    cnt_gb_file = os.path.join(CONF['dir_gram'], 'googlebooks/1gram/cnt_gb.obj')
    cnt_pmc_file = os.path.join(CONF['dir_gram'], 'pmc/1gram/cnt_pmc.obj')

    # selected years of documents
    years = list(range(1970, 2010, 5))
    years.append(2012)  # add the most complete data from googlebooks
    ignore_pattern = '0123456789' + CONF_QUERY['punctuation']

    if d == 'gb':  # read googlebooks files
        logging.info("creating google books n-gram file")
        files = list()
        cnt_gb = Counter()
        for ch in string.ascii_lowercase:
            f = 'googlebooks-eng-all-1gram-20120701-' + ch
            if not os.path.isfile(os.path.join(gb_path, f)):
                logging.warning("file not found:", f)
                continue
            with open(os.path.join(gb_path, f)) as f_in:
                for idx, l in enumerate(f_in):
                    stat = l.split('\t')
                    if int(stat[1]) not in years:
                        continue
                    if len(stat) != 4:
                        logging.warning("incorrect format:", l)
                        continue
                    term = stat[0].split('_')[0]
                    cnt_gb[term] += int(stat[2])
                    if idx % 10000 == 0:
                        print("reading {} lines {}\r".format(f, idx), end='')
            logging.info("reading {} lines from {} completed".format(idx, f))
        # store counter
        pickle.dump(cnt_gb, open(cnt_gb_file, 'wb'))
        logging.info("gb n-gram distribution object created and stored")
        return cnt_gb

    if d == 'pmc':  # read pmc files
        logging.info("creating pmc n-gram file")
        files = list()
        cnt_pmc = Counter()
        for f in os.listdir(pmc_path):
            grp = re.findall(r'(\d{4})', f)
            if f.endswith(".tsv") and int(
                    grp[0]) in years:  # tab separated values
                files.append(f)
        # read lines
        for f in files:
            with open(os.path.join(pmc_path, f)) as f_in:
                for idx, l in enumerate(f_in):
                    # ignore some patterns to be consistent
                    # - starts with digits, punctuations,
                    stat = l.split('\t')
                    if len(stat) != 4:
                        logging.warning('incorrect format: ' + l)
                        continue
                    if stat[0][0] in ignore_pattern:
                        continue
                    cnt_pmc[stat[0]] += int(stat[2])
                    if idx % 10000 == 0:
                        print("reading {} lines {}\r".format(f, idx), end='')
            logging.info("reading {} lines from {} completed".format(idx, f))
        # store counter
        pickle.dump(cnt_pmc, open(cnt_pmc_file, 'wb'))
        logging.info("pmc n-gram distribution object created and stored")
        return cnt_pmc


def _run_terrier_retrieve():
    """run Terrier retrieval mode (-r)"""
    logging.info("start/ terrier retrieval")
    cmd = [TERRIER['bin_run'], '-r',
           '-Dtrec.model=' + TERRIER['matching_model']]

    if TERRIER['enable_qe']:
        cmd.insert(2, '-q')
        cmd.append('-Dexpansion.terms=' + str(CONF_QUERY['expansion.terms']))
        cmd.append('-Dexpansion.documents=' +
                   str(CONF_QUERY['expansion.documents']))

    run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for l in run.stdout.decode('utf-8').splitlines():
        logging.debug(l)

    return run.check_returncode()


def _run_terrier_evaluate(interactive=False,
                          qrel=CONF['file_qrel'], qrel_s=CONF['file_qrel_inf']):
    """run Terrier retrieval mode (-e)"""
    logging.info("start/ terrier evaluation")
    dir_results = os.path.join(TERRIER['dir_var'], 'results')

    resid = None
    if interactive:
        # read files in var/results and user select which one to evaluate
        results = list()
        for f in os.listdir(dir_results):
            filename, ext = os.path.splitext(f)
            if ext == '.res':
                results.append(f)
        for i, f in enumerate(results):
            print("[{}] {}".format(i, f))
        ans = input("select the number of result file to evaluate "
                    "[0-{}]:".format(len(results)))
        resid = results[int(ans)]
    else:  # else get the most recent one
        recent = max(glob.iglob(dir_results + '/*.res.settings'),
                     key=os.path.getctime)
        resid = recent[:-9].split('/')[-1]

    logging.info("evaluating " + resid)
    # use trec_eval directly
    path_res = os.path.join('terrier/var/results/', resid)
    cmd = [TERRIER['bin_eval'], qrel, path_res]
    run1 = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    for l in run1.stdout.decode('utf-8').splitlines():
        logging.info(l)
    cmd = ['./sample_eval.pl', qrel_s, os.path.join(dir_results, resid)]
    run2 = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    metrics = ['infAP', 'infNDCG']
    for l in run2.stdout.decode('utf-8').splitlines():
        if any(w in l for w in metrics):
            logging.info(l)
    return run1.check_returncode() and run2.check_returncode()


def _run_terrier_setup():
    logging.info("start/ terrier setup")
    cmd = [TERRIER['bin_setup'], CONF['dir_pool']]
    run = subprocess.run(cmd)
    return run.check_returncode()


def _run_terrier_index():
    logging.info("start/ terrier indexing")
    # remove pre-existing index files
    dir_index = os.path.join(TERRIER['dir_var'], 'index')
    indexfiles = os.listdir(dir_index)
    for f in indexfiles:
        os.remove(os.path.join(dir_index, f))
    logging.info("removing {} files in index directory".format(len(indexfiles)))

    # copy terrier.properties
    run = subprocess.run(['cp', 'share/terrier.properties',
                          'terrier/etc/terrier.properties'])
    if run.check_returncode():
        logging.info("copied terrier.properties template")

    # run terrier indexer
    run = subprocess.run([TERRIER['bin_run'], '-i'])
    return run.check_returncode()


def _run_test():
    """run retrieval and evaluation in batch with different parameter
    settings."""
    logging.info("start/ testing")

    # testing with different weight factors and common words percentage
    cfg = copy.deepcopy(CONF_QUERY)

    wf = [3]
    nw_cm = [0]

    for w in wf:
        for p in nw_cm:
            logging.info('=' * 70)
            logging.info("testing with weight-factor: {}, nw_comm: {}"
                         .format(w, p))
            cfg['weight-factor'] = w
            cfg['noweight_common_perc'] = p
            _run_parse_query(cfg)
            _run_terrier_retrieve()
            _run_terrier_evaluate()


def _run_test_27():
    """testing on one topic for query expansion"""
    logging.info("start/ testing 27")

    # parse query
    cfg = copy.deepcopy(CONF_QUERY)
    # cfg['run'] = 'Da'
    _run_parse_query(cfg)
    _run_terrier_retrieve()
    _run_terrier_evaluate()


def _run_test_28():
    """run and evaluate with document class scores"""
    score_file = TERRIER['dir_var'] + '/results/scores.pkl'
    res_file = TERRIER['dir_var'] + \
               '/results/BM25b0.75_Bo1bfree_d_5_t_20_0.res'
    if os.path.exists(score_file):
        with open(score_file, 'rb') as f:
            scores = pickle.load(f)
    else:
        pmidx_file = '/root/projects/TREC/utils/cnnclf/data/pm.idx'

        if not os.path.exists(res_file):
            raise SystemError("res file does not eixsts. [{}]".format(res_file))
        if not os.path.exists(pmidx_file):
            raise SystemError("pm index file does not eixsts. [{}]".
                              format(pmidx_file))

        # construct scores
        scores = dict()
        # scores = {
        #   '1-2799065': {
        #       'q_no':         '1',
        #       'pmcid':        '2799065',
        #       'pmid':         '17242961',
        #       'raw_score':    110.3309697896708,
        #       'file_path':    '/root/projects/TREC/data/TREC/2015/pmc-text-01/36',
        #       'snippet':      'concatenated text of the title and abstract'
        #       'qt_loss':      0.781734314
        #   }

        # read res file
        pmcids = []
        with open(res_file) as f:
            for line in f:
                t = line.split()
                # if t[0] not in ['1', '2']:
                #     continue
                scores[t[0]+'-'+t[2]] = {
                    'q_no': t[0],
                    'pmcid': t[2],
                    'raw_score': float(t[4])
                }
                pmcids.append(t[2])

        # load file_path
        paths = {}
        with open(pmidx_file) as f:
            for line in f:
                t = line.split()
                id = t[1].split('.')[0]
                paths[id] = t
        logging.info("reading and parsing document")
        cnt = 0
        for k, res in scores.items():  # length: 30000
            id = res['pmcid']
            print("reading {} [{} / {}]\r".format(id, cnt, len(scores)), end='')
            cnt += 1
            # fill in file_path and pmid
            res['file_path'] = paths[id][0]
            res['pmid'] = paths[id][2]
            # read doc file and fill in snippet
            file_doc = os.path.join(res['file_path'], paths[id][1])
            if os.path.exists(file_doc):
                try:
                    doc_xml = et.parse(file_doc)
                except et.XMLSyntaxError as e:
                    logging.warning("unable to parse document file [{}]".
                                    format(file_doc))
                xpath_terms = {
                    'TITLE': '//front//article-title',
                    'ABSTRACT': '//abstract',
                    'KEYWORDS': '//kwd-group'
                }
                snippet = ''
                for tag, path in xpath_terms.items():
                    elms = doc_xml.xpath(path)
                    for e in elms:
                        snippet += ''.join([x for x in e.itertext()])
                res['snippet'] = snippet
        del paths

        # restore tensorflow model
        import tensorflow as tf
        from cnnclf.model import MeshCNN
        import cnnclf.data_prep as dp
        # CNN setup
        sequence_len = 58  # max number of sentences in a document
        embedding_dim = 200  # word embedding size (using pre-trained word2vec)
        filter_sizes = "3,4,5"        # comma-separated filter sizes
        num_filters = 128             # number of filters per filter size
        l2_reg_lambda = 0.0           # l2 regularization lambda (optional)
        model_path = './cnnclf/runs/1491937785/checkpoints/model-1844000'
        file_w2v = './cnnclf/data/PMC-w2v.bin'

        with tf.Graph().as_default():
            # define a session
            session_conf = tf.ConfigProto(allow_soft_placement=True)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # instantiate MeshCNN model
                cnn = MeshCNN(
                    sequence_length=sequence_len,
                    num_classes=4,
                    embedding_size=embedding_dim,
                    filter_sizes=list(map(int, filter_sizes.split(','))),
                    num_filters=num_filters,
                    l2_reg_lambda=l2_reg_lambda
                )
                cnn.inference()
                cnn.loss_accuracy()

                # restore session with checkpoint data
                if not os.path.exists(model_path + '.meta'):
                    SystemError("checkpoint path does not exist [{}]".
                                format(model_path))

                print("=== restoring a model ===")
                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(sess, model_path)
                print("model restored [{}]".format(model_path))

                # for each document run classifier and get doc type scores
                for k, res in scores.items():
                    emb = dp.sent_embedding(file_w2v, res['snippet'],
                                            sequence_len, embedding_dim)

                    if int(res['q_no']) < 11:
                        y = [1, 0, 0, 0]
                    elif int(res['q_no']) < 21:
                        y = [0, 1, 0, 0]
                    elif int(res['q_no']) < 31:
                        y = [0, 0, 1, 0]
                    else:  # this doesn't happen
                        y = [0, 0, 0, 0]

                    score, loss = sess.run([cnn.scores, cnn.loss],
                                     {cnn.input_x: [emb],
                                      cnn.input_y: [y],
                                      cnn.dropout_keep_prob: 1.0})
                    res['qt_loss'] = loss
        with open(score_file, 'wb') as f:
            pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)

    # write the results
    weights = [0, 0.05, 1, 2]
    with open(res_file) as f:
        for w in weights:
            f.seek(0)
            new_rank = [[] for x in range(30)]  # 30 topics
            print(w)
            for line in f:
                t = line.split()
                # if t[0] not in ['1', '2']:
                #     continue
                # 1 Q0 2799065 0 110.3309697896708 In_expC2c1.0_Bo1bfree_d_10_t_50
                # compute new score
                try:
                    new_score = float(t[4]) + \
                                w / float(scores[t[0]+'-'+t[2]]['qt_loss'])
                except ZeroDivisionError:
                    new_score = float(t[4])

                new_line = (t[2], new_score, t[5])
                new_rank[int(t[0])-1].append(list(new_line))

            # rerank
            for i, q in enumerate(new_rank):
                new_rank[i] = sorted(q, key=lambda r: r[1], reverse=True)

            # write a new res file
            newres = TERRIER['dir_var'] + '/results/w' + str(w) + '.res'
            with open(newres, 'w') as f_nr:
                for qi, q in enumerate(new_rank):
                    for li, l in enumerate(q):
                        f_nr.write("{} Q0 {} {} {} {}\n".\
                                   format(qi+1, l[0], li, l[1], l[2]))
            # run evaluation
            print("evaluating with new res file")
            cmd = ['./sample_eval.pl', CONF['file_qrel_inf'], newres]
            run = subprocess.run(cmd,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL)
            metrics = ['infAP', 'infNDCG']
            for l in run.stdout.decode('utf-8').splitlines():
                if any(w in l for w in metrics):
                    logging.info(l)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="run different subtasks",
                        choices=['parseDoc', 'parseQuery', 'retrieve',
                                 'evaluate', 'test', 'terrier_setup',
                                 'terrier_index'])
    parser.add_argument('-d', '--debug', dest='debug',
                        help="enable debug mode", action='store_true')
    args = parser.parse_args()

    logging.basicConfig(filename=CONF['file_logfile'], filemode='w',
                        level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s:: %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s:: %(message)s"))
    logging.getLogger('').addHandler(console)
    logging.info("logfile created: {}".format(CONF['file_logfile']))

    if args.command == 'parseDoc':
        _run_parse_doc()
    elif args.command == 'parseQuery':
        _run_parse_query()
    elif args.command == 'retrieve':
        _run_terrier_retrieve()
    elif args.command == 'evaluate':
        _run_terrier_evaluate(interactive=True)
    elif args.command == 'terrier_setup':
        _run_terrier_setup()
    elif args.command == 'terrier_index':
        _run_terrier_index()
    elif args.command == 'test':
        _run_test_28()
