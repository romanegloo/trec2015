#!/usr/bin/env python3

import os
from lxml import etree as et
from nltk.corpus import stopwords
import argparse

topic_file = '/home/jiho/projects/TREC/trec_eval/eval2015/topics2015A.xml'
processed_file= '/home/jiho/projects/TREC/trec_eval/eval2015/p_topics2015A.xml'
stop = set(stopwords.words('english'))

# TODO: no periods, quotation marks (. " ' < , :)

def convSingle():
    # read in
    tp_tree = et.parse(topic_file)
    with open(processed_file, 'w') as out:
        for elm in tp_tree.xpath('//topic'):
            tp_id = elm.attrib['number']
            text = ' '.join([x for x in elm.itertext()])
            text = [i for i in text.lower().split() if i not in stop]
            out.write(' '.join([tp_id] + text) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('format', choices=['single', 'sgml'], default='single')

    convSingle()
