#!/usr/bin/env python3
"""
nxml_parser.py
author: Jiho (jiho@cs.uky.edu)
description: parser reads nxml file, outputs an xml file containing the content
of a few important nodes such as [docid], [body], [additional info]
"""
# import xml.etree.ElementTree as et
from __future__ import print_function
from lxml import etree as et
import os
import logging
import argparse
import re
import shutil



# settings
collection_path = '/media/jiho/datasets/TREC/2015'
collection_subdirs = re.compile(r".*(pmc-text-\d+/\d+).*")
pool_path = '/media/jiho/datasets/TREC/2015/pool/'
document_ext = '.nxml'

xpath_docid = "//article-id[@pub-id-type='pmc']"
xpath_terms = {
    'TITLE': '//front//article-title',
    'ABSTRACT': '//abstract',
    'KEYWORDS': 'kwd-group',
    'BODY': '//body',
    'REF': '//back//article-title'
}

num_files = -1  # sampling for toy data, set to -1 if you want full indexing


def xml2simple(file, pool_dir):
    """
    read nxml file, extracts nodes of interest and reformat the document into a
    simpler xml file
    """
    try:
        d_from = et.parse(file)
    except:
        logging.warning('unable to parse file [{}]'.format(file))
        return 0

    # create new xml tree
    doc = et.Element("DOC")

    # read nodes and attach to the new tree
    for tag, path in xpath_terms.items():
        text = ''
        elms = d_from.xpath(path)
        for elm in elms:
            text += "".join([x for x in elm.itertext()])
        ch = et.SubElement(doc, tag)
        ch.text = text
    #     doc.text = text
    docid = d_from.xpath(xpath_docid)[0]
    docid.tag = 'DOCNO'
    del docid.attrib['pub-id-type']
    doc.append(docid)

    with open(os.path.join(pool_dir, docid.text + '.xml'), 'wb') as out:
        out.write(et.tostring(doc, pretty_print=True))
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true')
    parser.add_argument('-c', '--continue', dest='cont', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    elif args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    cnt_parsing = [0, 0, 0]  # [total, failed]

    # remove pool directory and re-create it, because I want new document
    #  pool each time
    if not args.cont and os.path.exists(pool_path):
        logging.debug("removing pool directory: {}".format(pool_path))
        shutil.rmtree(pool_path)
    # traverse the document directory
    for root, subdirs, files in os.walk(collection_path):
    # for root, subdirs, files in os.walk('.'):
        m = collection_subdirs.match(root)
        if not m:
            continue
        # check if pool sub-directory exist
        pool_sub = os.path.join(pool_path, m.group(1))
        if not os.path.exists(pool_sub):
            os.makedirs(pool_sub)

        for file in files:
            filename, ext = os.path.splitext(file)
            if ext == document_ext:
                cnt_parsing[0] += 1
                filepath = os.path.join(root, file)
                if args.cont and os.path.isfile(os.path.join(pool_sub, file)):
                    cnt_parsing[2] += 1
                    print("skipping {} [totla {cnt[0]}, failed {cnt[1]} "
                          "skipped {cnt[2]}]\r".
                          format(file, cnt=cnt_parsing), end="")
                    continue
                rst = xml2simple(filepath, pool_sub)
                if rst <= 0:
                    cnt_parsing[1] += 1
                else:
                    if num_files > 0:
                        num_files -= 1
                    elif num_files == 0:
                        break
                print("parsing {} [total {cnt[0]}, failed {cnt[1]}, "
                      "skipped {cnt[2]}]\r".
                      format(file, cnt=cnt_parsing), end="")
        else:
            continue
        break
    print("\nparsing completed", '='*70)
