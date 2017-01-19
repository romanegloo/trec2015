"""
txt_preproc.py
author: Jiho (jiho@cs.uky.edu)
"""
import xml.etree.ElementTree as et
import os


def xml2text(xml, elms):
    """
    description: converts xml string stream into textual content. It only
    extracts elements of interest and ignores other fields selectively

    inputs:
        xml - string or filename, entire xml stream
        elms - list of string, node names that need to be extracted
    outputs: string, document content
    """
    out = list()
    doc = None

    if len(xml) == 0 or len(elms) == 0:
        return
    if type(xml) is str:
        # read content and parse as an xml tree for validity
        if os.path.isfile(xml):  # check if it's a filename
            # TODO: read and parse asdf
            pass
        else:
            doc = et.fromstring(xml)

    for elm in elms:
        try:
            out.append(" ".join(doc.find(elm).itertext()))
        except:
            pass

    return " ".join(out)

if __name__ == '__main__':
    xml = '<body><elm1>hello and <sub>2 test</sub></elm1> here is more</body>'
    elms = ['sub']
    print(xml2text(xml, elms))
