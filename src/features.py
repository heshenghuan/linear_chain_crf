#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 2017

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""


import numpy as np
from parameters import START, END

# Separator of field values.
separator = "\t"

# Field names of the input data.
fields = ['w', 'y']
pmi_fields = ['w', 'p', 'y']

# Attribute templates.
templates = (
    (('r', -2), ),
    (('r', -1), ),
    (('r',  0), ),
    (('r',  1), ),
    (('r',  2), ),
    (('r', -2), ('r', -1)),
    (('r', -1), ('r',  0)),
    (('r',  0), ('r',  1)),
    (('r',  1), ('r',  2)),
    (('r', -1), ('r',  1)),
    )

pmi_templates = (
    (('r', -2), ),
    (('r', -1), ),
    (('r',  0), ),
    (('r',  1), ),
    (('r',  2), ),
    (('r', -2), ('r', -1)),
    (('r', -1), ('r',  0)),
    (('r',  0), ('r',  1)),
    (('r',  1), ('r',  2)),
    (('r', -1), ('r',  1)),
    (('p', -2), ),
    (('p', -1), ),
    (('p',  0), ),
    (('p',  1), ),
    (('p',  2), ),
    (('p', -2), ('p', -1)),
    (('p', -1), ('p',  0)),
    (('p',  0), ('p',  1)),
    (('p',  1), ('p',  2)),
    (('p', -1), ('p',  1)),
    )


def readiter(data, names):
    """
    Return an iterator for item sequences read from a file object.
    This function reads a sequence from a file object L{fi}, and
    yields the sequence as a list of mapping objects. Each line
    (item) from the file object is split by the separator character
    L{sep}. Separated values of the item are named by L{names},
    and stored in a mapping object. Every item has a field 'F' that
    is reserved for storing features.

    @type   data:     array
    @param  data:     The data array.
    @type   names:  tuple
    @param  names:  The list of field names.
    @rtype          list of mapping objects
    @return         An iterator for sequences.
    """
    X = []
    for line in data:
        if len(line) != len(names):
            raise ValueError('Too many/few fields (%d) for %r\n' % (len(line), names))
        item = {'F': []}    # 'F' is reserved for features.
        for i in range(len(names)):
            item[names[i]] = line[i]
        X.append(item)
    return X


def apply_templates(X, templates=templates):
    """
    Generate features for an item sequence by applying feature templates.
    A feature template consists of a tuple of (name, offset) pairs,
    where name and offset specify a field name and offset from which
    the template extracts a feature value. Generated features are stored
    in the 'F' field of each item in the sequence.

    @type   X:  A list of dict{'w':w, 'r':r, 'y':y, F:[]}
    @param  X:  The item sequence.
    @type   template:   tuple of (str, int)
    @param  template:   The feature template.
    """
    # print 'in apply templates! input:', X
    length = len(X)
    for template in templates:
        name = '|'.join(['%s[%d]' % (f, o) for f, o in template])
        for t in range(len(X)):
            values = []
            for field, offset in template:
                p = t + offset
                if p < 0:
                    values.append(START)
                elif p >= length:
                    values.append(END)
                else:
                    values.append(X[p][field])
            X[t]['F'].append('%s=%s' % (name, '|'.join(values)))
    return X


def escape(src):
    """
    Escape colon characters from feature names.

    @type   src:    str
    @param  src:    A feature name
    @rtype          str
    @return         The feature name escaped.
    """
    return src.replace(':', '__COLON__')


def output_features(fo, X, field=''):
    """
    Output features (and reference labels) of a sequence in CRFSuite
    format. For each item in the sequence, this function writes a
    reference label (if L{field} is a non-empty string) and features.

    @type   fo:     file
    @param  fo:     The file object.
    @type   X:      list of mapping objects
    @param  X:      The sequence.
    @type   field:  str
    @param  field:  The field name of reference labels.
    """
    for t in range(len(X)):
        if field:
            fo.write('%s' % X[t][field])
        for a in X[t]['F']:
            if isinstance(a, str):
                fo.write('\t%s' % escape(a))
            else:
                fo.write('\t%s:%f' % (escape(a[0]), a[1]))
        fo.write('\n')
    fo.write('\n')


def feature_extractor(X, templates=templates):
    # Apply attribute templates to obtain features (in fact, attributes)
    return apply_templates(X, templates)
