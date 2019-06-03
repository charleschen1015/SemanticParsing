# ==============================================================================
#From Physician Queries to Logical Forms for Efficient Exploration of Patient Data.
#Charles Chen, Sadegh Mirshekarian, Razvan Bunescu and Cindy Marling. 
#IEEE 13th International Conference on Semantic Computing (ICSC 2019), 
# Jan. 30-Feb. 1, 2019, Newport Beach, California.
# ==============================================================================

import argparse
import os
import numpy as np

rnd = np.random

#rnd.seed(4321)

def expand_type(type_word, type_dict):
    """
    Expands the type word recursively using the provided type_dict.
    """
    ret = type_word

    if type_word in type_dict:
        ret = type_dict[type_word]
        if '/' in ret:
            p = ret.partition('/')
            ret = expand_type(p[0].rstrip(), type_dict) + '/' + expand_type(p[2].lstrip(), type_dict)

    return ret


def parse_template(tmplt, type_dict, all_permutations=False, codes={}, read_only_codes=False,
                   c=1, w='', must_choose_ind=None):
    """
    Parses a template and generates one random or all of its permutations.
    :param tmplt: the template string
    :param type_dict: a dictionary holding types (can be empty), e.g. discrete_event, daily_intervals
    :param all_permutations: whether to output all the permulations of the given template or just one random
    :param codes: a dictionary holding the bracket codes, e.g. when you use [x/y] in a template, this bracket's code is
        saved in "codes" and passed on. Later, we can use [$1:X/Y] if needed and the parser knows this should match [x/y]
        from the recorded "codes" dictionary. Usually left at {} when calling on a natural language template.
    :param read_only_codes: if True, codes will not be updated. Usually used for logical form templates.
    :param c: keeps track of the bracket number. Leave it as default.
    :param w: keeps track of the code prefix. Leave it as default.
    :param must_choose_ind: if provided, this index is used instead of a random index when choosing between options
        separated by '/'. Leave as default.

    :return: a list containing the produced entries.
    """
    if tmplt.startswith('$'):
        if ':' in tmplt:
            _cv = tmplt[1:].partition(':')
            if _cv[0] in codes:
                return parse_template(_cv[2], type_dict, False, codes, read_only_codes, c, w, codes[_cv[0]][0])
            else:
                raise Exception("Provided code {} not in codes dictionary in {}.".format(_cv[0], tmplt))
        elif tmplt[1:] in codes:
            return [codes[tmplt[1:]][1]]
        else:
            raise Exception("Invalid format: expected ':' when starting with '$' for input", tmplt)
    i = 0
    s = len(tmplt)
    sep_inds = []  # alternative values separted by '/'
    open_brackets = 0
    while i < s:
        if tmplt[i] == '/' and open_brackets <= 0:
            sep_inds.append(i)
        elif tmplt[i] == '[':
            open_brackets += 1
        elif tmplt[i] == ']':
            open_brackets -= 1
        i += 1

    if len(sep_inds) > 0:  # some '/' found outside brackets
        sep_inds = [-1] + sep_inds + [s]
        if all_permutations:
            res = []
            for i in range(1, len(sep_inds)):
                _t = tmplt[sep_inds[i - 1] + 1:sep_inds[i]]
                if i == 1:
                    _t = _t.rstrip()
                elif i == len(sep_inds)-1:
                    _t = _t.lstrip()
                else:
                    _t = _t.strip()
                res += parse_template(_t, type_dict, True)
            return res
        else:
            if must_choose_ind is not None:
                i = must_choose_ind
            else:
                i = rnd.randint(1, len(sep_inds))

            _t = tmplt[sep_inds[i - 1] + 1:sep_inds[i]]

            if i == 1:
                _t = _t.rstrip()
            elif i == len(sep_inds)-1:
                _t = _t.lstrip()
            else:
                _t = _t.strip()

            if not read_only_codes:
                codes[w[:-1]] = (i, _t)

            return parse_template(_t, type_dict, False, codes, read_only_codes, c, w)

    i = open_brackets = 0
    a = b = -1
    while i < s:
        if tmplt[i] == '[':
            open_brackets += 1
            if a == -1:
                a = i
        elif tmplt[i] == ']':
            open_brackets -= 1
            if a != -1 and open_brackets == 0:
                b = i
                break
        i += 1

    if i < s:  # some stuff found inside brackets
        if all_permutations:
            res = []
            for rright in parse_template(tmplt[b + 1:], type_dict, True):
                for rmid in parse_template(tmplt[a + 1:b], type_dict, True):
                    _rright = rright
                    _rmid = rmid
                    res.append(tmplt[:a] + _rmid + _rright)
            return res
        else:
            return [tmplt[:a]
                    + parse_template(tmplt[a + 1:b], type_dict, False, codes, read_only_codes, 1, w+str(c)+'_')[0]
                    + parse_template(tmplt[b + 1:], type_dict, False, codes, read_only_codes, c+1, w)[0]]

    # no '/' or brackets found up to this point
    if tmplt in type_dict:
        tmplt = expand_type(tmplt, type_dict)
        return parse_template(tmplt, type_dict, all_permutations, codes, read_only_codes, c, w, must_choose_ind)
    elif tmplt.startswith('range'):
        _range = eval(tmplt)
        _val = str(rnd.randint(_range.start, _range.stop))
        if not read_only_codes:
            codes[w[:-1]] = (1, _val)
        return [_val]
    elif tmplt.startswith('clocktime'):
        if '(' in tmplt:
            _h, _m = eval(tmplt.partition('(')[2].partition(')')[0])
        else:
            _h = rnd.randint(1, 24)
            _m = rnd.randint(0, 60)

        if _h > 12:
            _h -= 12
            _tag = 'pm'
        else:
            _tag = 'am'

        _val = "{:01}:{:02}{}".format(_h, _m, _tag)
        if not read_only_codes:
            codes[w[:-1]] = (1, _val)
        return [_val]
    else:
        return [tmplt]


# Generation loop
def prepare_tex_file(fname, to_the_end=False):
    """
    Adds the necessary stuff at the beginning or the end of the output tex file.
    """
    if to_the_end:
        with open(fname, 'a') as f:
            f.write("\n\n\end{enumerate}\n\end{document}")
    else:
        with open(fname, 'w') as f:
            f.write(r"""\documentclass[11pt]{article}

\marginparwidth 0.5in
\oddsidemargin 0.25in
\evensidemargin 0.25in
\marginparsep 0.25in
\topmargin 0.25in
\textwidth 6in \textheight 8 in

\newcommand{\key}[1]{\textcolor{lightgray}{#1}}

\newcounter{CQuery}
\newcounter{CStatement}
\newcounter{CClick}

\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{xcolor}

\begin{document}
\author{}
\title{Synthetic Data}
\maketitle

\setcounter{CQuery}{1}
\setcounter{CStatement}{1}
\setcounter{CClick}{1}

\begin{enumerate}""")

