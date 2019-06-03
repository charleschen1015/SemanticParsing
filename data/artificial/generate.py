# ==============================================================================
#From Physician Queries to Logical Forms for Efficient Exploration of Patient Data.
#Charles Chen, Sadegh Mirshekarian, Razvan Bunescu and Cindy Marling. 
#IEEE 13th International Conference on Semantic Computing (ICSC 2019), 
# Jan. 30-Feb. 1, 2019, Newport Beach, California.
# ==============================================================================


import argparse
import os
import numpy as np
from utils import parse_template, prepare_tex_file

rnd = np.random
#rnd.seed(4321)


parser = argparse.ArgumentParser()
parser.add_argument('--template_dir',
                    type=str,
                    default='templates',
                    help="root directory for templates")

parser.add_argument('--types_file',
                    type=str,
                    default='templates/types',
                    help="the optional file that contains types")

parser.add_argument('--out_tex_file',
                    type=str,
                    default='3.tex',
                    help="the tex file to which generated data should be saved")

parser.add_argument('--line_length',
                    type=int,
                    default='80',
                    help="the maximum length allowed when writing annotations in multline environment")

parser.add_argument('--test_key',
                    type=str,
                    default=None,
                    help="if provided, forces the use of this key while generating examples")

parser.add_argument('--n_examples',
                    type=int,
                    default='1000',
                    help="the total number of sentences to create")

args = parser.parse_args()
## Read templates
template_desc = {}  # stores a tree of description labels for the templates
templates = {}  # stores templates with keys
tags = {}  # stores a set of keys per each tag

#import pdb; pdb.set_trace()

for cur_folder, cur_subfolders, cur_files in os.walk(args.template_dir):
    for fname in [os.path.join(cur_folder, x) for x in cur_files]:
        cur_key = ''
        cur_desc = []
        for elem in os.path.normpath(fname).split(os.path.sep):
            if elem[0].isdigit():
                cur_key += elem[0]
                cur_desc.append(elem[2:])
            else:
                if cur_key != '':
                    break

        assert (len(cur_key) == len(cur_desc))

        if cur_key != '':  # there is a valid file here
            print("File:", fname, ", FileKey:", cur_key, ", CurDesc", cur_desc)
            _d = template_desc
            for i in range(len(cur_key)):
                c = cur_key[i]
                if c not in _d:
                    _d[c] = {'label': cur_desc[i]}
                _d = _d[c]

            # read the file
            with open(fname, 'r') as f:
                lines = f.readlines()
                have_read_a_template = False
                cur_key_full = None
                for L in [x.strip() for x in lines]:
                    if not L or L.startswith('#'):
                        continue

                    if L.lower().startswith('key'):
                        L = L.partition(':')[2].strip()
                        assert (L.startswith(cur_key))
                        cur_key_full = L
                        if cur_key_full not in templates:
                            templates[cur_key_full] = []
                    elif L.startswith('-->'):  # line contains a logical form
                        if have_read_a_template:
                            p = L[3:].strip()
                            if p.startswith('"'):
                                p = p[1:-1]
                            templates[cur_key_full][-1][-1] = p
                            have_read_a_template = False
                        else:
                            raise Exception("Logical form provided without a template, in file ", fname)
                    elif L.startswith('...'):
                        have_read_a_template = True
                        p = L[3:].strip()
                        if p.startswith('"'):
                            p = p[1:-1]
                        templates[cur_key_full][-1] += [p, ""]
                    else:  # line contains a template
                        if L.startswith('('):
                            tag, _, p = L[1:].partition(')')
                            p = p.lstrip()
                        else:
                            tag, p = None, L

                        if p.startswith('"'):
                            p = p[1:-1]

                        if tag not in tags:
                            tags[tag] = set()

                        have_read_a_template = True
                        templates[cur_key_full].append([p, ""])
                        tags[tag].add(cur_key_full)


## Read types
types = {}
if args.types_file is not None:
    with open(args.types_file, 'r') as f:
        lines = f.readlines()
        for L in [x.strip() for x in lines]:
            if L and not L.startswith('#'):
                a = L.partition('=')
                types[a[0].strip()[1:-1]] = a[2].strip()[1:-1]


## Generation loop
prepare_tex_file(args.out_tex_file)

with open(args.out_tex_file, 'a') as f:
    print("Writing {} examples to '{}'...".format(args.n_examples, args.out_tex_file))
    for n in range(args.n_examples):
        available_keys = [x for x in templates.keys() if templates[x] and not x.endswith('*')]

        while True:
            if args.test_key:
                selected_key = args.test_key
            else:
                selected_key = rnd.choice(available_keys)

            if templates[selected_key]:
                # this is where a template is selected from the list of templates in 'selected_key'.
                selected_template = templates[selected_key][rnd.choice(len(templates[selected_key]))]
                break

        selection_codes = {}
        _tmplt = selected_template.copy()
        _key = selected_key
        i = 0
        while True:
            f.write("\n\\item\n")

            if _tmplt[i].startswith('@'):
                _key = _tmplt[i][1:]
                if _key + '*' in templates:
                    _key = _key + '*'
                _replacement = templates[_key][rnd.choice(len(templates[_key]))]
                _tmplt = _tmplt[:i] + _replacement + _tmplt[i+2:]

            f.write("\\textbf{{{0}\\theC{0}}} \\key{{({1})}} \\addtocounter{{C{0}}}{{1}}\n"
                    .format(template_desc[_key[0]]['label'], _key))

            res = parse_template(_tmplt[i], types, False, selection_codes)[0]
            print('\t'+res)
            f.write("\\textcolor{{blue}}{{ {} }}\n".format(res[:args.line_length]))

            res = parse_template(_tmplt[i+1], types, False, selection_codes, True)[0]
            print('\t'+res+'\n')

            f.write("\\begin{multline*}")
            while res:
                _respart = res[:args.line_length].rpartition(' ')
                f.write('\n')
                if _respart[0] and len(res) > args.line_length:
                    f.write("{} \\\\ ".format(_respart[0]))
                    res = _respart[2] + res[args.line_length:]
                else:
                    f.write("{} \\\\ ".format(res[:args.line_length]))
                    res = res[args.line_length:]

            f.write("\n\\end{multline*}\n\n")

            i += 2
            if i >= len(_tmplt):
                break

prepare_tex_file(args.out_tex_file, to_the_end=True)

