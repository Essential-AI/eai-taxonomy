#!/usr/bin/env python
DESCRIPTION = """
compute nmi or npmi over a set of annotator/category combinations
* if NMI, the options {categorey_paths} -a {annotator_paths} will cause the following to be computed:
  - each annotator/category pair will be a row in a NMI matrix
  - each annotator/category pair will be a column in a NMI matrix
  - rowi/columnj will be the NMI between the ith and jth annotator/category pair
* if NPMI, the options {categorey_paths} -a {annotator_paths} will cause the following to be computed:
  - the total number of annotator/category combinations should be 2
  - the labels of the first annotator/category pair, and the labels of the second annotator/category pair will be the row/column labels of the NPMI matrix
* input file format:
  - file format can be jsonl or parquet
  - each line or row in the input file should represent data that was annotated by multiple annotators.
  - inside each line/row is a dictionary with the following structure:
    $PREFIX_PATH/ # common prefix path to all annotators and their annotations
      $ANNO_1_PATH/ # name or path to the first annotator
        $CATEGORY_1_PATH/ # name or path to the first category labeled by the first annotator
          primary_code: int # primary label identifier
          secondary_code: int # optional secondary label identifier
          ...
        $CATEGORY_N_PATH/ # name or path to the nth category labeled by the first annotator
          primary_code: int # primary label identifier
          secondary_code: int # optional secondary label identifier
          ...
      ...
      $ANNO_M_PATH/ # name or path to the mth annotator
        $CATEGORY_1_PATH/ # name or path to the first category labeled by the mth annotator
          primary_code: int # primary label identifier
          secondary_code: int # optional secondary label identifier
          ...
        $CATEGORY_N_PATH/ # name or path to the nth category labeled by the mth annotator
          primary_code: int # primary label identifier
          secondary_code: int # optional secondary label identifier
          ...
* output file format examples
  - NPMI matrix:
    input:
    ```
    ./nmi.py stem.jsonl taxon_v1/dds1 taxon_v2/education_level  
             -a sonnet 
             --prefix taxonomy_annotations/parsed 
             --pointwise
    ```
    output:
    ```
                               Indeterminate	Graduate /    General     High School    Undergraduate 
                                                Expert Level  Audience    Level	         Level
    Arts and recreation        0.218        	-0.557         0.192      -0.048         -0.142             
    Religion                   -0.831       	0.198          -0.376     -0.281         0.160              
    History and geography      -0.834       	-0.204         -0.161     0.190          -0.112             
    Language                   -0.852       	0.184          -0.216     -0.121         0.136              
    Technology                 0.116        	-0.173         0.150      0.060          -0.158             
    Philosophy and psychology  -0.010       	0.134          -0.153     -0.091         0.148              
    Science                    -0.876       	0.132          -0.128     -0.098         0.143              
    Social sciences            -0.876       	-0.166         0.083      0.115          -0.151             
    Computer science,               	
      information, and 
      general works	           0.086        	0.030          -0.023     -0.020         0.026              
    Literature                 -0.844       	-0.001         0.044      0.049          -0.174       
    ```
  - NMI matrix:
    input:
    ```
    nmi.py stem.jsonl dds1 dds2 dds3 reasoning_depth education_level 
           -a eai-distill-0.5b/distill_prompt  
           --prefix taxonomy_annotations/parsed
    ```
    output:
    ```
                    dds1 	dds2 	dds3 	reasoning_depth	education_level
    dds1           	1.000	0.689	0.577	0.048       	0.075
    dds2           	0.689	1.000	0.859	0.076       	0.100
    dds3           	0.577	0.859	1.000	0.120       	0.147
    reasoning_depth	0.048   0.076   0.120   1.000          	0.186          
    education_level	0.075   0.100   0.147   0.186          	1.000
```
"""
import argparse
import math
import json
import collections
import daft
import os
import sys
import re
from textwrap import dedent

def fastget(d, key):
    """                                                                                                                                                                                                                   
    get a value from a dictionary using a key with slashes to separate the keys                                                                                                                                           
    fast version of dpath.get. doesnt do lists, doesnt do wildcards                                                                                                                                                       
    """
    keys = [k for k in key.split("/") if k != ""]
    for k in keys:
        d = d[k]
    return d

def plus1_smooth(catmap, exclude=set([])):
    return 1e-20 * math.prod(len(v) for k, v in catmap.items() if k not in exclude)


def counts(data, count_field="tokens", **kwargs):
    count = 0
    for d in data:
        include = True
        for k, v in d["taxon"].items():
            if v["label"] is None:
                include = False
                break
        for k, v in kwargs.items():

            if d["taxon"][k]["code"] != v or not include:
                include = False
                break
        if include:
            count += d[count_field]
    return count


class TblType:
    def __init__(self, length):
        self.length = length
    def __call__(self):
        if self.length == 0:
            return 0
        else:
            return collections.defaultdict(TblType(self.length - 1))


def tbltype(length):
    if length == 0:
        return 0
    else:
        return collections.defaultdict(TblType(length - 1))


def counts_table(data, fields=[], count_field="tokens", use_secondary=False):
    tbl = tbltype(len(fields))
    for d in data:
        if len(fields) > 0:
            entry = tbl
            for f in fields[:-1]:
                v = d["taxon"][f]["code"]
                entry = entry[v]
            entry[d["taxon"][fields[-1]]["code"]] += d[count_field]
        else:
            tbl += d[count_field]
    return tbl


def cats(data):
    catmap = collections.defaultdict(set)
    for d in data:
        for k, v in d["taxon"].items():
            if v["label"] is not None:
                catmap[k].add(v["code"])
    catmap = {k: sorted(list(catmap[k])) for k in catmap}
    return catmap


def cat_labels(data):
    catmap = collections.defaultdict(dict)
    for d in data:
        for k, v in d["taxon"].items():
            if v["label"] is not None:
                catmap[k][v["code"]] = v["label"]
    return catmap


def nmi(data, catmap, class1, class2, use_secondary=False):
    total = counts_table(data, use_secondary=use_secondary) + plus1_smooth(catmap)
    counts1 = counts_table(data, [class1], use_secondary=use_secondary)
    counts2 = counts_table(data, [class2], use_secondary=use_secondary)
    counts12 = counts_table(data, [class1, class2], use_secondary=use_secondary)

    def H(data, catmap, countstbl, labelclass, total):
        h = 0
        smooth = plus1_smooth(catmap, [labelclass])
        for y in catmap[labelclass]:
            n = countstbl[y] + smooth
            p = n / total
            h += -(p * math.log2(p))
        return h

    def CH(data, catmap, counts2, counts12, class1, class2, total):
        smooth_2 = plus1_smooth(catmap, [class2])
        smooth_12 = plus1_smooth(catmap, [class1, class2])
        ch = 0
        for y in catmap[class2]:
            ytotal = counts2[y] + smooth_2
            hcy = 0
            for x in catmap[class1]:
                n = counts12[x][y] + smooth_12
                p = n / ytotal
                hcy += p * math.log2(p)
            hcy = - (ytotal / total) * hcy
            ch += hcy
        return ch

    h1 = H(data, catmap, counts1, class1, total)
    h2 = H(data, catmap, counts2, class2, total)

    i12 = h1 - CH(data, catmap, counts2, counts12, class1, class2, total)
    return (2 * i12) / (h1 + h2)


def npmi(data, catmap, t, f, use_secondary=False):
    smooth = plus1_smooth(catmap)
    smooth_f = plus1_smooth(catmap, [f])
    smooth_t = plus1_smooth(catmap, [t])
    smooth_tf = plus1_smooth(catmap, [t, f])
    counts_t = counts_table(data, [t], use_secondary=use_secondary)
    counts_f = counts_table(data, [f], use_secondary=use_secondary)
    counts_tf = counts_table(data, [t, f], use_secondary=use_secondary)
    total = counts_table(data, use_secondary=use_secondary) + smooth
    #print(f"s factors: {smooth/total}, {smooth_f/total}, {smooth_tf/total}, {total}", file=sys.stderr)
    mat = {}
    for kd in catmap[t]:
        nt = counts_t[kd] + smooth_t
        pt = nt / total
        row = {}
        for cp in catmap[f]:
            ntf = counts_tf[kd][cp] + smooth_tf
            ptf = ntf / total

            if ptf > 0:
                pf = (counts_f[cp] + smooth_f) / total
                npmi = - math.log2(ptf/(pt * pf)) / math.log2(ptf)
                row[cp] = npmi
            else:
                row[cp] = 0.
        mat[kd] = row
    return mat


def extract(prefix, d, ps=True):
    for s in ("primary", "secondary"):
        codepath = f"{prefix}/{s}_code"
        lblpath = f"{prefix}/{s}_code_label"
        try:
            pth = codepath
            codeval = fastget(d, codepath)
            pth = lblpath
            lblval = fastget(d, lblpath)
            if codeval is not None and codeval >= 0:
                yield {"code": codeval, "label": lblval}
        except KeyError as e:
            #print(f"key error: {pth}", file=sys.stderr)
            pass


def transform(datum, prefix, p1, p2): 
    #print(f"transform: {prefix}, {p1}, {p2}", file=sys.stderr)
    dp1, dp2 = [f"{prefix}/{p}" for p in [p1, p2]]
    tokens = len(datum["text"])
    for x, d1 in enumerate(extract(dp1, datum)):
        for y, d2 in enumerate(extract(dp2, datum)):
            yield {
              "taxon": {
                 p1: d1,
                 p2: d2,
              },
              "tokens": tokens if x == y else (0.01 * tokens),
              "docs": 1,
            }


def unwrap(prompt, breakchar="\b"): 
    r = prompt.replace(f"{breakchar}\n", breakchar)
    return re.sub(f"\\s*{breakchar}\\s*", " ", r)

class UnwrapHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _leading_space(self, text):
        leading_space = ""
        for c in text:
            if c.isspace():
                leading_space += c
            else:
                break
        return leading_space
    
    def _fill_text(self, text, width, indent):
        text = unwrap(dedent(text))
        v = []
        for line in text.splitlines():
            if len(indent) + len(line) <= width:
                v.append(indent + line)
            else:
                start = True
                leading_space = self._leading_space(line)
                for sline in super()._split_lines(indent + line.lstrip(), width - len(leading_space)):
                    v.append(leading_space + sline)
        return "\n".join(v)

    def _split_lines(self, text, width):
        text = unwrap(dedent(text))
        lines = []
        for line in text.splitlines():
            if len(line) < width:
                lines.append(line)
            else:
                for sline in super()._split_lines(line, width):
                    lines.append(sline)
        return lines

if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=UnwrapHelpFormatter, description=DESCRIPTION)
    ap.add_argument("data", help="input file name. see above description for input file format.")
    ap.add_argument("category_paths", nargs="+", help="list of category paths to include in nmi/npmi computation. see above description for more details.")
    ap.add_argument("-a", "--annotator-paths", nargs="+", help="list of annotator paths to include in nmi/npmi computation. see above description for more details.")
    ap.add_argument("--pointwise", action="store_true", help="by default NMI is computed. this flag selects NPMI.")
    ap.add_argument("--prefix", default="taxonomy_annotations/parsed", help="prefix path in input file lines/rows to all annotators and their annotations.")
    ap.add_argument("--output", type=argparse.FileType('w'), default=sys.stdout, help="output file name.")
    args = ap.parse_args()

    try:
        pdata = daft.read_parquet(args.data).to_pylist()
    except Exception as e1:
        try:
            with open(args.data, "r") as f:
                pdata = [json.loads(line) for line in f]
        except Exception as e2:
            print(f"error reading {args.data} as parquet or jsonl:\n{e1}\n{e2}", file=sys.stderr)
            raise e1

    if args.pointwise:
        assert len(args.annotator_paths) * len(args.category_paths) == 2
        t, f = [f"{a}/{c}" for a in args.annotator_paths for c in args.category_paths]
        data = []
        for d in pdata:
            for dt in transform(d, args.prefix, t, f):
                data.append(dt)
        catmap = cats(data)
        catlbl = cat_labels(data)
        mat = npmi(data, catmap, t, f)
        sorted_ts = sorted(
            catmap[t],
            key=lambda c: max(mat[c][d] for d in catmap[f]),
            reverse=True,
        )
        sorted_fs = sorted(
            catmap[f],
            key=lambda d: max(mat[c][d] for c in catmap[t]),
            reverse=True,
        )
        label_ts = [catlbl[t][c] for c in sorted_ts]
        label_fs = [catlbl[f][d] for d in sorted_fs]
        paddings = [max(len(lt) for lt in label_ts)] + [max(len(lf),5) for lf in label_fs]
        print(" "*paddings[0], *[lf.ljust(paddings[i+1]) for i, lf in enumerate(label_fs)], sep="\t", file=args.output)
        for lt, t in zip(label_ts, sorted_ts):
            row = [mat[t][f] for f in sorted_fs]
            print(f"{lt:{paddings[0]}}", *[f"{c:.3f}".ljust(paddings[i+1]) for i, c in enumerate(row)], sep="\t", file=args.output)
    else:
        paths = [f"{a}/{c}" for c in args.category_paths for a in args.annotator_paths]
        common = os.path.commonpath(paths)
        def nrmslash(s):
            return "/".join(ss for ss in s.split("/") if ss != "")
        nms = [nrmslash(p[len(common):]) for p in paths]
        
        paddings = [max(len(nm) for nm in nms)] + [max(len(nm),5) for nm in nms]
        pairings = [((i, j),(t, f)) for i, t in enumerate(paths) for j, f in enumerate(paths)]
        mat = [[None for _ in range(len(paths))] for _ in range(len(paths))]
        for (i, j), (t, f) in pairings:
            if i == j:
                mat[i][j] = 1.0
            elif j < i:
                mat[i][j] = mat[j][i]
            else:
                data = []
                for d in pdata:
                    for dt in transform(d, args.prefix, t, f):
                        data.append(dt)
                catmap = cats(data)
                mat[i][j] = nmi(data, catmap, t, f)
        print(f"paddings: {paddings}", file=sys.stderr)
        padchar = " "
        print(padchar*paddings[0], *[p.ljust(paddings[i+1],padchar) for i, p in enumerate(nms)], sep="\t", file=args.output)
        for i, p in enumerate(nms):
            print(p.ljust(paddings[0],padchar), *[f"{c:.3f}".ljust(paddings[j+1],padchar) for (j,c) in enumerate(mat[i])], sep="\t", file=args.output)
