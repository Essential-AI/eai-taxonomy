#!/usr/bin/env python

DESCRIPTION = """
compute overlapping kappa for a set of annotators.
* all kappas can either be weighted or unweighted
* kappa can be based onfleiss or cohen
  - cohen can either be 
    + average cohen of first annotator with other annotators
    + average cohen of all annotators with each other

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
* output file format:
  - tsv file with the following columns:
    - category path
    - P_o # observed agreement
    - P_e # expected agreement
    - kappa # (P_o - P_e) / (1 - P_e)
    - std_kappa # standard deviation of kappa, if bootstrapping is requested

example invocation:
```
./agreement.py stem.jsonl dds1 dds2 dds3 
               --annotator-paths dsv3/taxon_v1 sonnet/taxon_v1
               --prefix taxonomy_annotations/parsed
               --pairwise
               --bootstrap
               --output agreement.tsv
               --input-format jsonl
```
produces output:
```
label 	P_o   	P_e   	kappa 	std_kappa
dds1	0.9644	0.3487	0.9453	0.0097
dds2	0.8588	0.0795	0.8466	0.0130
dds3	0.6958	0.0255	0.6878	0.0161
```
"""

from textwrap import dedent
import re
import argparse
import numpy as np
from collections import defaultdict
import dpath
import sys
import yaml
import logging
from tqdm import tqdm
import random
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def fastget(d, path):
    path = [p for p in path.split("/") if p != ""]
    for p in path:
        if d is not None and p in d:
            d = d[p]
        else:
            raise KeyError(f"key {p} not found in {d}")
    return d


def fastset(d, path, val):
    path = [p for p in path.split("/") if p != ""]
    for p in path[:-1]:
        d = d[p]
    d[path[-1]] = val


def fastdel(d, path):
    path = [p for p in path.split("/") if p != ""]
    for p in path[:-1]:
        d = d[p]
    del d[path[-1]]


def extract(prefix, d, top_pick=False):
     lbls = set([])
     esses = ["primary", "secondary"]
     for s in esses:
         path = f"{prefix}/{s}_code"
         try:
             val = fastget(d, path)
             if val is not None and val >= 0:
                 lbls.add(val)
                 if top_pick:
                     break
         except KeyError:
             pass
     return list(lbls)

def taxonomy_overlap(anno1, anno2):
    if len(anno1) == 0:
        return len(anno2) == 0
    else:
        return len(set(anno1).intersection(set(anno2))) > 0


def taxonomy_pe(fertilities0, weights0 , fertilities1, weights1, weighted):
    """
    P_e for cohen's kappa, overlap-based agreement, weighted or unweighted
    if P_e for fleiss kappa is desired, pass same fertilities and weights for 
    both annotators
    """
    F = [fertilities0, fertilities1]
    W = [w for w in zip(weights0, weights1)]

    if weighted:
        pw = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.5],
            [0.0, 0.5, 1/3],
        ]
        double_count_weight = 1/3
    else:
        pw = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        double_count_weight = -1.0

    # useful table for calculations involving sums of 
    # [y ! = x, x]
    R = [[
        sum(w[i] / (1 - w[i]) for (y, w) in enumerate(W) if x != y)
        for (x, _) in enumerate(W)
    ] for i in range(2)]
    
    # p[x][y] = probability of match given that 
    # the first annotation is length x and the second annotation is length y
    p = [[0]*3 for _ in range(3)]

    # count matches of the form [], []
    p[0][0] = 1 * pw[0][0]

    # count matches of the form [x], [x]
    p[1][1] = sum(w[0] * w[1] for w in W) * pw[1][1]

    # count matches of the form [x], [x, z] and [x], [z, x]
    p[1][2] = sum(w[0] * w[1] * (
        1 +     # [x], [x, z]
        R[1][x] # [x], [z, x]
    ) for (x, w) in enumerate(W)) * pw[1][2]

    # count matches of the form [x, z], [x] and [z, x], [x]
    p[2][1] = sum(w[0] * w[1] * (
        1 +     # [x, z], [x]
        R[0][y] # [z, x], [x]
    ) for (y, w) in enumerate(W)) * pw[2][1]

    # count matches of the form 
    # [x, y], [x, z]
    # [x, y], [z, x]
    # [x, y], [y, z]
    # [x, y], [z, y]
    p[2][2] = sum(w[0] * w[1] * (
        1 +               # [x, y], [x, z]
        R[0][x] +         # [x, y], [z, x]
        R[1][x] +         # [x, y], [y, z]
        R[0][x] * R[1][x] # [x, y], [z, y]
    ) for (x, w) in enumerate(W)) * pw[2][2]

    # we have double counted items of the form 
    # [x, y], [y, x]
    # [x, y], [x, y] 
    # so we subtract them
    for x, wx in enumerate(W):
        for y, wy in enumerate(W):
            if y != x:
                p0xy = wx[0] * (wy[0] / (1 - wx[0]))
                p1yx = wy[1] * (wx[1] / (1 - wy[1]))
                p1xy = wx[1] * (wy[1] / (1 - wx[1]))
                p[2][2] += p0xy * (p1yx + p1xy) * double_count_weight

    return sum(F[0][x] * F[1][y] * p[x][y] for x in range(3) for y in range(3))


def overlap_percent(x, y):
    if len(x) == 0 or len(y) == 0:
        return 1.0 if len(x) == len(y) else 0.0
    return len(set(x).intersection(set(y))) / len(set(x).union(set(y)))


def observed_agreement(samples, weighted: bool):
    po = 0.
    n_samples = 0 #len(samples)
    for sample in samples:
        n = 0
        a = 0
        for a1 in range(len(sample)):
            for a2 in range(len(sample)):
                if a1 != a2:
                    if weighted:
                        a += overlap_percent(sample[a1], sample[a2])
                    else:
                        if taxonomy_overlap(sample[a1], sample[a2]):
                            a += 1
                    n += 1
        if n > 0:
            po += (a/n)
            n_samples += 1
    return po / n_samples


def estimate_random_shared(L):
    """
    build a shared fertility, weights table for a set of annotators,
    for fleiss kappa P_e calculation
    """
    fertilities = [0, 0, 0]
    fn = 0
    label_counts = defaultdict(int)
    wn = 0
    for sample in L:
        for anno in sample:
            fertilities[len(anno)] += 1
            fn += 1
            for lbl in anno:
                label_counts[lbl] += 1
                wn += 1
    labels = sorted(label_counts.keys())
    weights = [label_counts[x] / wn for x in labels]
    fertilities = [f / fn for f in fertilities]
    label_map = {label: x for (x, label) in enumerate(labels)}

    return fertilities, weights, label_map


def estimate_random_per_annotator(L):
    """
    build a fertility, weights table for each annotator,
    for cohen's kappa P_e calculation
    """
    fertilities = [[0, 0, 0] for _ in range(len(L[0]))]
    fn = [0 for _ in range(len(L[0]))]
    label_counts = [defaultdict(int) for _ in range(len(L[0]))]
    wn = [0 for _ in range(len(L[0]))]
    for sample in L:
        for i, anno in enumerate(sample):
            fertilities[i][len(anno)] += 1
            fn[i] += 1
            for lbl in anno:
                label_counts[i][lbl] += 1
                wn[i] += 1
    labels = set([])
    for x in label_counts:
        for y in x.keys():
            labels.add(y)
    labels = list(labels)
    labels.sort()
    label_map = {label: x for (x, label) in enumerate(labels)}
    weights = [[] for _ in range(len(L[0]))]
    for i in range(len(L[0])):
        weights[i] = [(label_counts[i][x] + 1) / (wn[i] + len(labels)) for x in labels]
        fertilities[i] = [(f + 1) / (fn[i] + 3) for f in fertilities[i]]
    return fertilities, weights, label_map


def estimate_random(L, pairwise=False):
    if pairwise:
        return estimate_random_per_annotator(L)
    else:
        return estimate_random_shared(L)


def index_annotations(L, label_map):
    IL = []
    for sample in L:
        isample = []
        for anno in sample:
            ianno = []
            for lbl in anno:
                ianno.append(label_map[lbl])
            isample.append(ianno)
        IL.append(isample)
    return IL


def get_annotator_names(data, prefix):
    """
    get the names of the annotators from an input data line
    if no annotators were specified at command line,
    this function is used to get the names of the annotators from the input data.
    assumes the annotators are the keys of the dictionary at the prefix path.
    """
    names = set([])
    try:
        prefixdata = fastget(data, prefix)
    except KeyError:
        return names
    for name in prefixdata.keys():
        names.add(name)
    return list(names)


def get_all_annotator_names(data):
    names = set([])
    for itm in data:
        names = names.union(get_annotator_names(itm))
    return list(names)


def annotations(lst, paths, catpath, prefix,top_pick=False, strip_empties=False, pairwise=False):
    L = []
    for itm in lst:
        keep = True
        annos = []
        for path in paths:
            key = f"{prefix}/{path}"
            try:
                coll = fastget(itm, key)
                if coll is not None:
                    ext = extract(catpath, coll, top_pick=top_pick)
                    if top_pick:
                        annos.append(ext[:1])
                    else:
                        annos.append(ext)
                else:
                    annos.append([])
            except KeyError as keyerror:
                logger.warning(f"keyerror: {keyerror}")
                if strip_empties or pairwise:
                    keep = False
        if strip_empties:
            for a in annos:
                if len(a) == 0:
                    keep = False
                    break
        if keep:
            L.append(annos)
    #print(f"L: {L[:10]}", file=sys.stderr)
    fertilities, weights, label_map = estimate_random(L, pairwise=pairwise)
    L = index_annotations(L, label_map)
    #print(f"L: {L[:10]}", file=sys.stderr)
    return L, fertilities, weights, label_map


def calculate_kappa(L, fertilities, weights, weighted, pairwise=False, all_pairs=False):
    if all_pairs:
        pairwise = True

    if not pairwise:
        pe = taxonomy_pe(fertilities, weights, fertilities, weights, weighted=weighted)
        po = observed_agreement(L, weighted=weighted)
        kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else po
        return po, pe, kappa
    else:
        if all_pairs:
            pairs = ((i, j) for i in range(len(L[0])) for j in range(len(L[0])) if i < j)
        else:
            pairs = ((0, i) for i in range(1, len(L[0])))
        pe = []
        po = []
        kappa = []
        
        for i, j in pairs:
            pe.append(taxonomy_pe(fertilities[i], weights[i], fertilities[j], weights[j], weighted=weighted))
            po.append(observed_agreement([[L[x][i], L[x][j]] for x in range(len(L))], weighted=weighted))
            kappa.append((po[-1] - pe[-1]) / (1 - pe[-1]) if (1 - pe[-1]) != 0 else po[-1])
        return sum(po) / len(po), sum(pe) / len(pe), sum(kappa) / len(kappa)


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
    ap.add_argument(
        "input", 
        help="input file name. see above description for input file format."
    )

    ap.add_argument(
        "category_paths", 
        nargs="+",
        help="list of category paths to calculate agreement for. see above description for more details."
    )
    ap.add_argument(
        "--output",
        type=argparse.FileType('w'),
        default=sys.stdout,
        help="output file name",
    )
    ap.add_argument(
        "--input-format",
        type=str,
        choices=["jsonl", "parquet"],
        default="parquet",
        help="input file format"
    )
    ap.add_argument(
        "-a",
        "--annotator-paths", 
        nargs="*", 
        help="list of annotator paths to calculate agreement for."
    )
    ap.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="taxonomy_annotations/parsed",
        help="prefix path in input file lines/rowsto all annotators and their annotations."
    )
    ap.add_argument(
        "--pairwise",
        action="store_true",
        help="by default, fleiss kappa is calculated for all annotators. "
        "when pairwise is set, averge cohen's kappa is calculated between the first annotator and "
        "each other annotator."
    )
    ap.add_argument(
        "--all-pairs",
        action="store_true",
        help="by default, cohen's kappa is calculated between the first annotator and "
        "each other annotator. when all-pairs is set, average cohen's kappa is calculated "
        "for each pair of annotators."
    )
    ap.add_argument(
        "-t", 
        "--top-pick", 
        action="store_true",
        help="by default, overlapping kappa for primary and secondary codes is calculated. "
        "when top-pick is set,only the primary code is used."
    )
    ap.add_argument(
        "-s",
        "--strip-empties",
        action="store_true",
        help="by default, if an annotator has no labels for a category, then it is given an empty label. "
        "with this flag set, this row/line will be removed from calculation for this category.")
    ap.add_argument(
        "--repetitions", 
        type=int, 
        default=100,
        help="number of bootstrap repetitions"
    )
    ap.add_argument(
        "--bootstrap", 
        action="store_true",
        help="calculate standard deviation of kappa using bootstrap.")
    ap.add_argument(
        "--weighted", 
        action="store_true",
        help="use weighted overlap for kappa calculation. instead of match={1,0}, use match=|A^B|/|AvB|")
    ap.add_argument(
        "-T",
        "--transform-mappings",
        type=open,
        help="""
        advanced usage. if you are trying to calculate agreement between two annotators, but the \b
        categories names are different for each annotator, you can supply a yaml file that \b
        remaps paths in your input file. the yaml file should have the following structure:
        - from: $PATH_TO_OLD_CATEGORY_NAME1
            to: $PATH_TO_NEW_CATEGORY_NAME1
        - from: $PATH_TO_OLD_CATEGORY_NAME2
            to: $PATH_TO_NEW_CATEGORY_NAME2
        ...
        transformations are applied relative to $PREFIX and $PREFIX/*.
        """
    )
    args = ap.parse_args()
    if args.all_pairs:
        args.pairwise = True

    if args.transform_mappings:
        transform_mappings = yaml.safe_load(args.transform_mappings)
    else:
        transform_mappings = []

    if args.input_format == "parquet":
        import daft
        daft.context.set_runner_native()
        lst = daft.read_parquet(args.input).to_pylist()
    else:
        import json
        with open(args.input, "r") as f:
            lst = [json.loads(line) for line in f]

    if args.annotator_paths:
        names = args.annotator_paths
        paths = names
    else:
        names = get_all_annotator_names(lst)
        paths = names

    if transform_mappings:
        for item in tqdm(lst, desc="transforming mappings"):
            for p in transform_mappings:
                fr, to = p["from"], p["to"]
                try: 
                    val = fastget(item, f"{args.prefix}/{fr}")
                    fastset(item, f"{args.prefix}/{to}", val)
                    fastdel(item, f"{args.prefix}/{fr}")
                except KeyError:
                    pass
                for key, dct in fastget(item, args.prefix).items():
                    for anno in paths:
                        try:
                            val = fastget(dct, fr)
                            fastset(dct, to, val)
                            fastdel(dct, fr)
                        except KeyError:
                            pass

    padlen = max(len(catpath) for catpath in args.category_paths)

    if args.bootstrap:
        print("label".ljust(padlen), "P_o   ", "P_e   ", "kappa ", "std_kappa", file=args.output, sep="\t", flush=True)
    else:
        print("label".ljust(padlen), "P_o   ", "P_e   ", "kappa ", file=args.output, sep="\t", flush=True)

    for catpath in args.category_paths:
        L, fertilities, weights, label_map = annotations(
            lst,
            paths,
            catpath,
            prefix=args.prefix,
            top_pick=args.top_pick,
            strip_empties=args.strip_empties,
            pairwise=args.pairwise,
        )

        po, pe, kappa = calculate_kappa(L, fertilities, weights, weighted=args.weighted, pairwise=args.pairwise, all_pairs=args.all_pairs)
        if args.bootstrap:
            kappa_values = []
            for _ in tqdm(list(range(args.repetitions)), desc="bootstrapping"):
                bootstrap_L = random.choices(L, k=len(L))
                #print(f"bootstrap_L: {bootstrap_L[:10]}", file=sys.stderr)
                fertilities, weights, indmap = estimate_random(bootstrap_L, pairwise=args.pairwise)
                bootstrap_L = index_annotations(bootstrap_L, indmap)
                #print(f"indmap: {list(indmap.items())[:10]}", file=sys.stderr)

                bootstrap_po, bootstrap_pe, bootstrap_kappa = calculate_kappa(bootstrap_L, fertilities, weights, weighted=args.weighted, pairwise=args.pairwise, all_pairs=args.all_pairs)
                kappa_values.append(bootstrap_kappa)
            kappa_values = sorted(kappa_values)
            stdKappa = np.std(kappa_values, ddof=1)
            print(catpath.ljust(padlen), f"{po:.4f}", f"{pe:.4f}", f"{kappa:.4f}", f"{stdKappa:.4f}", file=args.output, sep="\t", flush=True)
        else:
            print(catpath.ljust(padlen), f"{po:.4f}", f"{pe:.4f}", f"{kappa:.4f}", file=args.output, sep="\t", flush=True)

    
