"""Three more random G1 null splits (seeds 1-3, default merge) for hdWGCNA signed vs signed correlation.

    python benchmarks/cell_cycle_signed_nullsplits.py path/to/Revelio/data
"""
import sys, functools, numpy as np, pandas as pd
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import cell_cycle as cc
import cell_cycle_signed as cs
cells_table, groups, batches, hd = cc.prepare(sys.argv[1], merge='default')
power = hd.keywords['power']
gene_sets = cc.read_gmt(cc.REACTOME); names = pd.Index(list(gene_sets))
is_cc = pd.Series(np.asarray(names.str.contains(cc.CELL_CYCLE)), index=names)
rows=[]
for seed in (1,2,3):
    half = cc.random_halves(batches['G1'], np.random.default_rng(seed))
    for kind in ('hdWGCNA signed','signed correlation'):
        build = functools.partial(cs.correlation_network, kind=kind, power=power)
        row,_ = cs.evaluate(f'null seed {seed}', kind, groups['G1'][half], groups['G1'][~half],
                            (batches['G1'][half], batches['G1'][~half]), build, gene_sets, is_cc)
        rows.append(row)
print(pd.DataFrame(rows)[['comparison','method','called','cell_cycle_called','genes_q<0.1']])
