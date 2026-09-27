"""Why does the default Borda (dims 4-24, euclidean + cosine) behave differently across co-expression networks?

Same factor model as correlation_uase_rate.py (1,000 genes, 29% with no module, 10% switch modules). For each
network, compares the default Borda with euclidean-only, cosine-only, Borda up to the profile-likelihood elbow and
a single dimension at the elbow: AUROC of changed genes, Spearman with the population ranking, and how many
module-less genes reach the top 100. Output in results/correlation_uase_borda.txt.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from node2vec2rank.embedding import uase, select_dimension
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances as cpd
from benchmarks.correlation_uase import loadings, sample, roc_auc_score
from scipy.stats import spearmanr
rng=np.random.default_rng(11)
def net(c, kind):
    c=c.copy()
    if kind=='hdWGCNA': a=((1+c)/2)**10
    elif kind=='abs10': a=np.abs(c)**10
    elif kind=='signed': a=c
    elif kind=='TOM':
        a=((1+c)/2)**10; np.fill_diagonal(a,0); k=a.sum(1); a=(a@a+a)/(np.minimum.outer(k,k)+1-a)
    np.fill_diagonal(a,0); return a
p=1000
b,a,changed,strength=loadings(p,rng,'mixed','switch')
moduleless = strength==0
print('module-less genes:', moduleless.sum(), 'changed:', changed.sum())
popcor=[L@L.T+np.diag(1-(L*L).sum(1)) for L in (b,a)]
D=list(range(4,25,2))
def scores(E, sv):
    dhat=max(select_dimension(sv),2)
    ups=[d for d in range(2,25,2) if d<=dhat] or [2]
    col=lambda d,m: cpd(E[0,:,:d],E[1,:,:d],m)
    return dhat, {
     'borda 4-24 euc+cos (default)': borda_aggregate(np.column_stack([col(d,m) for d in D for m in ('euclidean','cosine')])),
     'borda 4-24 euclidean': borda_aggregate(np.column_stack([col(d,'euclidean') for d in D])),
     'borda 4-24 cosine': borda_aggregate(np.column_stack([col(d,'cosine') for d in D])),
     'borda up to elbow euc+cos': borda_aggregate(np.column_stack([col(d,m) for d in ups for m in ('euclidean','cosine')])),
     'single dim at elbow, euclidean': col(dhat,'euclidean'),
    }
for kind in ('hdWGCNA','TOM','abs10','signed'):
    pop=[net(c,kind) for c in popcor]
    T,psv=uase(pop,24,random_state=0,return_singular_values=True)
    norms=np.linalg.norm(T[0],axis=1)
    print(f'{kind}: population position norm of module-less genes (median) {np.median(norms[moduleless]):.2e} vs module genes {np.median(norms[~moduleless]):.2e}; population elbow {select_dimension(psv)}')
    with np.errstate(all='ignore'):
        _,tsc=scores(T,psv)
    for n in (150,500,2000):
        res={}
        for r in range(3):
            G=[net(np.corrcoef(sample(L,n,rng),rowvar=False),kind) for L in (b,a)]
            E,sv=uase(G,24,random_state=r,return_singular_values=True)
            dhat,sc=scores(E,sv)
            for k,v in sc.items():
                t=tsc[k]
                res.setdefault(k,[]).append((roc_auc_score(changed,v), spearmanr(v[changed],t[changed])[0],
                    np.mean(np.argsort(np.argsort(-v))[moduleless] < 100), dhat))
        for k,v in res.items():
            v=np.array(v,float)
            print(f'  n={n:5d} {k:32s} AUROC {v[:,0].mean():.3f}  spearman-to-pop {np.nanmean(v[:,1]):.2f}  frac module-less in top100 {v[:,2].mean():.3f}  elbow {v[:,3].mean():.1f}',flush=True)
