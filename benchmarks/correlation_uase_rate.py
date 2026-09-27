"""Does UASE converge on co-expression networks as the number of samples grows?

Checks the entrywise-perturbation argument in the correlation-UASE write-up. Mixed-loading factor
model (benchmarks/correlation_uase.py), 10% of genes switch factors. Networks: hdWGCNA-style signed
((1+cor)/2)^10, TOM of it, unsigned |cor|^10 and plain correlation, all with a zero diagonal.
For each, the dimension at the largest relative eigengap of the population network, plus 5, 12 and
24, and n = 150 to 8000 samples (3 replicates):
- worst per-gene embedding error after Procrustes alignment, relative to the largest population
  position norm (should shrink like sqrt(log p / n) at an eigengap);
- Spearman between the estimated and population distances of the changed genes, per dimension and
  for the default Borda (dims 4-24, euclidean + cosine);
- AUROC of the changed genes by the default Borda.

    python benchmarks/correlation_uase_rate.py [num_genes]
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from node2vec2rank.embedding import uase
from node2vec2rank.model_utils import borda_aggregate, compute_pairwise_distances
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
def procrustes(Y,T):
    u,_,vt=np.linalg.svd(Y.reshape(-1,Y.shape[-1]).T@T.reshape(-1,T.shape[-1])); return Y@(u@vt)
p=int(sys.argv[1]) if len(sys.argv)>1 else 1000
b,a,changed,_=loadings(p,rng,'mixed','switch')
popcor=[L@L.T+np.diag(1-(L*L).sum(1)) for L in (b,a)]
DIMS=list(range(4,25,2))
for kind in ('hdWGCNA','TOM','abs10','signed'):
    pop=[net(c,kind) for c in popcor]
    sv=np.linalg.svd(np.hstack(pop),compute_uv=False)[:40]
    gap=int(np.argmax(sv[1:30]/sv[2:31]))+2
    dims=sorted({gap,5,12,24})
    T=uase(pop,24,random_state=0)
    tgt={d:compute_pairwise_distances(T[0,:,:d],T[1,:,:d],'euclidean') for d in dims}
    tb=borda_aggregate(np.column_stack([compute_pairwise_distances(T[0,:,:d],T[1,:,:d],m) for d in DIMS for m in ('euclidean','cosine')]))
    print(f'{kind}: gap at d={gap} (ratio {sv[gap-1]/sv[gap]:.2f}); ratios at 5,12,24: {sv[4]/sv[5]:.2f} {sv[11]/sv[12]:.2f} {sv[23]/sv[24]:.2f}',flush=True)
    for n in (150,500,2000,8000):
        res={d:[] for d in dims}; rk={d:[] for d in dims}; rb=[]; au=[]
        for r in range(3):
            G=[net(np.corrcoef(sample(L,n,rng),rowvar=False),kind) for L in (b,a)]
            E=uase(G,24,random_state=0)
            for d in dims:
                Y=procrustes(E[:,:,:d],T[:,:,:d]); scale=np.linalg.norm(T[:,:,:d],axis=2).max()
                res[d].append(np.linalg.norm(Y-T[:,:,:d],axis=2).max()/scale)
                e=compute_pairwise_distances(E[0,:,:d],E[1,:,:d],'euclidean')
                rk[d].append(spearmanr(e[changed],tgt[d][changed])[0])
            eb=borda_aggregate(np.column_stack([compute_pairwise_distances(E[0,:,:d],E[1,:,:d],m) for d in DIMS for m in ('euclidean','cosine')]))
            rb.append(spearmanr(eb[changed],tb[changed])[0]); au.append(roc_auc_score(changed,eb))
        print(f'  n={n:5d} sqrt(logp/n)={np.sqrt(np.log(p)/n):.3f} | worst-gene err '+' '.join(f'd{d}:{np.mean(res[d]):.3f}' for d in dims)
              +' | spearman(changed) '+' '.join(f'd{d}:{np.mean(rk[d]):.2f}' for d in dims)+f' borda:{np.mean(rb):.2f} | AUROC borda {np.mean(au):.3f}',flush=True)
