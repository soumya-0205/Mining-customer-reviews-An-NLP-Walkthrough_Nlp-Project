import pandas as pd, numpy as np, warnings, itertools; warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
df=pd.read_csv('data/reviews.csv');df=df[df['Review Text'].notna()].reset_index(drop=True)
df['text']=(df['Title'].fillna('')+'. '+df['Review Text']).str.strip('. ')
tv=TfidfVectorizer(stop_words='english',min_df=10,max_df=.4,max_features=6000);X=tv.fit_transform(df.text)
cv=CountVectorizer(vocabulary=tv.vocabulary_,stop_words=None);Cn=cv.fit_transform(df.text)
voc=np.array(tv.get_feature_names_out());idx={w:i for i,w in enumerate(voc)}
Bin=(Cn>0).tocsc().astype(np.float32);Nd=Bin.shape[0];df_=np.asarray(Bin.sum(0)).ravel()
def npmi(words,eps=1e-12):
    ii=[idx[w] for w in words];s=[]
    for a,b in itertools.combinations(ii,2):
        co=Bin[:,a].multiply(Bin[:,b]).sum()/Nd;pa=df_[a]/Nd;pb=df_[b]/Nd
        s.append(np.log((co+eps)/(pa*pb))/(-np.log(co+eps)))
    return np.mean(s)
def tops(H,n=10): return [list(voc[np.argsort(-h)[:n]]) for h in H]
def coh(H): return np.mean([npmi(t) for t in tops(H)])
print('K sweep (NMF): K, NPMI coherence')
res={}
for K in (4,6,8,10,12,14):
    m=NMF(K,random_state=0,init='nndsvda',max_iter=400).fit(X);res[K]=coh(m.components_);print(K,round(res[K],3),'recon err %.1f'%m.reconstruction_err_)
# LDA vs NMF at K=8
lda=LatentDirichletAllocation(8,random_state=0,learning_method='batch',max_iter=15,n_jobs=-1).fit(Cn)
nm=NMF(8,random_state=0,init='nndsvda',max_iter=400).fit(X)
print('K=8 NPMI  NMF %.3f  LDA %.3f'%(coh(nm.components_),coh(lda.components_)))
print('LDA top words:');[print(' ',t[:7]) for t in tops(lda.components_)]
# stability across seeds (random init) via best-match Jaccard of top-10 words
def jac(a,b): a,b=set(a),set(b);return len(a&b)/len(a|b)
ref=tops(NMF(8,random_state=0,init='nndsvda',max_iter=400).fit(X).components_)
sc=[]
for sd in range(1,6):
    t=tops(NMF(8,random_state=sd,init='random',max_iter=400).fit(X).components_)
    sc.append(np.mean([max(jac(r,x) for x in t) for r in ref]))
print('seed stability (mean best-match Jaccard, 5 seeds):',np.round(sc,2),'mean %.2f'%np.mean(sc))
# bootstrap stability
rng=np.random.default_rng(1);sc=[]
for _ in range(4):
    i=rng.integers(0,X.shape[0],X.shape[0]);t=tops(NMF(8,random_state=0,init='nndsvda',max_iter=400).fit(X[i]).components_)
    sc.append(np.mean([max(jac(r,x) for x in t) for r in ref]))
print('bootstrap stability: %s mean %.2f'%(np.round(sc,2),np.mean(sc)))
