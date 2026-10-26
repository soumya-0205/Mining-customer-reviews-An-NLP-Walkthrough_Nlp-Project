import pandas as pd, numpy as np, warnings, json; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score, brier_score_loss
df=pd.read_csv('data/reviews.csv');df=df[df['Review Text'].notna()].reset_index(drop=True)
df['text']=(df['Title'].fillna('')+'. '+df['Review Text']).str.strip('. ')
df['label']=df['Rating'].map(lambda r:'negative' if r<=2 else 'neutral' if r==3 else 'positive')
R={}
b=df[df.Rating!=3];yb=(b.Rating>=4).astype(int)
Xtr,Xte,ytr,yte=train_test_split(b.text,yb,test_size=.2,random_state=42,stratify=yb)
Xtr3,Xte3,ytr3,yte3=train_test_split(df.text,df.label,test_size=.2,random_state=42,stratify=df.label)
def lr(cw='balanced'): return LogisticRegression(max_iter=2000,C=4,class_weight=cw)
def run(name,vec,clf=None,cw='balanced'):
    out=[]
    for (a,bb,c,d) in [(Xtr,Xte,ytr,yte),(Xtr3,Xte3,ytr3,yte3)]:
        v=vec();m=clf() if clf else lr(cw);m.fit(v.fit_transform(a),c);p=m.predict(v.transform(bb))
        out.append(round(f1_score(d,p,average='macro')*100,1))
    print('%-34s binary F1 %.1f | 3-class F1 %.1f'%(name,out[0],out[1]));R[name]=out
T=lambda **k:(lambda:TfidfVectorizer(min_df=3,sublinear_tf=True,**k))
run('TF-IDF unigram',T())
run('TF-IDF uni+bigram (chosen)',T(ngram_range=(1,2)))
run('TF-IDF uni+bi+trigram',T(ngram_range=(1,3)))
run('Counts uni+bigram',lambda:CountVectorizer(min_df=3,ngram_range=(1,2)))
run('TF-IDF no sublinear',lambda:TfidfVectorizer(min_df=3,ngram_range=(1,2)))
run('TF-IDF + stopword removal',T(ngram_range=(1,2),stop_words='english'))
run('Char n-grams 2-5',lambda:TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=3,sublinear_tf=True,max_features=150000))
run('No class weighting',T(ngram_range=(1,2)),cw=None)
# significance binary LR vs NB
v=TfidfVectorizer(ngram_range=(1,2),min_df=3,sublinear_tf=True);A=v.fit_transform(Xtr);B=v.transform(Xte)
m1=lr().fit(A,ytr);m2=MultinomialNB(alpha=.3).fit(A,ytr);p1=m1.predict(B);p2=m2.predict(B);y=yte.values
c1=p1==y;c2=p2==y;b01=int((c1&~c2).sum());b10=int((~c1&c2).sum())
from scipy.stats import binomtest
print('McNemar LR-right-NB-wrong %d, NB-right-LR-wrong %d, p=%.2g'%(b01,b10,binomtest(b01,b01+b10,.5).pvalue))
rng=np.random.default_rng(0);d=[]
for _ in range(1000):
    i=rng.integers(0,len(y),len(y));d.append(f1_score(y[i],p1[i],average='macro')-f1_score(y[i],p2[i],average='macro'))
print('macroF1 LR-NB diff %.3f 95%% CI [%.3f, %.3f]'%(np.mean(d),*np.percentile(d,[2.5,97.5])))
# ROC / PR / calibration
pr=m1.predict_proba(B)[:,1];print('AUC %.4f  PR-AUC(neg) %.4f  Brier %.4f'%(roc_auc_score(y,pr),average_precision_score(1-y,1-pr),brier_score_loss(y,pr)))
def ece(y,p,n=10):
    bins=np.linspace(0,1,n+1);e=0
    for i in range(n):
        k=(p>=bins[i])&(p<bins[i+1]+(i==n-1))
        if k.sum():e+=k.mean()*abs(y[k].mean()-p[k].mean())
    return e
print('ECE raw %.4f'%ece(y,pr))
fp,mp=calibration_curve(y,pr,n_bins=8,strategy='quantile');print('calib bins (mean pred -> actual):',[(round(a,2),round(b_,2)) for a,b_ in zip(mp,fp)])
cal=CalibratedClassifierCV(lr(),method='isotonic',cv=5).fit(A,ytr);pc=cal.predict_proba(B)[:,1]
print('ECE isotonic %.4f Brier %.4f AUC %.4f'%(ece(y,pc),brier_score_loss(y,pc),roc_auc_score(y,pc)))
# learning curve (binary)
for frac in (.1,.25,.5,.75,1.0):
    n=int(len(Xtr)*frac);vv=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True);mm=lr().fit(vv.fit_transform(Xtr[:n]),ytr[:n])
    print('train %5d: macroF1 %.3f'%(n,f1_score(yte,mm.predict(vv.transform(Xte)),average='macro')))
# top features with coefficients
fn=np.array(v.get_feature_names_out());co=m1.coef_[0]
print('POS',[(fn[i],round(co[i],1)) for i in np.argsort(-co)[:12]])
print('NEG',[(fn[i],round(co[i],1)) for i in np.argsort(co)[:12]])
print('features',len(fn))
json.dump(R,open('ablation.json','w'))
