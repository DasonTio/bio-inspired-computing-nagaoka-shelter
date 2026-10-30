import sys, shapefile, pandas as pd
p=sys.argv[1]
for enc in ['cp932','utf-8']:
    try:
        r=shapefile.Reader(p, encoding=enc); recs=r.records(); break
    except Exception as e: print(enc,'fail',e)
fields=[f[0] for f in r.fields[1:]]
print('encoding',enc,'fields',fields, 'n',len(recs))
df=pd.DataFrame([list(x) for x in recs],columns=fields)
pts=[s.points[0] for s in r.shapes()]
df['lon']=[q[0] for q in pts]; df['lat']=[q[1] for q in pts]
print(df.head(3).to_string())
ng=df[df['P20_001'].astype(str).str.startswith('15202')]
print('\nNagaoka rows',len(ng))
for c in ['P20_004','P20_007']: print(ng[c].value_counts().head(15))
cap=pd.to_numeric(ng['P20_005'],errors='coerce')
print('capacity: missing/zero',(cap.isna()|(cap<=0)).sum(),'sum',cap.sum(),'median',cap.median(), 'max',cap.max())
print('area nonempty', (ng['P20_006'].astype(str).str.strip()!='').sum(), ng['P20_006'].head(10).tolist())
print('lat range',ng.lat.min(),ng.lat.max(),'lon',ng.lon.min(),ng.lon.max())
ng.to_csv(sys.argv[2],index=False,encoding='utf-8-sig')
