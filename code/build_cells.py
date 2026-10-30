import csv, glob, sys, numpy as np, pandas as pd, shapefile
base=sys.argv[1]
def center(k):
    p,u=int(k[:2]),int(k[2:4]); q,v,r,w,m=map(int,k[4:9])
    lat=p/1.5+q*5/60+r*30/3600+(15/3600 if m in(3,4) else 0)+7.5/3600
    lon=u+100+v*7.5/60+w*45/3600+(22.5/3600 if m in(2,4) else 0)+11.25/3600
    return lat,lon
rows=[]
for f in sorted(glob.glob(f'{base}/mesh/*.txt')):
    for r in list(csv.reader(open(f,encoding='cp932')))[2:]:
        lat,lon=center(r[0])
        rows.append(dict(key=r[0],htksyori=r[1],htksaki=r[2],pop=int(r[4]) if r[4].isdigit() else None,
                         pop65=int(r[22]) if r[22].isdigit() else None,hh=int(r[37]) if r[37].isdigit() else None,lat=lat,lon=lon))
c=pd.DataFrame(rows)
# all Niigata facilities as proxy for municipal territory
sh=shapefile.Reader(f'{base}/p20/P20-12_15.shp',encoding='cp932')
fac=pd.DataFrame([dict(city=str(rec[0]),lon=s.points[0][0],lat=s.points[0][1]) for rec,s in zip(sh.records(),sh.shapes())])
def km(lat1,lon1,lat2,lon2):
    p=np.pi/180;a=np.sin((lat2-lat1)*p/2)**2+np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
    return 12742*np.arcsin(np.sqrt(a))
c=c[(c.lat.between(37.15,37.80))&(c.lon.between(138.55,139.20))].copy()
D=km(c.lat.values[:,None],c.lon.values[:,None],fac.lat.values[None],fac.lon.values[None])
c['nearest_city']=fac.city.values[D.argmin(1)]; c['nearest_km']=D.min(1)
ng=c[c.nearest_city=='15202'].copy()
print('cells in box',len(c),'| nearest facility in Nagaoka:',len(ng),'pop',ng['pop'].sum())
print('pop65 missing',ng.pop65.isna().sum(),'flags',ng.htksyori.value_counts().to_dict())
print('nearest_km p50/p95/max',ng.nearest_km.quantile([.5,.95]).round(2).tolist(),round(ng.nearest_km.max(),2))
ng.to_csv(f'{base}/nagaoka_cells_2020.csv',index=False)
