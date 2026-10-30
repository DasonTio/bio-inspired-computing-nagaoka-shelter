import numpy as np, pandas as pd, sys
base=sys.argv[1]
c=pd.read_csv(f'{base}/nagaoka_cells_2020.csv'); s=pd.read_csv(f'{base}/nagaoka_shelters_raw.csv')
s['area']=pd.to_numeric(s.P20_006,errors='coerce'); s['cap_off']=pd.to_numeric(s.P20_005,errors='coerce')
s=s[(s.P20_004!='信濃川等のはんらん時避難場所')&(s.area>0)].copy()
s['cap35']=(s.area/3.5).astype(int)
def km(a,b,c_,d):
    p=np.pi/180;x=np.sin((c_-a)*p/2)**2+np.cos(a*p)*np.cos(c_*p)*np.sin((d-b)*p/2)**2
    return 12742*np.arcsin(np.sqrt(x))
D=km(c.lat.values[:,None],c.lon.values[:,None],s.lat.values[None],s.lon.values[None])*1.3  # road detour factor
near=D.argmin(1); load=np.bincount(near,weights=c['pop'].values,minlength=len(s))
print('shelters',len(s),'total cap35',s.cap35.sum(),'official cap',int(s.cap_off.sum()),'pop',c['pop'].sum())
for name,cap in [('3.5 m2/person',s.cap35.values),('official 2012 capacity',s.cap_off.values)]:
    over=np.maximum(load-cap,0)
    print(f'\n[{name}] overloaded shelters: {(load>cap).sum()} of {len(s)} | people over capacity: {int(over.sum())} ({over.sum()/c["pop"].sum():.1%})')
    r=load/np.maximum(cap,1); print('  load/capacity  p50 %.2f  p90 %.2f  max %.1f'%(np.median(r),np.quantile(r,.9),r.max()))
d=D[np.arange(len(c)),near]; w=c['pop'].values
print('\nwalk km (pop-weighted, x1.3 detour): mean %.2f  p90 %.2f  max %.2f'%(np.average(d,weights=w),np.quantile(np.repeat(d,w.astype(int)),.9),d.max()))
print('demand / cap35 ratio: %.2f'%(c['pop'].sum()/s.cap35.sum()))
