from pathlib import Path
import json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
reports=[('Initial routes\nbbc54c3', 'search_final/test.json'),('Lane reservation\na7e420d', 'search_verified/test.json'),('Direct KIT route\nf506bc2', 'search_direct_west/test.json')]
colors=['#aab4be','#aab4be','#1c7c82']
fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.05,1]})
fig.patch.set_facecolor('#faf8f3')
for ax in axes: ax.set_facecolor('#faf8f3'); ax.spines[['top','right']].set_visible(False)
for i,(label,name) in enumerate(reports):
 d=json.loads((ROOT/'output/best_design'/name).read_text()); rows=d['episodes']; n=len(rows); k=sum(r['success'] for r in rows); p=k/n; z=1.96
 center=(p+z*z/(2*n))/(1+z*z/n); half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
 axes[0].bar(i,p*100,color=colors[i],width=.55)
 axes[0].errorbar(i,p*100,yerr=[[(p-center+half)*100],[(center+half-p)*100]],fmt='none',ecolor='#29343b',capsize=5)
 axes[0].text(i,(center+half)*100+2,f'{k}/{n}',ha='center',fontsize=11,fontweight='bold')
axes[0].set(xticks=range(len(reports)),xticklabels=[r[0] for r in reports],ylim=(0,108),ylabel='Complete 160-point missions (%)',title='Development tests, each at its recorded source')
last=json.loads((ROOT/'output/best_design/search_direct_west/test.json').read_text()); times=[r['declaration_seconds'] for r in last['episodes'] if r['success']]
axes[1].hist(times,bins=16,color='#1c7c82',edgecolor='#faf8f3'); axes[1].axvline(120,color='#b05234',ls='--',lw=1.5,label='120 s limit')
axes[1].set(xlabel='Declaration time (simulated seconds)',ylabel='Successful missions',title='297 successful missions, f506bc2'); axes[1].legend(frameon=False)
fig.suptitle('KIT route correction removes the repeated crossing',x=.05,ha='left',fontsize=17,fontweight='bold')
fig.text(.05,.02,'Native MuJoCo on G4. Error bars: Wilson 95% intervals. Failed missions remain in success rates; timing uses successes only.\nThese tests informed subsequent recovery changes. They are development evidence for the final controller.',fontsize=9,color='#49555e')
fig.tight_layout(rect=(.025,.10,.99,.91)); out=ROOT/'output/best_design/development_reliability.png';fig.savefig(out,dpi=160,facecolor=fig.get_facecolor());print(out)
