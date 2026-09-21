"""Schematic: keep score aggregation, optional gating and audit paths explicit."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'_FINAL_Latex').exists() or (p/'results').exists())


def main():
    fig,ax=plt.subplots(figsize=(8.0,4.8));ax.set(xlim=(0,12),ylim=(0,7.2));ax.axis('off')
    def box(x,y,w,h,t,c='#e3edf4',size=9):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',fc=c,ec='#46515a',lw=1))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=size)
    def arrow(a,b,dash=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#46515a',
                                    linestyle='--' if dash else '-',lw=1.2))
    box(.15,3.4,2.3,1.65,'Frozen predictions\nPrimary + flip\n+ secondary')
    box(3.0,5.4,2.5,1.15,'Score averaging\nPrimary ranking baseline','#dcebdc')
    box(3.0,2.9,2.5,1.6,'Complementary evidence\nAgreement / stability\nDisagreement / count\nPrior conflict (SDCA)','#dcebdc',8.5)
    box(3.0,.65,2.5,1.1,'Fixed lesion-pair prior\nDirect lookup; no GAT','#f7e7ca',8.5)
    box(6.2,3.0,2.3,1.4,'Optional MLP gate\nEvidence + raw score\nor averaged score','#f7e7ca',8.5)
    box(9.15,5.3,2.55,1.25,'Selected ranker\nRetain / suppress','#eee1ef',9)
    box(9.15,2.8,2.55,1.4,'Prior-defined flag\nFour tiers + certificate\nNo clinical guarantee','#eee1ef',8.5)
    box(6.1,.5,2.55,1.2,'Exploratory anatomy\nAPS = 0.5 in primary\nExternal model: ablation','#f3f3f3',8)
    arrow((2.45,4.8),(3.0,5.75));arrow((2.45,3.95),(3.0,3.75))
    arrow((4.25,1.75),(4.25,2.9));arrow((5.5,3.7),(6.2,3.7))
    arrow((5.5,5.95),(9.15,5.95));arrow((5.5,5.45),(6.6,4.4))
    arrow((8.5,4.05),(9.15,5.4));arrow((10.4,5.3),(10.4,4.2))
    ax.plot([4.25,4.25,10.4],[.65,.15,.15],color='#46515a',lw=1.2)
    arrow((10.4,.15),(10.4,2.8));arrow((7.35,1.7),(7.35,3.0),True)
    ax.text(7.25,6.35,'Averaging can bypass the gate',ha='center',fontsize=8,color='#46515a')
    ax.text(7.5,2.2,'optional',ha='left',fontsize=7,color='#666')
    fig.tight_layout()
    for folder in [ROOT/'journal/figures',ROOT/'_FINAL_Latex/figures']:
        folder.mkdir(parents=True,exist_ok=True)
        for ext in ['png','pdf']:fig.savefig(folder/f'fig1_architecture.{ext}',dpi=220,bbox_inches='tight')
    print('Rendered ranking/audit architecture.')


if __name__=='__main__':main()
