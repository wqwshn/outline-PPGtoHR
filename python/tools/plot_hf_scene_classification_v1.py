"""Render all predeclared exploration views and paired LOSO evidence."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0,r"C:\Users\26541\.codex\skills\nature-figure\scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/experiments/hf_scene_classification_v1/p1_20261005_01"
FIG=OUT/"figures";FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":7,"axes.titlesize":8,"axes.labelsize":7,"xtick.labelsize":6,"ytick.labelsize":6,"legend.fontsize":6,"pdf.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
ARMS=["MIMU","HF","MIMU_HF"];DISPLAY=["MIMU","HF","MIMU + HF"]
ACT=["HW","RS","HG","TYP","RUN","JJ","BUR","PCH"];SUB=[f"subject-{i}" for i in range(1,7)]
BIN=["0%","(0,10%]","(10,25%]",">25%"]
palette=["#4477AA","#EE6677","#228833","#CCBB44","#66CCEE","#AA3377","#BBBBBB","#332288"]
subject_colors=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#777777"]
missing_colors=["#B9C9CF","#599CAE","#B78058","#6D333D"]
method_colors=["#397F96","#A77B79","#B95450"]
qa=[]
def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=FIG/f"{name}.alignment.json",tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    for ext in ("pdf","svg","png"):
        fig.savefig(FIG/f"{name}.{ext}",dpi=600)
    fig.savefig(FIG/f"{name}.preview.png",dpi=150)
    plt.close(fig);qa.append(name)
sm=pd.read_csv(OUT/"exploration_samples.csv");emb=np.load(OUT/"embeddings.npz")
settings=["PCA","tSNE_p15_s20261005","tSNE_p15_s20261006","tSNE_p40_s20261005","tSNE_p40_s20261006"]
for setting in settings:
    fig,axs=plt.subplots(3,3,figsize=(183/25.4,185/25.4),squeeze=False)
    fig.subplots_adjust(left=.10,right=.97,bottom=.20,top=.90,wspace=.28,hspace=.30)
    for i,arm in enumerate(ARMS):
        xy=emb[f"{arm}_{setting}"]
        for j,(field,levels,colors) in enumerate([("activity",ACT,palette),("subject",SUB,subject_colors),("missing_bin",BIN,missing_colors)]):
            ax=axs[i,j]
            for lab,c in zip(levels,colors):
                mask=sm[field]==lab
                ax.scatter(xy[mask,0],xy[mask,1],s=4,c=c,alpha=.7,linewidths=0,rasterized=True)
            ax.set_xticks([]);ax.set_yticks([])
            if j==0:ax.set_ylabel(DISPLAY[i],fontsize=8)
            if i==0:ax.set_title(["Activity","Anonymous subject","Missing sample fraction"][j],pad=9)
            ax.text(-.10,1.02,chr(97+i*3+j),transform=ax.transAxes,fontweight="bold",fontsize=8)
    for j,(levels,colors) in enumerate([(ACT,palette),(SUB,subject_colors),(BIN,missing_colors)]):
        handles=[Line2D([],[],marker="o",linestyle="",color=c,markersize=3,label=l) for l,c in zip(levels,colors)]
        fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.215+j*.303,.155),ncol=2,frameon=False,columnspacing=.6,handletextpad=.4)
    fig.suptitle(setting.replace("_","  "),y=.975,fontsize=9)
    fig.text(.10,.025,"952 windows: 8 non-overlapping windows per record. Exploratory; distances across views are not comparable.",fontsize=6)
    save(fig,f"exploration_{setting}")
scores=pd.read_csv(OUT/"subject_metrics.csv");d=pd.read_csv(OUT/"paired_subject_deltas.csv")
fig,axs=plt.subplots(1,2,figsize=(183/25.4,85/25.4))
fig.subplots_adjust(left=.10,right=.97,bottom=.24,top=.83,wspace=.40)
for _,row in d.iterrows():axs[0].plot([0,1,2],[row.MIMU,row.HF,row.MIMU_HF],color="#BBBBBB",linewidth=.6,zorder=1)
for i,arm in enumerate(ARMS):axs[0].scatter(np.full(6,i),d[arm],s=22,color=method_colors[i],zorder=2)
axs[0].set_xticks([0,1,2],DISPLAY);axs[0].set_ylim(0,1);axs[0].set_ylabel("Weighted macro F1")
axs[0].set_title("Six held-out subjects",pad=10)
axs[1].axhline(0,color="#888888",linewidth=.7)
axs[1].scatter(np.arange(6),100*d.delta,s=25,color=method_colors[2],zorder=2)
axs[1].set_xticks(np.arange(6),[f"S{i}" for i in range(1,7)])
axs[1].set_ylabel("Paired F1 difference (percentage points)")
axs[1].set_title("MIMU + HF minus MIMU",pad=10)
for i,ax in enumerate(axs):ax.text(-.14,1.06,chr(97+i),transform=ax.transAxes,fontweight="bold",fontsize=8)
fig.text(.10,.075,"Each point is one held-out subject; lines pair the same subject. S1-S6 denote subject-1 to subject-6.",fontsize=6)
fig.text(.10,.025,f"Mean paired difference: {100*d.delta.mean():+.3f} percentage points. No window-level significance test.",fontsize=6)
save(fig,"loso_paired_results")
cms=pd.read_csv(OUT/"confusion_matrices.csv")
fig,axs=plt.subplots(1,3,figsize=(183/25.4,78/25.4))
fig.subplots_adjust(left=.10,right=.97,bottom=.26,top=.81,wspace=.30)
for i,arm in enumerate(ARMS):
    cm=cms[cms.arm==arm].groupby(["true_activity","predicted_activity"]).weight.mean().unstack().reindex(index=ACT,columns=ACT).to_numpy()
    cm=cm/cm.sum(axis=1,keepdims=True)
    axs[i].imshow(cm,vmin=0,vmax=1,cmap="Blues",aspect="auto",interpolation="nearest")
    axs[i].set_xticks(np.arange(8),ACT,rotation=45,ha="right",rotation_mode="anchor")
    axs[i].set_yticks(np.arange(8),ACT);axs[i].set_title(DISPLAY[i],pad=10)
    axs[i].set_xlabel("Predicted activity")
    axs[i].text(-.19,1.06,chr(97+i),transform=axs[i].transAxes,fontweight="bold",fontsize=8)
axs[0].set_ylabel("True activity")
fig.text(.10,.035,"Subject-equally weighted confusion matrices, row normalized. Fixed color range: white 0 to dark blue 1.",fontsize=6)
save(fig,"loso_confusion")
(FIG/"figure_contract.json").write_text(json.dumps(dict(backend="Python matplotlib",width_mm=183,scope="Exploratory report; not submitted manuscript figures",units="subjects for primary effects; windows for descriptive embeddings",uncertainty="All six subject observations shown; no CI/p-value claimed",figure_names=qa,embedding_warning="No t-SNE clustering, no map distance/area claims across arms, all four settings delivered"),indent=2),encoding="utf-8")
print(json.dumps(qa))
