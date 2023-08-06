import numpy as np
from matplotlib import pyplot
from sklearn import metrics
import seaborn as sns
def time_depend_roc(predict,event,time,Timelist,filename):
    """predict: 风险函数的输出
    event: 事件是否发生
    time: 最后以此观察的时间
    T: 截断时间
    filename: 图像存储文件名
    supress: 是否只输出AUC值"""
    
    pyplot.figure()
    lw = 2
    for i,T in enumerate(Timelist):
        risk = []
        label = []
        for r, e, t in zip(predict,event,time):
            if ((t < T) and (e == 1)) or ((t > T) and (e==0)):
                risk.append(r)
                label.append(e)
        fpr, tpr, thresholds = metrics.roc_curve(label, risk, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        pyplot.plot(
            fpr,
            tpr,
            color=sns.color_palette()[i],
            lw=lw,
            label="ROC curve for T = %.1f(area = %0.3f)" % (T,roc_auc),
        )
    pyplot.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("Receiver operating characteristic")
    pyplot.legend(loc="lower right")
    pyplot.savefig(filename+'.png')
    pyplot.show()
    return 0
