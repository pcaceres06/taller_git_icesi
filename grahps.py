import numpy as np
import pandas as pd
import pycircular as pc
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

def circular_graph(data:pd.DataFrame,
                   title:str,
                   time_segment:str='hour'):

    """Función para realizar grafica de variables periodicas

    Arguments
    ---------
    data: dataframe
    title: titulo del gráfico
    time_segment: hour, dayweek, daymonth (default hour)
    
    Output
    -------
    Retorna un grafico
    """    

    time_segment = time_segment
    radians = pc.utils._date2rad(data.fecha_hora, time_segment=time_segment)
    freq_arr, times = pc.utils.freq_time(data.fecha_hora, 
                                         time_segment=time_segment)
    rad_mean, rad_std = pc.stats.periodic_mean_std(radians)

    fig, ax = pc.plots.base_periodic_fig(freq_arr[:,0], 
                                        freq_arr[:,1], 
                                        time_segment=time_segment)
    ax.bar([rad_mean], [1], width=0.1, label='Media Periodica')
    ax.legend(bbox_to_anchor=(-0.3, 0.05), loc='upper left', borderaxespad=0)
    ax.set_title(title, size=14)
    plt.show()
    
    
def biplot(data, loadings, index1, index2, labels=None):
    """
    Función para realizar graficas componentes principales
    """
    plt.figure(figsize=(15, 7))
    xs = data[:,index1]
    ys = data[:,index2]
    n=loadings.shape[0]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        plt.arrow(0, 0, loadings[i,index1], loadings[i,index2],color='r',alpha=0.5)
        if labels is None:
            plt.text(loadings[i,index1]* 1.15, loadings[i,index2] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(loadings[i,index1]* 1.15, loadings[i,index2] * 1.15, labels[i], color='r', ha='center', va='center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(index1))
    plt.ylabel("PC{}".format(index2))
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.grid()
    

def live_plot(data_dict, figsize=(7,5), title='', win_size: int = 100):
    """
    Función para mostrar en tiempo real el progreso de la optmización bayesiana.
    """
    # clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        if len(data) > win_size:
            data = data[-win_size:]
            iterations = np.arange(len(data))[-win_size:] 
        else:
            iterations = np.arange(len(data))
        plt.plot(iterations, data, label=label)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.legend(loc='center left') # the plot evolves to the right
    plt.show()