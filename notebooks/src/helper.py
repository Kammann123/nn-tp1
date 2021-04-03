import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def analyze_variable(data, var):
    
    # Create grid for figures
    fig, axs = plt.subplots(1, 4, figsize=(24, 8))
    fig.suptitle(var, fontsize=20)    
    
    # Plot global histogram
    axs[0].hist(data[var], bins=20)
    axs[0].set_title('Distribución', fontsize=15)
    axs[0].set_xlabel(var)
    axs[0].set_ylabel('Frecuencia')
    
    # Plot diabetic histogram
    axs[1].hist(data[var][data['Outcome'] == 1], bins=24)
    axs[1].set_title('Distribución dado que son diabéticos', fontsize=15)
    axs[1].set_xlabel(var)
    axs[1].set_ylabel('Frecuencia')
    
    # Plot non diabetic histogram
    axs[2].hist(data[var][data['Outcome'] == 0], bins=24)
    axs[2].set_title('Distribución dado que no son diabéticos', fontsize=15)
    axs[2].set_xlabel(var)
    axs[2].set_ylabel('Frecuencia')
    
    # Boxplot
    data.boxplot(column=[var], ax=axs[3])
    axs[3].set_title('Boxplot', fontsize=15)
    #axs[3].set_ylabel(var)
    plt.show()
    
def get_outliers(data, var):
    # Usa criterio de "Outlier Leve"
    # extraído de https://es.wikipedia.org/wiki/Valor_at%C3%ADpico 
    q1 = data[var].quantile(0.25)
    q3 = data[var].quantile(0.75)
    iqr = q3 - q1
    mean = data[var].mean()
    ret = []
    for value in data[var]:
        if value < (q1 - 1.5 * iqr) or value > (q3 + 1.5 * iqr):
            ret.append(value)
    return ret

def remove_outliers(data, var): 
    outliers = get_outliers(data, var)
    for outlier in outliers:
        data[var].replace(outlier, np.nan, inplace=True)
        