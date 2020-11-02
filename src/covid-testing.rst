***************************************
US State Corona Virus Testing and Cases
***************************************

  **Authors**
  - `Paul Schrimpf *UBC* <https://economics.ubc.ca/faculty-and-staff/paul-schrimpf/>`_
  - `Peifan Wu *UBC* <https://economics.ubc.ca/faculty-and-staff/peifan-wu/>`_

**Prerequisites**

- :doc:`../applications/regression`
- :doc:`../applications/visualization_rules`
- :doc:`../applications/covid-trends`

**Outcomes**

- Visualize data on Covid test numbers and results

.. contents:: Outline
    :depth: 2

.. literalinclude:: ../_static/colab_full.raw


Introduction
============

This notebook works with daily data on Covid-19 tests and results.

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import (
        linear_model, metrics, neural_network, pipeline, model_selection, preprocessing
    )

    %matplotlib inline
    # activate plot theme
    import qeds
    qeds.themes.mpl_style();


Data
====

We will use data from the `Covid Tracking
Project. <https://covidtracking.com/>`_ This site gathers data
on Covid-19 tests and results in the US. It has information at the
state level. It is updated daily.

.. code-block:: python

    sinfo = pd.read_csv('https://covidtracking.com/api/states/info.csv')
    sdaily = pd.read_csv('https://covidtracking.com/api/states/daily.csv')
    sdaily["date"] = pd.to_datetime(sdaily["date"].map(str),  format="%Y%m%d")


Visualizations
==============

Let's see the total number of tests and their outcomes vs time.

.. code-block:: python

   fig, ax = plt.subplots(2,1, sharex=True)
   colors = qeds.themes.COLOR_CYCLE
   tot = sdaily.groupby("date").sum()
   ax[0].plot(tot.index, tot["total"], color=colors[0])
   i = round(0.7*len(tot.index))
   slope = (np.log(tot["total"][i])-np.log(tot["total"][0]))/i
   angle = np.tan(slope)*360/(2*np.pi)
   ax[0].annotate("Total", (tot.index[i], tot["total"][i]*1.05), color=colors[0], rotation=angle)
   
   ax[0].plot(tot.index, tot["positive"], color=colors[1])
   i = round(0.6*len(tot.index))
   ax[0].annotate("Positive", (tot.index[i], tot["positive"][i]*1.05), color=colors[1], rotation=angle)
   
   ax[0].plot(tot.index, tot["negative"], color=colors[2])
   i = round(0.6*len(tot.index))
   ax[0].annotate("Negative", (tot.index[i], tot["negative"][i]*0.94), color=colors[2], rotation=angle)

   ax[0].plot(tot.index, tot["pending"], color=colors[3])
   i = round(0.7*len(tot.index))
   ax[0].annotate("Pending", (tot.index[i], tot["pending"][i]*1.05), color=colors[3],rotation=angle)

   ax[0].set_yscale('log')
   ax[0].set_ylabel("Count (log scale)")

   n = tot["positive"].diff()+tot["negative"].diff()
   prate = tot["positive"].diff()/n
   se = np.sqrt(prate*(1-prate)/n)
   ax[1].plot(tot.index, prate, color=colors[4])
   ax[1].fill_between(tot.index, prate+1.96*se, prate-1.96*se, alpha=0.2, color=colors[4])
   ax[1].set_ylabel("Daily positive test rate")
   fig.suptitle("Covid-19 testing in the US")
   fig.tight_layout()

And now some state level plots. 

.. code-block:: python
      
    def stateplot(focusstate, var, ax):
        for state, _df in sdaily.groupby("state"):
            df = _df.copy()
            df = df.set_index("date").sort_index()
            if (state==focusstate):
                alpha = 1.0
                color=colors[1]
                i = round(len(df.index)*0.75)
                ax.annotate(focusstate, (df.index[i], df[var].iloc[i]*1.05), color=color, rotation=40)
            else:
                alpha = 0.1
                color="black"                
            ax.plot(df.index, df[var], alpha=alpha, color=color)
            ax.set_yscale('log')
            ax.xaxis.set_tick_params(rotation=60)


    fig, ax = plt.subplots(3,3, sharex=True, sharey=True)
    states=list(sdaily.groupby("state").max().sort_values("positive").tail(9).index)
    i = 0
    for j in range(len(ax)):
        for k in range(len(ax[j])):
            stateplot(states[i], "positive", ax[j][k])
            i = i + 1
    fig.suptitle("Positive Covid-19 tests by state")
    fig.tight_layout()
    
.. code-block:: python
  
    fig, ax = plt.subplots(3,3, sharex=True, sharey=True)
    states=list(sdaily.groupby("state").max().sort_values("positive").tail(9).index)
    i = 0
    for j in range(len(ax)):
        for k in range(len(ax[j])):
            stateplot(states[i], "death", ax[j][k])
            i = i + 1
    fig.suptitle("Deaths by state")
    fig.tight_layout()
    
                
.. code-block:: python

    latest = sdaily.loc[sdaily["date"]==sdaily["date"].max(),:].copy()
    latest.loc[:,"focus"]= latest["state"].map(lambda x: x in states)
    def scatterlabels(xvar, yvar, ax):
        for f, df in latest.groupby("focus"):
            if f :           
                df.plot(xvar,yvar, kind="scatter", color=colors[1], alpha=1, ax=ax)
                for i in df.index:
                    ax.annotate(df.loc[i,"state"], (df.loc[i,xvar], df.loc[i,yvar]), color=colors[1])
            else:
                df.plot(xvar, yvar, kind="scatter", color="black", alpha=0.1, ax=ax)
        return(ax)

    fig, ax = plt.subplots()
    ax = scatterlabels("total","positive",ax)
    ax.set_xlabel('total tests')
    ax.set_ylabel('positive tests')

.. code-block:: python

    fig, ax = plt.subplots()
    ax = scatterlabels("positive","death",ax)
    ax.set_xlabel('positive tests')
    ax.set_ylabel('deaths')
    

.. code-block:: python

    latest['prate'] = latest['positive']/(latest['positive']+latest['negative'])
    latest['prate'].hist(bins=20)
    latest.sort_values('total').tail(9)

    
    


..
   .. exercise::


       .. code-block:: python

           # your code here


   References
   ==========


   .. bibliography:: applications.bib
       :cited:
       :labelprefix: covidtest
       :keyprefix: covidtest-

..
   Exercises
   =========

   .. exerciselist::
