
*************************
Covid Spread and Movement
*************************

  **Authors**
  - `Paul Schrimpf *UBC* <https://economics.ubc.ca/faculty-and-staff/paul-schrimpf/>`_

**Prerequisites**

- :doc:`../applications/visualization_rules`
- :doc:`../applications/maps`
- :doc:`../applications/regression`
- :doc:`../applications/covid-trends`


**Outcomes**

- 

.. contents:: Outline
    :depth: 2

.. literalinclude:: ../_static/colab_full.raw

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from sklearn import (
        linear_model, metrics, neural_network, pipeline, model_selection, preprocessing
    )
    import datetime
    import requests
    import os
    from urllib.request import urlopen
    import json
    import wget
    from IPython.display import HTML
   
    %matplotlib inline
    # activate plot theme
    import qeds
    qeds.themes.mpl_style();
    import plotly.express as px
  
   
            
   
Introduction
============


Data
====

Case and death data
-------------------

We will use case and death numbers by county from JHU CSSE.

.. code-block:: python

    confirmed = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv?raw=true')
    deaths = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv?raw=true')
    ids = ["UID","iso2","iso3","code3","FIPS","Admin2","Province_State","Country_Region", "Lat", "Long_", "Combined_Key"]
    
    confirmed=confirmed.melt(id_vars=ids, var_name="Date", value_name="cases")
    deaths=deaths.melt(id_vars=ids + ["Population"],var_name="Date", value_name="deaths")
    covid = pd.merge(confirmed, deaths, on=ids + ["Date"], how='outer')
    covid["Date"] = pd.to_datetime(covid["Date"])
    covid["FIPS"] = covid["FIPS"].map(lambda x : -1 if np.isnan(x) else int(x))

    
PlaceIQ movement data
---------------------

Code to download PlaceIQ data on movement from
https://github.com/COVIDExposureIndices/COVIDExposureIndices
Each day of data uses about 6MB of disk space. 

.. code-block:: python
 
    datadir = './lex_data'
    if not os.path.isdir(datadir):
        os.mkdir(datadir)

    # download data if needed
    start = datetime.date(2020,1,20)
    end = datetime.date.today()
    def downloadlex(dates) :
        for day in dates : 
            filename = 'county_lex_' + day.strftime("%Y-%m-%d") + '.csv.gz'
            lfile = datadir+'/'+filename
            if not os.path.isfile(lfile) :
                try : 
                    url = 'https://github.com/COVIDExposureIndices/COVIDExposureIndices/blob/master/lex_data/' + filename + '?raw=true'
                    r = requests.get(url)
                    if r.status_code == 200 : 
                        with open(lfile, 'wb') as f:
                            f.write(r.content)
                    else :
                        print(filename + f" returned {r.status_code}")
                except :
                    print("Failed to download " + filename)
                    None
        None
        
    # read data into memory
    def loadlexdata(dates) :
        dflist = list()
        for day in dates :
            print(f"working on {day}")
            filename = datadir+'/county_lex_' + day.strftime("%Y-%m-%d") + '.csv.gz'     
            if os.path.isfile(filename):            
                df = pd.read_csv(filename, compression='gzip', header=0)
                df["date"] = day
                df = df.melt(id_vars=['COUNTY_PRE','date'], var_name='COUNTY', value_name='LEX')
                df["COUNTY"] = df["COUNTY"].astype(int)
                dflist.append(df)
        lexdf = pd.DataFrame().append(dflist)
        lexdf.sort_values('date', inplace=True)
        return(lexdf)

    # Just load in one days of data
    downloadlex([start])
    lex = loadlexdata([start])
    lex.info()
    
Notice the size of the location data. We only loaded the one oldest
day of data. If you try running this code on the all days until today
we will need about

.. code-block:: python

    155*(datetime.date.today()-start).days

MB of memory. To actually analyze the data, it will be useful to have
considerably more memory. When faced with such a situation, the best
option is usually to simply find a machine with enough memory. The
extra programming time that it will take to deal with memory
constraints is usually far more costly than buying or renting a larger
server.

Nonethless, let's suppose we cannot use a server with more memory for
some reason. There are a number of packages and frameworks for dealing
with data that cannot fit into RAM. A simple approach that will work
for our purposes, is to just operate on each day of data, aggregate to
a smaller dataset of what we need, and then combine.

Movement Index
--------------

The movement data should have number of counties squared times number
of days of observations. The main variable is "LEX."  On each day and
for each pair of countains, "LEX" is the share of devices in county
"COUNTY" on "date" that were in county "COUNTY_PRE" anytime in the
past 14 days (not including 'date'). Rather than keeping all roughly
2000 times 2000 of these values for every day, let's just keep summary measures.

In particular, we will keep "own_lex" which is the "LEX" for "COUNTY"
and "COUNTY_PRE" being the same. One minus "own_lex" is then the
portion of devices in a county on a given data, that were **not** in
the county at all in the past 14 days. In other words, 1-"own_lex" is
the portion of devices in a county that are showing there for the time
in two weeks.

The other measure that we will keep is

$$
sum\_other\_lex_{COUNTY} \equiv \sum_{COUNTY_PRE \neq COUNTY} LEX
$$

This is the sum across all other counties of the portion of devices in
"COUNTY" on a given date, appeared in a given other county in the past
two weeks. Since a single device can visit multiple counties in the
past two weeks, this sum can be greater 1. A bigger number means more
devices visited more different counties in the past two weeks.

.. code-block:: python

   def loadlexdata_aggregate(dates) :
        dflist = list()
        for day in dates :
            print(f"working on {day}")
            filename = datadir+'/county_lex_' + day.strftime("%Y-%m-%d") + '.csv.gz'     
            if os.path.isfile(filename):            
                df = pd.read_csv(filename, compression='gzip', header=0)
                df["date"] = day
                df = df.melt(id_vars=['COUNTY_PRE','date'], var_name='COUNTY', value_name='LEX')
                df["COUNTY"] = df["COUNTY"].astype(int)
                own = df.query("COUNTY==COUNTY_PRE").copy()
                own.rename(columns={"LEX":"own_lex"}, inplace=True)
                own.drop(columns="COUNTY_PRE", inplace=True)
                other = df.query("COUNTY!=COUNTY_PRE").groupby("COUNTY").sum()
                other.reset_index(inplace=True)
                other.drop(columns="COUNTY_PRE", inplace=True)
                other.rename(columns={"LEX":"sum_other_lex"}, inplace=True)
                own=own.merge(other, on="COUNTY")
                dflist.append(own)
        lexdf = pd.DataFrame().append(dflist)
        lexdf.sort_values('date', inplace=True)
        return(lexdf)

   if (False) :
       downloadlex(pd.date_range(start, datetime.date.today()))
       lex = loadlexdata_aggregate(pd.date_range(start,datetime.date.today()))
       lex.to_pickle("lex_aggregate.wdi.gz")
   else :
       url = "https://github.com/ubcecon/ECON323_2020/blob/master/extra_notebooks/lex_aggregate.wdi.gz?raw=true"
       wget.download(url)
       lex = pd.read_pickle("lex_aggregate.wdi.gz")

       
Note that the code as written only downloads a pre-aggregated cache of
data. If you want to rerun the data aggregation code, you should
change False to True in the code above. 

Let's look at some exploratory figures for the movement index data.

.. code-block:: python

    lex.plot("own_lex","sum_other_lex", kind="scatter")


An animated plot showing how the relationship between our two summary
measures of movement changes with time.
                
.. code-block:: python

    from matplotlib.animation import FuncAnimation

    def animated_scatter(df, x, y, t):
        fig, ax = plt.subplots(2, 1,  gridspec_kw={'height_ratios': [5, 1]})
        ts = df[t].unique()
        ts.sort()
        ax[0].set_xlabel(x)
        ax[0].set_ylabel(y)
        scat=ax[0].scatter(df[x], df[y])
        line,=ax[1].plot(df[t], np.linspace(0,1,len(df[t])))
        ax[1].set_xlabel(t)
        ax[1].yaxis.set_visible(False)
        plt.setp(ax[1].spines.values(), visible=False)

        
        def init():
            sdf = df.loc[df[t]==ts[0],:]
            scat.set_offsets(np.array(sdf[[x,y]]))
            line.set_data([ts[0],ts[0]],[0,1])
            return scat,line

        def animate(tval):
            sdf = df.loc[df[t]==tval,:]            
            scat.set_offsets(np.array(sdf[[x,y]]))
            line.set_data([tval,tval],[0,1])
            return scat,line

        fig.tight_layout()
        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=ts, interval=200, blit=True)
        return(anim)

    anim=animated_scatter(lex,"own_lex","sum_other_lex", "date")
    HTML(anim.to_html5_video())
                         
Hmm, I'm not sure that animation is particularly successful, but it at
least illustrates another matplotlib feature.

Let's look at averages over time. So that both movement measures
increase with more movement, let's define "new_visits" = 1 - "own_lex"
and look at that instead of "own_lex".

.. code-block:: python

    fig, ax = plt.subplots(2,1)
    colors = qeds.themes.COLOR_CYCLE
    lex["new_visits"] = 1 - lex["own_lex"]
    avg=lex.groupby("date").mean().reset_index()
    avg.plot("date","new_visits", ax=ax[0], color=colors[0])
    avg.plot("date","sum_other_lex", ax=ax[1], color=colors[1])

We can see some cyclicality related to weekends.
    
                
Mapping
=======

To further explore the data, let's map it. We'll use plotly.express
for mapping this time.

.. code-block:: python

    # download some map data
    import plotly.express as px
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

        

First a map of current cases in each county.

.. code-block:: python
   
    # get only the most recent day
    covid = covid.sort_values(by='Date')    
    df = covid.drop_duplicates('FIPS', keep='last')
    
    df["log10cases"] = np.log10(df["cases"]+0.01)
    
    fig = px.choropleth(df, geojson=counties, locations='FIPS', color='log10cases',
                        color_continuous_scale="thermal",
                        range_color=(0, df['log10cases'].max()),
                        scope="usa",
                        labels={'log10cases':'log10(cases)'},
                        hover_name="Admin2",
                        hover_data=["cases","deaths"],
                        title=f"Confirmed cases as of {df.Date.max().strftime('%Y-%m-%d')}")
    fig.show()

We can see that county level data is missing from a few states.     

    
A map of movement indices. First from January 20th (the earliest date available).               

    
.. code-block:: python
    
    # get only the most recent day
    df = lex.drop_duplicates('COUNTY', keep='first')
       
    fig = px.choropleth(df, geojson=counties, locations='COUNTY', color='sum_other_lex',
                        color_continuous_scale="Viridis",
                        #range_color=(0, df['log10cases'].max()),
                        scope="usa",
                        #labels={'log10cases':'log10(cases)'},
                        hover_name="COUNTY",
                        hover_data=["sum_other_lex", "own_lex"],
                        title=f"Index of movement across counties on {df.date[1].strftime('%Y-%m-%d')}")
    fig.show()

    
    
.. code-block:: python
    
    # get only the most recent day
    df = lex.drop_duplicates('COUNTY', keep='last')
       
    fig = px.choropleth(df, geojson=counties, locations='COUNTY', color='sum_other_lex',
                        color_continuous_scale="Viridis",
                        #range_color=(0, df['log10cases'].max()),
                        scope="usa",
                        #labels={'log10cases':'log10(cases)'},
                        hover_name="COUNTY",
                        hover_data=["sum_other_lex","own_lex"],
                        title=f"Index of movement across counties on {df.date[1].strftime('%Y-%m-%d')}")
    fig.show()


Relationship between mobility and cases
=======================================

.. code-block:: python

    lex = lex.merge(covid, left_on=["COUNTY", "date"], right_on=["FIPS","Date"], how="outer");

    # we only need a single county and date variable. Drop the extra to avoid confusion
    ms=pd.isna(lex["Date"])
    lex.loc[ms,"Date"] = lex.loc[ms, "date"]
    lex = lex.drop(columns="date")
    ms=pd.isna(lex["FIPS"])
    lex.loc[ms,"FIPS"] = lex.loc[ms, "COUNTY"]
    lex = lex.drop(columns="COUNTY")
    lex["FIPS"]=lex["FIPS"].map(lambda x: "{:05.0f}".format(x))
    

Let's now explore the relationship betwen mobility and cases.

.. code-block:: python

    fig, ax = plt.subplots(1,2)
    colors = qeds.themes.COLOR_CYCLE    
    markersize=1
    ax[0].scatter(lex.new_visits, np.log10((lex.cases+0.9)), color=colors[0], s=markersize)
    ax[0].set_ylabel("log_10(cases)")
    ax[0].set_xlabel("new visit index")
    ax[1].scatter(lex.sum_other_lex, np.log10((lex.cases+0.9)), color=colors[1], s=markersize)
    ax[1].set_ylabel("log_10(cases)")
    ax[1].set_xlabel("sum other lex")
    fig.suptitle("Movement indices and cases")

From this, it appears that there's a *negative* relationship between
movement and cases.  However, this plot includes all dates. It is
combining the upward trend in cases and downward trend in movement
with whatever the cross-sectional relationship between cases and
movement on a given date might be.

.. code-block:: python

    lex["log10cases"] = np.log(lex["cases"]+0.9)
    anim=animated_scatter(lex,"new_visits","log10cases", "Date")
    HTML(anim.to_html5_video())

.. code-block:: python

    anim=animated_scatter(lex,"sum_other_lex","log10cases","Date")
    HTML(anim.to_html5_video())

From this we see that even conditional on date, there remains a
negative correlation between these movement indices and county. This
is a bit puzzling. There are two important confounding factors.

One is that most measures of movement, including these, are negatively
correlated with population density. People in dense cities rarely have
to cross county boundaries to go shopping or to work. People in rural
areas often do. Density will tend to be positively related to case
numbers.

The second confounder is that since there is some delay between
infections occurring and them showing up in confirmed case numbers, we
should expect not current movement, but past movement to increase
current case numbers.

Related to the second point, we can see in these animations that the
points tend to drift up and left. As current confirmed cases increase,
movement tends to decrease.

Thus, if we want to use this movement data to predict cases, we should :

1. Control for density and perhaps other county characteristics.

2. Look at lagged movement.


County Population Density
-------------------------

We download data on county area and population from the US Census Bureau.

.. code-block:: python

    # download county population data
    import wget
    filename = "co-est2019-alldata.csv"
    if not os.path.isfile(filename) :
        url = "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv"
        wget.download(url, filename)
                    
    cpop=pd.read_csv(filename, encoding="iso-8859-1")

    cpop["countyFIPS"] = (cpop["STATE"].map(lambda x: "{:02d}".format(x)) +
                          cpop["COUNTY"].map(lambda x: "{:03d}".format(x)))
    cpop["population"] = cpop["POPESTIMATE2019"]
    
    # get county areas
    import geopandas as gpd
    areas = gpd.read_file("http://www2.census.gov/geo/tiger/GENZ2016/shp/cb_2016_us_county_5m.zip")
    cpop = cpop.merge(areas,left_on="countyFIPS", right_on="GEOID", how="outer", validate="1:1")
    cpop["density"] = cpop["population"]/cpop["ALAND"]*1000*1000

    lex=lex.merge(cpop[["countyFIPS","density","population","ALAND"]],
                  left_on="FIPS", right_on="countyFIPS", how="left");

    import seaborn as sns
    lex["logdensity"] = np.log10(lex["density"])
    sns.pairplot(lex, vars=["log10cases", "new_visits", "sum_other_lex", "logdensity"])

    
From the last column we see that density is positively correlated with
cases and negatively correlated with movement.

Regressions
-----------

We will use regression to examine the relationship between cases and
movement conditional on density.

The delay between infection and showing in case numbers is uncertain,
so will regress case growth on many lags of our movement measures. To
reduce colinearity, and to eliminate the weekly cyclicality in
movement, we'll create weekly sums of the movement indices.

.. code-block:: python

    lex=lex.reset_index().set_index(["FIPS","Date"])

    # predict growth rate in cases
    y = lex["log10cases"] - lex.groupby("FIPS")["log10cases"].shift(1)
    def createx(lags, vars=["new_visits","sum_other_lex"]):
        X = lex[["logdensity"]].copy()
        X["constant"] = 1
        t0=lex.reset_index()["Date"].min()
        X["t"] = np.array((lex.reset_index()["Date"]-t0).dt.days)
        X["log10casesL1"]=lex.groupby("FIPS")["log10cases"].shift(1)
        for l in lags:
            week, day = np.divmod(l,7)
            for v in vars:
                if day==0 : 
                    X["{}LW{:02d}".format(v,week)] = lex.groupby("FIPS")[v].shift(l)
                else :
                    X["{}LW{:02d}".format(v,week)] = (X["{}LW{:02d}".format(v,week)] +
                                                       lex.groupby("FIPS")[v].shift(l))
        return(X)
    
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    X = createx(range(7,36), vars=["sum_other_lex"])
    reg=sm.OLS(y, X, missing='drop').fit()
    reg.summary()
    
.. code-block:: python

    X = createx(range(7,36), vars=["new_visits"])
    reg=sm.OLS(y, X, missing='drop').fit()
    reg.summary()
