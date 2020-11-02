************************
Visualizing Corona Virus
************************

  **Authors**
  - `Paul Schrimpf *UBC* <https://economics.ubc.ca/faculty-and-staff/paul-schrimpf/>`_
  - `Peifan Wu *UBC* <https://economics.ubc.ca/faculty-and-staff/peifan-wu/>`_

**Prerequisites**

- :doc:`../applications/visualization_rules`

**Outcomes**

- Visualize current data on Corona virus

.. contents:: Outline
    :depth: 2

.. literalinclude:: ../_static/colab_full.raw


Introduction
============

This notebook works with daily data on Covid-19 cases by country and region.

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

We will use data from `Johns Hopkins University Center for Systems
Science and
Engineering <https://github.com/CSSEGISandData/COVID-19>`_ . It is
gathered from a variety of sources and updated daily. JHU CSSE uses
the data for `this interactive
website. <https://coronavirus.jhu.edu/map.html>`_

JHU CSSE has the data on github. It gets updated daily.

There are three csv files containing daily counts of confirmed cases,
recoveries, and deaths for each country (and provinces within some
countries).

.. code-block:: python

    confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    recoveries = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

    confirmed.head()

The data comes in "wide" format with one row per area and different
dates in columns. Let's reshape to "long" format with one country-date
combination per row and counts in a single column.

.. code-block:: python

    ids = ["Province/State","Country/Region", "Lat","Long"]
    confirmed=confirmed.melt(id_vars=ids, var_name="Date", value_name="cases")
    deaths=deaths.melt(id_vars=ids,var_name="Date", value_name="deaths")
    recoveries=recoveries.melt(id_vars=ids,var_name="Date", value_name="recoveries")

We can now merge the confirmed cases, recoveries, and deaths into a
single data frame.

.. code-block:: python

    covid = pd.merge(confirmed, deaths, on=ids.append("Date"), how='outer')
    covid = pd.merge(covid, recoveries, on=ids.append("Date"), how='left')

    covid["Date"] = pd.to_datetime(covid["Date"])
    covid["Province/State"]=covid["Province/State"].fillna("")

We can see the most recent data for Canada with the following.

.. code-block:: python

   covid.groupby("Country/Region").get_group("Canada").groupby("Province/State").tail(1)


Visualization
=============

Let’s make a couple figures showing the evolution of cases in Canada.

.. code-block:: python

    def countryplot(countryname):
        df=covid.groupby("Country/Region").get_group(countryname)
        ax = df.groupby("Date").sum().reset_index().plot(x='Date', y=["cases","deaths","recoveries"], title=f"Covid Cases in {countryname}")
        return(ax)

    countryplot("Canada")

.. code-block:: python

    canada = covid.groupby("Country/Region").get_group("Canada")
    fig, ax = plt.subplots(figsize=(10,6))
    for prov, df in canada.reset_index().groupby("Province/State"):
        df.plot(x="Date",y="cases", ax=ax,label=prov)
    ax.set_title("Cases by Province")

.. exercise::

    Visualize the evolution of cases in other countries and their provinces or states.

    .. code-block:: python

        # your code here


Comparing Confirmed Case Trajectories
-------------------------------------

Corona virus began its spread in different areas at different dates.

To compare the growth trajectory of confirmed cases across countries,
we will plot case counts vs the days since the number of cases in an
area reach 50.

.. code-block:: python

    # Create column with days since cases reached basecount
    basecount=50
    baseday = covid.query(f"cases>={basecount}").groupby(["Country/Region","Province/State"])["Date"].min().reset_index().rename(columns={"Date":"basedate"})
    df = pd.merge(covid, baseday.reset_index(), on=["Country/Region","Province/State"])
    df["dayssince"] = (df["Date"]-df["basedate"]).dt.days;

    # To make the plot readable, we'll just show these countries
    sdf=df[df["Country/Region"].isin(["Canada","China","Iran","Italy","France","US","Korea, South", "Spain", "United Kingdom"])];


.. code-block:: python

    fig, ax = plt.subplots(figsize=(12,8))

    ymax = 3000
    gdf=sdf.groupby(["Country/Region","Province/State"])
    colors = qeds.themes.COLOR_CYCLE
    cmap = dict(zip(sdf["Country/Region"].unique(), colors))
    for k, g in gdf :
        alpha = 1.0 if k[0]=="Canada" or g["cases"].max()>3000 else 0.3
        g.plot(x="dayssince", y="cases", ax=ax, title="Evolution of Confirmed Cases", legend=False, color=cmap[k[0]], xlim=[-1, 21], ylim=[0,ymax], alpha=alpha)
        if g["cases"].max()>=ymax :
            y = ymax*.7
            x = g.query(f"cases<{y}")["dayssince"].max()
            y = g.query(f"dayssince>={x}")["cases"].min()
            ax.annotate(f"{k[1]}, {k[0]}",
            (x, y), rotation=80, color=cmap[k[0]])
        elif g["cases"].max()>=300 or (k[0]=="Canada" and g["cases"].max()>=100):
            x=g["dayssince"].max()
            y=g["cases"].max()
            ax.annotate(f"{k[1]}, {k[0]}", (x,y), color=cmap[k[0]], annotation_clip=True)

    ax.annotate("Other Provinces, China", (15,730),color=cmap["China"])
    ax.set_xlabel(f"days since {basecount} cases")
    ax.set_ylabel("confirmed cases");

From these trajectories, we can get some idea of what might happen in
countries where the epidemic is in its early stages (like Canada) by
looking at what happened in countries where the epidemic is further
along (like China, South Korea, and Italy).

It can be helpful to helpful to look at a similar graph on a log
scale. Differences in logs are approximately equal to growth rates, so
the slopes on the graph below are the growth rates of cases.

.. code-block:: python

    fig, ax = plt.subplots(figsize=(12,8))
    ymax = covid["cases"].max()
    xmax=30
    for k, g in gdf :
        alpha = 1.0 if k[0]=="Canada" or g["cases"].max()>3000 else 0.3
        g.plot(x="dayssince", y="cases", ax=ax, title="Evolution of Confirmed Cases",
               legend=False, color=cmap[k[0]], xlim=[-1, 21], ylim=[0,ymax], alpha=alpha)
        if g["cases"].max()>=ymax :
            y = ymax*.2
            x = g.query(f"cases<{y}")["dayssince"].max()
            y = g.query(f"dayssince>={x}")["cases"].min()
            ax.annotate(f"{k[1]}, {k[0]}",
                        (x, y), rotation=20, color=cmap[k[0]])
        elif g["cases"].max()>=2000 or (k[0]=="Canada" and g["cases"].max()>=100):
            x=min(g["dayssince"].max(),xmax-5)
            y=g.query(f"dayssince>={x}")["cases"].min()
            ax.annotate(f"{k[1]}, {k[0]}", (x,y), color=cmap[k[0]], annotation_clip=True)

    ax.annotate("Other Provinces, China", (15,730),color=cmap["China"])
    ax.set_xlabel(f"days since {basecount} cases")
    ax.set_ylabel("confirmed cases (log scale)");
    ax.set_ylim((basecount,covid["cases"].max()))
    ax.set_xlim((-1,xmax))
    ax.set_yscale('log')

`This Financial Times article
<https://www.ft.com/content/a26fbf7e-48f8-11ea-aeb3-955839e06441>`_
has some more polished figures looking at log cases.

Mortality
---------

We can get an estimate of the mortality rate by taking deaths divided
confirmed cases. If there are many unconfirmed cases, then we will be
overstating the mortality rate.

Medical care, population comorbidities, and testing availability vary
among countries. This can lead to differences in mortality rate per
confirmed case.

Here are the mortality rates per confirmed case in countries with at
least 50 deaths.

.. code-block:: python

    country=covid.groupby(["Country/Region","Date"]).sum()
    latest=country.groupby("Country/Region").last().drop(columns=["Lat","Long"])
    latest['mortalityrate'] = latest['deaths']/latest['cases']
    latest.query('deaths>50')

Below is a scatter plot of deaths vs cases. If the mortality rate were
the same across countries, then the points would all be on a line.

.. code-block:: python

    fig, ax = plt.subplots(figsize=(8, 6))
    latest.plot(x="cases", y="deaths", kind="scatter",title="Confirmed Cases and Deaths", ax=ax)

    for r in latest.query('deaths>=100').itertuples():
        ax.annotate(r.Index, (r.cases, r.deaths))


More Detailed Data
==================

China was the first country hit by COVID-19. After two months of lockdown,
there are almost no new cases anymore. Therefore we might learn more about
the complete COVID-19 epidemic dynamics by looking into Chinese panel data.

JHU CSSE collects data on province/state level for different countries
across the world which forms an unbalanced panel. In particular,
JHU CSSE gathered Chinese data from `DingXiangYuan <https://ncov.dxy.cn/ncovh5/view/en_pneumonia>`_
(DXY, it means "clove garden" in Chinese) starting from Jan 21st.
However, DXY manually collects more detailed prefecture- or city-level
data and some open-source web crawlers are updating these data every half
hour. We can utilize this panel data for further analysis.

.. code-block:: python

    # Use the city-level web crawler data form DingXiangYuan (DXY)
    dt_china = pd.read_csv('https://raw.githubusercontent.com/BlankerL/DXY-COVID-19-Data/master/csv/DXYArea.csv')

    # DXY collects data around the world but we use the Chinese data only
    dt_china = dt_china[dt_china["countryEnglishName"] == "China"]
    # Convert string to datetime
    dt_china["updateTime"] = pd.to_datetime(dt_china["updateTime"])
    # Drop NaN and unknown area reports on city level
    dt_china = dt_china.dropna(subset = ["cityEnglishName"])
    dt_china = dt_china[dt_china["cityEnglishName"] != "Area not defined"]

    dt_china.head()

Since DXY provides both Chinese and English webpages, the scraped data
consists of variables in both languages. We will use the English ones.
There are four variables of main interest: confirmed cases, suspected cases,
cases that are cured, and mortality.

Visualization, continued
========================

For most of the European and North American data we observe the expanding
trajectories. COVID-19 spreads out exponentially. We plot some similar
trajectories with Chinese data. Below we plot a trajectory for all cities
in the same province.

.. code-block:: python

    province_name = "Jiangsu" # For Jiangsu Province, you can change to "Hubei", "Shanghai", etc.
    cur_province = dt_china.groupby("provinceEnglishName").get_group(province_name)
    fig, ax = plt.subplots(figsize=(16,9))
    for city, df in cur_province.reset_index().groupby("cityEnglishName"):
        df.plot(x="updateTime",y="city_confirmedCount", ax = ax, label = city)

    # ax.legend().set_visible(False)
    ax.set_title(f"Confirmed Cases for {province_name}")
    ax.set_xlabel("Dates")
    ax.set_ylabel("Persons")

We can see that the growth rate is high at the beginning, and then the growth
slows down. Eventually those trajectories converge, meaning that there are
no new confirmed cases any more.

Now let's have a look at Hubei province (Wuhan is the capital of Hubei province
and most of the economic activities in Hubei are associated with Wuhan).

.. code-block:: python

    province_name = "Hubei"
    cur_province = dt_china.groupby("provinceEnglishName").get_group(province_name)

    # Use different color to emphasize
    emp_color = (0.95, 0.05, 0.05)
    nom_color = (0.8, 0.8, 0.8)

    fig, ax = plt.subplots(figsize=(16,9))
    for city, df in cur_province.reset_index().groupby("cityEnglishName"):
        if (city == "Wuhan"):
            cur_color = emp_color
        else:
            cur_color = nom_color
        df.plot(x="updateTime",y="city_confirmedCount", ax = ax, label = city, color = cur_color)

    ax.legend().set_visible(False)
    ax.set_title(f"Confirmed Cases for {province_name}")
    ax.set_xlabel("Dates")
    ax.set_ylabel("Persons")

Hubei stands out compared to other provinces from our previous graph, and
Wuhan stands out within Hubei province. Notice that there is a sudden jump
on Feb 13th. Do you know why? Wuhan starts lockdown from late January,
therefore the sudden jump doesn't make much sense.

The reason behind this sudden jump is that the Chinese CDC changes its measure
for confirmed cases. Since COVID-19 lab tests take 2-3 days which might be
slow, the Chinese CDC includes all clinically diagnosed cases (mostly through
Computed Tomography scans) as confirmed cases. This measure overestimates the
confirmed cases by a small amount.

Therefore, a drastic change in data might be some interesting phenomenon,
but it could also be just a change in measurement.

Curve in "Flatten the Curve"
----------------------------

You might have heard the phrase "flatten the curve" frequently these days.
In epidemiology, the idea of slowing a virus' spread so that fewer people
need to seek treatment at any given time, which brings less burden to the
whole medical system. That's why so many countries are implementing
"social distancing" and "shelter in place" orders.

To "flatten the curve", we first analyze the curve itself. Essentially,
the curve characterizes how many confirmed cases that are not cured yet.

.. code-block:: python

    # Plot Wuhan out
    dt_wuhan = dt_china[dt_china["cityEnglishName"] == "Wuhan"]
    fig, ax = plt.subplots(figsize = (16, 9))
    ax.plot(dt_wuhan["updateTime"], dt_wuhan["city_confirmedCount"], label = "Confirmed Cases")
    ax.plot(dt_wuhan["updateTime"], dt_wuhan["city_curedCount"], label = "Cured Cases")

    ax.legend()
    ax.set_title('Cases in Wuhan')
    ax.set_xlabel("Time")

.. code-block:: python

    # Compute current cases
    dt_china["city_current"] = dt_china["city_confirmedCount"] - dt_china["city_curedCount"]

    colors = qeds.themes.COLOR_CYCLE
    cmap = dict(zip(dt_china["cityEnglishName"].unique(), colors * (int(len(dt_china["cityEnglishName"].unique()) / 7) + 1)))
    nom_color = (0.8, 0.8, 0.8)

    fig, ax = plt.subplots(figsize=(16,9))
    # dt_hubei = dt_china.groupby("provinceEnglishName").get_group("Hubei")
    for city, df in dt_china.reset_index().groupby("cityEnglishName"):
        if (city != "Wuhan"):
            if (df["city_current"].max() >= 1000):
                df.plot(x="updateTime",y="city_current", ax = ax, label = city, color = cmap[city])
                y = df["city_current"].max()
                x = df["updateTime"][df["city_current"].idxmax()]
                ax.annotate(f"{city}", (x,y), color=cmap[city], annotation_clip=True)
            else:
                df.plot(x="updateTime",y="city_current", ax = ax, label = city, color = nom_color)

    ax.legend().set_visible(False)
    ax.set_xlabel("Time")


.. exercise::

    Plot the trajectory for all cities other than Wuhan in Hubei province.
    Compare the magnitudes of confirmed cases in Hubei province to other
    provinces (e.g. Jiangsu above)

    .. code-block:: python

        # your code here

References
==========


.. bibliography:: applications.bib
    :cited:
    :labelprefix: covid
    :keyprefix: covid-

Exercises
=========

.. exerciselist::
