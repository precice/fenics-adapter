---
title: The FEniCS adapter
permalink: adapter-fenics.html
keywords: adapter, fenics
summary: "A general adapter for the open source computing platform FEniCS"
---

## What is FEniCS

From the FEniCS website: _FEniCS is a popular open-source (LGPLv3) computing platform for solving partial differential equations (PDEs). FEniCS enables users to quickly translate scientific models into efficient finite element code. With the high-level Python and C++ interfaces to FEniCS, it is easy to get started, but FEniCS offers also powerful capabilities for more experienced programmers. FEniCS runs on a multitude of platforms ranging from laptops to high-performance clusters._ More details can be found at [fenicsproject.org](https://fenicsproject.org/).

## How to get FEniCS

You can install FEniCS on your system by several ways as mentioned on [fenicsproject.org](https://fenicsproject.org/download/). The simplest way to install FEniCS on Ubuntu is using the official PPA repository of FEniCS:

```bash
sudo apt install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install --no-install-recommends fenics
```

## Aim of this adapter

This adapter supports the Python interface of FEniCS and offers an API that allows the user to use FEniCS-style data structures for solving coupled problems. We provide usage example for the adapter for heat transport, conjugate heat transfer and fluid-structure interaction. However, the adapter is designed in a general fashion and can be used to couple any code using the FEniCS library.

## How to install the adapter

The adapter requires FEniCS and preCICE version 2.0 or greater and the preCICE language bindings for Python.

### Use `pip`

The adapter is [published on PyPI](https://pypi.org/project/fenicsprecice/). After installing preCICE and the python language bindings, run `pip3 install --user fenicsprecice` to install the adapter via your Python package manager.

### Use `conda`

You can alternatively use `conda` to install the adapter. Please refer to [`conda-forge/fenicsprecice`](https://anaconda.org/conda-forge/fenicsprecice) for installation instructions. Bonus, if you use `conda`: You don't have to worry about the dependencies, because `conda` takes care of this for you.

### Something special?

Please refer to the installation instructions provided [here](https://github.com/precice/fenics-adapter#installing-the-package) for alternative installation procedures.

## Examples for coupled codes

The following tutorials can be used as a usage example for the FEniCS adapter:

* Solving the heat equation in a partitioned fashion (heat equation solved via FEniCS for both participants)
* Flow over plate (heat equation solved via FEniCS for solid participant)
* Perpendicular flap (structure problem solved via FEniCS)
* Cylinder with flap (structure problem solved via FEniCS)
* Solving a chemical reaction process in a flow simulation (both reaction-advection-diffusion and fluid flow solved via FEniCS)

For more details please consult the references given in the [reference section](#related-literature).

## How can I use my own solver with the adapter

The FEniCS adapter does not couple your code out-of-the-box, but you have to call the adapter API from within your code. You can use the tutorials from above as an example. The API of the adapter and the design is explained and usage examples are given in the [reference paper](#how-to-cite).

## You need more information?

Please don't hesitate to ask questions about the FEniCS adapter on [discourse](https://precice.discourse.group/) or in [Matrix]({{ site.matrix_url }}).

## How to cite

If you are using our adapter, please consider citing our paper:

{% for pub in site.publications %}

{% if pub.title == "FEniCS–preCICE: Coupling FEniCS to other simulation software" %}

<div class="row">
<div class="col-md-10 col-md-offset-1">
  <div class="panel panel-primary panel-precice">
    <div class="panel-heading-precice">
      <strong>{{ pub.title }}</strong>
    </div>
    <div class="panel-body">
      <p>
        <em>{{ pub.authors }}</em>,
        {{ pub.journal.name }},
        Volume {{ pub.journal.volume }},
        {{ pub.journal.publisher }},
        {{ pub.year }},
        <a href="https://www.doi.org/{{pub.doi}}">doi:{{pub.doi}}</a>.
      </p>
      <a href="{{pub.pub-url}}">Publisher's site</a>&nbsp;&nbsp;
      <a href="assets/{{ pub.bibtex }}">Download BibTeX &nbsp;<i class="fas fa-download"></i></a>
    </div>
  </div>
</div>
</div>

{% endif %}

{% endfor %}

## Related literature

{% assign years = site.publications | group_by:"year" | sort:"title" %}
{% for year in years reversed %}

<div class="row">
{% for pub in year.items %}
{% if pub.tag contains "fenics" %}
<div class="col-md-10 col-md-offset-1">
  <div class="panel panel-primary panel-precice">
    <div class="panel-heading-precice">
      <strong>{{ pub.title }}</strong>
    </div>
    <div class="panel-body">
      <p>
        <em>{{ pub.authors }}</em>
        {% if pub.journal.name %}{{ pub.journal.name }}, {% endif %}
        {% if pub.journal.volume %}Volume {{ pub.journal.volume }}, {% endif %}
        {% if pub.journal.publisher %}{{ pub.journal.publisher }}, {% endif %}
        {{ pub.year }}{% if pub.doi %}, <a href="https://www.doi.org/{{pub.doi}}">doi:{{pub.doi}}</a>{% endif %}.
      </p>
      {% if pub.pub-url %}
      <a href="{{pub.pub-url}}">Publisher's site</a>&nbsp;&nbsp;
      {% endif %}
      {% if pub.bibtex %}
      <a href="assets/{{ pub.bibtex }}">Download BibTeX &nbsp;<i class="fas fa-download"></i></a>
      {% endif %}
    </div>
  </div>
</div>
{% endif %}
{% endfor %}
</div>
{% endfor %}
