# delfi

**Density Estimation Likelihood-Free Inference** with neural density estimators and adaptive acquisition of simulations. The implemented methods are based on [Papamakarios, Sterratt and Murray 2018](https://arxiv.org/pdf/1805.07226.pdf), [Lueckmann et al 2018](https://arxiv.org/abs/1805.09294) and [Alsing, Wandelt and Feeney, 2018](https://academic.oup.com/mnras/article-abstract/477/3/2874/4956055?redirectedFrom=fulltext), and described in detail in Alsing, Charnock, Feeney and Wandelt 2018 (the paper to go with the code, apearing soon). Please cite these papers if you use this code!

**Dependencies:** [tensorflow](https://www.tensorflow.org), [getdist](http://getdist.readthedocs.io/en/latest/), [emcee](http://dfm.io/emcee/current/), [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (if MPI required).

**Usage:** Once everything is installed, try out either `cosmic_shear.ipynb` or `jla_sne.ipynb` as example templates for how to use the code (plugging in your own simulator), or `cosmic_shear_pre-ran_sims.ipynb` if you have a set of pre-ran simulations you'd like to throw in rather than running sims on-the-fly. While in development we'll try to keep these three notebook examples up-to-date (the rest are us messing about with new things so may not work at any given time).

The code is not documented yet (paper and documentation coming imminently), but if you are interested in applying it to your problem please get in touch with us (at justin.alsing@fysik.su.se) and we are happy to collaborate!
