# delfi

**Density Estimation Likelihood-Free Inference**: based on Alsing, Wandelt and Feeney, 2018 ([arXiv](https://arxiv.org/abs/1801.01497), [published](https://academic.oup.com/mnras/article-abstract/477/3/2874/4956055?redirectedFrom=fulltext)), Papamakarios, Sterratt and Murray 2018 ([arXiv](https://arxiv.org/pdf/1805.07226.pdf)), and Alsing, Charnock, Feeney and Wandelt 2018 (apearing soon). Please cite these papers if you use this code!

**Dependencies:** [tensorflow](https://www.tensorflow.org), [getdist](http://getdist.readthedocs.io/en/latest/), [emcee](http://dfm.io/emcee/current/), [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (if MPI required).

**Usage:** Once everything is installed, try out one of the bundled cosmic shear ipython notebooks (`cosmic_shear*.ipynb`) or type `mpirun -np <n_procs> python snl_cmb_lens_tom_class_mpi.py` to run the CMB lensing tomography example.
