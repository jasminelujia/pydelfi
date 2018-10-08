# delfi

**Density Estimation Likelihood-Free Inference**: as described in Alsing, Wandelt and Feeney, 2018 ([arXiv](https://arxiv.org/abs/1801.01497), [published](https://academic.oup.com/mnras/article-abstract/477/3/2874/4956055?redirectedFrom=fulltext)). Please cite this paper when using the code!

**Dependencies:** [tensorflow](https://www.tensorflow.org/install/), [getdist](http://getdist.readthedocs.io/en/latest/), [emcee](http://dfm.io/emcee/current/), [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (if MPI required).

**Usage:** Once everything is installed, try out one of the bundled cosmic shear ipython notebooks (`cosmic_shear*.ipynb`) or type `mpirun -np <n_procs> python snl_cmb_lens_tom_class_mpi.py` to run the CMB lensing tomography example.
