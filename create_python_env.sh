module load python/3.10
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy jax dm-haiku matplotlib chex optax flax jaxlib orbax dill pybind11 cmake tqdm networkx
