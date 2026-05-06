module load python/3.11
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_drac.txt

echo "Run this to activate in your current shell:"
echo "  source $ENVDIR/bin/activate"