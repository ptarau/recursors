./clean.sh
pip install setuptools
pip install twine
python3 setup.py sdist
python3 -m twine upload dist/*
