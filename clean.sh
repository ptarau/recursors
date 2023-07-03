rm -r -f dist build
rm -r -f *.egg-info
pushd .
cd deepllm
rm -r -f __pycache__
popd
pushd .
cd deepllm
cd apps
rm -r -f __pycache__
popd
cd deepllm
cd demos
rm -r -f __pycache__
popd
cd deepllm
cd tests
rm -r -f __pycache__
popd

