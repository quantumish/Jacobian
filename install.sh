brew install eigen
python 3 -m pip install pytest
git clone https://github.com/pybind/pybind11.git
mkdir build
cd build
cmake ..
make check -j 4
make
sudo make install
