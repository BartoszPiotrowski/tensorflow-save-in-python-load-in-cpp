# Train Tensorflow model in Python and load it in C++

The repo contains two simplistic examples of training model in Python and
loading it in C++, according to two available methods.


## Requirements on Python side
- tensorflow 1.13 -- 1.15


## Requirements on C++ side
First make sure you have Bazel installed. Then, clone the Tensorflow repo:
```
cd ..
git clone https://github.com/tensorflow/tensorflow
```
and apply a small patch by running:
```
cd tensorflow
curl -L https://github.com/tensorflow/tensorflow/compare/master...hi-ogawa:grpc-backport-pr-18950.patch | git apply
```
Then, follow approximately these installation instructions, up to
`sudo ldconfig`:

https://tuanphuc.github.io/standalone-tensorflow-cpp/

(Building with Bazel will take ~10 minutes.) Place `lib` and `include` folders
with relevant content in this repo.

Check if you can compile a test example with this command:
```
g++ -Llib -Iinclude -Iinclude/third_party \
-ltensorflow_cc -ltensorflow_framework \
-o test/test1 test/test1.cc
./test/test1
```


Set the `LD_LIBRARY_PATH` environmental variable:
```
export LD_LIBRARY_PATH=lib
```

## Examples

There are two similar examples where a small network is trained in Python on
MNIST examples and then loaded in C++.


### `example_1`:
Older interface for loading and saving models. (Mirek uses this method is his
GNN code.)

Training in Python:
```
mkdir -p models/example_1
python3 example_1/train_and_save.py models/example_1/
```

Compiling C++ code:
```
g++ -Llib -Iinclude -Iinclude/third_party \
-ltensorflow_cc -ltensorflow_framework \
-o example_1/load_and_predict \
example_1/load_and_predict.cc
```

Running C++ code:
```
./example_1/load_and_predict \
	models/example_1/model.meta \
	models/example_1/model
```

The first and the second argument are a path to meta-graph and a path to values
of variables in the graph, respectively.

Because the input is not initialized you should see some random `0`s, `1`s or
`nan`s as an output.)

References:
- https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
- https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-c


### `example_2`
Newer, recommended interface of saving and loading models.

Training in Python:
```
mkdir -p models/example_2
python3 example_2/train_and_save.py models/example_2
```

Compiling C++ code:
```
g++ -Llib -Iinclude -Iinclude/third_party \
-ltensorflow_cc -ltensorflow_framework \
-o example_2/load_and_predict \
example_2/load_and_predict.cc
```

Running C++ code:
```
./example_2/load_and_predict \
	models/example_2
```

(Because input is not initialized you should see some random `0`s, `1`s or
`nan`s as an output.)

References:
- https://www.tensorflow.org/guide/saved_model
- https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-c


