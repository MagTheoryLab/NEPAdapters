# Repository Agent Notes

## Python Environment

Use the `mysci` conda environment for Python-facing work:

```sh
source /Users/superbing/miniconda3/etc/profile.d/conda.sh
conda activate mysci
```

For CMake builds that enable the Python frontend, pass:

```sh
-DPython3_EXECUTABLE=/Users/superbing/miniconda3/envs/mysci/bin/python
```
