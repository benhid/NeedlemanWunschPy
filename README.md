# NeedlemanWunschPy
Implementation of the Needleman-Wunsch algorithm in Python 3.x

## Dependencies
We use NumPy and pandas for matrix manipulations.

```bash
$ pip install pandas
$ pip install numpy
```

However, we recommend downloading the latest version of [Anaconda](https://www.continuum.io/downloads) as it includes several packages (like NumPy and pandas) by default.

## Usage
An example is shown in `example.py`.

## Output
The output will be created on a directory called `output/`. This will include:

1. The score matrix (`score_matrix.csv`) only if `save_score_matrix_to_file` was set to `True`.
2. A log file (`output.log`) including the time of execution of the program (you can also set `level` to `DEBUG` inside `logconfig.ini/[handler_default]` to show the traceback path).