# PowerGraph Raw Data

The PowerGraph benchmark is installed locally from the Figshare archive:
https://figshare.com/articles/dataset/PowerGraph/22820534?file=50081700

Each dataset keeps the original `<dataset_name>/raw` layout so that the
`PowerGrid` loader in `data/prepare_data/powergrid.py` can process the files
directly. Use the helpers in `data/utils_data.py` to list available datasets or
instantiate the loader without referencing any hard-coded paths.
