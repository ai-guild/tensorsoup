# TODO

- [ ] Migrate from `tasks/cbt/dictionary.py` to `tproc/dictionary.py`
- [ ] Write a module for post-processing indexed data
	- [ ] include learning rate and mode as default exta placeholders
	- [ ] padding
	- [ ] reindexing
	- [ ] including masks
- [ ] Create a based class `Model` for all models
	- to facilitate data parallelism
	- more structured inference and optimization
