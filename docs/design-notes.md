
## System Architecture

**DATA FORMAT**

- Different models require differently processed data
- Use a format for binding DataLoader and Model in Trainer
- Format is an unordered list of names, or handles to data
- Different types of data are pickled separately and the filenames are accessible using keys, in metadata
- Make DataLoader a static class, common to all tasks
- Different tasks have different processing pipelines (proc)
- Update `make_parallel` and `build_feed_dict_multi` to accomodate the changes

**SUBTASKS**

- Sometimes trainer requires to train/evaluate model on a subset of data
- `proc` builds and returns this subset of data depending on an id or name
- Dynamically bind new subset of data to DataLoader (`update_data`)

**DATA FEED**

- Build a data feed, that will accept data dictionary
- Instantiate and maintain 2 copies of data feed, one for train and one for test (in app)
