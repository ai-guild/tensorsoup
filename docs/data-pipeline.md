
## A Memory Efficient Algorithm


```python
# global vocab
vocab = []
for file in files:
	vocab_f = build_vocab(file)
	# read string from file
	#  obtain words from string ->
	#   make use of external nlp tools, say nltk

# now we have a long list of words -> vocab
#  build word2index dict
w2i = dictionary(vocab)

statistics = {}
data_dict = {}
file_keys = []

# iterate through files again
for i,file in enumerate(files):
	filename = file.get_filename()
	string = file.read()	
	data_item = string_to_data_item(string)
	# string_to_data_item
	#  tokenize, vectorize, structure(v) 
	
	# update statistics
	statistics.update(data_item.stats())

	# update data dictionary
	data_dict.update(data_item)

	file_keys.append(file.key())

	# clear data dictionary every n/100 samples
	if i%100 == 0:
		write_to_file(data_dict)
		data_dict = {}

```
