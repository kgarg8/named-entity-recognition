"""Build vocubularies of words and tags from datasets"""

import argparse
from collections import Counter
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help='Minimum count for words in the dataset', type=int)
parser.add_argument('--min_count_tag', default=1, help='Minimum count for tags in the dataset', type=int)

# Additional tokens
PAD_WORD = '<pad>'
PAD_TAG = 'O'
UNK_WORD = 'UNK'

def save_vocab_to_txt_file(vocab, txt_path):
	"""Save vocabulary to txt file"""

	with open(txt_path, 'w') as f:
		for token in vocab:
			f.write(token + '\n')


def save_dict_to_json(d, path):
	"""Save properties of dataset to json file"""

	with open(path, 'w') as f:
		dictionary = {k:v for k, v in d.items()}
		json.dump(dictionary, f, indent=4)


def update_vocab(txt_path, vocab):
	"""Build words and tags vocabularies"""

	with open(txt_path) as f:
		for i, line in enumerate(f):
			vocab.update(line.strip().split(' '))

	return i + 1


if __name__ == '__main__':
	args = parser.parse_args()

	# Build word vocab with train and test datasets
	print("Building words vocabulary...")
	words = Counter()
	size_train_sentences = update_vocab(os.path.join('data/', 'train/sentences.txt'), words)
	size_val_sentences = update_vocab(os.path.join('data/', 'val/sentences.txt'), words)
	size_test_sentences = update_vocab(os.path.join('data/', 'test/sentences.txt'), words)
	print("- done.")

	print("Building tags vocabulary...")
	tags = Counter()
	size_train_labels = update_vocab(os.path.join('data/', 'train/labels.txt'), tags)
	size_val_labels = update_vocab(os.path.join('data/', 'val/labels.txt'), tags)
	size_test_labels = update_vocab(os.path.join('data/', 'test/labels.txt'), tags)
	print("- done.")

	assert size_train_sentences == size_train_labels
	assert size_val_sentences == size_val_labels
	assert size_test_sentences == size_test_labels

	# Only keep frequent tokens
	words = [tok for tok, count in words.items() if count >= args.min_count_word]
	tags = [tok for tok, count in tags.items() if count >= args.min_count_tag]

	# Add pad tokens
	if PAD_WORD not in words: words.append(PAD_WORD)
	if PAD_TAG not in tags: tags.append(PAD_TAG)

	# Add Word for unknown words
	words.append(UNK_WORD)

	# Save vocabularies to file
	print('Saving vocabularies to file...')
	save_vocab_to_txt_file(words, os.path.join('data/', 'words.txt'))
	save_vocab_to_txt_file(tags, os.path.join('data/', 'tags.txt'))
	print('- done.')

	# Save dataset properties in json file
	sizes = {
		'train_size': size_test_sentences,
		'val_size': size_val_sentences,
		'test_size': size_test_sentences,
		'vocab_size': len(words),
		'number_of_tags': len(tags),
		'pad_word': PAD_WORD,
		'pad_tag': PAD_TAG,
		'unk_word': UNK_WORD
	}
	
	save_dict_to_json(sizes, os.path.join('data/', 'dataset_params.json'))

	# Logging sizes
	to_print = '\n'.join("- {}: {}".format(k, v) for k, v in sizes.items())
	print('Dataset Properties:\n{}'.format(to_print))