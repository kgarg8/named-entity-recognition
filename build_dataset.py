import csv
import os
import sys

def load_dataset(path_csv):
	"""Load dataset from csv file"""

	with open(path_csv, encoding="windows-1252") as f:
		csv_file = csv.reader(f, delimiter=',')
		dataset = []; words = []; tags = []

		# Each line of csv corresponds to one word
		for idx, row in enumerate(csv_file):
			if idx == 0: continue	# first row has only labels
			sentence, word, pos, tag = row
			
			# Non-empty sentence marks beginning of new sentence
			if len(sentence) != 0:
				if len(words) > 0:
					assert len(words) == len(tags)
					dataset.append((words, tags))
					words, tags = [], []
			# try:
			word, tag = str(word), str(tag)
			words.append(word)
			tags.append(tag)

	return dataset

def save_dataset(dataset, save_dir):
	"""Create splits from dataset"""

	print("Saving in {}...".format(save_dir))
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Export the dataset
	with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
		with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
			for words, tags in dataset:
				file_sentences.write("{}\n".format(" ".join(words)))
				file_labels.write("{}\n".format(" ".join(tags)))
	print("- done.")

if __name__ == "__main__":

	dataset_path = 'data/ner_dataset.csv'
	msg = '{} file not found. Make sure you have downloaded the right dataset'.format(dataset_path)
	assert os.path.isfile(dataset_path), msg

	# Load the dataset
	print("Loading dataset...")
	dataset = load_dataset(dataset_path)
	print("- done.")

	# Split the dataset into train, val and test
	train_dataset = dataset[:int(0.7*len(dataset))]
	val_dataset = dataset[int(0.7*len(dataset)) : int(0.85*len(dataset))]
	test_dataset = dataset[int(0.85*len(dataset)):]

	save_dataset(train_dataset, 'data/train')
	save_dataset(val_dataset, 'data/val')
	save_dataset(test_dataset, 'data/test')