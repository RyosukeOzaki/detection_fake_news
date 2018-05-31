import numpy as np
import copy
import csv

class Batcher(object):
    def __init__(self, word2vec):
        self._headline = []
        self._body = []
        self._targets = []
        self._word2vec = word2vec
        self._embedding_dim = len(self._word2vec["beer"])
        self._out_of_voc_embedding = (2 * np.random.rand(self._embedding_dim) - 1) / 20
        self._delimiter = self._word2vec["_"]
        

    def batch_generator(self, dataset, data_ids, num_epochs, batch_size, sequence_length, return_body=False):
        body = []
        headline = []
	target = []
        x_id = []
        ids = range(len(dataset["targets"]))
        for epoch in range(num_epochs):
            permutation = np.random.permutation(ids)
            for i, idx in enumerate(permutation):
                x_id.append(data_ids["ID"][idx])
                self._headline.append(self.preprocess(sequence=dataset["headline"][idx], sequence_length=sequence_length))
                self._body.append(self.preprocess(sequence=dataset["body"][idx], sequence_length=sequence_length, is_delimiter=True)) 
		self._targets.append(dataset["targets"][idx])
                body.append(self.preprocess(sequence=dataset["body"][idx], sequence_length=sequence_length, is_delimiter=True, return_p_seq=True))
		headline.append(self.preprocess(sequence=dataset["headline"][idx], sequence_length=sequence_length, is_delimiter=True, return_p_seq=True))
		target.append(dataset["targets"][idx])

                if len(self._targets) == batch_size or (i == (len(permutation) - 1) and epoch == (num_epochs - 1)):
                    batch = {
                                "headline": self._headline,
                                "body": self._body,
                                "targets": self._targets,
                            }
                    self._headline = []
                    self._body = []
                    self._targets = []
                    if not return_body:
                        yield batch, epoch
                    else:
			header = []
			header_target = [] 
                        targets = []
                        target = map(str,target)
                        for targets_num in range(len(target)):
                            if target[targets_num] == "0":
                               targets.append(["True"])
			    if target[targets_num] == "1":
                               targets.append(["False"])
			    #if target[targets_num] == "2":
                               #targets.append(["Can not judge"])
			    #if target[targets_num] == "3":
                               #targets.append(["Unrelated"])
                        for header_len in range(sequence_length):
		            header.append("{}".format(header_len))
                        with open('../runs/attention_lstm/log/bodies.csv', "w") as f_body:
                             writer_body = csv.writer(f_body, lineterminator="\n")
                             writer_body.writerow(header)
                             writer_body.writerows(body)
			with open('../runs/attention_lstm/log/headlines.csv', "w") as f_headline:
                             writer_headline = csv.writer(f_headline, lineterminator="\n")
                             writer_headline.writerow(header)
                             writer_headline.writerows(headline)
			with open('../runs/attention_lstm/log/targets.csv', "w") as f_target:
                             writer_target = csv.writer(f_target, lineterminator="\n")
                             writer_target.writerow("0")
                             writer_target.writerows(targets)
                        np.savetxt('../runs/attention_lstm/log/ids.csv', x_id, delimiter=',', header="0", comments='')
                        yield batch, epoch

    def preprocess(self, sequence, sequence_length, is_delimiter=False, return_p_seq=False):
        p_seq = copy.deepcopy(sequence)
        preprocessed = []
        diff_size = len(p_seq) - sequence_length + int(is_delimiter)
        if diff_size  > 0:
            start_index = 0#np.random.randint(diff_size + 1)
            p_seq = p_seq[start_index: (start_index + sequence_length - int(is_delimiter))]
        for word in p_seq:
            try:
                embedding = self._word2vec[word]
            except KeyError:
                embedding = self._out_of_voc_embedding
            finally:
                preprocessed.append(embedding)
        if is_delimiter:
            preprocessed = [self._delimiter] + preprocessed
        for i in range(sequence_length - len(preprocessed)):
            preprocessed.append(np.zeros(self._embedding_dim))
        if not return_p_seq:
            return preprocessed
        else:
            return p_seq

