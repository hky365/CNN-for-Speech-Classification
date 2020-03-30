# CNN-for-Speech-Classification
This code was used as part of a group project (2 members) for the course in "Machine Learning" as part of the Master's program in Cognitive Science and Artificial Intelligence at Tilburg University.
This assignment was submitted on the 29th of November, 2019 and is hence the reflective of my knowledge at that time.

Intro
Our group participated in a speech classification challenge, using single-word audio files, extracted Mel-Frequency Cepstral Coefficients (MFCC) features and train labels to predict the word for each audio file in the test set. We ended up using a Convolutional Neural Network for the purposes of this challenge.                                       

Data Description
Speech Classification in this challenge the task is to learn to recognize which of several English words is pronounced in an audio recording. The following files were provided to us:
feat.npy: anarray with Mel-frequency cepstral coeffcients extracted from each wav ﬁle. The features at index i in this array were extracted from the wav ﬁle at index i of the array in the ﬁle path.npy.
path.npy: an array with the order of wav ﬁles in the feat.npy array.
train.csv: this ﬁle contains two columns: path with the ﬁlename of the recording and word with word which was pronounced in the recording. This is the training portion of the data.
test.csv: This is the testing portion of the data, and it has the same format as the ﬁle train.csv except that the column word is absent.
Note - Due to the university's policy and regulations, the dataset cannot be made publicly available.

Evaluation Metric
Classification Accuracy - My team achieved a test accuracy of 93.7%.
