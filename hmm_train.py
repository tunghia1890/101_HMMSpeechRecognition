import os
import pickle

import hmmlearn.hmm as hmm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import preprocessing


class HMMTraining:
    def __init__(self):
        self.class_names = ['cothe', 'khong', 'nhung']
        self.states = [9, 9, 9]
        self.dataset_path = 'datasets'

        self.X = {'train': {}, 'test': {}}
        self.y = {'train': {}, 'test': {}}

        self.model = {}
        self.model_path = 'models_train'

    def train(self):
        length = 0
        for cn in self.class_names:
            length += len(os.listdir(f"{self.dataset_path}/{cn}"))
        print('Total samples:', length)

        all_data = {}
        all_labels = {}
        for cname in self.class_names:
            file_paths = [os.path.join(self.dataset_path, cname, i) for i in os.listdir(
                os.path.join(self.dataset_path, cname)) if i.endswith('.wav')]
            data = [preprocessing.get_mfcc(file_path) for file_path in file_paths]
            all_data[cname] = data
            all_labels[cname] = [self.class_names.index(cname) for _ in range(len(file_paths))]

        for cname in self.class_names:
            x_train, x_test, y_train, y_test = train_test_split(
                all_data[cname], all_labels[cname],
                test_size=0.33,
                random_state=42
            )

            self.X['train'][cname] = x_train
            self.X['test'][cname] = x_test
            self.y['test'][cname] = y_test

        total_train = 0
        total_test = 0
        for cname in self.class_names:
            train_count = len(self.X['train'][cname])
            test_count = len(self.X['test'][cname])
            print(cname, 'train:', train_count, '| test:', test_count)
            total_train += train_count
            total_test += test_count
        print('train samples:', total_train)
        print('test samples', total_test)

        for idx, cname in enumerate(self.class_names):
            start_prob = np.full(self.states[idx], 0.0)
            start_prob[0] = 1.0
            trans_matrix = np.full((self.states[idx], self.states[idx]), 0.0)
            p = 0.5
            np.fill_diagonal(trans_matrix, p)
            np.fill_diagonal(trans_matrix[0:, 1:], 1 - p)
            trans_matrix[-1, -1] = 1.0

            # trans matrix
            print(cname)
            print(trans_matrix)

            self.model[cname] = hmm.GaussianHMM(
                n_components=self.states[idx],
                verbose=True,
                n_iter=300,
                startprob_prior=start_prob,
                transmat_prior=trans_matrix,
                params='stmc',
                init_params='mc',
                random_state=42
            )
            self.model[cname].fit(X=np.vstack(self.X['train'][cname]),
                                  lengths=[x.shape[0] for x in self.X['train'][cname]])

    def save_model(self):
        for cname in self.class_names:
            name = f'{self.model_path}/model_{cname}.pkl'
            with open(name, 'wb') as file:
                pickle.dump(self.model[cname], file)

    def evaluation(self):
        print('====== Evaluation ======')
        y_true = []
        y_pred = []
        for cname in self.class_names:
            for mfcc, target in zip(self.X['test'][cname], self.y['test'][cname]):
                scores = [self.model[cname].score(mfcc) for cname in self.class_names]
                pred = np.argmax(scores)
                y_pred.append(pred)
                y_true.append(target)
            print(f'{cname}:', (np.array(y_true) == np.array(y_pred)).sum() / len(y_true))
        print('======')
        print('Confusion matrix:')
        print(confusion_matrix(y_true, y_pred))



if __name__ == '__main__':
    hmm_train = HMMTraining()
    hmm_train.train()
    hmm_train.save_model()
    hmm_train.evaluation()
