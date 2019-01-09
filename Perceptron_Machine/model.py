import numpy as np
class Model:
    def __init__(self, feature_dim):
        self.w = np.ones(feature_dim,dtype=np.float32)
        self.b = 0
        self.learn_rate = 0.1
    def sign(self, x, w, b):
        y_output = np.dot(x, w) + b
        return y_output
    def fit(self,train_feature,train_label):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(train_feature)):
                current_f = train_feature[d]
                current_label = train_label[d]
                if current_label * self.sign(current_f, self.w, self.b) <= 0:
                    self.w = self.w + self.learn_rate * np.dot(current_label, current_f)
                    self.b = self.b + self.learn_rate * current_label
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
        return 1
    def get_config(self):
        return self.w, self.b

