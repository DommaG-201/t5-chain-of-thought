EPOCH_NUM = 3
USE_GPU = True
REMOVED_QUESTIONS_FILENAME = './data/removed_questions'

import os
import sys
from glob import glob

import datasets
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tqdm import tqdm

from FileWriter import write_deleted_questions, write_questions
from Question import Question


def arguments():
    model_size = "t5-base"
    adjust_questions = False
    max_value = 1000
    min_value = 1
    if __name__ == "__main__":
        if len(sys.argv) > 1:
            model_size = str(sys.argv[1])
        if len(sys.argv) > 2:
            adjust_questions = bool(sys.argv[2])
        if len(sys.argv) > 3:
            max_value = int(sys.argv[3])
        if len(sys.argv) > 4:
            min_value = int(sys.argv[4])

    return model_size, adjust_questions, max_value, min_value


def get_train_and_test_data(adjust_questions, max_value, min_value):
    train_dataset = datasets.load_dataset("gsm8k", 'main', split="train", cache_dir="./mycache")
    test_dataset = datasets.load_dataset("gsm8k", 'main', split="test", cache_dir="./mycache")
    # train_dataset = datasets.load_dataset("gsm8k", 'main', split="train")
    # test_dataset = datasets.load_dataset("gsm8k", 'main', split="test")

    train_dict = {'source_text': train_dataset['question'],
                  'target_text': train_dataset['answer']}
    test_dict = {'source_text': test_dataset['question'],
                 'target_text': test_dataset['answer']}

    train_questions = convert_to_questions(train_dict, "train", adjust_questions, max_value, min_value)
    test_questions = convert_to_questions(test_dict, "test", adjust_questions, max_value, min_value)

    train_df = pd.DataFrame(train_dict)
    test_df = pd.DataFrame(test_dict)

    return train_df, test_df, train_questions, test_questions


def convert_to_questions(dict, name, adjust_questions, max_value, min_value):
    questions = []
    temp = 0
    deleted_questions = []
    filename = REMOVED_QUESTIONS_FILENAME + '_' + name + '.csv'
    print('Processing ' + name + ' questions and answers:')
    for i in tqdm(range(len(dict['source_text']))):
        question = Question(dict['source_text'][i], dict['target_text'][i])
        attempts = 0
        processed = False
        while not processed:
            try:
                if adjust_questions:
                    question.update_values(min_value, max_value)
                questions.append(question)
                processed = True
            except Exception as ex:
                if attempts >= 50:
                    deleted_questions.append(ex.args[0])
                    temp += 1
                    processed = True
                else:
                    attempts += 1
    print("removed ", temp, " from dataset")
    write_deleted_questions(deleted_questions, filename)
    write_questions(questions, filename)
    return questions


class T5Classifier:

    def __init__(self):
        self.model_size, self.adjust_questions, self.max_value, self.min_value = arguments()
        self.run_model()

    def evaluate(self, true, predicted, class_name):
        print("For ", class_name, ":")
        # use libaries to find accuracy and f1 overall
        print("Accuracy: ")
        print(accuracy_score(true, predicted))

        print("F1: ")
        print(f1_score(true, predicted, average='macro'))
        print("---------------------------------------")

    # tried this using fancy code, didn't work so doing it here
    def to_complexity_dictionary(self, question_list):
        q_list_2d = []
        for q in question_list:
            complexity = q.get_complexity()
            while complexity + 1 > len(q_list_2d):
                q_list_2d.append([])
            q_list_2d[complexity].append(q)
        return q_list_2d

    def run_model(self):
        print(self.model_size)
        print(self.adjust_questions)
        print(self.max_value)
        print(self.min_value)
        train_df, test_df, train_questions, test_questions = get_train_and_test_data(self.adjust_questions,
                                                                                     self.max_value, self.min_value)
        # model = SimpleT5()
        # self.train_model(model, train_df, test_df)
        # self.test_model(model, test_questions)

    # Note currently using extrapolation for ans q's, so we do not change questions trained on (can do this, maybe discuss w/ supervisor)
    def train_model(self, model, train_df, test_df):
        # load model
        model.from_pretrained(model_type="t5", model_name=self.model_size)
        # train model
        model.train(train_df=train_df,
                    ##ask about this - would passing in test data as eval metric change outcome
                    eval_df=test_df,
                    source_max_token_len=128,
                    target_max_token_len=50,
                    batch_size=8,
                    max_epochs=EPOCH_NUM,
                    outputdir="outputs",
                    use_gpu=USE_GPU
                    )

    def test_model(self, model, test_questions):
        # fetch the path to last model
        last_epoch_model = None
        for file in glob("./outputs/*"):
            if ('epoch-' + str(EPOCH_NUM - 1)) in file:
                last_epoch_model = file
        # load the last model
        model.load_model("t5", last_epoch_model, use_gpu=USE_GPU)

        # for each test data perform prediction
        true = []
        predicted = []
        print("Testing model:")
        for question in tqdm(test_questions):
            prediction = model.predict(question.question)[0]
            question.set_prediction(prediction)
            true.append(question.get_true_final_answer())
            predicted.append(question.get_predicted_final_answer())

        self.evaluate(true, predicted, "overall")
        q_dict = self.to_complexity_dictionary(test_questions)

        # for question_list in q_dict.values():
        for question_list in q_dict:
            true_ans = [q.get_true_final_answer() for q in question_list]
            pred_ans = [q.get_predicted_final_answer() for q in question_list]
            self.evaluate(true_ans, pred_ans, question_list[0].get_complexity())


os.environ['TRANSFORMERS_CACHE'] = '/mnt/lime/homes/dg707/mycache'
model_size1, adjust_questions1, max_value1, min_value = arguments()
t5_classifier = T5Classifier()
