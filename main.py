EPOCH_NUM = 3
MODEL_SIZE = "t5-11b"
USE_GPU = True

from glob import glob
import pandas as pd
from simplet5 import SimpleT5
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import re
import string

# utility method, used to take a written answer and find the final numerical answer given
# first looks for ###, used in training data to indicate where the answer is
# then looks for final number in given text
# all else fails, return arbutary number for now (e.g. 99999)
def extract_final_answer(text):
    ans = text.split("### ", 1)
    if len(ans) > 1:
        # removes punctuation, mostly incase any numbers use commas e.g. 1,321
        ans_temp = ans[1].translate(str.maketrans('', '', string.punctuation))
        return int(ans_temp)

    if re.search(r'\d', text):
        last_num_predicted = re.findall(r'\d+', text)[-1]
        last_num_predicted = last_num_predicted.translate(str.maketrans('', '', string.punctuation))
        return int(last_num_predicted)

    return 999


# Moving question to new class, allowing us to easily edit q and a's to increase values later in a single class
class Question:

    def __init__(self, question, true_answer):
        self.question = question
        self.true_answer = true_answer
        self.prediction = ""

    def get_complexity(self):
        return self.true_answer.count('<<')

    def get_true_final_answer(self):
        return extract_final_answer(self.true_answer)

    def get_predicted_final_answer(self):
        if self.prediction != "":
            return extract_final_answer(self.prediction)

    # first check for answer after ### like in test,
    # then assumes final digits are answer
    # otherwise false - checking answer in both the same, rather than same logic
    def check_answer(self, predicted, true, question):
        predicted_ans = predicted.split("### ", 1)
        if len(predicted_ans) > 1:
            return int(predicted_ans[1]) == self.get_final_answer

        if re.search(r'\d', predicted):
            last_num_predicted = re.findall(r'\d+', predicted)[-1]
            return self.get_final_anwer == int(last_num_predicted)

        # something is missing a number so does not fulfill these conditions - have a look at question and answers
        print('Question - ', question)
        print('True - ', true)
        print('Predicted - ', predicted)
        print('------------------------------------------------------------')
        return False

    def set_prediction(self, prediction):
        self.prediction = prediction

def get_train_and_test_data():
    train_dataset = load_dataset("gsm8k", 'main', split="train")
    test_dataset = load_dataset("gsm8k", 'main', split="test")

    train_dict = {'source_text': train_dataset['question'],
                  'target_text': train_dataset['answer']}
    test_dict = {'source_text': test_dataset['question'],
                 'target_text': test_dataset['answer']}

    train_questions = []
    for i in range(len(train_dict['source_text'])):
        train_questions.append(Question(train_dict['source_text'][i], train_dict['target_text'][i]))

    test_questions = []
    for i in range(len(test_dict['source_text'])):
        test_questions.append(Question(test_dict['source_text'][i], test_dict['target_text'][i]))

    train_df = pd.DataFrame(train_dict)
    test_df = pd.DataFrame(test_dict)

    return train_df, test_df, train_questions, test_questions


class T5Classifier:

    def __init__(self):
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
        train_df, test_df, train_questions, test_questions = get_train_and_test_data()
        model = SimpleT5()
        self.train_model(model, train_df, test_df)
        self.test_model(model, test_questions)

    # Note currently using extrapolation for ans q's, so we do not change questions trained on (can do this, maybe discuss w/ supervisor)
    def train_model(self, model, train_df, test_df):
        # load model
        model.from_pretrained(model_type="t5", model_name=MODEL_SIZE)
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
        for question in test_questions:
            prediction = model.predict(question.question)[0]
            question.set_prediction(prediction)
            true.append(question.get_true_final_answer())
            predicted.append(question.get_predicted_final_answer())

        self.evaluate(true, predicted, "overall")

        # question_dictionary = {
        #   x.get_complexity(): x for x in test_questions
        # }

        # question_dictionary = dict([(q.get_complexity(), q) for q in test_questions])

        q_dict = self.to_complexity_dictionary(test_questions)

        # for question_list in q_dict.values():
        for question_list in q_dict:
            true_ans = [q.get_true_final_answer() for q in question_list]
            pred_ans = [q.get_predicted_final_answer() for q in question_list]
            self.evaluate(true_ans, pred_ans, question_list[0].get_complexity())


t5_classifier = T5Classifier()