EPOCH_NUM = 3
MODEL_SIZE = "t5-11b"
USE_GPU = True

ADJUST_QUESTIONS = False
MAXIMUM_VALUE = 100
MINIMUM_VALUE = 0

from glob import glob
import pandas as pd
from simplet5 import SimpleT5
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from Question import Question
from datasets import load_dataset

def get_train_and_test_data():
    train_dataset = load_dataset("gsm8k", 'main', split="train")
    test_dataset = load_dataset("gsm8k", 'main', split="test")

    train_dict = {'source_text': train_dataset['question'],
                  'target_text': train_dataset['answer']}
    test_dict = {'source_text': test_dataset['question'],
                 'target_text': test_dataset['answer']}

    train_questions = []
    for i in range(len(train_dict['source_text'])):
        question = Question(train_dict['source_text'][i], train_dict['target_text'][i])
        if ADJUST_QUESTIONS:
            question.update_values(MINIMUM_VALUE, MAXIMUM_VALUE)
        train_questions.append(question)

    test_questions = []
    for i in range(len(test_dict['source_text'])):
        question = Question(test_dict['source_text'][i], test_dict['target_text'][i])
        if ADJUST_QUESTIONS:
            question.update_values(MINIMUM_VALUE, MAXIMUM_VALUE)
        test_questions.append(question)

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