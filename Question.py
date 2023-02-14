import re
import string
from Adjusting import adjust_question

#utility method, used to take a written answer and find the final numerical answer given
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

    def update_values(self, minimum, maximum):
        self.question, self.true_answer = \
            adjust_question(self.question, self.true_answer, minimum, maximum)

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