import random
import re

question1 = '''Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.
How many clips did Natalia sell altogether in April and May?'''
answer1 = '''Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72'''

question2 = '''Weng earns $12 an hour for babysitting. 
Yesterday, she just did 50 minutes of babysitting. How much did she earn?'''
answer2 = '''Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10'''

question3 = '''Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, 
a pound of cream cheese that cost half the price of the cheddar cheese, 
and a pack of cold cuts that cost twice the price of the cheddar cheese. 
How much does he spend on the ingredients?'''
answer3 = '''A pound of cream cheese cost $10 / 2 = $<<10/2=5>>5.
A pack of cold cuts cost $10 x 2 = $<<10*2=20>>20.
Jasper spent $10 + $5 + $20 = $<<10+5+20=35>>35 on the ingredients.
#### 35'''


#   This method below takes a question and answer from the format provided by gsm8k and updates the values
#   First it generates numbers in the range for each number in the question
#   for each pair of original and random replacement it adds it to the updated values queue
#   then it replaces the text in the question with the new values
#   The following loops until the updated values queue is empty:
#      Take the first old/new value pair
#      replace the values in the answer
#      run the method check equations
#      This method goes through every equation in the answer (between << and >>) and checks it is correct
#      If it isn't correct (so one of the values were updated), find the true answer to the equation
#      Replace the false equation with the correct one, using the correct answer found earlier
#      Add the old, false answer and the new, true answer to the back of the updated values queue

# Note that this isn't 100% accurate, something to take into account for the evaluation.
# Additionally, it sometimes goes against logical reasoning i.e. resulting in negatives
# for things that cannot really be negative e.g. pages left to read in a book.

def adjust_question(question, answer, lower_bound, upper_bound):
    q_org_numbers = re.findall(r"(?<![a-zA-Z:])[-+]?\d*\.?\d+", question)
    org_question = question
    org_answer = answer
    updated_values = []
    for number in q_org_numbers:
        rand_num = random.randint(lower_bound, upper_bound)
        updated_values.append([number, rand_num])
        question = question.replace(str(number), str(rand_num))

    count = 0
    while len(updated_values) > 0:
        count += 1
        if count > 100 or (not check_values_size(updated_values, upper_bound)):
            raise Exception([org_question, org_answer, question, answer])
        if updated_values[0][0] in answer:
            answer = answer.replace(str(updated_values[0][0]), str(updated_values[0][1]))
            answer, ans_updated_values = update_equations(answer)
            updated_values = updated_values + ans_updated_values
        updated_values.pop(0)

    if not check_equations(answer):
        raise Exception([org_question, org_answer, question, answer])
    return question, answer


def update_equations(answer):
    answer = remove_second_decimal(answer)
    equations = re.compile('<<(.*?)>>', re.DOTALL | re.IGNORECASE).findall(answer)
    updated_values = []
    if equations:
        for equation in equations:
            equation = remove_second_decimal(equation)
            true_ans = eval(equation.partition('=')[0])
            true_ans = round(true_ans, 3)
            stripped = equation.split('=', 1)[0]
            old_answer = equation.split('=', 1)[1]
            stripped += '='
            stripped += str(true_ans)
            answer = answer.replace(equation, stripped)
            if str(old_answer) != str(true_ans):
                updated_values.append([old_answer, true_ans])
    return answer, updated_values


def check_equations(answer):
    ans, updated_values = update_equations(answer)
    if not updated_values:
        return True
    return False


def remove_second_decimal(answer):
    numbers = re.findall(r"(?<![a-zA-Z:])[-+]?\d*\.?\d+\.\d+", answer)
    for number in numbers:
        if number.count('.') > 1:
            index = number.index('.', number.index('.') + 1)
            answer = answer.replace(number, number[:index])
    return answer


# makes sure no values are too large, and slow down processing. Returns false if there's an issue
def check_values_size(updated_values, max):
    for i in updated_values:
        for j in i:
            if float(j) > (max * max):
                return False

    return True

# new_q, new_ans = adjust_question(question2, answer2, 1000, 1000000)
# print(new_q)
# print(new_ans)
