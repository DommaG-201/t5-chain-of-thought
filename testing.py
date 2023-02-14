import random
import re

question1 = '''Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.
How many clips did Natalia sell altogether in April and May?'''
answer1 = '''Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72'''

question2 = '''Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?'''
answer2 = '''Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10'''


# recursive methods? i.e. for numbers in q, replace matching in a. Then solve replaced <<equations>> in a.
# Then take updated ans and replace old ans with new ans. Then solve any mismatched equations (if both numbers on left changed, ignore eq from now on).
# repeat until no more equations are updated

def adjust_question(question, answer, lower_bound, upper_bound):
    q_org_numbers = re.findall(r'\d+', question)
    updated_values = []
    for number in q_org_numbers:
        rand_num = random.randint(lower_bound, upper_bound)
        updated_values.append([number, rand_num])
        question = question.replace(str(number), str(rand_num))

    while len(updated_values) > 0:
        answer = answer.replace(str(updated_values[0][0]), str(updated_values[0][1]))
        answer, ans_updated_values = check_equations(answer)
        updated_values.pop(0)
        updated_values = updated_values + ans_updated_values

    # check equations, update answers, add updated answers to new_numbers queue
    return question, answer


def check_equations(answer):
    equations = re.compile('<<(.*?)>>', re.DOTALL | re.IGNORECASE).findall(answer)
    updated_values = []
    if equations:
        for equation in equations:
            true_ans = eval(equation.partition('=')[0])
            stripped = equation.split('=', 1)[0]
            old_answer = equation.split('=', 1)[1]
            stripped += '='
            stripped += str(true_ans)
            answer = answer.replace(equation, stripped)
            if str(old_answer) != str(true_ans):
                updated_values.append([old_answer, true_ans])
    return answer, updated_values


new_q, new_ans = adjust_question(question2, answer2, 0, 1000)
print(new_q)
print(new_ans)
