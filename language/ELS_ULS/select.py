import random

def select_random_numbers(seed_value):
    random.seed(seed_value)
    return random.sample(range(1, 11), 2)

seed_value = 42
# seed_value = 43
# seed_value = 44

# 测试函数
random_numbers = select_random_numbers(seed_value)
print("The randomly selected numbers are:", random_numbers)
