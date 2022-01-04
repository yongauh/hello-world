# farrell.py

prices = {'MacBook 13': 1300, 'MacBook 15': 2100, 'ASUS ROG': 1600 }

cart = {}

while True:
    _continue = input('Would you like to continue shopping? [y/n] ')
    if _continue == 'y':
        print(f'Available products and prices: {prices}')
        new_item = input('Which product would you like to add to your cart? ')
        if new_item in prices:
            if new_item in cart:
                cart[new_item] += 1
            else:
                cart[new_item] = 1
        else:
            print('Please only choose from the available products.')
    elif _continue == 'n':
        break
    else:
        print('Please only enter "y" or "n".')

# calculation of total bill

sum([prices[x] * cart[x] for x in cart])
sum(prices[x] * cart[x] for x in cart)

running_sum = 0
for item in cart:
    running_sum += cart[item] * prices[item]

print(f'Your final cart is:')
for item in cart:
    print(f'- {cart[item]} {item}(s)')
print(f'Your final bill is: {running_sum}')

def greet(name):
    print(f'Hello, {name}!')

def get_first_even(lst):
    for n in lst:
        if n % 2 == 0:
            return n
    return False

# Ex 1.05
def get_max(lst):
    running_max_index = 0
    for index, item in enumerate(lst):
        if item > lst[running_max_index]:
            running_max_index = index
    return running_max_index, lst[running_max_index]

#get_max([1, 2, 3, 2])

""" Looks like the system handles bignums automatically
"""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

# Ex 1.06 The tower of Hanoi

def solve(n):
    if n <= 1:
        return 1
    return 2 * solve(n-1) + 1

for n in range(1, 4):
    print(f'solve({n}) = {solve(n)}')
    

#def hanoi(lst1, lst2, lst3):
#

# n queens

def display_solution(board):
    N = len(board)
    for i in range(N):
        for j in range(N):
            print(board[i][j], end=' ')
        print()
    
def generate_solution(N):
    """
    """
    def check_next(board, row, col):
        # check the current column
        for i in range(row):
            if board[i][col] == 1:
                return False
        # check the upper-left diagonal
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        # check the upper-right diagonal
        for i, j in zip(range(row, -1, -1), range(col, N)):
            if board[i][j] == 1:
                return False    
                
        return True
        
    def recur_generate_solution(board, row_id):
        # return if we have reached the last row
        if row_id >= N:
            return True
        # iteratively try out the cells in the current row
        for i in range(N):
            if check_next(board, row_id, i):
                board[row_id][i] = 1
                # 
                final_board = recur_generate_solution(board, row_id + 1)
                if final_board:
                    return True
                # cancel
                board[row_id][i] = 0
        return False
    
    # start 
    the_board = [[0 for _ in range(N)] for _ in range(N)]
    final_solution = recur_generate_solution(the_board, 0)
    
    # display
    if final_solution is False:
        print('No solution is found.')
    else:
        display_solution(the_board)

# recording positions of the pieces could be more efficient


import unittest

class SampleTest(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(2**3-1, 7)
        self.assertEqual('Hello, world!', 'Hello, ' + 'world!')
        
    def test_true(self):
        self.assertTrue(2**3 < 3**2)
        for x in range(10):
            self.assertTrue(- x**2 <= 0)
        
# place breakpoint()