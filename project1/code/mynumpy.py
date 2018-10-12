import sys

def pad(matrix, up=0, down=0, left=0, right=0, val=0):
    """
    Purpose:
        do the matrxi padding.
    Input:
        matrix: a two dimension list
        up, down, left, right: int, where and how much the matrix need to be padded
        val: pad things
    Output:
        None
    """
    if up > 0 or down > 0:
        col_len = len(matrix[0])
        pad_horizontal = []
        for i in range(col_len):
            pad_horizontal.append(val)
        for i in range(up):
            matrix.insert(0,pad_horizontal[:])
        for i in range(down):
            matrix.append(pad_horizontal[:])
            
    if left > 0 or right > 0:   
        for row in matrix:
            for i in range(left):
                row.insert(0,val)
            for i in range(right):
                row.append(val)
    return matrix
    

def zeros(row_num, col_num):
    res = []
    row = []
    for i in range(col_num):
        row.append(0.0)
    for i in range(row_num):
        res.append(row[:]) 
    return res

def flip(matrix):
    res = []
    for row in reversed(matrix):
        res.append([i for i in reversed(row)])
    return res

def inner_product(mat1, mat2):
    res = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] * mat2[i][j])
        res.append(row)
    return res

def sum_all(mat):
    res = 0
    for row in mat:
        for val in row:
            res += val
    return res

def min_all(mat):
    res = sys.maxsize
    for row in mat:
        for val in row:
            if val < res:
                res = val
    return res

def max_all(mat):
    res = -sys.maxsize - 1
    for row in mat:
        for val in row:
            if val > res:
                res = val
    return res

def min_all_count(mat):
    res = sys.maxsize
    count = 0
    for row in mat:
        for val in row:
            if val < res:
                count = 0
                res = val
            if val == res:
                count += 1
    return res, count


def max_all_count(mat):
    res = -sys.maxsize - 1
    count = 0
    for row in mat:
        for val in row:
            if val > res:
                count = 0
                res = val
            if val == res:
                count += 1
    return res, count


def abs_all(mat):
    res = []
    for row in mat:
        res_row = [abs(val) for val in row]
        res.append(res_row)
    return res

def power(x, y): 
    """
    Cite:
        https://www.geeksforgeeks.org/write-a-c-program-to-calculate-powxn/
    Purpose:
        Function to calculate x 
    """
    if (y == 0): return 1
    elif (int(y % 2) == 0): 
        return (power(x, int(y / 2)) * power(x, int(y / 2))) 
    else: 
        return (x * power(x, int(y / 2)) * power(x, int(y / 2))) 


def resize_shrink(matrix, fx, fy):
    """
    Purpose:
        resize a matrix given fx and fy.
    Input:
        fx: resize on column
        fy: resize on row
    Output:
        Resized matrix list
    """
    
    fx_inv = int(1 / fx)
    fy_inv = int(1 / fy)
    res = []
    
    for i in range(0, len(matrix), fy_inv):
        res_row = []
        for j in range(0, len(matrix[i]), fx_inv):
            res_row.append(matrix[i][j])
        res.append(res_row)
    
    return res






