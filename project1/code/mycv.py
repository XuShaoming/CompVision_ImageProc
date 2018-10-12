def resize_shrink(matrix, fx, fy):
    """
    Purpose:
        shrink a matrix given fx and fy.
    Input:
        fx: resize on column
        fy: resize on row
    Output:
        shrink matrix list
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