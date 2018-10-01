def calDeterminant(mat):
    deter = 0
    try: 
        if len(mat[0]) != 2 and len(mat[1] != 2) :
            raise Exception
    except Exception:
            print('INFO - INPUT MISMATCH: Matrix must be of size 2x2 to calculate determinant')
        
    else:
        deter = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

    finally:
        return deter


if __name__ == '__main__':
    mat =[[3, 5], [-1, 1]]
    mat2 =[[3, 5, 3], [-1, 1, 4]]
    print(calDeterminant(mat))
    print(calDeterminant(mat2))
