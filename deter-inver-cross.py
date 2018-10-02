class DterminantInverse:
    
    def __init__(self, mat):
        self.mat = mat

    def calDeterminant(self, mat):
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

    def inverseOfMatrix(self, mat):
        det = self.calDeterminant(mat)
        fracDet = 1/det

        tmp = mat[0][0]
        mat[0][0] = mat[1][1]
        mat[1][1] = tmp

        mat[0][1] = self.checkForNeg(mat[0][1])
        mat[1][0] = self.checkForNeg(mat[1][0])
        
        inverseMat = [[0 for y in range(len(mat[0]))]for x in range(len(mat[1]))]
        
        for i in range(len(mat[0])):
            for j in range(len(mat[1])):
                inverseMat[i][j] += fracDet * mat[i][j]

        return inverseMat

    def checkForNeg(self, x):
        newNum = 0
        if x > 0: 
            newNum = -x
        elif x < 0:   
            newNum = abs(x)
        return newNum

if __name__ == '__main__':
    mat =[[3, 5], [-1, 1]]
    matInverse =[[2, 4], [1, 3]]
    mat2 =[[3, 5, 3], [-1, 1, 4]]

    print(DterminantInverse(mat).calDeterminant(mat))
    print(DterminantInverse(matInverse).inverseOfMatrix(matInverse))
