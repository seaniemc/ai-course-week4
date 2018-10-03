from operator import add
from operator import sub
#Tried to implement SRP "Single Responsibility Principal" for all classes in this module

#Class DeterminantInverse is used to calculate the 
#determinant and inverse of a 2x2 matrix
class DeterminantInverse:
    #constructor function
    def __init__(self, mat):
        self.mat = mat

    #function calculates the determinant of a 2x2 matrix
    def calDeterminant(self, mat):
        deter = 0
        try: 
            if len(mat[0]) != 2 and len(mat[1] != 2) :
                raise Exception
        except Exception:
                print('\nINFO - INPUT MISMATCH: Matrix must be of size 2x2 to calculate determinant') 
        else:
            #calculates determinant
            deter = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        finally:
            return deter

    #function calculates the inverse of a 2x2 matrix
    def inverseOfMatrix(self, mat):
        #get the determinant of mat
        det = self.calDeterminant(mat)
        fracDet = 0
        #create a 2x2 matrix containing 0
        inverseMat = [[0 for y in range(len(mat[0]))]
                      for x in range(len(mat[1]))]
        #if determinant == 0 return matrix containing zero
        if det == 0:
            return inverseMat
        else:
            #divide by 1
            fracDet = 1/det

        try:
            if len(mat[0]) == 2 and len(mat[1]) == 2:
                #changes the places of matrix
                tmp = mat[0][0]
                mat[0][0] = mat[1][1]
                mat[1][1] = tmp
                #changes from positive to negative and vice versa
                mat[0][1] = self.checkForNeg(mat[0][1])
                mat[1][0] = self.checkForNeg(mat[1][0])
                
                for i in range(len(mat[0])):
                    for j in range(len(mat[1])):
                        inverseMat[i][j] += fracDet * mat[i][j]
        except:
            print(
                '\nINFO - INPUT MISMATCH: Inverse can only be calculated on a 2X2 matrix')
        finally:
            return inverseMat

    def checkForNeg(self, x):
        newNum = 0
        #print('before if {}'.format(x))
        if x > 0:
            #print('x is > 0 {}'.format(x)) 
            newNum = -x
            #print(newNum)
        elif x < 0: 
            #print('x is < 0 {}'.format(x))
            newNum = abs(x)
            #print(newNum)
        return newNum

#Class performs addition and subtraction on matrix's of equal height and length
class AddOrSubMatrix:
    def __init__(self, matrix1, matrix2):
        self.matrix1 = matrix1
        self.matrix2 = matrix2
    
    # function which adds or subtracts matrix
    def matrixAddOrSubtr(self, matrix1, matrix2, mathSign):
        #create a array containing only zero, which is the
        # height of mat1 and length of mat2
        sumAns = [[0 for y in range(len(matrix2))]for x in range(len(matrix1))]

        try:  # Verify the length of the Column / Row of the Matrix
            if len(matrix1) != len(matrix2[0]) or len(matrix1[0]) != len(matrix2):
                raise Exception

        except Exception:
                print('\nINFO - INPUT MISMATCH: Columns of matrix 1 not the same size as rows of matrix 2')

        else:
            for i in range(len(matrix1)):
                for j in range(len(matrix2[0])):
                    # pass the numbers at mat1[i][j] and mat2[i][j] to the built in function
                    # which will add or subtract the values, depending on what is passed into function.
                    sumAns[i][j] += mathSign(matrix1[i][j], matrix2[i][j])

        finally:  # Return Result as a list
            return sumAns

#Class performs matrix multiplication of any 2 matrix's 
#of same height and length
class MatrixMulti: 
    def __init__(self, matrix1, matrix2):
        self.matrix1 = matrix1
        self.matrix2 = matrix2
    
    #function which multiplies matrix's
    def matriceMulti(self, mat1, mat2):
        mat1Row = len(mat1)
        mat1Columns = len(mat1[0])
        mat2Rows = len(mat2)
        mat2Columns = len(mat2[0])
        #create a array containing only zero, which is the
        # height of mat1 and length of mat2
        sumAns = [[0 for y in range(mat2Columns)]for x in range(mat1Columns)]
        try:
            if mat1Columns == mat2Rows:
                #https://stackoverflow.com/questions/17623876/matrix-multiplication-using-arrays
                for i in range(mat1Row):
                    for j in range(mat2Columns):
                        for k in range(mat1Columns):
                            sumAns[i][j] += mat1[i][k] * mat2[k][j]
        except:
            print('\nINFO - INPUT MISMATCH: Columns of matrix 1 not the same size as rows of matrix 2')
        finally:
            return sumAns

#Class multiplies a matrix and a vector
class MatrixVectorMulti:
    def __init__(self, matrix, vector):
        self.matrix = matrix
        self.vector = vector

    #function multiplies matrix and a vector
    def matriceXVector(self, mat, vect):
        matLen = len(mat)
        vetLen = len(vect)
        #list gets populated with multiplication values
        multiAns = []
        #list gets populated with sum of multiplication values
        sumAns = []
        #print(len(mat[0]) == len(vect))
        try:
            if len(mat[0]) == vetLen:
                for i in range(matLen):
                    for j in range(vetLen):
                        multiAns.append(mat[i][j] * vect[j])

                sumAns = self.sumOfVectorMulti(multiAns)
        except:
            print(
                '\n INFO: Columns of the matrix must be same length as the rows of the vector')
        finally:
            return sumAns

    #Helper function which performs addition of the multiplied
    # vector/matrix multiplication.
    def sumOfVectorMulti(self, multiAns):
        sumAns = []
        try:
            if len(multiAns) == 4:
                sumAns = self.calcuList(multiAns, 2)
            elif len(multiAns) == 6:
                sumAns = self.calcuList(multiAns, 3)
            elif len(multiAns) == 16:
                sumAns = self.calcuList(multiAns, 4)
        except:
            print('\n INFO: Input mismatch, function can only operate on '
                + 'tensors with elements of length 2, 3 or 4. Please check your Matrix input length ')
        finally:
            return sumAns

    #helper function which splits the list
    #into smaller section of 2/3/4 elements and then sums each
    #individual list
    def calcuList(self, multiAns, increment):
        sumAns = []
        #split multiAns list in to smaller lists based of increment passed in e.g 2/3/4
        composite_list = [multiAns[x:x+increment]
                        for x in range(0, len(multiAns), increment)]
        for i in composite_list:
            #sum each list and add to sumAns list
            sumAns.append(sum(i))
        return sumAns

class CrossProduct:
    def __init__(self, vector1, vector2):
        self.vector1 = vector1
        self.vector2 = vector2

    def calculateCrossProduct(self, vector1, vector2):
        ansVector = []
        ansVector.append(vector1[1] * vector2[2] - vector1[2] * vector2[1])
        ansVector.append(vector1[2] * vector2[0] - vector1[0] * vector2[2])
        ansVector.append(vector1[0] * vector2[1] - vector1[1] * vector2[0])
        return ansVector

if __name__ == '__main__':
    mat =[[3, 5], [-1, 1]]
    matInverse =[[2, 4], [1, 3]]
    mat2 =[[3, 5, 3], [-1, 1, 4]]

    #4 x4 Matrices
    matrix3 = [[1, 6, 4, 5], [4, -4, 8, 6], [4, -4, 8, 7], [4, -4, 8, -9]]
    matrix4 = [[2, -6, 9, -4], [4, 5, -1, -3], [4, 5, -1, 7], [4, -2, 5, -1]]

    #vector x vector cross product
    inverseVector1 = [1, -7, 1]
    inverseVector2 = [5, 2, 4]

    #Matrix of 4x4 and vector of 4 elements
    testMat4 = [[2, 6, -9, 1], [-1, 8, 3, 1], [-2, 4, 4, -2], [2, 5, -7, 1]]
    vect4 = [2, -9, -6, 1]

    answerDeter = 8
    answerInverse = [[1.5, -2.0], [-0.5, 1.0]]
    answerMatVec = [5, -91, -66, 2]
    answerCrossProd = [-30, 1, 37]
    answerMatAdd = [[3, 0, 13, 1], [8, 1, 7, 3],
                    [8, 1, 7, 14], [8, -6, 13, -10]]
    answerMatSub = [[-1, 12, -5, 9], [0, -9, 9, 9],
                    [0, -9, 9, 0], [0, -2, 3, -8]]
    answerMatMulti = [[62, 34, 24, 1],
                     [48, -16, 62, 46],
                     [52, -18, 67, 45],
                     [-12, 14, -13, 61]]

    #print(DeterminantInverse(mat).calDeterminant(mat))
    print(DeterminantInverse(matInverse).inverseOfMatrix(matInverse))

    if answerDeter == DeterminantInverse(mat).calDeterminant(mat):
        print('1). Determinant of {} = {} == True\n'.format(
            mat, answerDeter))
    else:
        print('False')

    #For some reason this returns false as we can see from the else its correct
    #if someone can have a look that would be great
    if answerInverse == DeterminantInverse(matInverse).inverseOfMatrix(matInverse):
        print('2). \nInverse of {} = {}  == True\n'.format(
            matInverse, answerInverse))
    else:
        print('2). == False')
        print(answerInverse == DeterminantInverse(
            matInverse).inverseOfMatrix(matInverse))

    if answerInverse == DeterminantInverse(mat2).inverseOfMatrix(mat2):
        print('3). \nInverse of {} = {}  == True\n'.format(
            matInverse, answerInverse))
    else:
        print('3). == False\n')

    if answerMatAdd == AddOrSubMatrix(matrix3, matrix4).matrixAddOrSubtr(matrix3, matrix4, add):
        print('4). {} \n + {} \n= {}  == True\n'.format(
            matrix3, matrix4, answerMatAdd))
    else:
        print('4). == False')

    if answerMatSub == AddOrSubMatrix(matrix3, matrix4).matrixAddOrSubtr(matrix3, matrix4, sub):
        print('5). {} \n - {} \n= {}  == True\n'.format(
            matrix3, matrix4, answerMatSub))
    else:
        print('5). == False')

    if answerMatMulti == MatrixMulti(matrix3, matrix4).matriceMulti(matrix3, matrix4):
        print('6). {} \n X {} \n= {}  == True\n'.format(
            matrix3, matrix4, answerMatSub))
    else:
        print('6). == False')

    if answerMatVec == MatrixVectorMulti(testMat4, vect4).matriceXVector(testMat4, vect4):
        print('7). {} \n X {} \n= {}  == True\n'.format(
            testMat4, vect4, answerMatVec))
    else:
        print('7). == False')

    if answerCrossProd == CrossProduct(inverseVector1,
                       inverseVector2).calculateCrossProduct(inverseVector1, inverseVector2):
        print('8). Cross Product of \n{} And {} = {}  == True\n'.format(
            inverseVector1, inverseVector2, answerCrossProd))
    else:
        print('8). == False')
