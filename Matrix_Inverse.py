""""
This program asks user to enter six real numbers:
a, b, c, d, e, and f. check whether the matrix is invertible. 
matrix: [a, 0, 0][b, c, 0][d, e, f]
matrix is invertible when determinant != 0
inverse found augmented identity matrix
"""
import numpy as np 

def AugmentMatrixInverse(A):
    invmatrix = np.zeros((len(A),2*len(A)))
    for i in range(0, len(A)):
        for j in range(0, 2*len(A)):
            if j < len(A) and i < len(A):
                invmatrix[i][j] = A[i][j]
    for i in range(0, len(A)):
        for j in range(len(A), 2*len(A)):
            if(j-len(A)==i):
                invmatrix[i][j] = 1
            else:
                invmatrix[i][j] = 0
    return invmatrix


def Inverse(A):
    """
    Calculates the inverse of matrix A
        : @pre-condition : A is a square matrix
        : @param A : user defined 3x3 matrix
        : @return inverse matrix
    """
    # calculate the matrix of minors,
    # then turn that into the matrix of cofactors,
    # then the adjugate, and
    # multiply that by 1/determinant.

    newA = AugmentMatrixInverse(A)
    n = len(newA)
    NROW = np.zeros(n)
    pfound = False
    for i in range(0, n):
        NROW[i] = i
    for i in range(0, n - 1):
        p = i
        maxim = 0
        for j in range(i, n):
            if (pfound == False):
                if (np.absolute(newA[int(NROW[j])][i]) > maxim):
                    maxim = np.absolute(newA[int(NROW[j])][i])
                if (np.absolute(newA[int(NROW[p])][i]) == maxim):
                    p = j
                    pfound = True
        # determinant = 0, then matrix is singular
        if (newA[int(NROW[p])][i] == 0):
            print('matrix is singular; not invertible')

        if (NROW[i] != NROW[p]):
            temp = NROW[i]
            NROW[i] = NROW[p]
            NROW[p] = temp
        m = np.zeros((n, n + 1))
        for j in range(i + 1, n):
            m[int(NROW[j])][i] = newA[int(NROW[j])][i] / newA[int(NROW[i])][i]
            newA[int(NROW[j])] = newA[int(NROW[j])] - (m[int(NROW[j])][i] * newA[int(NROW[i])])
    
    # determinant = 0, then matrix is singular
    if (newA[int(NROW[n - 1])][n - 1] == 0):
        print('matrix is singular; not invertible')

    for i in reversed(range(0, n)):
        for j in reversed(range(0, i + 1)):
            # print(i, ' ', j)
            if j == i:
                newA[int(NROW[i])] = newA[int(NROW[i])] / newA[int(NROW[i])][j]
                largesti = int(NROW[i])
                for k in reversed(range(0, i)):
                    newA[int(NROW[k])] = newA[int(NROW[k])] - newA[int(NROW[k])][j] * newA[largesti]
            else:
                newA[int(NROW[i])] = newA[int(NROW[i])] - newA[int(NROW[i])][j] * newA[largesti]
        # print(A1)
    Inverse = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            Inverse[i][j] = newA[int(NROW[i])][j + n]
    return Inverse   


# Driver Code
def main():
    # reads six numbers from input and typecasts them to int using 
    # list comprehension e.g. 1 2 3 4 5 6
    print("Numbers should be entered with spaces, e.g. 1 2 3 4 5 6")
    a, b, c, d, e, f = [int(a) for a in input("Enter six real numbers:").split()]

    # initialize matrix array
    mtx = [[a, 0, 0], [b, c, 0], [d, e, f]]
    # print(mtx)

    print(Inverse(mtx))

    
main()
    
