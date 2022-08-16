


class Mathlib(object):
    def __init__(self):
        pass
    def identity(self, size):
        self.matrix = []
        for i in range(size):
            self.matrix.append([])
            for j in range(size):
                if i == j:
                    self.matrix[i].append(1)
                else:
                    self.matrix[i].append(0)
                    
        return self.matrix
    
    def toMatrix(self, matrix):
        matrixFinal = []
        for i in range(len(matrix)):
            matrixFinal.append(matrix[i])
        return matrixFinal
    
    def negative(self, matrix):
        matrixFinal = []
        for i in range(len(matrix)):
            matrixFinal.append(-matrix[i])
        return matrixFinal
    
    def multiply(self, matrix_list):
        self.matrix = []
        matrix1 = matrix_list[0]
        matrix2 = matrix_list[1]
        
        for i in range(len(matrix1)):
            self.matrix.append([])
            for j in range(len(matrix2[0])):
                self.matrix[i].append(0)
                for k in range(len(matrix2)):
                    self.matrix[i][j] += matrix1[i][k] * matrix2[k][j]
                    
                    
        matrix_list[0] = self.matrix
        matrix_list.pop(1)

        if len(matrix_list) > 1:
            self.multiply(matrix_list)

        return self.matrix
    
    def HorVert(self, matrix):
        matrixFinal = []
        for i in range(len(matrix)):
            matrixFinal.append([matrix[i]])
        return matrixFinal

    def VertHor(self, matrix):
        matrixFinal = []
        for i in range(len(matrix)):
            matrixFinal.append(matrix[i][0])
        return [matrixFinal]
    
    def dotProduct(self, matrix1, matrix2):
        matrixFinal = []
        res = 0
        for i in range(len(matrix1)):
            matrixFinal.append(matrix1[i] * matrix2[i])
        for i in range(len(matrixFinal)):
            res += matrixFinal[i]
        
        return res
    
    def dot(self, matrix1, matrix2):
        matrixFinal = []
        for i in range(len(matrix1)):
            matrixFinal.append(matrix1[i] * matrix2[i])
        return matrixFinal

    def cross(self, matrix1, matrix2):
        matrixFinal = []
        matrixFinal.append(matrix1[1] * matrix2[2] - matrix1[2] * matrix2[1])
        matrixFinal.append(matrix1[2] * matrix2[0] - matrix1[0] * matrix2[2])
        matrixFinal.append(matrix1[0] * matrix2[1] - matrix1[1] * matrix2[0])
        return matrixFinal

    def add(self, matrix1, matrix2):
        matrixFinal = []
        for i in range(len(matrix1)):
            matrixFinal.append(matrix1[i] + matrix2[i])
        return matrixFinal
    
    def substract(self, matrix1, matrix2):
        matrixFinal = []
        for i in range(len(matrix1)):
            matrixFinal.append(matrix1[i] - matrix2[i])
        return matrixFinal
    
    def linalgNorm(self, matrix):
        if len(matrix) == 1:
            matrix = self.VertHor(matrix)
        else:
            matrixFinal = 0
            for i in range(len(matrix)):
                matrixFinal += matrix[i] ** 2
            return matrixFinal ** 0.5

    

    def getCofactor(self, A, temp, p, q, n):
    
        i = 0
        j = 0
    
        # Looping for each element of the matrix
        for row in range(n):
    
            for col in range(n):
    
                # Copying into temporary matrix only those element
                # which are not in given row and column
                if (row != p and col != q):
    
                    temp[i][j] = A[row][col]
                    j += 1
    
                    # Row is filled, so increase row index and
                    # reset col index
                    if (j == n - 1):
                        j = 0
                        i += 1


# Recursive function for finding determinant of matrix.
#  n is current dimension of A[][].
    def determinant(self, A, n, N):
    
        D = 0   # Initialize result
    
        # Base case : if matrix contains single element
        if (n == 1):
            return A[0][0]
    
        temp = []   # To store cofactors
        for i in range(N):
            temp.append([None for _ in range(N)])
    
        sign = 1   # To store sign multiplier
    
        # Iterate for each element of first row
        for f in range(n):
    
            # Getting Cofactor of A[0][f]
            self.getCofactor(A, temp, 0, f, n)
            D += sign * A[0][f] * self.determinant(temp, n - 1, N)
    
            # terms are to be added with alternate sign
            sign = -sign
    
        return D
        
        
        # Function to get adjoint of A[N][N] in adj[N][N].
    def adjoint(self, A, adj, N):
    
        if (N == 1):
            adj[0][0] = 1
            return
    
        # temp is used to store cofactors of A[][]
        sign = 1
        temp = []   # To store cofactors
        for i in range(N):
            temp.append([None for _ in range(N)])
    
        for i in range(N):
            for j in range(N):
                # Get cofactor of A[i][j]
                self.getCofactor(A, temp, i, j, N)
    
                # sign of adj[j][i] positive if sum of row
                # and column indexes is even.
                sign = [1, -1][(i + j) % 2]
    
                # Interchanging rows and columns to get the
                # transpose of the cofactor matrix
                adj[j][i] = (sign)*(self.determinant(temp, N-1, N))
    
    
    # Function to calculate and store inverse, returns false if
    # matrix is singular
    def inverse(self, A, N, inverse):
        

            
        # Find determinant of A[][]
        det = self.determinant(A, N, N)
        if (det == 0):
            print("Singular matrix, can't find its inverse")
            return False
    
        # Find adjoint
        adj = []
        for i in range(N):
            adj.append([None for _ in range(N)])
        self.adjoint(A, adj, N)
    
        # Find Inverse using formula "inverse(A) = adj(A)/det(A)"
        for i in range(N):
            for j in range(N):
                inverse[i][j] = adj[i][j] / det
    
        return True