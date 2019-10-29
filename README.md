# Parallelism-of-Gaussian-Elimination-Method-using-CUDA
In linear algebra, Gaussian elimination method also known as row reduction is an algorithm used to solve systems of linear equations. 
Generally acknowledged as a sequence of operations performed on the corresponding matrix of coefficients. The solutions for the rank 
of a matrix, to calculate the determinant of a matrix, to calculate the inverse of an invertible square matrix can also be obtained 
from Gaussian elimination method. The Gaussian elimination method is named after Carl Friedrich Gauss (1777-1855), before that it 
was known to Chinese mathematicians in the era of 179 A.D.
The row reduction is performed by using elementary row operations, divided into two parts. The first part also called as forward 
elimination reduces a given system to “row echelon form”, which yields if there are no solutions, a unique solution, or infinitely 
many solutions. The second part is called as back substitution further use row operations until the solution is found. The back 
substitution put the matrix into “reduced row echelon form”. 
In this project, the first part i.e. forward elimination is parallelized to obtain a row echelon form of a system. 
Further the matrix is solved using serialized code of back substitution step to obtain unknown variables. 
