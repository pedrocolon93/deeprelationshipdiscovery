from numpy import zeros, sqrt, dot, diag, ceil, log
from numpy.random import randn
from numpy.linalg import svd, qr
from scipy.sparse import lil_matrix as sparse_matrix
from scipy.sparse import csc_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from failed_tests.cnscraper import *
from failed_tests.matrixSketcherBase import MatrixSketcherBase


# simultaneous iterations algorithm
# inputs: matrix is input matrix, ell is number of desired right singular vectors
# outputs: transpose of approximated top ell singular vectors, and first ell singular values
from failed_tests.pmf import convert_to_coo_sparse_matrix
from failed_tests.vectorcomparer import find_vectors


def simIter(matrix, ell):
    [m,d] = matrix.shape
    num_of_iter = int(ceil(4 * log(m)))
    init_vectors = randn(m, ell)
    matrix = csc_matrix(matrix)
    matrix_trans = matrix.transpose()

    for i in range(num_of_iter):
        init_vectors = matrix.dot((matrix_trans).dot(init_vectors))

    [Q,_] = qr((matrix_trans).dot(init_vectors))
    M = matrix.dot(Q)

    [_,S,U] = svd(M, full_matrices = False)

    return (U[:,:ell].transpose()).dot(Q.transpose()), S[:ell]

  
# sparse frequent directions sketcher
class SparseSketcher(MatrixSketcherBase):

    def __init__(self, d, ell):
        self.class_name = 'SparseSketcher'
        self.d = d
        self.ell = ell
        self._sketch = zeros( (2*self.ell, self.d) )
        self.sketch_nextZeroRow = 0 

        self.buffer_ell = self.d
        self.buffer = sparse_matrix( (self.buffer_ell, self.d) )
        self.buffer_nnz = 0
        self.buffer_nextZeroRow = 0
        self.buffer_nnz_threshold = 2 * self.ell * self.d

    
    def append(self, vector): 
        if vector.nnz == 0:
            return

        if (self.buffer_nextZeroRow >= self.buffer_ell or self.buffer_nnz >= self.buffer_nnz_threshold):
            print("Rotating")
            self.__rotate__()

        self.buffer[self.buffer_nextZeroRow,:] = vector
        self.buffer_nnz += vector.nnz
        self.buffer_nextZeroRow +=1
      

    def __rotate__(self):
        # First shrink the buffer
        [Vt, s] = simIter(self.buffer, self.ell)

        # insert the shrunk part into the sketch
        if len(s) >= self.ell:
            sShrunk = sqrt(s[:self.ell]**2 - s[self.ell-1]**2)
            self._sketch[self.ell:,:] = dot(diag(sShrunk), Vt[:self.ell,:])
        else:
            self._sketch[self.ell : self.ell+len(s),:] = dot(diag(s), Vt[:len(s),:])


        # resetting the buffer matrix
        del self.buffer
        self.buffer = sparse_matrix( (self.buffer_ell, self.d) )
        self.buffer_nnz = 0
        self.buffer_nextZeroRow = 0

        # A dense shrink of the sketch
        [_,s,Vt] = svd(self._sketch, full_matrices = False)
        if len(s) >= self.ell:
            sShrunk = sqrt(s[:self.ell]**2 - s[self.ell-1]**2)
            self._sketch[:self.ell,:] = dot(diag(sShrunk), Vt[:self.ell,:])
            self._sketch[self.ell:,:] = 0
        else:
            self._sketch[:len(s),:] = dot(diag(s), Vt[:len(s),:])
            self._sketch[len(s):,:] = 0


    def get(self):
        self.__rotate__()
        return self._sketch[:self.ell,:]
  

if __name__ == '__main__':
    print("Starting test")
    # n = 100
    # d = 20
    #Singular vals to approx.
    ell = 50
    # A = rand(n, d, density=0.001, format='lil')
    print("Loading edges")
    rss = load_local_edgelist(limit=100000)
    print("Splitting")
    conceptlist, featurelist, weightlist = split_features(rss)
    print("Creating csm")
    conceptmap, featuremap, A = convert_to_coo_sparse_matrix(conceptlist, featurelist, weightlist)
    A = A.tocsr()
    print("Normalizing")
    normed_matrix = normalize(A, axis=1)

    n,d = normed_matrix.shape
    print(n,d)
    print(normed_matrix)
    sketcher = SparseSketcher(d, ell)
    print("Going through sketch")
    for idx,v in enumerate(normed_matrix):
        if idx%1000==0:
            print(idx)
        sketcher.append(v)
    sketch = sketcher.get()
    print("TSVD")
    svd = TruncatedSVD(n_components=50)
    svd.fit(sketch)
    print(svd.singular_values_)
    print(sketch)
    print("Done")
    print("Done")
    print(svd.singular_values_)
    nb_scores = []
    skecth_scores = []
    pairs = [("cat", "dog"), ("good", "bad"), ("motivation", "inspiration"), ("girl", "chick"), ("body", "girl"),
             ("britain", "united_kingdom"), ("warrior", "war")]
    for pair in pairs:
        pref = "/c/en/"
        c1 = pair[0]
        c2 = pair[1]
        print("Comparing")
        print(c1)
        print(c2)
        truck_index = conceptmap[pref + c1]
        car_index = conceptmap[pref + c2]

        truck_row = normed_matrix[truck_index].toarray()
        car_row = normed_matrix[car_index].toarray()

        truck_low_dim = svd.transform(truck_row)[:, 0]
        car_low_dim = svd.transform(car_row)[:, 0]
        testdotres = dot(truck_low_dim, car_low_dim.transpose())
        skecth_scores.append(testdotres)
        print(testdotres)
        print("Comparing against Numberbatch")
        v1, v2 = find_vectors(c1, c2)
        nbdotres = dot(v1, v2)
        nb_scores.append(nbdotres)
        print(nbdotres)

    print(np.cov([skecth_scores,nb_scores]))