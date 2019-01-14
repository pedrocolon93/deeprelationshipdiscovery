from numpy import zeros, sqrt, dot, diag, ceil, log
from numpy.random import randn
from numpy.linalg import norm, svd, qr, eigh, lstsq, pinv
from scipy.linalg import orth
from scipy.sparse import lil_matrix as sparse_matrix
from scipy.sparse import csc_matrix, rand
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from cnscraper import *
from matrixSketcherBase import MatrixSketcherBase


# simultaneous iterations algorithm
# inputs: matrix is input matrix, ell is number of desired right singular vectors
# outputs: transpose of approximated top ell singular vectors, and first ell singular values
from pmf import convert_to_coo_sparse_matrix


  
# sparse frequent directions sketcher
class GeneralSketcher(MatrixSketcherBase):

    def __init__(self, A, k, l):
        self.class_name = 'General Sketcher'
        #Init
        m,n = A.shape
        self.capital_ares = np.random.normal(size=(n,k)) #nxk
        self.capital_psy = np.random.normal(size=(l,m)) #lxm
        #Ortho
        # self.capital_ares = orth(self.capital_ares)
        # self.capital_psy_star = orth(self.capital_psy.conj().T)
        #Initial sketch and co sketch
        self.Y = np.matmul(A,self.capital_ares)
        self.W = np.matmul(self.capital_psy,A)
        del A

    def update(self,H,epsilon=0.01,n=0.01):
        self.Y = epsilon*self.Y + n*np.matmul(H,self.capital_ares)
        self.W = epsilon*self.W + n*np.matmul(self.capital_psy,H)

    def simple_low_rank_approx(self):
        Q = orth(self.Y)
        t =np.matmul(self.capital_psy,Q)
        X = lstsq(t,self.W)
        return Q,X

    def low_rank_approx(self):
        Q,_ = qr(self.Y)
        del _
        mult = np.matmul(self.capital_psy,Q)
        U,T = qr(mult)
        X = np.matmul(pinv(T),(np.matmul(U,self.W)))
        return Q,X

    def fixed_rank_approx(self,rank):
        Q,X = self.low_rank_approx()
        U, Sigma, VT = randomized_svd(X,
                                      n_components=rank,
                                      n_iter=5,
                                      random_state=None)
        Q = np.matmul(Q,U)
        return Q,np.diagflat(Sigma),VT



if __name__ == '__main__':
    print("Starting test")
    # n = 100
    # d = 20
    #Singular vals to approx.
    ell = 50
    # A = rand(n, d, density=0.001, format='lil')
    print("Loading edges")
    rss = load_local_edgelist(limit=1000)
    print("Splitting")
    conceptlist, featurelist, weightlist = split_features(rss)
    print("Creating csm")
    conceptmap, featuremap, A = convert_to_coo_sparse_matrix(conceptlist, featurelist, weightlist)
    A = A.toarray()
    print("Normalizing")
    normed_matrix = normalize(A, axis=1)

    n,d = normed_matrix.shape
    print(n,d)
    print(normed_matrix)
    A = normed_matrix
    sketcher = GeneralSketcher(A, ell,ell)
    print("Going through sketch")
    # for idx,v in enumerate(normed_matrix):
    #     if idx%1000==0:
    #         print(idx)
    #     sketcher.update(v)
    Q,X = sketcher.low_rank_approx()
    sketch = np.matmul(Q,X)
    # Q,S,VT = sketcher.fixed_rank_approx(50)
    # sketch = np.matmul(Q,S)
    # sketch = np.matmul(sketch,VT)
    print("TSVD")
    svd = TruncatedSVD(n_components=50)
    svd.fit(sketch)
    print(svd.singular_values_)
    print(sketch)
    print("Done")
    # print("Done")
    # print(svd.singular_values_)

    c1 = "/c/en/motivation"
    c2 = "/c/en/inspiration"
    print("Comparing")
    print(c1)
    print(c2)
    truck_index = conceptmap[c1]
    car_index = conceptmap[c2]

    truck_row = normed_matrix[truck_index]
    car_row = normed_matrix[car_index]
    print(dot(truck_row,car_row))
    truck_low_dim = svd.transform([truck_row])
    car_low_dim = svd.transform([car_row])
    # truck_low_dim = np.matmul(sketch,truck_row)
    # car_low_dim = np.matmul(sketch,car_row)

    print(dot(truck_low_dim, car_low_dim.transpose()))
