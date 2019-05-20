from numpy import dot
from numpy.linalg import qr
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from failed_tests.cnscraper import *
from failed_tests.matrixSketcherBase import MatrixSketcherBase


# simultaneous iterations algorithm
# inputs: matrix is input matrix, ell is number of desired right singular vectors
# outputs: transpose of approximated top ell singular vectors, and first ell singular values
from failed_tests.pmf import convert_to_coo_sparse_matrix


  
# sparse frequent directions sketcher
from failed_tests.vectorcomparer import find_vectors


def Orthogonal(next_x, Q_j):
    next_x = next_x.reshape(len(next_x),1)
    inpt = Q_j
    q,r = qr(inpt)
    m0 = np.matmul(q,q.transpose())
    mult = np.matmul(m0,next_x.reshape(len(next_x),1))
    diff = next_x - mult
    return diff



def Orthonormal(next_x, Q_j):
    r = Orthogonal(next_x,Q_j)
    mag = np.linalg.norm(r)
    res = r
    if mag != 0:
        res = r/mag
    res = res.transpose()
    return res
    


class IsolsExact(MatrixSketcherBase):

    def __init__(self):
        self.class_name = 'ISOLS exact sketcher Sketcher'
        #Init



    def update(self,H,epsilon=0.01,n=0.01):
        self.Y = epsilon*self.Y + n*np.matmul(H,self.capital_ares)
        self.W = epsilon*self.W + n*np.matmul(self.capital_psy,H)



    def fixed_rank_approx(self,X,k):

        m, n = X.shape
        N = n
        # Y = np.random.normal(size=(m, N))  # lxm
        Y = X
        u = []
        v = []
        f = []
        #INitialization
        print("INitializing params")
        def fun(u,v):
            return u/v
        i_star = 0

        for col_i in range(n):
            if col_i %1000 == 0:
                print(col_i)
            u.append(np.linalg.norm(np.matmul(Y.transpose(),X[:,col_i]),2))
            v.append(np.linalg.norm(X[:,col_i], 2))
            f.append(fun(u[col_i],v[col_i]))
            i_star = np.argmax(f)
        picked = [i_star]
        x_s = X[:,i_star]
        S = [x_s]
        q_j = x_s/np.linalg.norm(x_s)
        q_j = np.array(q_j).reshape(len(q_j),1)
        Q_j = np.array(q_j)

        #Iteration
        beta = []
        gamma = [[]]
        print("Starting")
        for j in range(0,k-1):
            print("Iteration of k ")
            print(j)
            c_j = np.matmul(np.matmul(Y,Y.transpose()),q_j)
            d_j = c_j-np.matmul(np.matmul(q_j,q_j.transpose()),c_j)
            b_j = np.matmul(q_j.transpose(),c_j)
            alpha_j = []
            gamma_j = []
            for col_i in range(0,n):
                ##
                tinyx = np.array(X[:,col_i])
                temp = np.matmul(np.array(q_j).reshape(len(q_j),1).transpose(),tinyx.reshape((len(tinyx),1)))
                alpha_j.append(temp[0][0])
                gamma_j.append(np.matmul(np.array(d_j).transpose(),X[:,col_i]))
                ##
                u[col_i] = u[col_i]+ alpha_j[col_i]**2*b_j-2*alpha_j[col_i]*gamma_j[col_i]
                v[col_i] = v[col_i]-alpha_j[col_i]**2
                # print("Here")
                f[col_i] = (u[col_i]/v[col_i])[0,0]
            # f_found = [(idx,val) for idx,val in enumerate(f)]
            f_restricted = np.delete(f,picked)
            i_star = np.argmax(f_restricted)
            start = len(picked)
            for idx, val in enumerate(f):
                if val == f_restricted[i_star] and idx not in picked:
                    picked.append(idx)
                    break
            end = len(picked)
            if start == end:
                raise Exception("No val found!!@<LK!")
            i_star = picked[-1]
            # while True:
            #     if i_star not in picked:
            #         picked.append(i_star)
            #         break
            #     else:




            # i_star = np.argmax(f)
            print(i_star)
            next_x = X[:,i_star]
            S.append(next_x)
            q_j = Orthonormal(next_x,Q_j)
            q_j = q_j.transpose()
            Q_j = np.hstack((Q_j,q_j))
            # add the next x
        return S

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
    A = A.toarray()
    print("Normalizing")
    normed_matrix = normalize(A, axis=1)

    n,d = normed_matrix.shape
    print(n,d)
    print(normed_matrix)
    A = normed_matrix
    sketcher = IsolsExact()
    print("Going through sketch")
    # for idx,v in enumerate(normed_matrix):
    #     if idx%1000==0:
    #         print(idx)
    #     sketcher.update(v)
    sketch = sketcher.fixed_rank_approx(A,ell)

    print("TSVD")
    svd = TruncatedSVD(n_components=50)
    svd.fit(sketch)
    print(svd.singular_values_)
    # print(sketch)
    print("Done")
    # print("Done")
    # print(svd.singular_values_)

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

        truck_row = sketch[truck_index]
        car_row = sketch[car_index]
        print(dot(truck_row, car_row.transpose()))
        truck_low_dim = svd.transform([truck_row])[:, 0]
        car_low_dim = svd.transform([car_row])[:, 0]

        testdotres = dot(truck_low_dim, car_low_dim.transpose())
        skecth_scores.append(testdotres)
        print(testdotres)
        print("Comparing against Numberbatch")
        v1, v2 = find_vectors(c1, c2)
        nbdotres = dot(v1, v2)
        nb_scores.append(nbdotres)
        print(nbdotres)

    print(np.cov([skecth_scores, nb_scores]))

    # c1 = "/c/en/motivation"
    # c2 = "/c/en/inspiration"
    # print("Comparing")
    # print(c1)
    # print(c2)
    # truck_index = conceptmap[c1]
    # car_index = conceptmap[c2]
    #
    # truck_row = sketch[truck_index]
    # car_row = sketch[car_index]
    # print(dot(truck_row,car_row.transpose()))
    # truck_low_dim = svd.transform([truck_row])[:,0]
    # car_low_dim = svd.transform([car_row])[:,0]
    # # truck_low_dim = np.matmul(sketch,truck_row)
    # # car_low_dim = np.matmul(sketch,car_row)
    #
    # print(dot(truck_low_dim, car_low_dim.transpose()))
