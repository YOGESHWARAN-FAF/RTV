import numpy as np
from scipy.sparse import csr_matrix

def get_laplacian_matrix(num_verts,edges):
    num_edges=edges.shape[0]
    row_list = []
    col_list = []
    v_list = []
    e_copy=edges.copy()
    #print(e_copy.dtype)
    for e in e_copy:
        e=e.sort()
    e_copy=np.unique(np.array(e_copy,np.int64),axis=0)
    #print(e_copy)
    for e in range(e_copy.shape[0]):
        id0=e_copy[e,0]
        id1=e_copy[e,1]
        row_list.append(id0)
        col_list.append(id0)
        v_list.append(1.0)
        row_list.append(id1)
        col_list.append(id1)
        v_list.append(1.0)
        row_list.append(id0)
        col_list.append(id1)
        v_list.append(-1.0)
        row_list.append(id1)
        col_list.append(id0)
        v_list.append(-1.0)

    row = np.array(row_list)
    col = np.array(col_list)

    # taking data
    data = np.array(v_list)

    # creating sparse matrix
    sparseMatrix = csr_matrix((data, (row, col)),
                              shape=(num_verts, num_verts), dtype=np.float32)
    #print(sparseMatrix)
    return sparseMatrix