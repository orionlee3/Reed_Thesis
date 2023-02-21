import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import block_diag

def TAMatrix(sin,cos):
    phases = (2*np.pi)*np.random.random_sample(3)
    
    matrix_one = np.array([[1,0],[0,np.exp(-1j*phases[0])]])
    matrix_two = np.array([[1,-sin],[-sin,1]])
    matrix_three = np.array([[np.exp(1j*phases[1]),0],[0,np.exp(-1j*phases[2])]])

    return (1/cos)*multi_dot([matrix_one,matrix_two,matrix_three])
            
def TBMatrix(sin,cos):   
    
    phases = (2*np.pi)*np.random.random_sample(3)
    
    matrix_one = np.array([[np.exp(-1j*phases[1]),0],[0,np.exp(1j*phases[0])]])
    matrix_two = np.array([[1,cos],[cos,1]])
    matrix_three = np.array([[-1,0],[0,np.exp(1j*phases[2])]])

    return (1/sin*multi_dot([matrix_one,matrix_two,matrix_three]))

def TAReplace(sin,cos):
    TAmatricies = [TAMatrix(sin,cos) for i in range(4)]
    TBmatricies = TBMatrix(sin,cos)
    
    M = multi_dot([
            block_diag(TAmatricies[0],TAmatricies[1]),
            block_diag(1,TBmatricies,1),
            block_diag(TAmatricies[2],TAmatricies[3])
    ])

    new_TA = np.zeros((2,2), dtype = 'complex_')
    denominator = (M[2,1]+M[2,2]-M[1,1]-M[1,2])
    new_TA[0,0] = M[0,0] + (((M[0,1] + M[0,2])*(M[1,0]-M[2,0]))/denominator)
    new_TA[0,1] = M[0,3] + (((M[0,1] + M[0,2])*(M[1,3]-M[2,3]))/denominator)
    new_TA[1,0] = M[3,0] + (((M[3,1] + M[3,2])*(M[1,0]-M[2,0]))/denominator)
    new_TA[1,1] = M[3,3] + (((M[3,1] + M[3,2])*(M[1,3]-M[2,3]))/denominator)

    return new_TA

def TBReplace(sin,cos):
    
    TBmatricies = [TBMatrix(sin,cos) for i in range(4)]
    TAmatricies = TAMatrix(sin,cos)
    
    M = multi_dot([
            block_diag(TBmatricies[0],TBmatricies[1]),
            block_diag(1,TAmatricies,1),
            block_diag(TBmatricies[2],TBmatricies[3])
    ])

    new_TB = np.zeros((2,2), dtype = 'complex_')
    denominator = (M[2,1]+M[2,2]-M[1,1]-M[1,2])
    new_TB[0,0] = M[0,0] + (((M[0,1] + M[0,2])*(M[1,0]-M[2,0]))/denominator)
    new_TB[0,1] = M[0,3] + (((M[0,1] + M[0,2])*(M[1,3]-M[2,3]))/denominator)
    new_TB[1,0] = M[3,0] + (((M[3,1] + M[3,2])*(M[1,0]-M[2,0]))/denominator)
    new_TB[1,1] = M[3,3] + (((M[3,1] + M[3,2])*(M[1,3]-M[2,3]))/denominator)

    return new_TB
    

