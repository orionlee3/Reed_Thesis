import numpy as np
from numpy.linalg import multi_dot
from numpy.linalg import qr
import numpy.linalg as npla
import scipy as sp
from scipy.linalg import block_diag, logm, eigvals
from scipy.linalg import lu



def TAMatrix(amat, sin,cos, phases):
        #phases should be a list with length 0,1,2
        #amat should be a 2x2 array
        matrix_one = np.array([[1,0],[0,np.exp(-1j*phases[0])]])
        matrix_three = np.array([[np.exp(1j*phases[1]),0],[0,np.exp(-1j*phases[2])]])
        return (1/cos)*multi_dot([matrix_one,amat,matrix_three])
            
def TBMatrix(bmat, sin,cos,phases):   

    matrix_one = np.array([[np.exp(-1j*phases[1]),0],[0,np.exp(1j*phases[0])]])
    matrix_three = np.array([[-1,0],[0,np.exp(1j*phases[2])]])
    return (1/sin*multi_dot([matrix_one,bmat,matrix_three]))

def TAReplace(ATAMatricies, ATBMatricies):
    TAmatricies = ATAMatricies
    TBmatricies = ATBMatricies

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

def TBReplace(BTAMatricies, BTBMatricies):

    TAmatricies = BTAMatricies
    TBmatricies = BTBMatricies

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

def TAS(probability_cutoff, TA_prob, TA_nodes, TAR_nodes):
            
            TA_counter = 0
            TAR_counter = 0
            Tslist = [0 for i in range(len(TA_nodes) + len(TAR_nodes))]
        #TA_prob will be a list the same length as strip_width
            for n, i in enumerate(TA_prob):
                if i < probability_cutoff:
                    Tslist[n] = TA_nodes[TA_counter]
                    TA_counter += 1
                else:
                    Tslist[n] = TAR_nodes[TAR_counter]
                    TAR_counter += 1
                
            return block_diag(*Tslist)
        
def TBS(probability_cutoff, TB_prob, TB_nodes, TBR_nodes): 
            strip_width = len(TB_nodes) + len(TBR_nodes)
            TB_counter = 0
            TBR_counter = 0 
            Tslist = [0 for i in range(strip_width-1)]
            
            for i in range(strip_width - 1):
                if TB_prob[i] < probability_cutoff:
                    Tslist[i] = TB_nodes[TB_counter]
                    TB_counter += 1
                else:
                    Tslist[i] = TBR_nodes[TBR_counter]
                    TBR_counter += 1
            
            if TB_counter < len(TB_nodes):
                extra = TB_nodes[TB_counter]
            else: 
                extra = TBR_nodes[TBR_counter]
            temp_mat = block_diag(extra[1,1],*Tslist,extra[0,0])    
            temp_mat[0,(2*strip_width)-1] = extra[1,0]
            temp_mat[(2*strip_width)-1,0] = extra[0,1]
            return temp_mat

def FullStrip(probability_cutoff,TA_prob,TB_prob, TA_nodes, TAR_nodes, TB_nodes, TBR_nodes): #np.array 2strip_width x 2strip_width
    
        #we construct TAS for TA type strips and multiply by TB type strips
        #this means our length is really 2x our strip_length followig the convention of CC
        #probability cutoff is first introduced here to replace specific nodes in both types of strips
        
        return np.matmul(TAS(probability_cutoff, TA_prob, TA_nodes, TAR_nodes),
                         TBS(probability_cutoff, TB_prob, TB_nodes, TBR_nodes))

def FullTransfer(probabilties, strip_length,strip_width, probability_cutoff, phases, replacement_num, theta):
    
    TA_prob = probabilties[0]
    TB_prob = probabilties[1]
    
    sin = np.sin(theta)
    cos = np.cos(theta)
    
    amat = np.array([[1,-sin],[-sin,1]])
    bmat = np.array([[1,cos],[cos,1]])  
    
    TAphases = phases[0]
    TBphases = phases[1]
    TAReplace_phases = phases[2]
    TBReplace_phases = phases[3]
    
    num_of_replaced_A_nodes = replacement_num[0]
    num_of_replaced_B_nodes = replacement_num[1]
    
    #creates the list of regular nodes
    TA_nodes = [[TAMatrix(amat,sin,cos,TAphases[j][i]) for i in range((strip_width) - num_of_replaced_A_nodes[j])] for j in range(strip_length)]
    TB_nodes = [[TBMatrix(bmat,sin,cos,TBphases[j][i]) for i in range((strip_width) - num_of_replaced_B_nodes[j])] for j in range(strip_length)]
    
    #creates the nodes needed to construct the replacement node
    TAReplaceA_nodes = [[[TAMatrix(amat,sin,cos,TAReplace_phases[i][j][k]) for k in range(4)]for j in range(num_of_replaced_A_nodes[i])] for i in range(strip_length)]
    TAReplaceB_nodes = [[TBMatrix(bmat,sin,cos,TAReplace_phases[i][j][4])for j in range(num_of_replaced_A_nodes[i])] for i in range(strip_length)]

    TBReplaceB_nodes = [[[TBMatrix(bmat,sin,cos,TBReplace_phases[i][j][k]) for k in range(4)]for j in range(num_of_replaced_B_nodes[i])] for i in range(strip_length)]
    TBReplaceA_nodes = [[TAMatrix(amat,sin,cos,TBReplace_phases[i][j][4])for j in range(num_of_replaced_B_nodes[i])] for i in range(strip_length)]


    #constructs the actual replacement nodes
    TAR_nodes = [[TAReplace(TAReplaceA_nodes[i][j], TAReplaceB_nodes[i][j]) for j in range(num_of_replaced_A_nodes[i])] for i in range(strip_length)]
    TBR_nodes = [[TBReplace(TBReplaceA_nodes[i][j], TBReplaceB_nodes[i][j]) for j in range(num_of_replaced_B_nodes[i])] for i in range(strip_length)]

    #group_val determines how many matricies are multipled before 
    group_val = 8
    


    #creating matricies
    matrix_strips = [FullStrip(probability_cutoff,TA_prob[i],TB_prob[i], TA_nodes[i], TAR_nodes[i], TB_nodes[i], TBR_nodes[i]) for i in range(strip_length)]
    #splitting matricies
    every_nth = [multi_dot(matrix_strips[i:i+group_val]) for i in range(int(strip_length/group_val))]

    #This step is proved by induction, find in literature
    Tone = matrix_strips[0]
    pone,lone,uone = lu(Tone)
    bigQ = np.matmul(pone,lone)
    rlog_one = np.log(np.absolute(uone.diagonal()))

    for n,i in enumerate(every_nth):
        matrixb = np.matmul(i,bigQ)
        p,l,u = lu(matrixb)
        bigQ = np.matmul(p,l)
        rlogs = np.log(np.absolute(u.diagonal()))
        rlog_one = np.add(rlogs,rlog_one)
        
        #autosave every 50000
        if ( n % 50000):
            np.save('LULogQ.npy', bigQ)
            np.save('LULogR.npy',rlog_one)
    return (rlog_one)