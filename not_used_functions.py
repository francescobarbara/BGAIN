def create_dataset(n,d):
  return np.random.uniform(size = (n,d))

def conditional_sigma2(m, sigma0, sigma1, d):
    #return (A) at page D, but for all i grouped together
    '''print(m)
    print(type(m))
    print(m.shape)
    print(type(m.shape))'''
    


    out = np.zeros(d)
    j_indices = m == 1
    print(j_indices)
    print(type(j_indices))
    sigma1_jj_inv = np.linalg.inv(sigma1[j_indices][:, j_indices])
    
    for i in range(d):
        
        sigma0_i_j = sigma0[i, j_indices].reshape(1, sum(j_indices))
        
        out[i] = sigma0[i,i] - np.matmul( np.matmul(sigma0_i_j, sigma1_jj_inv) , np.transpose(sigma0_i_j))
    print('good')  
    return out


def conditional_mu2(m, sigma0, sigma1, d):
    #return (..) in (B) at page D it returns a list of 1xsum(m) vectors
    #m = np.array(m) #doesn't work
    ''' print(m)
    print(type(m))
    print(m.shape)
    print(type(m.shape))'''
    


    out = np.zeros((d, d))
    j_indices = (m == 1)
    
    sigma1_jj_inv = np.linalg.inv(sigma1[j_indices][:, j_indices])
    
    for i in range(d):
        
        sigma0_i_j = sigma0[i, j_indices].reshape(1, sum(j_indices))
        
        out[i] = np.matmul(sigma0_i_j, sigma1_jj_inv)
    print('goog')    
    return out


def matmul(tensor1, tensor2):
  tensor1 = tf.cast(tensor1, dtype='float64')
  tensor2 = tf.cast(tensor2, dtype='float64')
  return tf.matmul(tensor1, tensor2)
