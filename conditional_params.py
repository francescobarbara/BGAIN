

def conditional(X, M, sigma0, sigma1, mu0, mu1):
    M_mat = tf.concat(values = [M for i in range(dim)], axis = 1) 
    M_mat = tf.reshape(M_mat,  shape = [tf.shape(M_mat)[0], dim, dim])
    M_mat = tf.cast(M_mat, dtype = 'float32')

    muc = mu0 + tf.reshape( tf.matmul( tf.math.multiply( tf.cast(M_mat, dtype = 'float32'), tf.cast(sigma0, dtype = 'float32')), \
                          tf.matmul(tf.cast(tf.linalg.inv(sigma1), dtype = 'float32') , tf.cast(tf.transpose(X), dtype = 'float32'))) , shape = [tf.shape(M_mat)[0], dim])
    #sigmac = sigma_diag - tf.reshape( tf.matmul( tf.math.multiply(M_mat, sigma0), tf.matmul(tf.linalg.inv(sigma1) , tf.transpose(matmul( tf.math.multiply(M_mat, sigma0)))) , shape = [tf.shape(M_mat)[0], dim])


    #calculating sigmac here
    temp = tf.matmul(  tf.math.multiply(M_mat, sigma0)    ,  tf.matmul(tf.linalg.inv(sigma1) , tf.transpose( tf.math.multiply(M_mat, sigma0) ) ) )
    temp = tf.transpose(temp)
    temp = sigma0 - temp

    indices = [[i, i] for i in range(dim)]
    temp = tf.gather_nd(tf.transpose(temp), indices, batch_dims=0, name=None)
    temp = tf.transpose(temp)
    sigmac = temp
    
    return (muc, sigmac)