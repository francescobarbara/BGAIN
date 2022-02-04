def gain (data_x, gain_parameters, priors):
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:     #this is a dictionary
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  ''' priors:
      priors['covariance'] is a dxd array
      priors['mean'] is a d array
      '''
  
  

  
  # Define mask matrix
  data_m = 1-np.isnan(data_x)
   
  
  
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  
  #NEW STUFF HERE
  sigma1 = create_sigma_1(norm_data, data_m) 
  mu1 = np.nanmean(data_x, axis = 0)  #should be all zero (sanity check)
  #ALSO, VOLENDO, MODIFY PRIORS USING NORM_PARAMETERS
  sigma0 = priors['covariance']
  mu0 = priors['mean']
  
  #CONVERT THEM TO TENSORS
  sigma1 = tf.cast(tf.convert_to_tensor(sigma1), dtype = 'float32')
  mu1 = tf.cast(tf.convert_to_tensor(mu1), dtype = 'float32')
  sigma0 = tf.cast(tf.convert_to_tensor(sigma0), dtype = 'float32')
  mu0 = tf.cast(tf.convert_to_tensor(mu0), dtype = 'float32')


  ## GAIN architecture   
  # Input placeholders
  # Data vector
  X = tf.placeholder(tf.float64, shape = [None, dim])     #changed to 64 !!!!!!!!!!!!
  # Mask vector 
  M = tf.placeholder(tf.float64, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float64, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs   #credo x_tilde, h as in Algo 1
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 =  tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 =  tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs   #the output dimension should be the same as the data (we are trying to replicate it)
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]   #set of params
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 =  tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 =  tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    print(type(x))
    print(type(m))
    inputs = tf.cast(tf.concat(values = [x, m], axis = 1), dtype = 'float32')
    print(type(inputs))
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)      #cambia qui!!! tipo elimina sigmoid
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    print(type(x))
    
    inputs = tf.cast(tf.concat(values = [tf.cast(x, dtype = 'float32'), tf.cast(h, dtype = 'float32')], axis = 1), dtype = 'float32')
    print(type(inputs))
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
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
      
      
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M)
  print(G_sample)
  print(type(G_sample))
 
  # Combine with observed data
  Hat_X = tf.cast(X, dtype = 'float32') * tf.cast(M, dtype = 'float32') + tf.cast(G_sample, dtype = 'float32') * tf.cast((1-M), dtype = 'float32')
  print('hat')
  print(Hat_X)
  print(type(Hat_X))
  #new here the vectors muc and sigmac
  conditional_mu, conditional_sigma = conditional(X, M, sigma0, sigma1, mu0, mu1)   #should put X instead of Hat_X
  print('cond_mu')
  print(conditional_mu)
  print(type(conditional_mu))
  
  
  
  # Discriminator
  D_prob = discriminator(Hat_X, H)
  
  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(tf.cast(M, dtype = 'float32') * tf.cast(tf.log(D_prob + 1e-8), dtype = 'float32') \
                                + tf.cast((1-M), dtype = 'float32') * tf.cast(tf.log(1. - D_prob + 1e-8) , dtype = 'float32'))
  
  G_loss_temp = -tf.reduce_mean(tf.cast((1-M), dtype = 'float32') * tf.cast(tf.log(D_prob + 1e-8), dtype = 'float32'))
  
  MSE_loss = \
  tf.reduce_mean((tf.cast(M, dtype = 'float32') * tf.cast(X, dtype = 'float32') - tf.cast(M, dtype = 'float32') * tf.cast(G_sample, dtype = 'float32'))**2) / \
  tf.cast(tf.reduce_mean(M), dtype = 'float32')
  
  #ADD THIRD LOSS HERE
  print('M')
  print(M.shape)
  print('Hat_X')
  print(Hat_X.shape)
  print('conditional_mu')
  print(conditional_mu.shape)
  print('conditional_sigma')
  print(conditional_sigma.shape)
  print('D_prob')
  print(D_prob.shape)
  
  loss_3 = \
  -tf.reduce_mean( tf.cast((1-M), dtype = 'float32') * tf.cast( tf.exp(  - (tf.cast(Hat_X, dtype = 'float32') - tf.cast(conditional_mu, dtype = 'float32'))**2 \
                  / (2*tf.cast(conditional_sigma, dtype = 'float32'))) , dtype = 'float32')  )
  print('loss_3')
  print(loss_3.shape)

  """
  j_index = M == 1
  loss_3 = 0"""
  """for i in range(dim):
      loss_3 += (1 - M[i])*tf.math.exp(-tf.math.pow( (Hat_X[i] - mu0[i] - 
                tf.tensordot( tf.convert_to_tensor(conditional_mu[i]), 
                    tf.boolean_mask(Hat_X[i], j_index) - tf.boolean_mask(mu1, j_index) ) ), 2) / (2*conditional_sigma[i])) 
        """
  
  
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss  + loss_3   #as defined in the paper, add beta
  
  ## GAIN solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
   
  # Start Iterations
  for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]     #tells you where data is missing (look top of script)
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
    print('checkpoint 1')  
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
    print('checkpoint 2')  
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    print('checkpoint 3') 
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})
    print('checkpoint 4')        
  ## Return imputed data      
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
  print('checkpoint 5')     
  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  print('checkpoint 6') 
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  print('checkpoint 7') 
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  print('checkpoint 8') 
  # Rounding
  imputed_data = rounding(imputed_data, data_x)  
  print('checkpoint 9')        
  return imputed_data

