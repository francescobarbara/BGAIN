def normalization (data, parameters=None):    #tutto ok, input semplice dataset
  '''Normalize data 
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    mean = np.nanmean(data, axis = 0)
    std = np.nanstd(data, axis = 0)
    
    # For each dimension
    for i in range(dim):
        
      norm_data[:,i] = norm_data[:,i] - mean[i]
      if std[i] != 0:
          norm_data[:,i] = norm_data[:,i] / std[i]   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'mean': mean,
                       'std': std}  
      
  return norm_data, norm_parameters

def renormalization (norm_data, norm_parameters): #norm_pars is a dict
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  mean = norm_parameters['mean']
  std = norm_parameters['std']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * std[i] + mean[i]   
    
  return renorm_data

