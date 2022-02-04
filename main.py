def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args['data_name']
  miss_rate = args['miss_rate']
  
  gain_parameters = {'batch_size': args['batch_size'],
                     'hint_rate': args['hint_rate'],
                     'alpha': args['alpha'],
                     'iterations': args['iterations']}
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)    #CHANGE HERE FOR MY EXPERIMENTS
  
  #nNEW HERE
  prior0 = np.zeros(ori_data_x.shape[1])
  prior1 = np.identity(ori_data_x.shape[1])
  priors = {'mean' : prior0, 'covariance' : prior1}
  # Impute missing data
  imputed_data_x = gain(miss_data_x, gain_parameters, priors)
  
  # Report the RMSE performance
  rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
  
  print()
  print('RMSE Performance: ' + str(np.round(rmse, 4)))
  
  return imputed_data_x, rmse



#EXAMPLE HERE

args = {'data_name': 'synthetic', 'miss_rate' : 0.2, 'batch_size' : 128, 'hint_rate' : 0.9, 'alpha' : 100, 'iterations' : 10}
main(args)

