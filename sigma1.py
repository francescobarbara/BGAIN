def create_sigma_1 (data_x, data_m): 
    
    #data_m = np.array(data_m)
    #data should already have been centered IMPORTANT
    n, d = data_x.shape
    sigma_1 = np.zeros((d, d))
    
    
    
    for i in range(d):
        for j in range(d):
            count = 0
            for row in range(n):
                if data_m[row, i] == 1 and data_m[row, j] == 1:
                    sigma_1[i,j] += data_x[row, i] * data_x[row, j]
                    count += 1
                    
            if count > 1:
                sigma_1[i,j] = sigma_1[i,j]/(count - 1)   #for unbiasedness
    print('goof')            
    return sigma_1
