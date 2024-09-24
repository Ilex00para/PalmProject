def cyclicLR_step_size(magnitude, n_samples, batch_size):
        ''' 
        Step size for the learning rate cycle https://arxiv.org/abs/1506.01186; 

        samples training data set: 212 trees x 131 samples = 27.772,0 
        iterations = samples training data set/batch size
        recommended step size/ magnitude = (2,3 or 4) x iterations
        '''
        step_size =  int(magnitude * (n_samples/batch_size))
        return step_size