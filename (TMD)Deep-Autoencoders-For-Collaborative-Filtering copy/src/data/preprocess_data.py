import pandas as pd
import numpy as np
import gc
import os
from pathlib import Path

p = Path(__file__).parents[1]

ROOT_DIR=os.path.abspath(os.path.join(p, '..', 'data/raw/'))

def convert(data, num_users, num_movies):
    ''' Making a User-Movie-Matrix'''
    
    new_data=[]
    
    for id_user in range(1, num_users+1):
        
        id_movie=data[:,1][data[:,0]==id_user]
        id_rating=data[:,2][data[:,0]==id_user]
        ratings=np.zeros(num_movies, dtype=np.uint32)
        ratings[id_movie-1]=id_rating
        if sum(ratings)==0:
            continue
        new_data.append(ratings)

        del id_movie
        del id_rating
        del ratings
        
    return new_data

def get_dataset_TMB():
    ''' For each train.dat and test.dat making a User-Movie-Matrix'''
    
    gc.enable()
    
    training_set=pd.read_csv(ROOT_DIR+'/train.txt', sep=' ', header=None, engine='python', encoding='latin-1')
    training_set=np.array(training_set, dtype=np.uint32)
    
    test_set=pd.read_csv(ROOT_DIR+'/test.txt', sep=' ', header=None, engine='python', encoding='latin-1')
    test_set=np.array(test_set, dtype=np.uint32)
    
      
    num_users=int(max(max(training_set[:,0]), max(test_set[:,0])))
    num_movies=int(max(max(training_set[:,1]), max(test_set[:,1])))


    training_set=convert(training_set,num_users, num_movies)
    test_set=convert(test_set,num_users, num_movies)

    print(num_users)
    print(num_movies)
    
    return training_set, test_set