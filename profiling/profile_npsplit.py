import cProfile
import timeit
import numpy as np

def profile_npsplit(mode='mine'):
    def profile_my_split(myarray):
        return myarray[:, :-1], myarray[:, -1:]

    def profile_numpy(myarray):
        return np.split(myarray, [4095], axis=1)
    
    myarray = np.concatenate( ( np.ones( (1, 2048) ), np.zeros( (1, 2048) ) 
                          ), 
                         axis=1)
    if mode == 'mine':
        profile_my_split(myarray)
    else:
        profile_numpy(myarray)

if __name__ == '__main__':
    import timeit
    # profile_vstack()

    # cProfile.run('profile_npsplit')
    print(timeit.timeit('profile_npsplit', setup='from __main__ import profile_npsplit', 
                    number=100000))

    # cProfile.run('profile_npsplit(mode="numpy")')
    print(timeit.timeit('profile_npsplit(mode="numpy")',  setup='from __main__ import profile_npsplit', 
                    number=100000))

