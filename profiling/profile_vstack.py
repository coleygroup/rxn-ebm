import cProfile
import timeit
import numpy as np

def profile_vstack(mode='mine'):
    def hopefully_faster_vstack(tuple_of_arrays):
        array_1, array_2 = tuple_of_arrays
        width = array_1.shape[1]
        length_1, length_2 = array_1.shape[0], array_2.shape[0]
    #         assert array_1.shape[1] == array_2.shape[1]
        array_out = np.empty((length_1 + length_2, width))
        array_out[:length_1] = array_1
        array_out[length_1:] = array_2
        return array_out

    def profile_my_vstack(tuple_of_arrays):
        return hopefully_faster_vstack(tuple_of_arrays)

    def profile_numpy(tuple_of_arrays):
        return np.vstack(tuple_of_arrays)
    
    mytuple = (np.ones((4, 4096)), np.zeros((1, 4096)))
    if mode == 'mine':
        profile_my_vstack(mytuple)
    else:
        profile_numpy(mytuple)

if __name__ == '__main__':
    import timeit
    # cProfile.run('profile_vstack')
    print(timeit.timeit('profile_vstack', setup='from __main__ import profile_vstack', 
                    number=100000))

    # cProfile.run('profile_vstack')
    print(timeit.timeit('profile_vstack(mode="numpy")',  setup='from __main__ import profile_vstack', 
                    number=100000))

