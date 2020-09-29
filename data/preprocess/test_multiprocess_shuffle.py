from concurrent.futures import ProcessPoolExecutor as Pool
import os 

def main():
    distributed = False
    if distributed: 
        try:
            num_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            num_workers = os.cpu_count()
        print(f'Parallelizing over {num_workers} cores')
    else:
        num_workers = 1
    
    mylist = list(range(20))
    def add_one(xs):
        for i, x in enumerate(xs):
            xs[i] += 1
        return xs

    with Pool(max_workers=num_workers) as client:
        future = client.map(add_one, mylist)    
    print(future.results())

if __name__ == '__main__':
    main()