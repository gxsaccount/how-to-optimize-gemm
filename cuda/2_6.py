import numpy as np 

m=k=n=512
a=np.array([i for i in range(m*k)]).reshape(m,k)
b=np.array([i for i in range(k*n)]).reshape(k,n) 
# a=np.ones(shape=[m,n]).reshape(m,k)
# b=np.ones(shape=[m,n]).reshape(k,n) 
# a = np.random.rand(m,k)
# b = np.random.rand(k,n)
c=np.zeros(shape=[m,n]) 
block = 128 
gold_c = np.matmul(a,b) 

def kernal(bx,by,tx,ty,m,n,k): 
    global gold_c
    sum=np.zeros(shape=[8,8])  

    import pdb 
    pdb.set_trace()
    for loop in range(0,k,8):
        # import pdb 
        # pdb.set_trace()
        # all bloack use same share 
        ashare = a[bx*128:bx*128+128,loop:loop+8]  # a[128*8]
        bshare = b[loop:loop+8,by*128:by*128+128] # b[8*128] 
        #sync 
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    sum[i][j] += ashare[i+tx*8][k] *bshare[k][j+ty*8] 

        #sync
    # 一共128*128 row:(tx*64)/128  col:(tx*64)%128 
    row = bx*128 + tx*8 
    col = by*128 + ty*8
    for i in range(8):
        for j in range(8):
            c[row+i][col+j] = sum[i][j] 
            if not (abs(c[row+i][col+j] - gold_c[row+i][col+j]) < 0.001):
                print("[{},{}] no pass".format(bx,by,tx,ty))
                import pdb 
                pdb.set_trace() 
                assert(False)
'''
check single thread 
'''
# kernal(0,0,0,0,m,n,k)
'''
check all thread 
'''
# gold_c = np.matmul(a,b) 
# for bx in range(int(m/128)):
#     for by in range(int(n/128)):
#         for tx in range(16):
#             for ty in range(16):
#                 kernal(bx,by,tx,ty,m,n,k)
#         print("[{},{}], has pass".format(bx,by))
# assert((gold_c.astype(int)==c.astype(int)).all())   



'''
check per block (share mem)
'''
def load_one_loop(bx,by,tx,ty,loop,ashare,bshare,m,n,k):
    _a = a.reshape(m*k)
    _b = b.reshape(k*n)
    threadNo = tx * 16 + ty
    aindex = bx*m*128 + int((threadNo*4)/ 8)*m + (threadNo*4)%8 # a:128*8
    bindex = by*128 + int((threadNo*4)/ 128)*k + (threadNo*4)%128 # b:8*128 
    print(bx,by,tx,ty)
    for i in range(4):
        print(i+aindex+loop)
        ashare[threadNo*4+i] = _a[i+aindex+loop]
        bshare[threadNo*4+i] = _b[i+bindex+loop*128*4]

'''
check single block  
'''
ashare=np.zeros(shape=[1024]) 
bshare=np.zeros(shape=[1024]) 
for loop in range(0,k,8): 
    for tx in range(16):
        for ty in range(16):
            load_one_loop(2,2,tx,ty,loop,ashare,bshare,m,n,k) 

'''
check all thread 
'''
# for bx in range(int(m/128)):
#     for by in range(int(n/128)):
#         ashare=np.zeros(shape=[1024]) 
#         bshare=np.zeros(shape=[1024]) 
#         for loop in range(0,k,8): 
#             for tx in range(16):
#                 for ty in range(16):
#                     load_one_loop(bx,by,tx,ty,loop,ashare,bshare,m,n,k) 
#             gold_a= a[bx*128:bx*128+128,loop:loop+8].astype(int).reshape(-1)
#             gold_b= b[loop:loop+8,by*128:by*128+128].astype(int).reshape(-1)
#             if not (ashare.astype(int)==gold_a).all() :
#                 import pdb 
#                 pdb.set_trace()
#                 assert(False)
#             if not (bshare.astype(int)==gold_b).all() :
#                 import pdb 
#                 pdb.set_trace()
#                 assert(False)
#         print("[{},{}], has pass".format(bx,by)) 

