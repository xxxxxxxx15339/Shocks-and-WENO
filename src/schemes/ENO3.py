def ENO3():
    def scheme(u):
        ep = 1E-6
        
        f1 =-1/2*u[:,0]+3/2*u[:,1]
        f2 = 1/2*u[:,1]+1/2*u[:,2]
        
        B1 = (u[:,0]-u[:,1])**2
        B2 = (u[:,1]-u[:,2])**2
        
        fl = np.zeros_like(f1)
        for i in range(0,len(fl)):
            if(B1[i]>B2[i]):
                fl[i] = f2[i]
            else:
                fl[i] = f1[i]
        #fl[B1>B2] = f2[B1>B2]
        #fl[B1<=B2] = f1[B1<=B2]

        return fl
    FVM = FiniteVolumeMethod(3, scheme)
    return FVM

