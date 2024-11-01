

def genData(which_sim, nhosp, hosp_size):
    
    import pandas as pd
    import numpy as np
    
    if which_sim == 1 : # binary treatment with linear decision boundary
        # number of hospital
        nhosp = nhosp
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        X1 = np.random.uniform(low = 0, high = 1, size = n)
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2

        ## Assigned treatment
        prob_a = 0.5
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = (X1>0.5) * (5+10*(A==1)+22*(S>=half)-24*(A==1)*(S>=half)) + (X1<=0.5)*(11+19*(A==1)+2*(S>=half)-32*(A==1)*(S>=half)) + noise
    
        ## Generate potential outcome
        A_p = 0
        Y0_true = (X1>0.5) * (5+10*(A_p==1)+22*(S>=half)-24*(A_p==1)*(S>=half)) + (X1<=0.5)*(11+19*(A_p==1)+2*(S>=half)-32*(A_p==1)*(S>=half))
        A_p = 1
        Y1_true = (X1>0.5) * (5+10*(A_p==1)+22*(S>=half)-24*(A_p==1)*(S>=half)) + (X1<=0.5)*(11+19*(A_p==1)+2*(S>=half)-32*(A_p==1)*(S>=half))
                
        
        # if X<=0.5, then S>10 is the vulnerable group, so d=-1
        # if X>0.5, then S<=10 is the vulnerable group, so d=1
        d = np.where((X1>0.5), 1, 0)
        d_exp = np.where((X1>0.5), 0, 1)

        is_vul = [999]*n
        for i in range(n):
            if X1[i]<=0.5 and S[i]>=half:
                is_vul[i] = 1
            elif X1[i]>0.5 and S[i]<half:
                is_vul[i] = 1
            else:
                is_vul[i] = 0
                
        obj = [999]*n
        for i in range(n):
            if X1[i]<=0.5:
                obj[i] = 11
            else:
                obj[i] = 13
                
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)

                
        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype(np.uint8)
        data["X1"] = X1
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true
        data["d"] = d
        data["d_exp"] = d_exp
        data["d_opt"] = d_opt
        data["is_vul"] = is_vul
        data['objective'] = obj

        dummies = pd.get_dummies(data['S'], dtype=np.uint8).rename(columns=lambda x: 'S' + str(x))
        data = pd.concat([data, dummies], axis=1)
        dummies_names = dummies.columns.values.tolist()

        Y_name = "Y"
        A_name = "A"
        X_names = ["X1"]
        S_names = None
        X_names_exp = X_names + dummies_names
        is_rct = True
        is_class = False
        s_type = "disc"
        qua_use = 0.25
        
    
    if which_sim == 2 : # binary treatment with nonlinear decision boundary
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        X1 = np.random.uniform(low = 0, high = 1, size = n)
        X2 = np.random.uniform(low = 0, high = 1, size = n)
        X3 = np.random.uniform(low = 0, high = 1, size = n)
        X4 = np.random.uniform(low = 0, high = 1, size = n)
        X5 = np.random.uniform(low = 0, high = 1, size = n)
        X6 = np.random.uniform(low = 0, high = 1, size = n)
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Assigned treatment
        prob_a = 1 / (1 + np.exp( 0.6*K - 0.6*X1 + 0.6*X2 - 0.6*X3 + 0.6*X4 - 0.6*X5 + 0.6*X6 ))
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        #Y = (0.5 + 1*A + np.exp(K) - 2.5*A*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A + 0.2*np.exp(K) - 3.5*A*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4)) + noise
        #Y = (np.sin(np.pi * X1 * X2) - np.cos(np.pi * X3 * X4)) * (np.exp(K-1) + 0.1*A) + K + 5 * X5 + noise
        Y = (X1*X2 - X3*X4) * (-np.exp(K-1) - A + A*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A + 3*A*K) + K + 5 + noise
        
        ## Generate potential outcome
        A_p = 0
        #Y0_true = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
        Y0_true = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5
        A_p = 1
        #Y1_true = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
        Y1_true = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            #Y0mat['Y_'+str(i)] = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
            Y0mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5
            A_p = 1
            #Y1mat['Y_'+str(i)] = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
            Y1mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5

        
        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)
        
        is_vul = [999]*n
        vul_hosp_list = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0
            vul_hosp_list[i] = vul_hosp[0]
        is_vul = np.array(is_vul)
            
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        d = np.where((g1-g0)>0, 1, 0)
        obj = np.maximum(g0, g1)
        
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)
        
        ## Generate d_exp
        A_p = 0
        K = 1
        Y0_K1 = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5
        K = 0
        Y0_K0 = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5
        Y0_avg = (Y0_K0 + Y0_K1)/2
        
        A_p = 1
        K = 1
        Y1_K1 = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5
        K = 0
        Y1_K0 = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5
        Y1_avg = (Y1_K0 + Y1_K1)/2
        
        d_exp = np.where((Y1_avg-Y0_avg>0), 1, 0)
        
        
        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["X1"] = X1
        data["X2"] = X2
        data["X3"] = X3
        data["X4"] = X4
        data["X5"] = X5
        data["X6"] = X6
        data["d"] = d
        data["d_exp"] = d_exp
        data["d_opt"] = d_opt
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true
        data["is_vul"] = is_vul
        data['objective'] = obj
        
        #vul_data = data[data["is_vul"]==1]
        #np.mean(vul_data["d_exp"]==vul_data['d'])

        dummies = pd.get_dummies(data['S'], dtype=int).rename(columns=lambda x: 'S' + str(x))
        data = pd.concat([data, dummies], axis=1)
        dummies_names = dummies.columns.values.tolist()

        Y_name = "Y"
        A_name = "A"
        X_names = ["X1", "X2", "X3", "X4", "X5", "X6"]
        S_names = None
        X_names_exp = X_names + dummies_names
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
        
        
    if which_sim == 3 : # multi-category treatment with linear decision boundary
        
        # number of hospital
        nhosp = nhosp
        # number of total subject
        n = nhosp*hosp_size

        # Generate data
        data = pd.DataFrame()
        X1 = np.random.uniform(low = 0, high = 1, size = n)

        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2

        ## Assigned treatment
        A = np.random.choice(np.arange(0, 3), size=n, p=[1/3,1/3,1/3])

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = [999]*n
        for i in range(n):
            if X1[i]<0.3 and S[i]<half and A[i]==0:
                Y[i] = 7
            elif X1[i]<0.3 and S[i]<half and A[i]==1:
                Y[i] = 4
            elif X1[i]<0.3 and S[i]<half and A[i]==2:
                Y[i] = 20
            if X1[i]<0.3 and S[i]>=half and A[i]==0:
                Y[i] = 8
            elif X1[i]<0.3 and S[i]>=half and A[i]==1:
                Y[i] = 13
            elif X1[i]<0.3 and S[i]>=half and A[i]==2:
                Y[i] = 0
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]<half and A[i]==0:
                Y[i] = 2
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]<half and A[i]==1:
                Y[i] = 9
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]<half and A[i]==2:
                Y[i] = 15
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]>=half and A[i]==0:
                Y[i] = 22
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]>=half and A[i]==1:
                Y[i] = 10
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]>=half and A[i]==2:
                Y[i] = 5
            elif X1[i]>=0.5 and S[i]<half and A[i]==0:
                Y[i] = 13
            elif X1[i]>=0.5 and S[i]<half and A[i]==1:
                Y[i] = 4
            elif X1[i]>=0.5 and S[i]<half and A[i]==2:
                Y[i] = 9
            elif X1[i]>=0.5 and S[i]>=half and A[i]==0:
                Y[i] = 6
            elif X1[i]>=0.5 and S[i]>=half and A[i]==1:
                Y[i] = 22
            elif X1[i]>=0.5 and S[i]>=half and A[i]==2:
                Y[i] = 12

        Y = Y + noise
        
        ## Generate potential outcome
        Y0_true = [999]*n
        for i in range(n):
            if X1[i]<0.3 and S[i]<half:
                Y0_true[i] = 7
            if X1[i]<0.3 and S[i]>=half:
                Y0_true[i] = 8
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]<half:
                Y0_true[i] = 2
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]>=half:
                Y0_true[i] = 22
            elif X1[i]>=0.5 and S[i]<half:
                Y0_true[i] = 13
            elif X1[i]>=0.5 and S[i]>=half:
                Y0_true[i] = 6
                
        Y1_true = [999]*n
        for i in range(n):
            if X1[i]<0.3 and S[i]<half:
                Y1_true[i] = 4
            if X1[i]<0.3 and S[i]>=half:
                Y1_true[i] = 13
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]<half:
                Y1_true[i] = 9
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]>=half:
                Y1_true[i] = 10
            elif X1[i]>=0.5 and S[i]<half:
                Y1_true[i] = 4
            elif X1[i]>=0.5 and S[i]>=half:
                Y1_true[i] = 22
                
        Y2_true = [999]*n
        for i in range(n):
            if X1[i]<0.3 and S[i]<half:
                Y2_true[i] = 20
            if X1[i]<0.3 and S[i]>=half:
                Y2_true[i] = 0
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]<half:
                Y2_true[i] = 15
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]>=half:
                Y2_true[i] = 5
            elif X1[i]>=0.5 and S[i]<half:
                Y2_true[i] = 9
            elif X1[i]>=0.5 and S[i]>=half:
                Y2_true[i] = 12


        d = [999]*n
        d_exp = [999]*n
        for i in range(n):
            if X1[i]<0.3:
                d[i]=0
                d_exp[i]=2
            elif X1[i]>=0.3 and X1[i]<0.5:
                d[i]=1
                d_exp[i]=0
            elif X1[i]>=0.5:
                d[i]=2
                d_exp[i]=1


        is_vul = [999]*n
        for i in range(n):
            if X1[i]<0.3 and S[i]>=half:
                is_vul[i] = 1
            elif X1[i]>=0.3 and X1[i]<0.5 and S[i]<half:
                is_vul[i] = 1
            elif X1[i]>=0.5 and S[i]<half:
                is_vul[i] = 1
            else:
                is_vul[i] = 0
                
        obj = [999]*n
        for i in range(n):
            if X1[i]<0.3:
                obj[i] = 7
            elif X1[i]>=0.3 and X1[i]<0.5:
                obj[i] = 9
            else:
                obj[i] = 9
        
        # Optimal treatment under the true Y0, Y1 and Y2
        Y_stack = np.stack((Y0_true, Y1_true, Y2_true))
        d_opt = np.argmax(Y_stack, axis=0)

        data['id'] = range(n)   
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["X1"] = X1
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true
        data["Y2_true"] = Y2_true
        data["d"] = d
        data["d_exp"] = d_exp
        data["d_opt"] = d_opt
        data["is_vul"] = is_vul
        data['objective'] = obj

        dummies = pd.get_dummies(data['S'], dtype=int).rename(columns=lambda x: 'S' + str(x))
        data = pd.concat([data, dummies], axis=1)
        dummies_names = dummies.columns.values.tolist()

        Y_name = "Y"
        A_name = "A"
        X_names = ["X1"]
        S_names = None
        X_names_exp = X_names + dummies_names
        is_rct = True
        is_class = False
        s_type = "disc"
        qua_use = 0.25
        
    
    if which_sim == 4 : # multi-category treatment with nonlinear decision boundary
        
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        X1 = np.random.uniform(low = 0, high = 1, size = n)
        X2 = np.random.uniform(low = 0, high = 1, size = n)
        X3 = np.random.uniform(low = 0, high = 1, size = n)
        X4 = np.random.uniform(low = 0, high = 1, size = n)
        X5 = np.random.uniform(low = 0, high = 1, size = n)
        X6 = np.random.uniform(low = 0, high = 1, size = n)
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Assigned treatment
        xbeta0 = 0.6*K - 0.6*X1 + 0.6*X2 - 0.6*X3 + 0.6*X4 - 0.6*X5 + 0.6*X6
        xbeta1 = 0.6*K + 0.6*X1 - 0.6*X2 + 0.6*X3 - 0.6*X4 + 0.6*X5 - 0.6*X6
        prob_a0 = np.exp(xbeta0) / (1 + np.exp(xbeta0) + np.exp(xbeta1))
        prob_a1 = np.exp(xbeta1) / (1 + np.exp(xbeta0) + np.exp(xbeta1))
        prob_a2 = 1-prob_a0-prob_a1
        
        ## Assigned treatment
        A = [999]*n
        for i in range(n):
            A[i] = np.random.choice(np.arange(0, 3), size=1, p=[prob_a0[i], prob_a1[i], prob_a2[i]])
        A = np.array(A).reshape((n,))
        

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = 0.05 * ((0.5 + 1*A + np.exp(K) - 2.5*A*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A + 0.2*np.exp(K) - 3.5*A*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))) + noise
        
        ## Generate potential outcome
        A_p = 0
        Y0_true = 0.05 * ((0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4)))
        A_p = 1
        Y1_true = 0.05 * ((0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4)))
        A_p = 2
        Y2_true = 0.05 * ((0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4)))
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        Y2mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            Y0mat['Y_'+str(i)] = 0.05 * ((0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4)))
            A_p = 1
            Y1mat['Y_'+str(i)] = 0.05 * ((0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4)))
            A_p = 2
            Y2mat['Y_'+str(i)] = 0.05 * ((0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4)))
        
        minMatrix = np.minimum(np.minimum(Y0mat, Y1mat), Y2mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)
        
        is_vul = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0
                
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        g2 = Y2mat.min(axis=1)
        obj = np.maximum(g0, g1, g2)
        
        q01 = g0 - g1
        q02 = g0 - g2
        q10 = g1 - g0
        q12 = g1 - g2
        q20 = g2 - g0
        q21 = g2 - g1
        
        q0 = q01 + q02
        q1 = q10 + q12
        q2 = q20 + q21
        
        d = []
        for i in range(n):
            max_val = max(q0[i], q1[i], q2[i])
            if max_val == q0[i]:
                d.append(0)
            elif max_val == q1[i]:
                d.append(1)
            else:
                d.append(2)
        
        # Optimal treatment under the true Y0, Y1 and Y2
        Y_stack = np.stack((Y0_true, Y1_true, Y2_true))
        d_opt = np.argmax(Y_stack, axis=0)
        
        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["X1"] = X1
        data["X2"] = X2
        data["X3"] = X3
        data["X4"] = X4
        data["X5"] = X5
        data["X6"] = X6
        data["d"] = d
        data["d_opt"] = d_opt
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true
        data["Y2_true"] = Y2_true
        data["is_vul"] = is_vul
        data['objective'] = obj

        dummies = pd.get_dummies(data['S'], dtype=int).rename(columns=lambda x: 'S' + str(x))
        data = pd.concat([data, dummies], axis=1)
        dummies_names = dummies.columns.values.tolist()

        Y_name = "Y"
        A_name = "A"
        X_names = ["X1", "X2", "X3", "X4", "X5", "X6"]
        S_names = None
        X_names_exp = X_names + dummies_names
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
    
    
    if which_sim == 5 : # binary treatment with nonlinear decision boundary (large p)
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        
        # number of covariates
        p = 50
        
        for i in range(p):
            data['X'+str(i+1)] = np.random.uniform(low = 0, high = 1, size = n)
        
        X1 = data['X1']
        X2 = data['X2']
        X3 = data['X3']
        X4 = data['X4']
        X5 = data['X5']
        X6 = data['X6']
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Assigned treatment
        prob_a = 1 / (1 + np.exp( 0.6*K - 0.6*X1 + 0.6*X2 - 0.6*X3 + 0.6*X4 - 0.6*X5 + 0.6*X6 ))
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = (0.5 + 1*A + np.exp(K) - 2.5*A*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A + 0.2*np.exp(K) - 3.5*A*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4)) + noise
        
        ## Generate potential outcome
        A_p = 0
        Y0_true = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
        A_p = 1
        Y1_true = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            Y0mat['Y_'+str(i)] = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
            A_p = 1
            Y1mat['Y_'+str(i)] = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
        
        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)
        
        is_vul = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0
        
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        d = np.where((g1-g0)>0, 1, 0)
        obj = np.maximum(g0, g1)
        
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)

        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["d"] = d
        data["d_opt"] = d_opt
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true

        dummies = pd.get_dummies(data['S'], dtype=int).rename(columns=lambda x: 'S' + str(x))
        data = pd.concat([data, dummies], axis=1)
        dummies_names = dummies.columns.values.tolist()

        Y_name = "Y"
        A_name = "A"
        X_names = ["X" + str(num) for num in (range(1,(p+1)))]
        S_names = None
        X_names_exp = X_names + dummies_names
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
        
        cols = ['id', 'S'] + X_names + ['d', 'd_opt', 'A', 'Y' ] + dummies_names + ["Y0_true", "Y1_true"]
        data = data[cols]
        data['is_vul'] = is_vul
        data['objective'] = obj
    
    
    if which_sim == 6 : # violation of the causal assumptions
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        
        # number of covariates
        p = 6
        
        for i in range(p):
            data['X'+str(i+1)] = np.random.uniform(low = 0, high = 1, size = n)
        
        X1 = data['X1']
        X2 = data['X2']
        X3 = data['X3']
        X4 = data['X4']
        X5 = data['X5']
        X6 = data['X6']
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Add random noise to X1
        X1_new = X1 + np.random.normal(0, 1, n)
        
        ## Assigned treatment
        prob_a = 1 / (1 + np.exp((-1.2)*( -K + X1_new - X2 + X3 - X4 + X5 - X6 )))
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = (X1*X2 - X3*X4) * (-np.exp(K-1) - A + A*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A + 3*A*K) + K + 5 + noise
        
        ## Generate potential outcome
        A_p = 0
        Y0_true = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5
        A_p = 1
        Y1_true = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            Y0mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5
            A_p = 1
            Y1mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-np.exp(K-1) - A_p + A_p*K) + (1- X2 - 2*X3) * (3*np.exp(K) - 2*A_p + 3*A_p*K) + K + 5
        
        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)
        
        is_vul = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0 
        
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        d = np.where((g1-g0)>0, 1, 0)
        obj = np.maximum(g0, g1)
        
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)

        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["d"] = d
        data["d_opt"] = d_opt
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true

        dummies = pd.get_dummies(data['S'], dtype=int).rename(columns=lambda x: 'S' + str(x))
        data = pd.concat([data, dummies], axis=1)
        dummies_names = dummies.columns.values.tolist()

        Y_name = "Y"
        A_name = "A"
        X_names = ["X" + str(num) for num in (range(1,(p+1)))]
        S_names = None
        X_names_exp = X_names + dummies_names
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
        
        cols = ['id', 'S'] + X_names + ['d', 'd_opt', 'A', 'Y' ] + dummies_names + ["Y0_true", "Y1_true"]
        data = data[cols]
        data['is_vul'] = is_vul
        data['objective'] = obj
    
    
    if which_sim == 7 : # no site variation of CATE, heterogeneous propsensity score
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        
        # number of covariates
        p = 6
        
        for i in range(p):
            data['X'+str(i+1)] = np.random.uniform(low = 0, high = 1, size = n)
        
        X1 = data['X1']
        X2 = data['X2']
        X3 = data['X3']
        X4 = data['X4']
        X5 = data['X5']
        X6 = data['X6']
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Assigned treatment
        prob_a = 1 / (1 + np.exp( 0.6*K - 0.6*X1 + 0.6*X2 - 0.6*X3 + 0.6*X4 - 0.6*X5 + 0.6*X6 ))
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = (X1*X2 - X3*X4) * (-A) + (1- X2 - 2*X3) * (-2*A) + K + 5 + noise
        
        ## Generate potential outcome
        A_p = 0
        Y0_true = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        A_p = 1
        Y1_true = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        
        ## Generate d_exp
        A_p = 0
        K = 1
        Y0_K1 = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        K = 0
        Y0_K0 = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        Y0_avg = (Y0_K0 + Y0_K1)/2
        
        A_p = 1
        K = 1
        Y1_K1 = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        K = 0
        Y1_K0 = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        Y1_avg = (Y1_K0 + Y1_K1)/2
        
        d_exp = np.where((Y1_avg-Y0_avg>0), 1, 0)
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            Y0mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
            A_p = 1
            Y1mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5

        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)

        is_vul = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0
        is_vul = np.array(is_vul)
        
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        d = np.where((g1-g0)>0, 1, 0)
        obj = np.maximum(g0, g1)
        
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)

        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["d"] = d
        data["d_opt"] = d_opt
        data["d_exp"] = d_exp
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true

        dummies = pd.get_dummies(data['S'], dtype=int).rename(columns=lambda x: 'S' + str(x))
        data = pd.concat([data, dummies], axis=1)
        dummies_names = dummies.columns.values.tolist()       

        Y_name = "Y"
        A_name = "A"
        X_names = ["X1", "X2", "X3", "X4", "X5", "X6"]
        S_names = None
        X_names_exp = X_names + dummies_names
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
        
        cols = ['id', 'S'] + X_names + ['d', 'd_opt', 'd_exp', 'A', 'Y' ] + dummies_names + ["Y0_true", "Y1_true"]
        data = data[cols]
        data['is_vul'] = is_vul
        data['objective'] = obj
    
    
    if which_sim == 8 : # missing treatment
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        
        # number of covariates
        p = 6
        
        for i in range(p):
            data['X'+str(i+1)] = np.random.uniform(low = 0, high = 1, size = n)
        
        X1 = data['X1']
        X2 = data['X2']
        X3 = data['X3']
        X4 = data['X4']
        X5 = data['X5']
        X6 = data['X6']
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Assigned treatment
        prob_a = (1 / (1 + np.exp( 0.6*K - 0.6*X1 + 0.6*X2 - 0.6*X3 + 0.6*X4 - 0.6*X5 + 0.6*X6 )))*(S!=3)*(S!=7)
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = (0.5 + 1*A + np.exp(K) - 2.5*A*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A + 0.2*np.exp(K) - 3.5*A*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4)) + noise
        
        ## Generate potential outcome
        A_p = 0
        Y0_true = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
        A_p = 1
        Y1_true = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            Y0mat['Y_'+str(i)] = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
            A_p = 1
            Y1mat['Y_'+str(i)] = (0.5 + 1*A_p + np.exp(K) - 2.5*A_p*K) * (1+X1 -X2 +X3**2 +np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*A_p*K) * (1+5*X1 -2*X2 +3*X3 +2*np.exp(X4))
        
        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)
        
        is_vul = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0 
        
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        d = np.where((g1-g0)>0, 1, 0)
        obj = np.maximum(g0, g1)
        
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)

        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["d"] = d
        data["d_opt"] = d_opt
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true

        dummies = pd.get_dummies(data['S'], dtype=int).rename(columns=lambda x: 'S' + str(x))
        data = pd.concat([data, dummies], axis=1)
        dummies_names = dummies.columns.values.tolist()
        
        Y_name = "Y"
        A_name = "A"
        X_names = ["X1", "X2", "X3", "X4", "X5", "X6"]
        S_names = None
        X_names_exp = X_names + dummies_names
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
        
        cols = ['id', 'S'] + X_names + ['d', 'd_opt', 'A', 'Y' ] + dummies_names + ["Y0_true", "Y1_true"]
        data = data[cols]
        data['is_vul'] = is_vul
        data['objective'] = obj
        
        
    if which_sim == 9 : # generalization with heterogeneous treatment, site variation of CATE # Setting 7
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        X1 = np.random.uniform(low = 0, high = 1, size = n)
        X2 = np.random.uniform(low = 0, high = 1, size = n)
        X3 = np.random.uniform(low = 0, high = 1, size = n)
        X4 = np.random.uniform(low = 0, high = 1, size = n)
        X5 = np.random.uniform(low = 0, high = 1, size = n)
        X6 = np.random.uniform(low = 0, high = 1, size = n)
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Assigned treatment
        prob_a = 1 / (1 + np.exp( 0.6*K - 0.6*X1 + 0.6*X2 - 0.6*X3 + 0.6*X4 - 0.6*X5 + 0.6*X6 ))
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y =  (0.5 + A + np.exp(K) - 2.5*K*A) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A + 0.2*np.exp(K) - 3.5*K*A) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4)) + noise
        
        ## Generate potential outcome
        A_p = 0
        Y0_true = (0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4))
        A_p = 1
        Y1_true = (0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4))
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            Y0mat['Y_'+str(i)] = (0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4))
            A_p = 1
            Y1mat['Y_'+str(i)] = (0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4))
        
        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)
        
        is_vul = [999]*n
        vul_hosp_list = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0
            vul_hosp_list[i] = vul_hosp[0]
            
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        d = np.where((g1-g0)>0, 1, 0)
        obj = np.maximum(g0, g1)
        
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)
        
        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["X1"] = X1
        data["X2"] = X2
        data["X3"] = X3
        data["X4"] = X4
        data["X5"] = X5
        data["X6"] = X6
        data["d"] = d
        data["d_opt"] = d_opt
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true
        data["is_vul"] = is_vul
        data['objective'] = obj


        Y_name = "Y"
        A_name = "A"
        X_names = ["X1", "X2", "X3", "X4", "X5", "X6"]
        X_names_exp = ['NO NAME']
        S_names = None
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
    
    
    if which_sim == 10 : # no site variation of CATE, heterogeneous propsensity score, generalizability # Setting 10
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        
        # number of covariates
        p = 6
        
        for i in range(p):
            data['X'+str(i+1)] = np.random.uniform(low = 0, high = 1, size = n)
        
        X1 = data['X1']
        X2 = data['X2']
        X3 = data['X3']
        X4 = data['X4']
        X5 = data['X5']
        X6 = data['X6']
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Assigned treatment
        prob_a = 1 / (1 + np.exp( 0.6*K - 0.6*X1 + 0.6*X2 - 0.6*X3 + 0.6*X4 - 0.6*X5 + 0.6*X6 ))
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = (X1*X2 - X3*X4) * (-A) + (1- X2 - 2*X3) * (-2*A) + K + 5 + noise
        
        ## Generate potential outcome
        A_p = 0
        Y0_true = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        A_p = 1
        Y1_true = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            Y0mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
            A_p = 1
            Y1mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5

        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)

        is_vul = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0
        
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        d = np.where((g1-g0)>0, 1, 0)
        obj = np.maximum(g0, g1)
        
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)

        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["d"] = d
        data["d_opt"] = d_opt
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true  
        data['is_vul'] = is_vul
        data['objective'] = obj

        Y_name = "Y"
        A_name = "A"
        X_names = ["X1", "X2", "X3", "X4", "X5", "X6"]
        S_names = None
        X_names_exp = ['NO NAME']
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
        
    
    if which_sim == 11 : # no site variation of CATE, covariate shift, generalizability # Setting 11
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        
        # number of covariates
        p = 6
        
        for i in range(p):
            X_S1 = np.random.uniform(low=0, high=1, size=int(n/2))
            X_S2 = np.random.uniform(low=0.5, high=1.5, size=int(n/2))
            data['X'+str(i+1)] = np.concatenate((X_S1, X_S2))
        
        X1 = data['X1']
        X2 = data['X2']
        X3 = data['X3']
        X4 = data['X4']
        X5 = data['X5']
        X6 = data['X6']
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Assigned treatment
        prob_a = 0.5
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = (X1*X2 - X3*X4) * (-A) + (1- X2 - 2*X3) * (-2*A) + K + 5 + noise
        
        ## Generate potential outcome
        A_p = 0
        Y0_true = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        A_p = 1
        Y1_true = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            Y0mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
            A_p = 1
            Y1mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5

        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)

        is_vul = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0
        
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        d = np.where((g1-g0)>0, 1, 0)
        obj = np.maximum(g0, g1)
        
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)

        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["d"] = d
        data["d_opt"] = d_opt
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true
        data['is_vul'] = is_vul
        data['objective'] = obj
        

        Y_name = "Y"
        A_name = "A"
        X_names = ["X1", "X2", "X3", "X4", "X5", "X6"]
        S_names = None
        X_names_exp = ['NO NAME']
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
    
    
    if which_sim == 12 : # no site variation of CATE, covariate shift, heterogeneous treatment, generalizability # Setting 12
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        
        # number of covariates
        p = 6
        
        for i in range(p):
            X_S1 = np.random.uniform(low=0, high=1, size=int(n/2))
            X_S2 = np.random.uniform(low=0.5, high=1.5, size=int(n/2))
            data['X'+str(i+1)] = np.concatenate((X_S1, X_S2))
        
        X1 = data['X1']
        X2 = data['X2']
        X3 = data['X3']
        X4 = data['X4']
        X5 = data['X5']
        X6 = data['X6']
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Assigned treatment
        prob_a = 1 / (1 + np.exp( 0.6*K - 0.6*X1 + 0.6*X2 - 0.6*X3 + 0.6*X4 - 0.6*X5 + 0.6*X6 ))
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = (X1*X2 - X3*X4) * (-A) + (1- X2 - 2*X3) * (-2*A) + K + 5 + noise
        
        ## Generate potential outcome
        A_p = 0
        Y0_true = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        A_p = 1
        Y1_true = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            Y0mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5
            A_p = 1
            Y1mat['Y_'+str(i)] = (X1*X2 - X3*X4) * (-A_p) + (1- X2 - 2*X3) * (-2*A_p) + K + 5

        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)

        is_vul = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0
        
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        d = np.where((g1-g0)>0, 1, 0)
        obj = np.maximum(g0, g1)
        
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)

        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["d"] = d
        data["d_opt"] = d_opt
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true
        data['is_vul'] = is_vul
        data['objective'] = obj
        

        Y_name = "Y"
        A_name = "A"
        X_names = ["X1", "X2", "X3", "X4", "X5", "X6"]
        S_names = None
        X_names_exp = ['NO NAME']
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
        
    
    if which_sim == 13 : # generalization with covariate shift, site variation of CATE # Setting 8
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        
        # number of covariates
        p = 6
        
        for i in range(p):
            X_S1 = np.random.uniform(low=0, high=1, size=int(n/2))
            X_S2 = np.random.uniform(low=0.5, high=1.5, size=int(n/2))
            data['X'+str(i+1)] = np.concatenate((X_S1, X_S2))
        
        X1 = data['X1']
        X2 = data['X2']
        X3 = data['X3']
        X4 = data['X4']
        X5 = data['X5']
        X6 = data['X6']
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Assigned treatment
        prob_a = 0.5
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = 0.5*((0.5 + A + np.exp(K) - 2.5*K*A) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A + 0.2*np.exp(K) - 3.5*K*A) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4))) + noise
        
        ## Generate potential outcome
        A_p = 0
        Y0_true = 0.5*((0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4)))
        A_p = 1
        Y1_true = 0.5*((0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4)))
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            Y0mat['Y_'+str(i)] = 0.5*((0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4)))
            A_p = 1
            Y1mat['Y_'+str(i)] = 0.5*((0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4)))
        
        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)
        
        is_vul = [999]*n
        vul_hosp_list = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0
            vul_hosp_list[i] = vul_hosp[0]
            
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        d = np.where((g1-g0)>0, 1, 0)
        obj = np.maximum(g0, g1)
        
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)
        
        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["X1"] = X1
        data["X2"] = X2
        data["X3"] = X3
        data["X4"] = X4
        data["X5"] = X5
        data["X6"] = X6
        data["d"] = d
        data["d_opt"] = d_opt
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true
        data["is_vul"] = is_vul
        data['objective'] = obj


        Y_name = "Y"
        A_name = "A"
        X_names = ["X1", "X2", "X3", "X4", "X5", "X6"]
        X_names_exp = ['NO NAME']
        S_names = None
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
        
    
    
    if which_sim == 14 : # generalization with heterogeneous treatment, covariate shift, site variation of CATE # Setting 9
        # number of hospital
        nhosp = nhosp
        hosp = range(nhosp)
        
        # number of total subject
        n = nhosp*hosp_size
        
        # Generate data
        data = pd.DataFrame()
        
        # number of covariates
        p = 6
        
        for i in range(p):
            X_S1 = np.random.uniform(low=0, high=1, size=int(n/2))
            X_S2 = np.random.uniform(low=0.5, high=1.5, size=int(n/2))
            data['X'+str(i+1)] = np.concatenate((X_S1, X_S2))
        
        X1 = data['X1']
        X2 = data['X2']
        X3 = data['X3']
        X4 = data['X4']
        X5 = data['X5']
        X6 = data['X6']
        
        ## Discrete S
        S = np.repeat(range(nhosp), [n/nhosp]*nhosp, axis=0)
        half = nhosp/2
        K = np.where((S>=half), 1, 0)
        
        ## Assigned treatment
        prob_a = 1 / (1 + np.exp( 0.6*K - 0.6*X1 + 0.6*X2 - 0.6*X3 + 0.6*X4 - 0.6*X5 + 0.6*X6 ))
        A = (np.random.binomial(n = 1, p = prob_a, size = n))

        ## Random noise 
        noise = np.random.normal(0, 1, n)

        ## Generate reward
        Y = (0.5 + A + np.exp(K) - 2.5*K*A) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A + 0.2*np.exp(K) - 3.5*K*A) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4)) + noise
        
        ## Generate potential outcome
        A_p = 0
        Y0_true = (0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4))
        A_p = 1
        Y1_true = (0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4))
        
        ## Generate vulnerable label
        Y0mat = pd.DataFrame()
        Y1mat = pd.DataFrame()
        for i in range(nhosp):
            K = np.where((i>=half), 1, 0)
            A_p = 0
            Y0mat['Y_'+str(i)] = (0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4))
            A_p = 1
            Y1mat['Y_'+str(i)] = (0.5 + A_p + np.exp(K) - 2.5*K*A_p) * (1 + X1 - X2 + X3**2 + np.exp(X4)) + (1 + 2*A_p + 0.2*np.exp(K) - 3.5*K*A_p) * (1 + 5*X1 - 2*X2 + 3*X3 + 2*np.exp(X4))
        
        minMatrix = np.minimum(Y0mat, Y1mat)
        minValue = pd.DataFrame(minMatrix).min(axis=1)
        
        is_vul = [999]*n
        vul_hosp_list = [999]*n
        for i in range(n):
            vul_S = np.where(minMatrix.iloc[i]<=minValue[i])
            vul_hosp = [hosp[j] for j in vul_S[0]]
            if S[i] in vul_hosp:
                is_vul[i] = 1
            else: 
                is_vul[i] = 0
            vul_hosp_list[i] = vul_hosp[0]
            
        g0 = Y0mat.min(axis=1)
        g1 = Y1mat.min(axis=1)
        d = np.where((g1-g0)>0, 1, 0)
        obj = np.maximum(g0, g1)
        
        # Optimal treatment under the true Y1 and Y0
        d_opt = np.where((Y1_true-Y0_true)>0, 1, 0)
        
        data['id'] = range(n)        
        data["S"] = S
        data["S"] = data["S"].astype('category')
        data["X1"] = X1
        data["X2"] = X2
        data["X3"] = X3
        data["X4"] = X4
        data["X5"] = X5
        data["X6"] = X6
        data["d"] = d
        data["d_opt"] = d_opt
        data["A"] = A
        data["Y"] = Y
        data["Y0_true"] = Y0_true
        data["Y1_true"] = Y1_true
        data["is_vul"] = is_vul
        data['objective'] = obj


        Y_name = "Y"
        A_name = "A"
        X_names = ["X1", "X2", "X3", "X4", "X5", "X6"]
        X_names_exp = ['NO NAME']
        S_names = None
        is_rct = False
        is_class = False
        s_type = "disc"
        qua_use = 0.25
        
    
        
    return Y_name, A_name, X_names, S_names, X_names_exp, is_rct, is_class, s_type, qua_use, data

    