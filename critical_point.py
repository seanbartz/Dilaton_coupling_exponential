import numpy as np
import matplotlib.pyplot as plt
from criticalZoom import critical_zoom 
from timebudget import timebudget
import pandas as pd

# this code searches for the critical point for a given lambda1 and ml
@timebudget
def critical_point_refined(lambda1,mu0,mu1,mu2,ml,tmin,tmax,numtemp,minsigma,maxsigma,mu_initial,delta_mu,mu_precision):
    mu=mu_initial
    #create a list to store the values of mu, Tc, and order
    mu_list=[]
    Tc_list=[]
    order_list=[]
    while round(delta_mu) >= mu_precision:
        print("current mu value is", mu)
        # see if the current mu value is in the list  mu_list
        if mu in mu_list:
            #find the index of the current mu value in the list
            index=mu_list.index(mu)
            #get the corresponding Tc and order values
            Tc=Tc_list[index]
            order=order_list[index]
            print("mu value already checked. Tc=",Tc,"order=",order)
        else:
            order, iterationNumber, sigma_list,temps_list,Tc=critical_zoom(tmin,tmax,numtemp,minsigma,maxsigma,ml,mu,lambda1,mu0,mu1,mu2)
            # create a dataframe to store sigma_list and temps_list
            df1=pd.DataFrame({'temps':temps_list,'sigma':sigma_list})
            # add order, mu, and Tc to the dataframe
            df1['order']=order
            df1['mu']=mu
            df1['Tc']=Tc
            #pickle the dataframe with the values of ml, lambda1, and mu in the filename
            df1.to_pickle("data/sigma_transition_mq_"+str(ml)+"_lambda_"+str(lambda1)+"_mu_"+str(mu)+".pkl")

            #add the current mu value to the list
            mu_list.append(mu)
            #add the corresponding Tc and order values to the lists
            Tc_list.append(Tc)
            order_list.append(order)


        if mu==mu_initial and order==1:
            print(" no critical point. Transition is always first-order for mu greater than", mu_initial)
            break
        if order==1:
            mu=mu-delta_mu+delta_mu/2
            delta_mu=delta_mu/2
            "!!! might need to adjust the temperature range and maxsigma here"
        else:
            mu=mu+delta_mu

            #adjust the temperature range if Tc is too close to the boundaries
            if Tc< tmin+50:
                tmin=max(0,tmin-50)
                tmax=max(tmin+50,tmax-50)
            #adjust maxsigma if it is too high
            if maxsigma>np.amax(sigma_list[0][:,0])+150:
                maxsigma=np.amax(sigma_list[0][:,0])-50
        if mu>5000:
            print( "This is taking too long. No critical point found for mu less than 5000")
            break
    #find the maximum of the first element of sigma_list
    actual_max_sigma=np.amax(sigma_list[0][:,0])

    #create a dataframe to store the values of mu, Tc, and order
    df=pd.DataFrame({'mu':mu_list,'Tc':Tc_list,'order':order_list})
    #include  the other parameters in the dataframe
    df['lambda1']=lambda1
    df['ml']=ml
    df['mu0']=mu0
    df['mu1']=mu1
    df['mu2']=mu2
    df['mu_precision']=mu_precision



    

    # find the indices where order is 2
    index=np.where(np.array(order_list)==2)
    #find the corresponding mu values
    mu_cross=np.array(mu_list)[index]
    #find the corresponding Tc values
    Tc_cross=np.array(Tc_list)[index]

    #find the indices where order is 1
    index=np.where(np.array(order_list)==1)
    #find the corresponding mu values
    mu_1storder=np.array(mu_list)[index]
    #find the corresponding Tc values
    Tc_1storder=np.array(Tc_list)[index]

    #finding the best estimate for the critical point
    #if mu first order is not empty (i.e. there is a critical point)
    if mu_1storder.size>0:
        # take the average of the largest value of mu_cross and the smallest value of mu_1storder
        mu_crit=(np.amax(mu_cross)+np.amin(mu_1storder))/2
        #find the corresponding Tc value
        Tc_crit=np.interp(mu_crit,mu_list,Tc_list)

        #add the critical point to the dataframe
        df['mu_crit']=mu_crit
        df['Tc_crit']=Tc_crit


    #pickle the dataframe with the values of ml, lambda1 in the filename
    df.to_pickle("data/phase_plot_zoom_mq_"+str(ml)+"_lambda_"+str(lambda1)+".pkl")

    #plot the Tc vs mu
    plt.plot(mu_cross,Tc_cross,label='crossover')
    plt.plot(mu_1storder,Tc_1storder,label='1st order')
    plt.xlabel('$\mu$ (MeV)')
    plt.ylabel('$T$ (MeV)')
    plt.legend()
    #save the plot
    plt.savefig("plots/phase_plot_zoom_mq_"+str(ml)+"_lambda_"+str(lambda1)+".png")
            
    return(mu,Tc,actual_max_sigma)

