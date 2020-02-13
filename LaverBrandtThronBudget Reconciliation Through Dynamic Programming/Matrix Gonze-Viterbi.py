import numpy as np # Numerical Python functions
#GGCACTGAACTGAATACAGC is our sequence: A,C,G,T=0,1,2,3
Seq = [2,2,1,0,1,3,2,0,0,1,3,2,0,1,3,3,0,1,0,2,1] #Sequence
HLSeq = np.zeros((len(Seq),2)) #Store optimal sequences as we progress
HLSeq[:,1] += 1 #Defaults are all low, all high
Hi = [.2,.3,.3,.2] #Probabilities for nucleotides in High state
Lo = [.3,.2,.2,.3] #Probabilities for nucleotides in Low state
HLT = [[.5,.5],[.4,.6]] #Matrix of transition probabilities between High&Low states
currProb = np.array([.5*Hi[Seq[0]],.5*Lo[Seq[0]]]) #Stores current H&L probabilities
currProb = currProb.reshape(2,1) #change to 2X1 matrix
Prob = [[1,1],[1,1]] #Stores all 4 calculated probabilities at each stage
##### Compute trellis
for k in range(1,len(Seq)):
    HiLo = np.array( [Hi[Seq[k]],Lo[Seq[k]]] ) #Current nucleotide probabilities in H&L
    HiLo = HiLo.reshape(1,2) #change to a 1X2 matrix
    Prob = (currProb@HiLo)*HLT #matrix mult. between currProb and HiLo, then item by item mult. by HLT
    currProb[0,0],currProb[1,0] = Prob[0,0],Prob[1,1] #set initial currProb
    if Prob[1,0]> Prob[0,0]: #Update sequence ending in 0; switch if necessary
        HLSeq[0:k,0] = HLSeq[0:k,1]
        currProb[0,0] = Prob[1,0]
    if Prob[0,1] > Prob[1,1]: #Update sequence ending in 0; switch if necessary
        HLSeq[0:k,1] = HLSeq[0:k,0]
        currProb[1,0] = Prob[0,1]
##### Finished: choose optimal from final two options
if currProb[0,0] >= currProb[1,0]:
    FinalSeq = HLSeq[:,0] #High/Low states history ending with 0
else:
    FinalSeq = HLSeq[:,1] #High/Low states history ending with 1
FinalSeq = list(FinalSeq)
for k in range(0,len(Seq)):
    if FinalSeq[k]==0:
        FinalSeq[k]="High"
    else:
        FinalSeq[k]="Low"
print(FinalSeq)
