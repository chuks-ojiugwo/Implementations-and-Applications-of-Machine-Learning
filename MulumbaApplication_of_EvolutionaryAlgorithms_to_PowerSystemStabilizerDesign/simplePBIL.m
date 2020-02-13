% PBIL
% PBIL generates random bitstrings, with control over the probability that a bit in a particular 
% position is a '1'. This control is effected by a probality vector, which contains the bit-probabilities
% (numbers in the range 0-1).

% To generate a randomised bitstring with the distribution specified by the PV, the latter 
% is 'sampled' by generating a uniformly random vector and comparing it element-by-element 
% with the PV. Wherever an element of the PV is greater than the corresponding random 
% element, a '1' is generated (otherwise a 0).

%To find a solution to an optimisation problem:

% 1. Initialise element of the PV to 0.5 (ensuring uniformly-random bitstrings)
%     Generate a population of uniformly-random bitstrings
%     Interpret each bitstring as a solution to the problem and evaluate its merit
%     in order to identify the "Best".

%2. Repeat the following:
%
% Adjust PV to favour the generation of bitstrings which resembe "Best"
% (slightly increase PV(i) if Best(i) =1 and decrease PV(i) if Best(i) =0)
% Generate a new population reflecting the modified distribution
% Interpret and evaluate each bitstring to find the new "Best"
% Until a satisfactory solution is found.

% Here is a very basic implementation, minimising the function f (fmax =0 at X=[1 2 3])
% NB: Using 7 bits per variable allows each variable to be expressed with 1/128 
%(better than 1%) precision

fitrec = [];										% null vector to hold record of fitness
x1 = [];
xx=x1;
LR = 0.1;   %learning rate
bestever = -inf;
bw = 2.^(14:-1:0);                                                   % vector of descending powers of 2 ('bitweights)
%PV = 0.5*ones(1,45);			 % initialise 45-element PV (15 bits per variable)
PV = 0.5*ones(1,30);
for g = 1:100									 % will run for 100 generations
   bestfit = -inf;	
for t =1:20                                                          % 20 trial solution (population size -20)

    
      ts = rand(size(PV)) <PV;                               % generate bitstring (if true value =1; otherwise=0)
     % x= reshape(ts, 3,15)*bw'/2^15;					    % quick way of converting binary to decimal
      x= reshape(ts,2,15)*bw'/2^15;
%ts
%PV
%pause

     % f= 1-((x(1)-1)^2+(x(2)-2)^2+(x(3)-3)^2) ;         % function to be optimised (replace by any other)
      f= (20+x(1)^2+ x(2)^2-10*(cos(2*pi*x(1))+cos(2*pi* x(2))));  
      %
      if f>bestfit									% if a better value has been found
         bestfit = f;									% record the fitness
         bestsol = ts;      % and the bitstring itself
         xx=x;
      end
      %%% former PV
      PV = (1-LR)*PV+LR*bestsol;					% update the probability vector
      PV = PV-0.005*(PV-0.5);					% relax toward neutral 0.5 to maintain diversity
   end
%     PV = 0.9*PV+0.1*bestsol;					% update the probability vector
%     PV = PV-0.005*(PV-0.5);					% relax toward neutral 0.5 to maintain diversity
   fitrec = [fitrec,bestfit];							% append the best fitness to fitness record
  x1=[x1,x] ;                                                         % append values of X
   if bestfit>bestever
      bestever = bestfit;
      besteversol = bestsol;						% keep track of best solution ever seen
   end
end
plot(fitrec)										% plot a graph showing how fitness improves
bestever										%display the best ever fitness found
xx