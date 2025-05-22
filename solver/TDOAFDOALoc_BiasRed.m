function [u,u_dot] = TDOAFDOALoc_BiasRed(s,s_dot,rd,rd_dot,Q_alpha),
%
% This program realizes the algorithm for localizing a moving source
% using TDOAs and FDOAs. The sensors are moving as well. The details of the
% algorithm development can be found in  K. C. Ho, "Bias reduction for an
% explicit solution of source localization using TDOA,"
% IEEE Trans. Signal Process., vol. 60, pp. 2101-2114, May 2012.
%
% Usage: [u,u_dot] = TDOAFDOALoc_BiasRed(s,s_dot,r,r_dot,Q_alpha);
%
% Input Parameter List:
% s:        2xM or 3xM sensor position matrix.
%           M: number of sensors.
%           s(:,i) is the known position of the ith sensor.
% s_dot:    2xM or 3xM sensor velocity matrix.
%           s_dot(:,i) is the known velocity of the ith sensor.
% rd:       (M-1)x1 TDOA (range difference) measurement vector.
% rd_dot:   (M-1)x1 FDOA (range rate difference) measurement vector.
% Q_alpha:  (2M-2)x(2M-2) covariance matrix of [rd;rd_dot].
%
% Need to un-comment line 98 for OmegaTilde if it is poor-rank 
%
% The program returns a 2x1 or 3x1 source location estimate u and a 2x1 or
% 3x1 source velocity estimate u_dot.
%
%
% The program can be used for 2D(Dim=2) or 3D(Dim=3) localization
%
% ********************************************************************
% CORRECTION IN THE PAPER:  The matrix W2 under (63) is not correct.
%                           Please see lines 119-122, 128-130 below for the
%                           correction.
% ********************************************************************
%
% K. C. Ho      07-01-2012
% K. C. Ho      03-21-2013
%
%       Copyright (C) 2012
%       Computational Intelligence Signal Processing Laboratory
%       University of Missouri
%       Columbia, MO 65211, USA.
%       hod@missouri.edu
%

Qa = Q_alpha;
M = size(s,2);           % Number of sensors.
N = size(s,1);           % Dimension of the localization problem.
iQa = inv(Qa);

% ========== First Stage ==========
ht = rd.^2-sum(s(:,2:end).^2)'+sum(s(:,1).^2);
hf = 2*( rd.*rd_dot - sum(s_dot(:,2:end).*s(:,2:end))' + s_dot(:,1)'*s(:,1) );
Gt = -2*[ (s(:,2:end)-s(:,1)*ones(1,M-1))', zeros(M-1,N), rd, zeros(M-1,1) ];
Gf = -2*[ (s_dot(:,2:end)-s_dot(:,1)*ones(1,M-1))', (s(:,2:end)-s(:,1)*ones(1,M-1))', rd_dot, rd];

h1 = [ht;hf];
G1 = [Gt;Gf];
W1 = iQa;

A=[-G1, h1];

% ----- improve the weighting matrix W1 -----
phi1 = inv(G1'*W1*G1)*G1'*W1*h1;
u = phi1(1:N);
u_dot = phi1(N+1:end-2);

B_tilde = diag(rd);
Bdot_tilde = diag(rd_dot);

for m = 1 : 1,           % repeat once or more to update W1.
    
    b = sqrt(sum((repmat(u,1,M-1)-s(:,2:end)).^2))';
    b_dot = (sum((repmat(u,1,M-1)-s(:,2:end)).*(repmat(u_dot,1,M-1)-s_dot(:,2:end)))'./b);
    B = 2 * diag(b);
    B_dot = 2 * diag(b_dot);
    
    B1 = [B,zeros(size(B));B_dot,B];
    iB1 = inv(B1);
    W1 = iB1'*iQa*iB1;
    
    % --------------------------
    E11 = zeros(size(W1));
    E11(M:end,1:M-1) = eye(M-1);
    E12 = zeros(size(W1));
    E12(1:M-1,1:M-1) = B_tilde;
    E12(M:end,1:M-1) = Bdot_tilde;
    E12(M:end,M:end) = B_tilde;
    
    OmegaTilde = 4*[trace(W1*Qa), trace(W1*E11*Qa), trace(W1*E12*Qa);
        0, trace(E11'*W1*E11*Qa), trace(E11'*W1*E12*Qa);
        0, 0, trace(E12'*W1*E12*Qa)];
    OmegaTilde(2,1) = OmegaTilde(1,2);
    OmegaTilde(3,1) = OmegaTilde(1,3);
    OmegaTilde(3,2) = OmegaTilde(2,3);

%     % the following instruction is needed if OmegaTilde is poor rank
%     % (possibly caused by poor localization geometry)
%     OmegaTilde = OmegaTilde + eye(length(OmegaTilde))*trace(OmegaTilde)/10;
%     % avoid poor rank of OmegaTilde
    
    At = A'*W1*A;
    A11 = At(1:2*N,1:2*N); A12 = At(1:2*N,2*N+1:end); A21 = A12'; A22 = At(end-2:end,end-2:end);
    Av1 = -inv(A11)*A12;
    Me = A22+A21*Av1;
    [v2,mnEigVal] = MinGenEigCmp(Me,OmegaTilde);
    v1 = Av1*v2;
    phi1 = [v1; v2(1:end-1)];
    
    u = phi1(1:N);
    u_dot = phi1(N+1:end-2);
    icov_phi1 = (G1'*W1*G1);
    
end;

% ========== Second Stage ==========
h2 = [(u-s(:,1)).^2;
    (u-s(:,1)).*(u_dot-s_dot(:,1));
    phi1(end-1)^2; ...
    phi1(end-1)*phi1(end)];
G2 = [eye(N),    zeros(N);...
    zeros(N),  eye(N);...
    ones(1,N), zeros(1,N);...
    zeros(1,N),ones(1,N)];
K = [eye(N), zeros(N,1), zeros(N), zeros(N,1); ...
    zeros(N), zeros(N,1), eye(N), zeros(N,1); ...
    zeros(1,N), 1, zeros(1,N), 0;
    zeros(1,N), 0, zeros(1,N), 1];

for m = 1 : 2           % repeat a few times to update W2.
    Bt =  diag([u-s(:,1);norm(u-s(:,1),2)]);
    Btd = diag([u_dot-s_dot(:,1);(u-s(:,1))'*(u_dot-s_dot(:,1))/norm(u-s(:,1),2)]);
    B2 = [2*Bt, zeros(N+1); Btd, Bt];
    B2 = K*B2*K';
    iB2 = inv(B2);
    W2 = iB2'*icov_phi1*iB2;
    phi2 = inv(G2'*W2*G2)*G2'*W2*h2;
    
    u = diag(sign(u-s(:,1)))*sqrt(abs(phi2(1:N))) + s(:,1);
    u_dot = (phi2(N+1:end)./(u-s(:,1)) + s_dot(:,1));
end;


