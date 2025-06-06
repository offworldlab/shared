% This program provides an example on how to call TDOAFDOALoc_BiasRed.m
%
% Reference: K. C. Ho, Bias reduction for an explicit solution of source
%  localization using TDOA," IEEE Trans. Signal Process., vol. 60,
%  pp. 2101-2114, May 2012.
%
% K. C. Ho      05-01-2012
%
%       Copyright (C) 2012
%       Computational Intelligence Signal Processing Laboratory
%       University of Missouri
%       Columbia, MO 65211, USA.
%       hod@missouri.edu
%

clear all;
warning off;            % program initialization

M=8;                    % number of sensors, at least 5

seed=17;                % noise seed

L = 2000;               % Number of ensemble runs

uo = [2000 2500 3000]';    % true source position
u_doto = [-20 15 40]';     % true source velocity

% --- true sensor positions ---
x = [-150 -200 200 100 -100 350 300 400]';
y = [150 250 -150 100 -100 200 500 150]';
z = [-130 -250 300 100 -100 100 200 100]';
so = [x, y, z]';        % true sensor position matrix

% --- true sensor velocities ---
xd = [20 -10 -7 15 -20 10 10 -30]';
yd = [0 -15 -10 15 10 20 -20 10]';
zd = [-20 10 20 -10 10 30 10 20]';
s_doto = [xd, yd, zd]';       % true sensor position and velocity

so=so(:,1:M); s_doto=s_doto(:,1:M);  %x%

N = size(so,1);                 % dimension of localization
M = size(so,2);                 % number of sensors

% --- noise covariance matrices ---
R = (eye(M-1)+ones(M-1))/2;           % covariance matrix
R = [R,zeros(M-1); zeros(M-1),R*0.01]; % of measurements
chol_R=chol(R);

numCnfg=1;

for cntCnfg=1:numCnfg,
    
    ro = sqrt(sum((uo*ones(1,M)-so).^2))';
    r_doto=(uo*ones(1,M)-so)'*(u_doto*ones(1,M)-s_doto);
    r_doto = diag(r_doto)./ro;
    d=[ro]; dd=[r_doto];
    
    % --- True range differences ---
    rdo = ro(2:end) - norm(uo-so(:,1));
    % --- True range rate differences ---
    rd_doto = r_doto(2:end) - (uo-so(:,1))'*(u_doto-s_doto(:,1))/norm(uo-so(:,1));
    
    % @@@@@@@@@@ make noise zero mean for accurate bias evaluation @@@@@@@@@@@@
    randn('state',seed);                        % initialize random number generator
    
    aveNse=0;
    for k=1:L, aveNse=aveNse+chol_R'*randn(2*(M-1),1); end;
    PP=aveNse/L;
    
    uAll=zeros(N,L);
    u_dotAll=zeros(N,L);
    uBRAll=zeros(N,L);
    u_dotBRAll=zeros(N,L);
    MsrNseAll=zeros(2*M,L);
    
    % --- noise power vector in log-scale ---
    sigma2_dVec=(-30:5:15);
    
    i = 1;                              % loop counter
    for sigma2_d = sigma2_dVec,         % main loop
        randn('state',seed);            % initialize random number generator
        fprintf('.');
        NseStd=10^(sigma2_d/20);
        Q_alpha = 10^(sigma2_d/10) * R;
        CRLB = TDOAFDOALocMvgSrcSenCRLB(so,s_doto,uo,u_doto,Q_alpha);
        crlbu(cntCnfg,i) = trace(CRLB(1:N,1:N));
        crlbudot(cntCnfg,i) = trace(CRLB(N+1:end,N+1:end));
        
        SimMSEu = 0; SimMSEudot = 0;
        SimMSEuBR = 0; SimMSEudotBR = 0;
        
        for k = 1 : L,
            % Noisy TDOA and FDOA measurements.
            Delta_r = (chol_R'*randn(2*(M-1),1)-PP)*NseStd;
            
            rd = rdo + Delta_r(1:M-1);
            rd_dot = rd_doto + Delta_r(M:end);
            
            % --- previous solution (Ho & Xu, T-SP, 2004)
            [u,u_dot] = TDOAFDOALocMvgSrcSen(so,s_doto,rd,rd_dot,Q_alpha);
            
            % --- bias reduction solution ---
            [uBR,u_dotBR] = TDOAFDOALoc_BiasRed(so,s_doto,rd,rd_dot,Q_alpha);
            
            SimMSEu = SimMSEu + norm(u-uo)^2;
            SimMSEudot = SimMSEudot + norm(u_dot-u_doto)^2;
            SimMSEuBR = SimMSEuBR + norm(uBR-uo)^2;
            SimMSEudotBR = SimMSEudotBR + norm(u_dotBR-u_doto)^2;
            uAll(:,k)=u;
            u_dotAll(:,k)=u_dot;
            uBRAll(:,k)=uBR;
            u_dotBRAll(:,k)=u_dotBR;
            
        end;
        
        mseu(cntCnfg,i) = SimMSEu/L;
        mseudot(cntCnfg,i) = SimMSEudot/L;
        mseuBR(cntCnfg,i) = SimMSEuBR/L;
        mseudotBR(cntCnfg,i) = SimMSEudotBR/L;
        biasu(cntCnfg,i)=(norm(mean(uAll,2)-uo));
        biasuBR(cntCnfg,i)=(norm(mean(uBRAll,2)-uo));
        biasudot(cntCnfg,i)=(norm(mean(u_dotAll,2)-u_doto));
        biasudotBR(cntCnfg,i)=(norm(mean(u_dotBRAll,2)-u_doto));
        
        i = i + 1;                        % Update loop counter.
    end;
    fprintf('\n');
    
end;    %cntCnfg


xLabelTxt='20 log(\\sigma_r)';

% Plot the result.
figure(11);
h=plot(sigma2_dVec,10*log10(mseuBR),'kx','MarkerSize',8); set(h,'linewidth',2); hold on; grid on;
h=plot(sigma2_dVec,10*log10(mseu),'ko','MarkerSize',8); set(h,'linewidth',2);
h=plot(sigma2_dVec,10*log10(crlbu),'k','LineWidth',1); set(h,'linewidth',2);
xlabel(xLabelTxt); ylabel('l0 log(MSE), Position');
legend('BiasRed','WMLS,Ho-Xu(2004)','CRLB',2);
hold off;

figure(12);
h=plot(sigma2_dVec,10*log10(mseudotBR),'kx','MarkerSize',8); set(h,'linewidth',2); hold on; grid on;
h=plot(sigma2_dVec,10*log10(mseudot),'ko','MarkerSize',8); set(h,'linewidth',2);
h=plot(sigma2_dVec,10*log10(crlbudot),'k','LineWidth',1); set(h,'linewidth',2);
xlabel(xLabelTxt); ylabel('10 log(MSE), Velocity');
legend('BiasRed','WMLS,Ho-Xu(2004)','CRLB',2);
hold off;

figure(13);
h=plot(sigma2_dVec,20*log10(biasuBR)/1,'kx','MarkerSize',8); set(h,'linewidth',2); hold on; grid on;
h=plot(sigma2_dVec,20*log10(biasu)/1,'ko','MarkerSize',8); set(h,'linewidth',2);
xlabel(xLabelTxt); ylabel('20 log(bias), Position');
legend('BiasRed','WMLS,Ho-Xu(2004)',2);
hold off;

figure(14);
h=plot(sigma2_dVec,20*log10(biasudotBR)/1,'kx','MarkerSize',8); set(h,'linewidth',2); hold on; grid on;
h=plot(sigma2_dVec,20*log10(biasudot)/1,'ko','MarkerSize',8); set(h,'linewidth',2);
xlabel(xLabelTxt); ylabel('20 log(bias), Velocity');
legend('BiasRed','WMLS,Ho-Xu(2004)',2);
hold off;

