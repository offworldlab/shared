function [mnEigVec,mnEigVal]  =  MinGenEigCmp(F,D),
%
% This program computes the minimum generalized eigenvalue and its
% corresponding generalized eigenvector of the pair (F,D), where
% F and D are 3x3 matrices.
%
% Usage: [mnEigVec,mnEigVal]  =  MinGenEigCmp(F,D);
%
%
% K. C. Ho      07-01-2012
%
%       Computational Intelligence Signal Processing Laboratory
%       University of Missouri
%       Columbia, MO 65211, USA.
%       hod@missouri.edu
%

Forg = F;

adjD(1,1) = D(2,2)*D(3,3)-D(3,2)*D(2,3);
adjD(1,2) = -(D(2,1)*D(3,3)-D(3,1)*D(2,3));
adjD(1,3) = D(2,1)*D(3,2)-D(3,1)*D(2,2);
adjD(2,1) = -(D(1,2)*D(3,3)-D(3,2)*D(1,3));
adjD(2,2) = D(1,1)*D(3,3)-D(3,1)*D(1,3);
adjD(2,3) = -(D(1,1)*D(3,2)-D(3,1)*D(1,2));
adjD(3,1) = D(1,2)*D(2,3)-D(2,2)*D(1,3);
adjD(3,2) = -(D(1,1)*D(2,3)-D(2,1)*D(1,3));
adjD(3,3) = D(1,1)*D(2,2)-D(2,1)*D(1,2);
adjD = adjD';
F = adjD*Forg;

a = -(F(1,1)+F(2,2)+F(3,3));
b = F(1,1)*F(2,2)+F(1,1)*F(3,3)+F(2,2)*F(3,3)-F(1,3)*F(3,1)-F(1,2)*F(2,1)-F(2,3)*F(3,2);
c = F(1,1)*F(2,3)*F(3,2)+F(1,2)*F(2,1)*F(3,3)+F(1,3)*F(2,2)*F(3,1) -F(1,1)*F(2,2)*F(3,3)-F(1,2)*F(2,3)*F(3,1)-F(1,3)*F(2,1)*F(3,2);

q = (3*b-a^2)/9;
r = (9*a*b-27*c-2*a^3)/54;
d = q^3+r^2;

if (d<0)
    rho = sqrt(-q^3);
    rhoR3 = rho^(1/3);
    theta = acos(r/rho);
    thetaD3 = theta/3;
    rt(1) = -a/(3)+2*rhoR3*cos(thetaD3);
    rt(2) = -a/(3)-rhoR3*cos(thetaD3)-sqrt(3)*rhoR3*sin(thetaD3);
    rt(3) = -a/(3)-rhoR3*cos(thetaD3)+sqrt(3)*rhoR3*sin(thetaD3);
    rt = abs(real(rt));
    rt = sort(rt);
else
    s = (r+sqrt(d))^(1/3);
    t = (r-sqrt(d))^(1/3);
    rt = -a/(3)+(s+t);
    rt = abs(real(rt));
end;

mnEigVal = rt(1);
P = det(D)*Forg-D*mnEigVal;
mnEigVec = -inv(P(1:2,1:2))*P(1:2,3);
mnEigVec = [mnEigVec;1];


