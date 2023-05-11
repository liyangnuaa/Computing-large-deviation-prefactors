clear;
clc;

xmin=-1.5;
xmax=0;
ymin=-0.8;
ymax=0.8;

alpha=0.5;
beta=3;

x1hat=[-1.5,-1.5,-1.5,-1,-1,-1,0,0,0];
x2hat=[-0.8,0,0.8,-0.8,0,0.8,-0.8,0,0.8];
%Vhat=0.5*x1hat.^4-2*x1hat.^2+(alpha-1)*x2hat.^2+0.5;
Vhat=0.5*x1hat.^4-2*x1hat.^2-2*x1hat+(alpha-1)*x2hat.^2-0.5;
V=0.5*x1hat.^4-x1hat.^2+alpha*x2hat.^2+0.5;
lx1=-alpha*beta*x1hat.*x2hat;
lx2=beta*(x1hat.^4-x1hat.^2);
lx1_x1=-alpha*beta*x2hat;
lx1_x2=-alpha*beta*x1hat;
lx2_x1=beta*(4*x1hat.^3-2*x1hat);
lx2_x2=0*x1hat;

%% vector field
n0=20;
x1=linspace(xmin,xmax,n0);
y1=linspace(ymin,ymax,n0);
[x,y]=meshgrid(x1,y1);
u=x-x.^3-alpha*beta*x.*y;
v=beta*(x.^4-x.^2)-alpha*y;
figure;
streamslice(x,y,u,v);
axis([xmin xmax ymin ymax])
figure;
plot([-0.5,-0.5],[-0.8,0.8]);

%% data processing
N=500;
xlin=linspace(xmin,xmax,N);
ylin=linspace(ymin,ymax,N);
[xmesh,ymesh]=meshgrid(xlin,ylin);

xtest=reshape(xmesh,1,N^2);
ytest=reshape(ymesh,1,N^2);
path = sprintf('xtest.mat');
save(path,'xtest');
path = sprintf('ytest.mat');
save(path,'ytest');

%% NN results
xmesh1=reshape(xtest,N,N);
ymesh1=reshape(ytest,N,N);
S=Stest(:,1)';
S=S+(xtest+1).^2+ytest.^2;
Smesh1=reshape(S,N,N);
lx=Stest(:,2)';
lxmesh1=reshape(lx,N,N);
ly=Stest(:,3)';
lymesh1=reshape(ly,N,N);

figure;
mesh(xmesh1,ymesh1,Smesh1);
figure;
mesh(xmesh1,ymesh1,lxmesh1);
figure;
mesh(xmesh1,ymesh1,lymesh1);

%% true results
Strue=0.5*xmesh1.^4-xmesh1.^2+alpha*ymesh1.^2+0.5;
lxtrue=-alpha*beta*xmesh1.*ymesh1;
lytrue=beta*(xmesh1.^4-xmesh1.^2);
figure;
mesh(xmesh1,ymesh1,Strue);
figure;
mesh(xmesh1,ymesh1,lxtrue);
figure;
mesh(xmesh1,ymesh1,lytrue);

%% approximation error
eS=max(max((Smesh1-Strue).^2))/max(max((Strue).^2));
elx=max(max((lxmesh1-lxtrue).^2))/max(max((lxtrue).^2));
ely=max(max((lymesh1-lytrue).^2))/max(max((lytrue).^2));
el=max(max((lxmesh1-lxtrue).^2+(lymesh1-lytrue).^2))/max(max((lxtrue).^2+(lytrue).^2));


%% characteristic boundary
%% Computing prefactor via NN
H_bar=[2,0;0,alpha];
H_star=[-1,0;0,alpha];
lambda_star=1;
h=0.001;
T=10;
Nstep=floor(T/h);
dx=xlin(2)-xlin(1);
dy=ylin(2)-ylin(1);

global alpha beta X Y Ux Uy
X=xmesh1;
Y=ymesh1;
[Ux,Uy]=gradient(Smesh1,dx,dy);
x_star=[0;0];
xnode=[-1;0];
delta=0.05;
t=-h;
x0=x_star-[delta;0];
MPEP=x0;

for i=1:Nstep
    x1=rk4(0,h,x0);
    if norm(x1-xnode)<=0.2*delta
        break;
    end
    x0=x1;
    MPEP=[x0 MPEP];
    t=[-(i+1)*h,t];
end
%MPEP=[xnode MPEP];
figure;
plot(MPEP(1,:),MPEP(2,:),'m-');

[L1_1,L1_2]=gradient(lxmesh1,dx,dy);
[L2_1,L2_2]=gradient(lymesh1,dx,dy);
Np=length(t);
divL=[];
normbx=[];
for i=1:Np
    x=MPEP(:,i);
    divL=[interp2(X,Y,L1_1,x(1),x(2),'linear')+interp2(X,Y,L2_2,x(1),x(2),'linear'),divL];
    normbx=[norm([[x(1)-x(1)^3-alpha*beta*x(1)*x(2);beta*(x(1)^4-x(1)^2)-alpha*x(2)]]),normbx];
end
integ=(sum(divL./normbx)-0.5*(divL(1)/normbx(1)+divL(end)/normbx(end)))*h;
prefactor=pi/lambda_star*sqrt(abs(det(H_star)))/sqrt(abs(det(H_bar)))*exp(integ);

%% Computing true prefactor
[Ux,Uy]=gradient(Strue,dx,dy);
delta=0.05;
t_true=-h;
x0=x_star-[delta;0];
MPEP_true=x0;

for i=1:Nstep
    x1=rk4(0,h,x0);
    if norm(x1-xnode)<=0.2*delta
        break;
    end
    x0=x1;
    MPEP_true=[x0 MPEP_true];
    t_true=[-(i+1)*h,t_true];
end
%MPEP=[xnode MPEP];
figure;
plot(MPEP_true(1,:),MPEP_true(2,:),'m-');

[L1_1_true,L1_2_true]=gradient(lxtrue,dx,dy);
[L2_1_true,L2_2_true]=gradient(lytrue,dx,dy);
Np_true=length(t_true);
divL_true=[];
normbx_true=[];
for i=1:Np_true
    x=MPEP_true(:,i);
    divL_true=[interp2(X,Y,L1_1_true,x(1),x(2),'linear')+interp2(X,Y,L2_2_true,x(1),x(2),'linear'),divL_true];
    normbx_true=[norm([[x(1)-x(1)^3-alpha*beta*x(1)*x(2);beta*(x(1)^4-x(1)^2)-alpha*x(2)]]),normbx_true];
end
integ_true=(sum(divL_true./normbx_true)-0.5*(divL_true(1)/normbx_true(1)+divL_true(end)/normbx_true(end)))*h;
prefactor_true=pi/lambda_star*sqrt(abs(det(H_star)))/sqrt(abs(det(H_bar)))*exp(integ_true);

%% computing mean exit time
epsilon_inv=linspace(5,25,1000);
MET_true=1.0121*exp(0.5081*epsilon_inv);
ep_inv=[5,10,15,20,25];
MET_MC=[16.8220,193.3117,2050.8,25739,307330];
figure;
plot(epsilon_inv,MET_true);
figure;
plot(ep_inv,MET_MC,'r*');


%% non-characteristic boundary
%% Computing prefactor via NN
% epsilon=1/25;
xb=-0.5;
[m,Ix]=min(abs(xlin-xb));
[m,Iy]=min(Smesh1(:,Ix));
x_star=[xlin(Ix);ylin(Iy)];
[Ux,Uy]=gradient(Smesh1,dx,dy);
miu_star=Ux(Iy,Ix);
[Uxy,Uyy]=gradient(Uy,dx,dy);
deth_star=Uyy(Iy,Ix);
delta=0.05;
t2=-h;
x0=x_star;
MPEP2=x0;

for i=1:Nstep
    x1=rk4(0,h,x0);
    if norm(x1-xnode)<=0.2*delta
        break;
    end
    x0=x1;
    MPEP2=[x0 MPEP2];
    t2=[-(i+1)*h,t2];
end
%MPEP=[xnode MPEP];
figure;
plot(MPEP2(1,:),MPEP2(2,:),'m-');

Np2=length(t2);
divL2=[];
normbx2=[];
for i=1:Np2
    x=MPEP2(:,i);
    divL2=[interp2(X,Y,L1_1,x(1),x(2),'linear')+interp2(X,Y,L2_2,x(1),x(2),'linear'),divL2];
    normbx2=[norm([[x(1)-x(1)^3-alpha*beta*x(1)*x(2);beta*(x(1)^4-x(1)^2)-alpha*x(2)]]),normbx2];
end
integ2=(sum(divL2./normbx2)-0.5*(divL2(1)/normbx2(1)+divL2(end)/normbx2(end)))*h;
prefactor2=1/miu_star*sqrt(2*pi*deth_star)/sqrt(abs(det(H_bar)))*exp(integ2);

%% Computing true prefactor
[Ux,Uy]=gradient(Strue,dx,dy);
miu2_star=Ux(Iy,Ix);
[Uxy,Uyy]=gradient(Uy,dx,dy);
deth2_star=Uyy(Iy,Ix);
delta=0.05;
t2_true=-h;
x0=x_star;
MPEP2_true=x0;

for i=1:Nstep
    x1=rk4(0,h,x0);
    if norm(x1-xnode)<=0.2*delta
        break;
    end
    x0=x1;
    MPEP2_true=[x0 MPEP2_true];
    t2_true=[-(i+1)*h,t2_true];
end
%MPEP=[xnode MPEP];
figure;
plot(MPEP2_true(1,:),MPEP2_true(2,:),'m-');

[L1_1_true,L1_2_true]=gradient(lxtrue,dx,dy);
[L2_1_true,L2_2_true]=gradient(lytrue,dx,dy);
Np2_true=length(t2_true);
divL2_true=[];
normbx2_true=[];
for i=1:Np2_true
    x=MPEP2_true(:,i);
    divL2_true=[interp2(X,Y,L1_1_true,x(1),x(2),'linear')+interp2(X,Y,L2_2_true,x(1),x(2),'linear'),divL2_true];
    normbx2_true=[norm([[x(1)-x(1)^3-alpha*beta*x(1)*x(2);beta*(x(1)^4-x(1)^2)-alpha*x(2)]]),normbx2_true];
end
integ2_true=(sum(divL2_true./normbx2_true)-0.5*(divL2_true(1)/normbx2_true(1)+divL2_true(end)/normbx2_true(end)))*h;
prefactor2_true=1/miu2_star*sqrt(2*pi*deth2_star)/sqrt(abs(det(H_bar)))*exp(integ2_true);

%% computing mean exit time
epsilon_inv=linspace(15,45,1000);
MET2_true=2.1525./sqrt(epsilon_inv).*exp(0.2835*epsilon_inv);
ep_inv=[15,20,25,30,35,40,45];
MET2_MC=[36.6181,134.7741,483.7051,1915.4,7216.5,28149,106500];
figure;
plot(epsilon_inv,MET2_true);
figure;
plot(ep_inv,MET2_MC,'r*');
