function y=fun(~,x)

global alpha beta X Y Ux Uy

bx=[x(1)-x(1)^3-alpha*beta*x(1)*x(2);beta*(x(1)^4-x(1)^2)-alpha*x(2)];
y=zeros(size(x));
y(1)=-(bx(1)+interp2(X,Y,Ux,x(1),x(2),'linear'))/norm(bx);
y(2)=-(bx(2)+interp2(X,Y,Uy,x(1),x(2),'linear'))/norm(bx);

