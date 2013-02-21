function [L] = laplacianFromBmDump(out, imW, imH)

N = imW*imH;
d = zeros(1,17);
d(8+1) = 0;
d(8+2) = 1;
d(8+3) = 2;
d(8+4) = imW;
d(8+5) = imW+1;
d(8+6) = imW+2;
d(8+7) = 2*imW;
d(8+8) = 2*imW+1;
d(8+9) = 2*imW+2;
d(8-0) = -1;
d(8-1) = -2;
d(8-2) = -imW;
d(8-3) = -(imW+1);
d(8-4) = -(imW+2);
d(8-5) = -(2*imW);
d(8-6) = -(2*imW+1);
d(8-7) = -(2*imW+2);

i = repmat([1:N]',[1,17]);
j = i + repmat(d,[N,1]);
j(j <= 0) = 1;
j(j > N) = 1;
L = sparse(i,j,out',N,N);

end