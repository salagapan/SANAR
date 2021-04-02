function   median  =  findEstimate (f, w, p)
%function for computing the weighted Euclidean median
% f is a matrix of size k x n, where k is the size
% of the patch, and n is the number of neighbors
%
% w is the weight vector of size n X 1
%
% median = arg min_f \sum_j w(j) (|| f(:, j) - g(:)||_2 )^p
%
% median is conputed using an IRLS-type alogrithm (first-order method)

% initialization
median = (f * w) / sum(w);
n    = length(w);
eps  = 1e-6;
err  = 1;

while err > 1e-4
    residuals = sum( (median * ones(1,n) - f).^2, 1);
    gamma     = 1./ (residuals + eps * ones(1,n)).^(1 - p / 2);
    ww        = gamma' .* w;
    f_hold    = (f * ww) / sum(ww);
    err       = norm(f_hold - median);
    %disp(err);
    median     = f_hold;
end