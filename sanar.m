function x_out = sanar(x_detrend, artifact_loc, N, fs, ...
    BFORDER, CUTOFF, SHRINKAGE, DIFFUSION)

% Function to implement Shape Adaptive Nonlocal Artifact Removal algorithm
% Ref: Alagapan et al. 2018, J Neural Engineering
% Author: Hau-tieng Wu, Duke University
% Dependencies:
%       OpenTSTool
%       scaleBeatHeights, findEstimate

%==========================================================================
% Inputs:
%       x_detrend     - detrended signal
%       artifact_loc  - artifact locations
%       N             - number of nearest neighbors (~ 30)
%       fs            - sampling rate
%       BFORDER       - Butterworth filter order
%       CUTOFF        - Butterworth filter cutoff
%       SHRINKAGE     - 0 or 1 - to use singular value optimal shrinkage
%       DIFFUSION     - 0 or 1 - to use diffusion distance
%
% Outputs:
%       x_out        - output signal
%==========================================================================

x_out = zeros(size(x_detrend));

L =  round( median( (artifact_loc(2:end) - artifact_loc(1:end-1)) )*3/8 )  ;
BT = 5 ;

if artifact_loc(1) < L
    artifact_loc = artifact_loc(2:end) ;
end

if artifact_loc(end)+ L > length(length(x_detrend))
    artifact_loc = artifact_loc(1:end-1) ;
end


% Highpass filter the signal
%=============================================
% You can well replace this high pass filter step by subtracting the original signal
% by the ICA reconstructed EEG signal. The goal is to get a "clean underlying
% artifact" to determine the neighbors so that the EEG signal is better preserved

[b_hp,a_hp] = butter(BFORDER, CUTOFF/(fs/2),'high');
x_highpass = filtfilt(b_hp,a_hp, x_detrend);
x_highpass_smooth = smooth(x_highpass, 'loess', fs*0.005) ;


% Get the height and widths of the artifacts
%=============================================
n_artifacts = length(artifact_loc) ;
fprintf(['\tOnly ',num2str(n_artifacts),' beats\n']) ;
artifacts = zeros(2*L+1, n_artifacts) ;
artifacts_hp = zeros(2*L+1, n_artifacts) ;
artifacts_hp_smooth = zeros(2*L+1, n_artifacts) ;
artifacts_idx = zeros(2*L+1, n_artifacts) ;
artifacts_height = zeros(1, n_artifacts) ;
artifacts_width = zeros(1, n_artifacts) ;

x_dummy = zeros(size(x_detrend)) ;

fprintf('\tPreparing dataset...\n') ;
for i_artifact = 1:n_artifacts
    idx = artifact_loc(i_artifact)-L : artifact_loc(i_artifact) + L  ;
    artifacts(:,i_artifact) = x_detrend(idx) ;
    artifacts_hp(:,i_artifact) = x_highpass(idx) ;
    artifacts_hp_smooth(:,i_artifact) = x_highpass_smooth(idx) ;
    artifacts_idx(:,i_artifact) = idx ;
    artifacts_height(:, i_artifact) = x_detrend(artifact_loc(i_artifact)) ;
    if i_artifact>1
        x_dummy(artifact_loc(i_artifact-1)+L+1:artifact_loc(i_artifact)-L-1) = ...
            x_detrend(artifact_loc(i_artifact-1)+L+1:artifact_loc(i_artifact)-L-1) ;
    end
end

for  i = 1:n_artifacts
    [~,~,tempWidth,temppkHeight] = findpeaks(artifacts_hp(:,i),...
        'widthreference','halfheight');
    [~,tempInd] = max(temppkHeight);
    if isempty(tempInd)
        gg=smooth(artifacts_hp(:,i), 110, 'loess');
        [~,~,tempWidth,temppkHeight] = findpeaks(artifacts_hp(:,i)-gg,...
            'widthreference','halfheight');
        [~,tempInd] = max(temppkHeight);
        
    end
    
    if tempInd==0
        artifacts_width(i) = median(artifacts_width(max(1,i-10):i-1)) ;
    else
        artifacts_width(i) = tempWidth(tempInd(1));
    end
end

%=============================================


% Run shrinkage and nearest neighbor estimation
%===============================================
fprintf('\tFinding neighbors...\n') ; t0=tic;
clear X ;

if SHRINKAGE
    fprintf('\t*** Use shrinkage..\n') ;
    sigma = std(x_dummy(x_dummy~=0)) ;
    beta = (2*L+1) ./ n_artifacts ;
    [u,l,v] = svd(artifacts_hp./sqrt(n_artifacts)./sigma);
    J = zeros(size(l));
    y = diag(l) ;
    eta = sqrt( (y.^2-beta-1).^2 - 4*beta) ./ y ;
    eta(y<=1+sqrt(beta)) = 0 ;
    tmp = min(size(J)) ; J(1:tmp,1:tmp) = diag(eta) ;
    Xc = (sigma*sqrt(n_artifacts))*u*J*v' ;
    X = Xc' ;
else
    fprintf('\t*** Use kernel smoothing..\n') ;
    X = artifacts_hp_smooth.';
end

% nn_prepare, nn_search are functions from openTStool
atria = nn_prepare(X);
[index,distance] = nn_search(X, atria, [1:n_artifacts].', N, -1, 0.0);
J1 = zeros(N,n_artifacts);
for k = 1 : N; J1(k,:) = 1:n_artifacts; end
J1 = J1(:);


J2 = index.'; J2 = J2(:);
DD = distance' ;
Z = sparse(J1, J2, DD(:), n_artifacts, n_artifacts, n_artifacts*N) ;
toc(t0)


% prepare data for diffusion distance
%===============================================
tmp = quantile(distance(:), .5) ;
QQ = 1./(1+(distance/tmp).^2) ;
QQ = QQ' ;
WW = sparse(J1, J2, QQ(:), n_artifacts, n_artifacts, n_artifacts*N) ;


% use height to segment artifacts
%===============================================
fprintf('\tAdjust neighbors...\n') ; t0=tic;
for i = 1:n_artifacts
    Zheight = artifacts_height(index(i,:)) - artifacts_height(i) ;
    idx = find(Zheight>2000) ;
    Z(i,index(i,idx)) = inf ;
    WW(i,index(i,idx)) = 0 ;
end


% Compute diffusion distance
%===============================================
if DIFFUSION
    WW = sparse(min(WW,WW')) ; DD = sum(WW,2);
    Dinv = sparse(1:length(DD),1:length(DD),1./DD);
    AA = Dinv*WW ;
    [UU,LL,VV] = svds(AA, 50, 'L') ;
    
    XX = UU*LL;
    atria = nn_prepare(XX);
    [indexD,distanceD] = nn_search(XX, atria, [1:n_artifacts].', N, -1, 0.0);
    J2 = indexD.'; J2 = J2(:); DDD = distanceD' ;
    Z = sparse(J1, J2, DDD(:), n_artifacts, n_artifacts, n_artifacts*N) ;
    toc(t0)
end


% prepare weight information for NLEM
%===============================================
ZW = zeros(N,n_artifacts) ; ZH = zeros(N,n_artifacts) ;
for i = 1:n_artifacts
    ZW(:,i) = abs(artifacts_width(index(i,:)) - artifacts_width(i)) ;
    ZH(:,i) = abs(artifacts_height(index(i,:)) - artifacts_height(i)) ;
end
Zwidth = sparse(J1, J2, ZW, n_artifacts, n_artifacts, n_artifacts*N) ;
Zheight = sparse(J1, J2, ZH, n_artifacts, n_artifacts, n_artifacts*N) ;
clear J1; clear J2;
toc(t0)




% Run NLEM
%===============================================
fprintf('\tRun NLEM...\n') ; t0=tic;
fprintf(['Total: ',num2str(n_artifacts), ' artifacts: ']) ;
fprintf('%05d',0) ;

xd0 = zeros(size(artifacts_idx)) ;
for i = 1:n_artifacts
    if ~mod(i,100) ; fprintf('\b\b\b\b\b') ; fprintf(['%05d'],i) ; end
    
    % don't take the artifact itself into account
    tmp = Z(i,:) ; tmp(find(tmp==0)) = inf ;
    z = sort(tmp,'ascend');
    th = z(N-1) ;
    thTemp(i) = th;
    if isinf(th)
        Nidx = find(~isinf(tmp)) ;
    else
        % don't count the artifact itself to avoid oversmoothing
        Nidx = find(tmp<=th) ;
    end
    
    Vx = artifacts(:,Nidx);
    if quantile(Zwidth(i,Nidx),0.95)>0
        w = exp(-Z(i, Nidx)/median(Z(i, Nidx))) .* exp(-Zwidth(i, Nidx)/quantile(Zwidth(i,Nidx), .95)) ;
    else
        w = exp(-Z(i, Nidx)/median(Z(i, Nidx))) ;
    end
    
    % Scale the height of the artifacts to that of neighbors
    VxScaled = scaleBeatHeights(Vx, artifacts_height(i), artifacts_height(Nidx));
    m = findEstimate (VxScaled, w', 1) ;
    
    % Section for taking care of 60 Hz noise - not needed in general case
    
    %     if i == 1 % first artifact
    %         left_overlap = 2;
    %         right_overlap = length(intersect(artifacts_idx(:,i+1),artifacts_idx(:,i)));
    %     elseif i == n_artifacts % last artifact
    %         left_overlap = length(intersect(artifacts_idx(:,i-1),artifacts_idx(:,i)));
    %         right_overlap = 2;
    %     else % all the other artifacts
    %         left_overlap = length(intersect(artifacts_idx(:,i-1),artifacts_idx(:,i)));
    %         right_overlap = length(intersect(artifacts_idx(:,i+1),artifacts_idx(:,i)));
    %     end
    %
    %     if left_overlap <= 1; left_overlap = BT; end
    %     if right_overlap <= 1; right_overlap = BT; end
    %
    %     W = ones(2*L+1,1);
    %     W(1:left_overlap) = sin(linspace(0,pi/2,left_overlap)).^2;
    %     W(end:-1:end-right_overlap+1) = sin(linspace(0,pi/2,right_overlap)).^2;
    
    xd0(:, i) = W.*m ;
    
end
fprintf('\n') ;
for i = 1:n_artifacts
    x_out(artifacts_idx(:,i)) = x_out(artifacts_idx(:,i)) + xd0(:,i) ;
end


