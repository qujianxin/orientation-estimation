% load('moredata.mat', 'result')
rsrp_all=[];

for m=1:36
    rsrp=[];
    for i=1:10
        for j=1:1000
            rsrp_each_ue = result(10*(m-1)+i).mreport(j).average_rsrp_w__tx_t;
            rsrp=[rsrp rsrp_each_ue];
        end
        10*(m-1)+i
    end
    rsrp_all=[rsrp_all rsrp];
end
rsrp_all = rsrp_all.';
%%
isLos_all=[];

for m=1:36
    isLos=[];
    for i=1:10
        for j=1:1000
            isLos_each_ue = result(10*(m-1)+i).mreport(j).is_los__tx_t;
            isLos=[isLos isLos_each_ue];
        end
        10*(m-1)+i
    end
    isLos_all=[isLos_all isLos];
end
isLos_all = isLos_all.';
%%
orientation=[];

for i=1:360
    orientation_each_ue = repelem(i-1,1000).';
    orientation=[orientation; orientation_each_ue];
end

%%
position_all=[];

for m=1:36
    position=[];
    for i=1:10
        for j=1:1000
            position_each_ue = result(10*(m-1)+i).receiverinfo(j).apl_node_idx;
            position=[position position_each_ue];
        end
        10*(m-1)+i
    end
    position_all=[position_all position];
end
position_all = position_all.';
%% scatter plot positions
p_all=[];

for m=1:36
    p=[];
    for i=1:10
        for j=1:1000
            p_each_ue = result(10*(m-1)+i).receiverinfo(j).position_m;
            p=[p p_each_ue];
        end
        10*(m-1)+i
    end
    p_all=[p_all p];
end
p_all = p_all.';

scatter(p_all(:,1), p_all(:,2))
%%
rsrp_all(isnan(rsrp_all))=0;
numelements = round(0.9*length(rsrp_all));
% get the randomly-selected indices
indices = randperm(length(rsrp_all));

train.indices = indices(1:numelements);
train.rsrp = rsrp_all(train.indices,:);
train.orientation=orientation(train.indices);
train.isLos = isLos_all(train.indices,:);


test.indices = indices(numelements+1:end);
test.rsrp = rsrp_all(test.indices,:);
test.orientation=orientation(test.indices);
test.isLos = isLos_all(test.indices,:);
%%
test.xyposition = p_all(test.indices,:);
train.xyposition = p_all(train.indices,:);
%%
train.position = position_all(train.indices,:);
test.position = position_all(test.indices,:);
figure(10)
subplot(2,1,1)
histogram(train.position, 1000)
title('Histogram of training UE position')
xlabel('apl_node_idx')
ylabel('count')

subplot(2,1,2)
histogram(train.orientation, 360)
title('Histogram of training UE degree')
xlabel('degree')
ylabel('count')
%% KNN classifier
K=3;
tr = [train.xyposition(:, 1:2) train.rsrp];
te = [test.xyposition(:, 1:2) test.rsrp];
Mdl = fitcknn(tr,train.orientation,'NumNeighbors',K);
test.predict = predict(Mdl,te);
[test.RMSE, test.error] = cal_RMSE(test.orientation, test.predict);

%% KNN Regression
RMSEs = [];
for K=1:5
    [test.neighborsIdx,~] = knnsearch([train.xyposition(:, 1:2) train.rsrp],[test.xyposition(:, 1:2) test.rsrp],'K',K,'Distance','euclidean');
    
    largeErrorIdx=[];
    test.neighborsDegree = [];
    for i=1:length(test.neighborsIdx)
        idx = test.neighborsIdx(i,:);
        neighborsDegree = train.orientation(idx);
        test.regressionPredict(i) = mean(neighborsDegree);
        test.neighborsDegree = [test.neighborsDegree; neighborsDegree.'];
        diffs= abs(diff(neighborsDegree));
        maxdiff=max( diffs );
        if maxdiff > 300
            largeErrorIdx= [largeErrorIdx i];
            test.regressionPredict(i) = neighborsDegree(1);
        end

    end
    [test.RMSE2, test.error2] = cal_RMSE(test.orientation, test.regressionPredict);
    RMSEs = [RMSEs, test.RMSE2];
end

%% Weighted neighbors

RMSEs = [];
K=3;
tr = [train.rsrp];
te = [test.rsrp];

[test.neighborsIdx,test.neighborsDistance] = knnsearch(tr,te,'K',K,'Distance','euclidean');
%%
test.neighborsDegree = train.orientation(test.neighborsIdx(:,:));
test.neighborsWeight = 1./ test.neighborsDistance;
test.neighborsWeight = test.neighborsWeight ./ sum(test.neighborsWeight, 2);
test.regressionPredictWeighted = sum(test.neighborsWeight .* test.neighborsDegree, 2);

for i=1:length(test.neighborsIdx)
    diffs= abs(diff(test.neighborsDegree(i, :) ));
    maxdiff=max( diffs );
    if maxdiff > 330
        test.regressionPredictWeighted(i) = test.neighborsDegree(i, 1);
    end
end

[test.RMSE2, test.error2] = cal_RMSE(test.orientation, test.regressionPredictWeighted);

%% Average neighbors
RMSEs = [];
K=3;
tr = [train.xyposition(:, 1:2) train.rsrp];
te = [test.xyposition(:, 1:2) test.rsrp];

[test.neighborsIdx,test.neighborsDistance] = knnsearch(tr, te, 'K',K,'Distance','euclidean');
test.neighborsDegree = train.orientation(test.neighborsIdx(:,:));
test.regressionPredict = mean(test.neighborsDegree, 2);


%%

largeErrorIdx=[];
for i=1:length(test.neighborsIdx)
    diffs= abs(diff(test.neighborsDegree(i, :)));
    maxdiff=max( diffs );
    if maxdiff > 300
        largeErrorIdx= [largeErrorIdx i];
        test.regressionPredict(i) = test.neighborsDegree(i, 1);
    end

end
[test.RMSE2, test.error2] = cal_RMSE(test.orientation, test.regressionPredict);
RMSEs = [RMSEs test.RMSE2]

%%

test.neighborsPosition = train.xyposition(test.neighborsIdx, 1:2);
test.neighborsPosition  = [test.neighborsPosition(1:36000,:) test.neighborsPosition(36001:72000,:) test.neighborsPosition(72001:end,:)];
%%
test.diff = abs(test.error2);
realDegreeAndDiff = [test.orientation test.diff sum(test.isLos,2) sum(~test.rsrp, 2)];
% realDegreeAndDiff = [];
% realDegreeAndDiff = [test.orientation test.diff test.neighborsDegree test.xyposition(:, 1:2) test.neighborsPosition];
% realDegreeAndDiff = [test.orientation test.diff test.xyposition];
[~, order] = sort(realDegreeAndDiff(:, 2));
realDegreeAndDiff = realDegreeAndDiff(order,:);
%%
temp = realDegreeAndDiff(32400:end,:);
figure(1)
cdfplot(realDegreeAndDiff(:,2))
title('CDF of error degrees after modification')
xlabel('errors')
ylabel('CDF')

%%

figure(2)
subplot(2,1,1)
histogram(temp(:,1), 360)
title('Histogram of real orientation degrees of bad performing cases')
xlabel('real degree')
ylabel('count')

subplot(2,1,2)
plot(32400:36000, temp(:,2))
title('Error degree of bad performing cases')
xlabel('index')
ylabel('errors')
%%

figure(3)
subplot(3,1,1)
plot(realDegreeAndDiff(:,3))
title('LoS number for sorted UE')
xlabel('UE')
ylabel('Count')

subplot(3,1,2)
plot(smoothdata(realDegreeAndDiff(:,3), 'gaussian', 200))
title('Smoothing LoS number for sorted UE')
xlabel('UE')
ylabel('Count')

subplot(3,1,3)
plot(realDegreeAndDiff(:,2))
title('Error degree for sorted UE')
xlabel('UE')
ylabel('degree')

%%

figure(4)
subplot(3,1,1)
plot(realDegreeAndDiff(:,4))
title('NaN number for sorted UE')
xlabel('UE')
ylabel('Count')

subplot(3,1,2)
plot(smoothdata(realDegreeAndDiff(:,4), 'gaussian', 300))
title('Smoothing NaN number for sorted UE')
xlabel('UE')
ylabel('Count')

subplot(3,1,3)
plot(realDegreeAndDiff(:,2))
title('Error degree for sorted UE')
xlabel('UE')
ylabel('degree')

%%
scatter(p_all(:,1), p_all(:,2), [], 'yellow')
hold on
scatter(realDegreeAndDiff(:,3), realDegreeAndDiff(:,4), [], 'blue')
hold on
scatter(realDegreeAndDiff(33000:end,3), realDegreeAndDiff(33000:end,4), [], 'red')
title('Bad performance UE location')
xlabel('x')
ylabel('y')
hold off


function [RMSE,err] = cal_RMSE(y, yhat)
    error = y - yhat;    % Errors
    err = arrayfun(@(x) mod((x + 180),360) - 180,error);
    %error = mod((error + 180),360) - 180;
    RMSE = sqrt(mean(err.^2));  % Root Mean Squared Error

end
 












