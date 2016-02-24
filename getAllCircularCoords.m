init;
cases = {'2SinesSlidingCommensurate', 'BeatingHeartSynthetic', 'JumpingJacks', 'SubtleMotion'};
%Need to do '2SinesSlidingNoncommensurate' by itself
for ii = 1:length(cases)
    s = cases{ii};
    fprintf(1, 'Doing %s...\n', s);
    Y = load(sprintf('%s/Y.txt', s));
    %Compute homology diagram, use to estimate cutoff
    [I, J] = rca1pc(Y, 1e9);
    [~, idx] = max(I(:, 2)-I(:, 1));
    cutoff = I(idx, 2);
    fprintf(1, 'Cutoff = %g\n', cutoff);
    system(sprintf('./rips-pairwise-cohomology %s/Y.txt -m %g -b %s/Y.bdry -c %s/Y -v %s/Y.vrt -d %s/Y.dgm', s, cutoff, s, s, s, s));
    %Load first cohomology diagram
    I = load(sprintf('%s/Y.dgm', s));
    I = I(I(:, 1) == 1, :);
    I = I(:, 2:3);
    [~, idx] = max(I(:, 2)-I(:, 1));
    if ~isinf(I(idx, 2))
        cutoff = I(idx, 2) - 0.001;
    else
        [I, ~] = rca1pc(Y, 1e9);
    end
    fprintf(1, '--------------------\n\n\nNew cutoff = %g\n', cutoff);
    system(sprintf('./rips-pairwise-cohomology %s/Y.txt -m %g -b %s/Y.bdry -c %s/Y -v %s/Y.vrt -d %s/Y.dgm', s, cutoff, s, s, s, s));
    system(sprintf('python cocycle.py %s/Y.bdry %s/Y-0.ccl %s/Y.vrt', s, s, s));
    circCoords = load(sprintf('%s/Y-0.val', s));
    circCoords = circCoords(:, 2);
    subplot(121);
    plotpersistencediagram(I);
    subplot(122);
    plot(circCoords);
    print('-dsvg', '-r100', sprintf('%s/PDCircCoords.svg', s));
    save(sprintf('%s/PDCircCoords.mat', s), 'I', 'circCoords');
end

%Prototype of code for outputting paper figures
frame1 = imread('BeatingHeartSynthetic/stack.png');
PCA = imread('BeatingHeartSynthetic/PCA.png');
subplot(141);
imagesc(frame1);
title('Sliding Window');
axis off;
subplot(142);
imagesc(PCA);
axis off;
title('3D PCA: 27.9% Variance Explained');
subplot(143);
plotpersistencediagram(I);
xlabel('Birth Time');
ylabel('Death Time');
title('1D Persistence Diagram');
subplot(144);
plot(circCoords);
xlabel('Frame Number');
ylabel('Circular Coordinate');
title('Cohomology Circular Coordinates');
print('-dsvg', '-r100', sprintf('%s/PDCircCoords.svg', s));