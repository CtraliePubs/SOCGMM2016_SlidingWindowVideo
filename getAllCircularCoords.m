init;
cases = {'2SinesSlidingCommensurate', '2SinesSlidingNoncommensurate', 'BeatingHeartSynthetic', 'JumpingJacks', 'SubtleMotion'};
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
    %Load cohomology diagram
    I = load(sprintf('%s/Y.dgm', s));
    I = I(:, 2:3);
    [~, idx] = max(I(:, 2)-I(:, 1));
    cutoff = I(idx, 2) - 0.1;
    fprintf(1, 'New cutoff = %g\n', cutoff);
    system(sprintf('./rips-pairwise-cohomology %s/Y.txt -m %g -b %s/Y.bdry -c %s/Y -v %s/Y.vrt -d %s/Y.dgm', s, cutoff, s, s, s, s));
    system(sprintf('python cocycle.py %s/Y.bdry %s/Y-0.ccl %s/Y.vrt', s, s, s));
    circCoords = load(sprintf('%s/Y-0.ccl', s));
    subplot(121);
    plotpersistencediagram(I);
    subplot(122);
    plot(circCoords);
    print('-dsvg', '-r100', sprintf('%s/PDCircCoords.svg', s));
    save(sprintf('%s/PDCircCoords.mat', s), 'I', 'circCoords');
end