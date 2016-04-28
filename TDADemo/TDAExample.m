init;

res = 2*pi/10;

c1 = [0, 0];
R1 = 2;
t1 = 0:res/R1:1.7*pi;
X1 = bsxfun(@plus, c1, R1*[cos(t1(:)) sin(t1(:))]);

R2 = 1;
c2 = [3 3];
t2 = 0:res/R2:2*pi;
X2 = bsxfun(@plus, c2, R2*[cos(t2(:)) sin(t2(:))]);

X = [X1; X2];
X = X + 0.06*randn(size(X));

D = squareform(pdist(X));
[II, JJ] = meshgrid(1:size(D, 1), 1:size(D, 2));
D(II <= JJ) = inf;
[~, idx] = sort(D(:));

[I, J] = rca1pc(X, 1e9);

NEdges = sum(D(:) <= max(I(:)));

%Enumerate triangle indices
[A, B, C] = ndgrid(1:size(D, 1), 1:size(D, 1), 1:size(D, 1));
idxp = ((A < B).*(A < C) == 1);
A = A(idxp);
B = B(idxp);
C = C(idxp);

plotidx = 1;
for ii = 1:NEdges
    clf;
    subplot(121);
    plot(X(:, 1), X(:, 2), '.');
    title('Point Cloud');
    hold on;
    cutoff = D(idx(ii));
    
    %Plot triangles
    for kk = 1:length(A)
        if (D(A(kk), B(kk)) <= cutoff) & (D(A(kk), C(kk)) <= cutoff) & (D(A(kk), C(kk)) <= cutoff)
            Ps = X([A(kk), B(kk), C(kk)], :);
            h = fill(Ps(:, 1), Ps(:, 2), [0, 0.8, 0]);
            set(h,'facealpha',.1);
        end
    end
    if (sum(I(:, 1) == cutoff) > 0)
        text(-2, 4, 'Birth!');
    elseif sum(I(:, 2) == cutoff) > 0
        text(-2, 4, 'Death!');
    end
    
    i = II(idx(1:ii));
    j = JJ(idx(1:ii));
    for kk = 1:length(i)
        color = 'b';
        lw = 1;
        if kk == length(i)
            color = 'r';
            lw = 3;
        end
        plot(X([i(kk), j(kk)], 1), X([i(kk), j(kk)], 2), color, 'LineWidth', lw);
    end

    
    
    xlim([-2.5, 4.5]);
    ylim([-2.5, 4.5]);
    subplot(122);
    
    thisI = I(I(:, 1) <= cutoff, :);
    thisI(thisI > cutoff) = cutoff;
    
    M=max(I(:));
    m=min(I(:));
    M=max(M,0);
    m=min(m,0);  % Ensure that the plot includes the origin.
    diagonal=linspace(1.2*m,1.2*M,2);
    plot([m, M], [m, M], 'r');
    axis([1.2*m 1.2*M 1.2*m 1.2*M]); % set axes to include all points, with a bit of space on both sides
    hold on;
    if ~isempty(thisI)
        scatter(thisI(:, 1), thisI(:, 2), 20, 'b', 'fill');
        for kk = 1:size(thisI, 1)
            if thisI(kk, 1) == cutoff || thisI(kk, 2) == cutoff
                scatter(thisI(kk, 1), thisI(kk, 2), 40, 'r', 'fill');
            end
        end
    end
    xlabel('Birth Time');
    ylabel('Death Time');
    title('Persistence Diagram');
    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 12 6])
    if sum(I(:) == cutoff) > 0
        for kk = 1:20
            print('-dpng', '-r100', sprintf('%i.png', plotidx));
            plotidx = plotidx + 1;
        end
    end
    print('-dpng', '-r100', sprintf('%i.png', plotidx));
    plotidx = plotidx + 1;
end