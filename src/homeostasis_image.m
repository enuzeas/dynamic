icc = [0.0501 0.0733 0.0612];
corr_mean = [0.0465 0.0781 0.0628];
names = categorical({'A','B','C'});

fig = figure('Color',[0.03 0.04 0.06],'Position',[100 100 900 520]);

subplot(1,2,1)
b = bar(names, icc, 0.55);
b.FaceColor = 'flat';
b.CData = [0.18 0.37 0.76; 0.83 0.56 0.12; 0.18 0.62 0.35];
ylim([0 1])
yline(0.5,'--','weak threshold','Color',[0.8 0.8 0.8],'LabelHorizontalAlignment','left')
title('Homeostasis: ICC','Color',[0.84 0.88 0.93])
ylabel('ICC','Color',[0.55 0.62 0.72])
set(gca,'Color',[0.07 0.08 0.11],'XColor',[0.55 0.62 0.72],'YColor',[0.55 0.62 0.72])
grid on

subplot(1,2,2)
b = bar(names, corr_mean, 0.55);
b.FaceColor = 'flat';
b.CData = [0.18 0.37 0.76; 0.83 0.56 0.12; 0.18 0.62 0.35];
ylim([0 1])
yline(0.5,'--','weak threshold','Color',[0.8 0.8 0.8],'LabelHorizontalAlignment','left')
title('Repeat Similarity: mean r','Color',[0.84 0.88 0.93])
ylabel('correlation','Color',[0.55 0.62 0.72])
set(gca,'Color',[0.07 0.08 0.11],'XColor',[0.55 0.62 0.72],'YColor',[0.55 0.62 0.72])
grid on

sgtitle('Low Homeostasis in Wrist-Speed Repeats','Color',[0.84 0.88 0.93],'FontWeight','bold')
exportgraphics(fig,'homeostasis_matlab.png','Resolution',180)
