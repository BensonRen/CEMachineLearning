#This function is for testing and single plotting purposes
import plotAnalysis as PA
#PA.plotonerun('results/20190923_125934')
dir1 = 'results/LR_schedule/20190923_133229'
dir2 = 'results/LR_schedule/20190923_133210'

PA.plotonerun(dir1)
PA.plotonerun(dir2)
PA.compare2testacc(dir1,dir2,'with decay','without decay')
#PA.compare2testacc('results/BNcompare/20190923_130727','results/BNcompare/20190923_130746','with BN','No BN')
#PA.HeatMapBVL(plot_x_name = 'initial_lr', plot_y_name='train batch size',
#                        title = 'Accuracy heatmap for batch size and lr',HeatMap_dir = "results/hyper-swipe-with-BN/",
#                        feature_2_name = 'initial_lr', feature_1_name = 'train_batch_size')
#PA.HeatMapBVL(plot_x_name = 'Stopping Loss', plot_y_name='bv_loss',save_name = "HaetMap of stop point.png",
#                        title = 'Tandem spectra reconstrcution Loss comparison vs stopping point ',HeatMap_dir = "../swipe_stop_point_compare",
#                        feature_1_name = 'stop_threshold', feature_2_name = None)#,condense_tuple2len =False)
